import argparse
import logging
import math
import os
import types
import warnings
from copy import deepcopy

import accelerate
import diffusers
import numpy as np
import torch
import torch._dynamo
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import xformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from diffusers.training_utils import EMAModel
from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph
from omegaconf import OmegaConf
from torchdiffeq import odeint_adjoint as odeint
from tqdm.auto import tqdm
import moviepy.editor as mpy
import imageio

import wandb
from echosyn.common import *
from echosyn.common.schedulers import StepBasedLearningRateScheduleWithWarmup
from echosyn.common.datasets import instantiate_dataset


allow_ops_in_compiled_graph()
torch._dynamo.config.suppress_errors = True
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="resource_tracker")

logger = get_logger(__name__, log_level="INFO")


def parse_args():
    parser = argparse.ArgumentParser(description="Train Latent Video Flow Matching model")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    parser.add_argument("--no_wandb", action="store_true", default=False, help="Disable wandb logging.")
    return parser.parse_args()


def log_validation(config, maybe_ema_denoiser, accelerator, weight_dtype, val_dataset, step=None):
    logger.info("Running validation... ")

    val_vae = instantiate(config.vae).eval().to(accelerator.device)
    vae_scaling = get_vae_scaler(config, accelerator.device)

    val_denoiser = instantiate_class_from_config(config.denoiser).eval()
    if not ("UNetSTIC" in config.denoiser.target):
        val_denoiser.enable_xformers_memory_efficient_attention()

    if config.get("use_ema", False):
        maybe_ema_denoiser.copy_to(val_denoiser.parameters())
    else:
        with torch.no_grad():
            for param, eval_param in zip(
                maybe_ema_denoiser.parameters(), val_denoiser.parameters()
            ):
                eval_param.copy_(param)
    val_denoiser.to(accelerator.device, weight_dtype)

    generator = (
        torch.Generator(device=accelerator.device).manual_seed(config.seed)
        if config.seed is not None
        else None
    )

    indices = torch.randint(
        0, len(val_dataset), (config.validation.samples,), device=accelerator.device
    ).tolist()

    B, C, T, H, W = (
        config.validation.samples,
        config.globals.latent_channels,
        config.validation.frames,
        config.globals.latent_res,
        config.globals.latent_res,
    )

    ref_elements = [val_dataset[i] for i in indices]

    ref_images = torch.stack([e["image"] for e in ref_elements], dim=0)
    ref_images = ref_images.to(accelerator.device, dtype=weight_dtype)
    ref_images = sample_latents(config, ref_images)
    ref_images = scale_latents(ref_images, vae_scaling)

    ref_videos = torch.stack([e["video"] for e in ref_elements], dim=0)
    ref_videos = ref_videos.to(device=accelerator.device)
    ref_videos = sample_latents(config, ref_videos)

    text_embeddings = torch.stack([e["embedding"] for e in ref_elements], dim=0)
    text_embeddings = text_embeddings.to(accelerator.device, dtype=weight_dtype)

    logger.info("Sampling... ")
    with torch.no_grad(), accelerator.autocast():
        z_1 = torch.randn(
            (B, C, T, H, W),
            device=accelerator.device,
            dtype=weight_dtype,
            generator=generator,
        )

        timesteps = torch.tensor([1.0, 0.0], dtype=weight_dtype, device=accelerator.device)

        val_denoiser.forward_original = val_denoiser.forward

        def new_forward(self, t, y, *args, **kwargs):
            kwargs = {
                **kwargs,
                "encoder_hidden_states": text_embeddings,
                "cond_image": ref_images,
            }
            return self.forward_original(y, t, *args, **kwargs).sample

        val_denoiser.forward = types.MethodType(new_forward, val_denoiser)

        synthetic_video = odeint(
            val_denoiser,
            z_1,
            timesteps,
            atol=1e-5,
            rtol=1e-5,
            adjoint_params=val_denoiser.parameters(),
        )[-1]

    with torch.no_grad():
        synthetic_video = rearrange(synthetic_video, "b c t h w -> (b t) c h w")
        synthetic_video = unscale_latents(synthetic_video, vae_scaling)
        synthetic_video = val_vae.decode(synthetic_video.float()).sample
        synthetic_video = synthetic_video * 306
        synthetic_video = synthetic_video.clamp(0, 255).to(torch.uint8).cpu()
        synthetic_video = rearrange(synthetic_video, "(b t) c h w -> b c t h w", b=B)

        ref_images = rearrange(ref_images, "b c t h w -> (b t) c h w")
        ref_images = unscale_latents(ref_images, vae_scaling)
        ref_images = val_vae.decode(ref_images.float()).sample
        ref_images = ref_images * 306
        ref_images = ref_images.clamp(0, 255).to(torch.uint8).cpu()
        ref_images = rearrange(ref_images, "(b t) c h w -> b c t h w", b=B)

        ref_videos = rearrange(ref_videos, "b c t h w -> (b t) c h w")
        ref_videos = val_vae.decode(ref_videos.float()).sample
        ref_videos = ref_videos * 306
        ref_videos = ref_videos.clamp(0, 255).to(torch.uint8).cpu()
        ref_videos = rearrange(ref_videos, "(b t) c h w -> b c t h w", b=B)

        videos = torch.cat([ref_images, ref_videos, synthetic_video], dim=3)

    videos = rearrange(videos, "b c t h w -> t c h (b w)")
    if step is not None:
        fname = f"sample_{step:07d}.mp4"
        os.makedirs(os.path.join(config.output_dir, "samples"), exist_ok=True)
        save_as_mp4(
            rearrange(videos, "t c h w -> t h w c"),
            os.path.join(config.output_dir, "samples", fname),
        )
    videos = videos.numpy()

    logger.info("Done sampling.")
    for tracker in accelerator.trackers:
        if tracker.name == "wandb":
            tracker.log({
                "validation": wandb.Video(
                    videos,
                    caption="Validation samples",
                    fps=config.validation.fps,
                )
            })
            logger.info("Samples sent to wandb.")

    del val_denoiser
    del val_vae
    torch.cuda.empty_cache()

    return videos


def main():
    print("Code starting...")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("medium")

    args = parse_args()
    config = OmegaConf.load(args.config)

    logger = get_logger(__name__, log_level="INFO")
    accelerator, logger = setup_accelerator_and_logging(config, logger)
    print(f"Accelerator ready for process {accelerator.process_index}")

    # Models, optimizer, data
    denoiser = instantiate_class_from_config(config.denoiser).train()
    if not ("UNetSTIC" in config.denoiser.target):
        denoiser.enable_xformers_memory_efficient_attention()
    vae_scaling = get_vae_scaler(config, accelerator.device)
    clamp_v = config.get("clamp_v", float("inf"))

    ema_denoiser = get_ema_model(denoiser) if config.use_ema else None
    optimizer = instantiate_class_from_config(config.optimizer, denoiser.parameters())

    train_dataset = instantiate_dataset(config.datasets, split=["TRAIN"])
    val_dataset = instantiate_dataset(config.val_datasets, split=["VAL"])
    train_dataloader = instantiate_class_from_config(config.dataloader, train_dataset)

    denoiser, optimizer, train_dataloader = accelerator.prepare(
        denoiser, optimizer, train_dataloader
    )
    lr_scheduler = instantiate_class_from_config(config.scheduler, optimizer)

    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / config.gradient_accumulation_steps
    )
    config.num_train_epochs = math.ceil(
        config.max_train_steps / num_update_steps_per_epoch
    )
    dtype, config = set_weight_dtype(accelerator, config)

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # Resume or fine-tune
    initial_global_step, first_epoch = load_checkpoint(
        config, accelerator, num_update_steps_per_epoch, ema_denoiser
    )
    if config.get("use_ema", False):
        ema_denoiser.to(accelerator.device)

    if_ft = bool(config.get("if_fine_tuned", False))
    if if_ft:
        global_step = 0
        optimizer = instantiate_class_from_config(config.optimizer, denoiser.parameters())
        lr_scheduler = instantiate_class_from_config(config.scheduler, optimizer)
    else:
        global_step = initial_global_step

    if global_step == 0:
        set_seed(config.seed)

    init_trackers(accelerator, config, args)
    log_training_info(config, accelerator, denoiser, train_dataset, logger)

    forward_kwargs = prepare_forward_kwargs(config, denoiser, accelerator)
    if config.max_grad_norm < 0:
        config.max_grad_norm = float("inf")

    infinite_loader = cycle(train_dataloader)
    progress_bar = tqdm(
        range(0, config.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_main_process,
        dynamic_ncols=True,
    )

    # Training loop
    while True:
        batch = next(infinite_loader)
        with accelerator.accumulate(denoiser):

            images = sample_latents(config, batch["image"]).detach()
            images = scale_latents(images, vae_scaling)
            images = images.clamp(-clamp_v, clamp_v)

            videos = sample_latents(config, batch["video"]).detach()
            videos = scale_latents(videos, vae_scaling)
            videos = videos.clamp(-clamp_v, clamp_v)

            B, C, T, H, W = videos.shape

            text_embeddings = batch["embedding"]
            forward_kwargs["encoder_hidden_states"] = text_embeddings

            # Condition image: current block with optional noise and dropout
            if config.get("noise_cond_image", 0.0) > 0.0:
                images += config.noise_cond_image * torch.randn_like(images)

            images_mask = (
                torch.rand_like(images[:, 0:1, 0:1, 0:1]) > config.p_drop_conditionning
            ).float()
            images = images * images_mask + (1.0 - images_mask) * 0.0
            forward_kwargs["cond_image"] = images

            # Sample timesteps for flow matching
            t = torch.rand(B, device=accelerator.device, dtype=dtype)
            t = t.view(-1, 1, 1, 1, 1)
            forward_kwargs["timestep"] = t[:, 0, 0, 0, 0]

            # Flow matching interpolation
            z_0 = videos
            z_1 = get_noise(videos, noise_offset=config.get("noise_offset", 0.0))
            offset = 1e-5
            z_t = (1 - t) * z_0 + (offset + (1 - offset) * t) * z_1
            u = (1 - offset) * z_1 - z_0  # target velocity

            with accelerator.autocast():
                v = denoiser(z_t, **forward_kwargs).sample
                loss = F.mse_loss(v, u)

            accelerator.backward(loss)
            if accelerator.sync_gradients:
                grad_norm = accelerator.clip_grad_norm_(
                    denoiser.parameters(), config.max_grad_norm
                )
                if config.max_grad_value > 0:
                    accelerator.clip_grad_value_(
                        denoiser.parameters(), config.max_grad_value
                    )
                lr_scheduler.step()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            if accelerator.sync_gradients:
                if config.use_ema:
                    ema_denoiser.step(denoiser.parameters())
                progress_bar.update(1)
            if accelerator.sync_gradients and global_step % 10 == 0:
                accelerator.log(
                    {
                        "train_loss": loss.mean().item(),
                        "lr": lr_scheduler.get_last_lr()[0],
                        "grad_norm": grad_norm.item(),
                        "mean_latent": videos.mean().item(),
                        "std_latent": videos.std().item(),
                    },
                    step=global_step,
                )

            if (
                accelerator.sync_gradients
                and accelerator.is_main_process
                and global_step % config.validation.steps == 0
            ):
                log_validation(
                    config,
                    ema_denoiser or denoiser,
                    accelerator,
                    dtype,
                    val_dataset,
                    step=global_step,
                )

            if (
                accelerator.sync_gradients
                and accelerator.is_main_process
                and global_step % config.checkpointing_steps == 0
            ):
                cleanup_checkpoints(config, logger)
                save_checkpoint(config, accelerator, logger, global_step, ema_denoiser)

            if global_step >= config.max_train_steps:
                break

            if accelerator.sync_gradients:
                global_step += 1

    cleanup_checkpoints(config, logger)
    accelerator.end_training()


if __name__ == "__main__":
    main()
