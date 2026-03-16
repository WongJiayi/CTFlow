# Standard library imports
import argparse
import importlib
import inspect
import json
import logging
import math
import os
import shutil
from enum import Enum
from functools import partial

# Third-party library imports
import cv2
import diffusers
import imageio
import numpy as np
import omegaconf
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration, set_seed
from einops import rearrange
from safetensors.torch import load_file, save_file
from torch.nn.utils import parameters_to_vector, vector_to_parameters
from tqdm.auto import tqdm as tqdm_std

import wandb


class Scheduler(Enum):
    EDM = 0
    EULER = 1
    OTHER = 2


logger = get_logger(__name__, log_level="INFO")


### Instantiation helper functions ###


def parse_klass_arg(value, full_config):
    if isinstance(value, str) and "." in value:
        if value.startswith("${") and value.endswith("}"):
            try:
                value = omegaconf.OmegaConf.resolve(full_config)[value[2:-1]]
            except Exception as e:
                logger.error(f"Error resolving OmegaConf interpolation {value}: {e}")
                return None

        parts = value.split(".")
        for i in range(len(parts) - 1, 0, -1):
            module_name = ".".join(parts[:i])
            attr_name = parts[i]
            try:
                module = importlib.import_module(module_name)
                result = module
                for j in range(i, len(parts)):
                    result = getattr(result, parts[j])
                return result
            except ImportError:
                continue
            except AttributeError as e:
                logger.warning(
                    f"Warning: Could not resolve attribute {parts[j]} from {module_name}, error: {e}"
                )
                continue
        return value
    return value


def instantiate_class_from_config(config, *args, **kwargs):
    module_name, class_name = config.target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    klass = getattr(module, class_name)

    config = omegaconf.OmegaConf.to_container(config, resolve=True)
    conf_kwargs = {
        key: parse_klass_arg(value, config) for key, value in config["args"].items()
    }
    all_args = list(args)
    all_kwargs = {**conf_kwargs, **kwargs}

    instance = klass(*all_args, **all_kwargs)
    return instance


### Accelerator and logging setup ###


def setup_accelerator_and_logging(config, logger):
    logging_dir = os.path.join(config.output_dir, config.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=config.output_dir, logging_dir=logging_dir
    )
    accelerator = Accelerator(
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        mixed_precision=config.mixed_precision,
        log_with=config.report_to,
        project_config=accelerator_project_config,
        step_scheduler_with_optimizer=False,
    )

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)

    return accelerator, logger


def set_weight_dtype(accelerator, config):
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
        config.mixed_precision = accelerator.mixed_precision
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
        config.mixed_precision = accelerator.mixed_precision
    return weight_dtype, config


def init_trackers(accelerator, config, args):
    if accelerator.is_main_process:
        tracker_config = omegaconf.OmegaConf.to_container(config, resolve=True)
        wandb_args = omegaconf.OmegaConf.to_container(config.wandb_args, resolve=True)
        accelerator.init_trackers(
            project_name=wandb_args.pop("project"),
            config=tracker_config,
            init_kwargs={
                "wandb": {
                    **wandb_args,
                    "mode": "disabled" if args.no_wandb else "online",
                    "resume": "allow",
                }
            },
        )
        config.wandb_args.id = wandb.run.id


def log_training_info(config, accelerator, model, train_dataset, logger):
    total_batch_size = (
        config.dataloader.args.batch_size
        * accelerator.num_processes
        * config.gradient_accumulation_steps
    )
    model_num_params = sum(p.numel() for p in model.parameters())
    model_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset):_}")
    logger.info(f"  Num Epochs = {config.num_train_epochs:_}")
    logger.info(f"  Batch size per device = {config.dataloader.args.batch_size:_}")
    logger.info(f"  Total train batch size (w. parallel & accumulation) = {total_batch_size:_}")
    logger.info(f"  Gradient Accumulation steps = {config.gradient_accumulation_steps:_}")
    logger.info(f"  Total optimization steps = {config.max_train_steps:_}")
    logger.info(
        f"  Model: Total params = {model_num_params:_} \t Trainable params = {model_trainable_params:_} "
        f"({model_trainable_params/model_num_params*100:.2f}%)"
    )


### Checkpointing and model saving/loading ###


def load_checkpoint(config, accelerator, num_update_steps_per_epoch, ema_model=None):
    first_epoch = 0
    if config.resume_from_checkpoint:
        if config.resume_from_checkpoint != "latest":
            path = os.path.basename(config.resume_from_checkpoint)
        else:
            dirs = os.listdir(config.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(
                f"Checkpoint '{config.resume_from_checkpoint}' does not exist. Starting a new training run."
            )
            config.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(config.output_dir, path))
            if ema_model is not None:
                ema_path = os.path.join(config.output_dir, path, "denoiser_ema")
                model_cls = accelerator._models[0].__class__
                ema_tmp = ema_model.__class__.from_pretrained(
                    ema_path, model_cls=model_cls
                )
                ema_model.__dict__.update(ema_tmp.__dict__)

            global_step = int(path.split("-")[1])
            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
            run_config = omegaconf.OmegaConf.load(
                os.path.join(config.output_dir, "config.yaml")
            )
            config.wandb_args.id = run_config.wandb_args.id
    else:
        initial_global_step = 0

    return initial_global_step, first_epoch


def cleanup_checkpoints(config, logger):
    if config.get("checkpoints_total_limit", None) is not None:
        checkpoints = os.listdir(config.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))

        if len(checkpoints) >= config.checkpoints_total_limit:
            num_to_remove = len(checkpoints) - config.checkpoints_total_limit + 1
            removing_checkpoints = checkpoints[0:num_to_remove]
            logger.info(f"Removing checkpoints: {', '.join(removing_checkpoints)}")
            for removing_checkpoint in removing_checkpoints:
                shutil.rmtree(os.path.join(config.output_dir, removing_checkpoint))

    elif config.get("checkpoints_to_keep", None) is not None:
        checkpoints_to_keep = omegaconf.OmegaConf.to_container(
            config.checkpoints_to_keep
        )
        checkpoints_to_keep = [f"checkpoint-{c}" for c in checkpoints_to_keep]

        checkpoints = os.listdir(config.output_dir)
        checkpoints = [d for d in checkpoints if d.startswith("checkpoint")]
        checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
        if len(checkpoints) == 0:
            return

        last_checkpoint = checkpoints[-1]
        for checkpoint in checkpoints:
            if checkpoint not in checkpoints_to_keep and checkpoint != last_checkpoint:
                logger.info(f"Removing checkpoint: {checkpoint}")
                shutil.rmtree(os.path.join(config.output_dir, checkpoint))


def save_checkpoint(config, accelerator, logger, global_step, ema_model=None):
    save_path = os.path.join(config.output_dir, f"checkpoint-{global_step}")
    accelerator.save_state(save_path)
    if ema_model is not None:
        ema_model.save_pretrained(os.path.join(save_path, "denoiser_ema"))
    omegaconf.OmegaConf.save(config, os.path.join(config.output_dir, "config.yaml"))
    logger.info(f"Saved state to {save_path}")


def save_model_hook(models, weights, output_dir):
    for i, model in enumerate(models):
        has_saved = False
        for nn_name in ["net", "encoder", "backbone"]:
            if hasattr(model, nn_name):
                state_dict = getattr(model, nn_name).state_dict()
                save_file(
                    state_dict, os.path.join(output_dir, f"{nn_name}.safetensors")
                )
                has_saved = True
                break

        if not has_saved:
            model.save_pretrained(os.path.join(output_dir, "denoiser"))

        weights.pop()


def load_model_hook(models, input_dir):
    for i in range(len(models)):
        model = models.pop()
        has_loaded = False
        for nn_name in ["net", "encoder", "backbone"]:
            if hasattr(model, nn_name):
                state_dict = load_file(
                    os.path.join(input_dir, f"{nn_name}.safetensors")
                )
                getattr(model, nn_name).load_state_dict(state_dict)
                has_loaded = True
                break

        if not has_loaded:
            load_model = model.__class__.from_pretrained(
                os.path.join(input_dir, "denoiser")
            )
            model.register_to_config(**load_model.config)
            model.load_state_dict(load_model.state_dict())
            del load_model
            model.from_pretrained(os.path.join(input_dir, "denoiser"))


### Data loading ###


def cycle(dl):
    while True:
        for batch in dl:
            yield batch


### Visualization helpers ###


def save_as_mp4(tensor, filename, fps=30):
    """Save a (T, H, W, C) uint8 tensor as MP4."""
    np_video = tensor.cpu().numpy()
    if np_video.dtype != np.uint8:
        raise ValueError("Tensor must be uint8")
    with imageio.get_writer(filename, fps=fps) as writer:
        for i in range(np_video.shape[0]):
            writer.append_data(np_video[i])


def save_as_gif(tensor, filename, fps=30):
    """Save a (T, H, W, C) uint8 tensor as GIF."""
    np_video = tensor.cpu().numpy()
    if np_video.dtype != np.uint8:
        raise ValueError("Tensor must be uint8")
    imageio.mimsave(filename, np_video, fps=fps, loop=0)


def loadvideo(filename: str, return_fps=False):
    """Load a video file into a (T, 3, H, W) tensor."""
    if not os.path.exists(filename):
        raise FileNotFoundError(filename)
    capture = cv2.VideoCapture(filename)
    fps = capture.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(torch.from_numpy(frame))
    capture.release()
    frames = torch.stack(frames, dim=0)
    if return_fps:
        return frames, fps
    return frames


### Other helpers ###


def instantiate(config_og, return_klass_kwargs=False):
    config = omegaconf.OmegaConf.create(config_og)
    module_path, class_name = config.target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    if "pretrained" in config:
        pretrained = config.pop("pretrained")
        fkwargs = filter_kwargs_for_func(cls.from_pretrained, {"use_safetensor": True, "torch_dtype": torch.float32})
        if "," in pretrained:
            pretrained, subfolder = pretrained.split(",")
            obj = cls.from_pretrained(pretrained, subfolder=subfolder, **fkwargs)
        else:
            obj = cls.from_pretrained(pretrained, **fkwargs)
    elif "weights" in config:
        weights_path = config.pop("weights")
        weights = torch.load(weights_path)
        obj = cls(**omegaconf.OmegaConf.to_container(config.args))
        obj.load_state_dict(weights)
    else:
        obj = cls(**omegaconf.OmegaConf.to_container(config.args))

    if return_klass_kwargs:
        return obj, cls, omegaconf.OmegaConf.to_container(config.args)
    return obj


def filter_kwargs_for_func(func, kwargs):
    init_signature = inspect.signature(func)
    valid_keys = init_signature.parameters.keys()
    return {k: v for k, v in kwargs.items() if k in valid_keys}


def get_dtype(config):
    dtypes = {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }
    return dtypes.get(config.mixed_precision, torch.float32)


def get_vae_scaler(config, device):
    try:
        vae_config_path = os.path.join(config.vae.pretrained, "config.json")
        with open(vae_config_path, "r") as f:
            vae_config = json.load(f)
        scaler = {
            "std": torch.tensor(vae_config.get("scaling_factor", 1.0)),
            "mean": torch.tensor(vae_config.get("shift_factor", 0.0)),
        }
        print(f"Loaded VAE scaler: scaling={scaler['std']}, shift={scaler['mean']}")
    except Exception:
        scaler = {
            "mean": torch.tensor(0),
            "std": torch.tensor(1),
        }
        print("WARNING: VAE scaling file not found. Using default mean=0, std=1.")
    return {k: v.to(device) for k, v in scaler.items()}


def get_noise(latents, noise_scheduler=None, noise_offset=0.0):
    noise = torch.randn_like(latents)
    if noise_offset > 0.0:
        noise_offset_shape = (latents.shape[0], latents.shape[1]) + (1,) * (
            len(latents.shape) - 2
        )
        noise = noise + noise_offset * torch.randn(
            noise_offset_shape, device=latents.device
        )
    return noise


def sample_latents(config, latents):
    B, C, *_ = latents.shape
    if config.sample_latents and C == 2 * config.globals.latent_channels:
        mean, std = latents.chunk(2, dim=1)
        latents = torch.randn_like(mean) * std + mean
    else:
        latents = latents[:, : config.globals.latent_channels]
    return latents


def scale_latents(latents, vae_scaling=None):
    latents -= vae_scaling["mean"]
    latents *= vae_scaling["std"]
    return latents


def unscale_latents(latents, vae_scaling=None):
    latents /= vae_scaling["std"]
    latents += vae_scaling["mean"]
    return latents


### EMA ###


def get_ema_model(denoiser):
    ema_model = diffusers.training_utils.EMAModel(
        denoiser.parameters(),
        model_cls=denoiser.__class__,
        model_config=denoiser.config,
    )
    return ema_model


def prepare_forward_kwargs(config, denoiser, accelerator):
    forward_kwargs = {"timestep": None}
    if "class_labels" in inspect.signature(denoiser.forward).parameters:
        forward_kwargs["class_labels"] = torch.zeros(
            (config.dataloader.args.batch_size,), device=accelerator.device
        ).long()
    if "encoder_hidden_states" in inspect.signature(denoiser.forward).parameters:
        forward_kwargs["encoder_hidden_states"] = torch.zeros(
            (
                config.dataloader.args.batch_size,
                1,
                config.denoiser.args.get("joint_attention_dim", 1),
            ),
            device=accelerator.device,
        )
    return forward_kwargs


def initialize_weights(module):
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0, std=0.02)


def zero_nan_inf_all_grad(model):
    grads = []
    for p in model.parameters():
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        grads.append(p.grad)
    flat_grad = parameters_to_vector(grads)
    torch.nan_to_num_(flat_grad, nan=0.0, posinf=0.0, neginf=0.0)
    vector_to_parameters(flat_grad, model.parameters())


def tensor_stat(tensor):
    summary = {
        "shape": list(tensor.shape),
        "dtype": tensor.dtype,
        "device": str(tensor.device),
        "mean": tensor.mean().item() if tensor.numel() > 0 else None,
        "std": tensor.std().item() if tensor.numel() > 1 else None,
        "min": tensor.min().item() if tensor.numel() > 0 else None,
        "max": tensor.max().item() if tensor.numel() > 0 else None,
    }
    print("Tensor Summary:")
    for key, value in summary.items():
        print(f"{key:15}: {value}")
