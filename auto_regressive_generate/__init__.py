import time
import types

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torchdiffeq import odeint_adjoint as odeint

from echosyn.common import *


class LatentAutoregressiveGenerator:
    """
    Auto-regressive generator for latent video volumes.

    Generates a sequence of latent blocks conditioned on:
    - A text/CT embedding (prompt_embeds)
    - The previously generated block (cond_image)

    Uses flow matching (Euler ODE) for denoising.
    """

    def __init__(
        self,
        denoiser,
        vae,
        device,
        vae_scaling,
        config,
        block_size: int = 16,
        overlap: int = 8,
        stop_value: float = 1.0,
        eps: float = 0.1,
    ):
        self.denoiser = denoiser
        self.vae = vae
        self.device = device
        self.vae_scaling = vae_scaling
        self.config = config
        self.block_size = block_size
        self.overlap = overlap
        self.stop_value = stop_value
        self.eps = eps
        self.dtype = torch.float32
        self.trim = True

        # Pre-encode zero and one frames as boundary references
        self.zero_latent = self.encode_image(torch.zeros(1, 3, 256, 256))

        one_latent = self.encode_image(torch.ones(1, 3, 256, 256))
        one_latent = sample_latents(self.config, one_latent)
        one_latent = scale_latents(one_latent, vae_scaling)
        self.one_latent = one_latent

    def encode_image(self, img):
        with torch.no_grad():
            img = img.to(self.device).to(self.dtype)
            img = self.vae.encode(img).latent_dist.sample()
        return img

    def decode_latent(self, latents, max_batch_size=64):
        with torch.no_grad():
            latents = unscale_latents(latents, self.vae_scaling)
            b, c, t, h, w = latents.shape
            latents = rearrange(latents, "b c t h w -> (b t) c h w")

            decoded_chunks = []
            for chunk in latents.split(max_batch_size, dim=0):
                decoded = self.vae.decode(chunk.float()).sample
                decoded_chunks.append(decoded)

            latents = torch.cat(decoded_chunks, dim=0)
            latents = latents * 306
            latents = latents.clamp(0, 255).to(torch.uint8).cpu()
            latents = rearrange(latents, "(b t) c h w -> b c t h w", b=b)
            print("Decoded latents shape:", latents.shape)

        return latents

    def is_stop_frame(self, latent: torch.Tensor):
        """Check if the last block is all-white (stop signal)."""
        last_frame = latent[:, :, 0, :, :]
        target = self.one_latent.expand_as(last_frame)
        if_stop = torch.mean((torch.abs(last_frame - target) < self.eps).float())
        return if_stop > 0.9

    def trim_stop_frames(self, latent: torch.Tensor):
        """Remove trailing all-white frames from the end of the volume."""
        B, C, T, H, W = latent.shape
        target = self.one_latent.expand(B, C, H, W)

        keep_until = T
        for i in reversed(range(T)):
            frame = latent[:, :, i, :, :]
            if torch.mean((torch.abs(frame - target) < self.eps).float()) > 0.9:
                keep_until -= 1
            else:
                break

        return latent[:, :, :keep_until, :, :]

    def generate(self, prompt_embeds, max_blocks=30, gt_first_block=None):
        """
        Generate a full latent volume auto-regressively.

        Args:
            prompt_embeds: Text/CT embedding [B, 1, D].
            max_blocks: Maximum number of blocks to generate.
            gt_first_block: Optional ground-truth first block for gt-head inference mode.

        Returns:
            Tensor of shape [B, C, T, H, W].
        """
        self.denoiser.eval()

        B = 1
        C = self.config.globals.latent_channels
        H = W = self.config.globals.latent_res
        T = self.block_size

        if gt_first_block is None:
            init_block = self.zero_latent.unsqueeze(0).permute(0, 2, 1, 3, 4)
            init_block = init_block.repeat(B, 1, self.block_size, 1, 1)
            init_block = scale_latents(init_block, self.vae_scaling)
        else:
            init_block = gt_first_block

        blocks = [init_block]
        cur_step = 0
        block_times = []

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype):
            while cur_step < max_blocks:
                cur_latent = torch.cat(blocks, dim=2)

                z = torch.randn((B, C, T, H, W), device=self.device, dtype=self.dtype)

                cond_image_latent = cur_latent[:, :, -self.block_size:, :, :]
                cond_image_latent = sample_latents(self.config, cond_image_latent)

                def rhs(t, y):
                    return self.denoiser(
                        y, t, encoder_hidden_states=prompt_embeds, cond_image=cond_image_latent
                    ).sample

                timesteps = torch.linspace(1.0, 0.0, steps=201, device=self.device, dtype=self.dtype)
                start_time = time.time()
                new_block = odeint(
                    rhs,
                    z,
                    timesteps,
                    atol=1e-5,
                    rtol=1e-5,
                    adjoint_params=self.denoiser.parameters(),
                    method="euler",
                )[-1]
                elapsed = time.time() - start_time
                block_times.append(elapsed)

                blocks.append(new_block)
                cur_step += 1

                if self.overlap > 0:
                    last_overlap = new_block[:, :, -self.overlap:, :, :]
                    blocks[-1] = new_block[:, :, :-self.overlap, :, :]
                    blocks.append(last_overlap)

                if self.is_stop_frame(new_block):
                    print("[Generator] Stop block detected. Ending generation.")
                    break

        full_latent = torch.cat(blocks, dim=2)
        if gt_first_block is None and full_latent.shape[2] > 16:
            full_latent = full_latent[:, :, 16:]  # remove the zero-padded init block
        if self.trim:
            full_latent = self.trim_stop_frames(full_latent)

        if block_times:
            avg_time = sum(block_times) / len(block_times)
            total_time = sum(block_times)
            print(
                f"[Timing] Avg block time: {avg_time:.3f}s | "
                f"Total: {total_time:.3f}s | Blocks: {len(block_times)}"
            )

        return full_latent

    def generate_next_block(self, prev_latent, prompt_embeds):
        """
        Generate a single next block (used in block-wise inference mode).

        Args:
            prev_latent: Previous block latent [B, C, T, H, W].
            prompt_embeds: Text/CT embedding [B, 1, D].

        Returns:
            Next block latent [B, C, T', H, W] with stop frames trimmed.
        """
        B = 1
        C = self.config.globals.latent_channels
        H = W = self.config.globals.latent_res
        T = self.block_size

        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=self.dtype):
            z = torch.randn((B, C, T, H, W), device=self.device, dtype=self.dtype)
            cond_image_latent = prev_latent

            if not hasattr(self.denoiser, "forward_original"):
                self.denoiser.forward_original = self.denoiser.forward

            def new_forward(self, t, y, *args, **kwargs):
                kwargs = {
                    **kwargs,
                    "encoder_hidden_states": prompt_embeds,
                    "cond_image": cond_image_latent,
                }
                return self.forward_original(y, t, *args, **kwargs).sample

            self.denoiser.forward = types.MethodType(new_forward, self.denoiser)

            timesteps = torch.tensor([1.0, 0.0], dtype=self.dtype, device=self.device)
            start_time = time.time()
            new_block = odeint(
                self.denoiser,
                z,
                timesteps,
                atol=1e-5,
                rtol=1e-5,
                adjoint_params=self.denoiser.parameters(),
            )[-1]
            elapsed = time.time() - start_time
            print(f"[Timing] Single block generation time: {elapsed:.3f}s")

        new_block = self.trim_stop_frames(new_block)
        return new_block
