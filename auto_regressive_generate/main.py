"""
Auto-regressive inference script for CT volume generation.

Usage:
    python auto_regressive_generate/main.py \
        --config /path/to/experiment/config.yaml \
        --ckpt   /path/to/checkpoint/denoiser_ema \
        --embedding /path/to/ct_embedding.pt \
        --output /path/to/output_dir \
        --type full-body

Inference modes:
    full-body   Generate the entire volume from scratch (default).
    gt-head     Use the ground-truth first block, then roll out auto-regressively.
    block-wise  Teacher-forcing: condition each step on the GT block.
"""

import argparse
import os

import imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from torchvision import transforms

from echosyn.common import *
from auto_regressive_generate import LatentAutoregressiveGenerator


def save_video_as_frames(video: torch.Tensor, output_dir: str, n_frames: int = 201):
    """
    Save a video tensor [C, T, H, W] as PNG frames.

    Args:
        video: Tensor of shape [C, T, H, W], pixel values in [0, 255].
        output_dir: Directory to save frames.
        n_frames: Total frames to save (pad last frame if needed).
    """
    os.makedirs(output_dir, exist_ok=True)

    video = video.permute(1, 0, 2, 3).contiguous()  # [T, C, H, W]
    T = video.shape[0]

    if T >= n_frames:
        video = video[:n_frames]
    else:
        pad = video[-1:].repeat(n_frames - T, 1, 1, 1)
        video = torch.cat([video, pad], dim=0)

    if video.dtype == torch.uint8:
        video = video.float() / 255.0
    else:
        video = video.clamp(0, 1)

    for i, frame in enumerate(video):
        pil_image = transforms.ToPILImage()(frame)
        pil_image.save(os.path.join(output_dir, f"frame_{i:03d}.png"), format="PNG", optimize=True)

    print(f"Saved {n_frames} frames to: {output_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Auto-regressive CT volume generation")
    parser.add_argument("--embedding", type=str, required=True, help="Path to CT embedding .pt file")
    parser.add_argument("--gt-latent", type=str, default=None, help="Path to GT latent .pt file (gt-head / block-wise modes)")
    parser.add_argument("--config", type=str, required=True, help="Path to training config file")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (denoiser_ema)")
    parser.add_argument("--output", type=str, default="output_frames", help="Output directory for generated frames")
    parser.add_argument(
        "--type",
        type=str,
        default="full-body",
        choices=["full-body", "gt-head", "block-wise"],
        help="Inference type",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load denoiser
    denoiser = instantiate_class_from_config(config.denoiser)
    denoiser = denoiser.from_pretrained(args.ckpt).to(device).eval()

    # Load VAE
    vae = instantiate(config.vae).eval().to(device)
    vae_scaling = get_vae_scaler(config, device)

    # Load embedding
    prompt_embedding = torch.load(args.embedding, map_location=device)
    prompt_embedding = prompt_embedding.unsqueeze(0)
    prompt_embedding = prompt_embedding / (prompt_embedding.norm(p=2) + 1e-6)

    # Init generator
    generator = LatentAutoregressiveGenerator(
        denoiser=denoiser,
        vae=vae,
        device=device,
        vae_scaling=vae_scaling,
        config=config,
    )

    # Run inference
    if args.type == "full-body":
        result_latent = generator.generate(
            prompt_embeds=prompt_embedding,
            max_blocks=20,
        )

    elif args.type == "gt-head":
        assert args.gt_latent is not None, "Must provide --gt-latent for gt-head mode"
        gt_latent = torch.load(args.gt_latent, map_location=device)
        if gt_latent.dim() == 4:
            gt_latent = gt_latent.unsqueeze(0)
        gt_latent = gt_latent.to(device)
        block_size = generator.block_size
        gt_first_block = gt_latent[:, :, block_size:2 * block_size, :, :]
        gt_first_block = scale_latents(gt_first_block, vae_scaling)
        result_latent = generator.generate(
            prompt_embeds=prompt_embedding,
            max_blocks=19,
            gt_first_block=gt_first_block,
        )

    elif args.type == "block-wise":
        assert args.gt_latent is not None, "Must provide --gt-latent for block-wise mode"
        gt_latent = torch.load(args.gt_latent, map_location=device)
        if gt_latent.dim() == 4:
            gt_latent = gt_latent.unsqueeze(0)
        B, C, T_total, H, W = gt_latent.shape
        block_size = generator.block_size
        num_blocks = T_total // block_size - 1

        blocks = []
        for i in range(num_blocks):
            gt_block = gt_latent[:, :, i * block_size:(i + 1) * block_size, :, :]
            gt_block = scale_latents(gt_block, vae_scaling)
            next_block = generator.generate_next_block(gt_block, prompt_embedding)
            if next_block.shape[2] == 0:
                print(f"[Block-wise] Stop at block {i} (all frames trimmed).")
                break
            blocks.append(next_block)
            if generator.is_stop_frame(next_block):
                print(f"[Block-wise] Stop block detected at block {i}")
                break

        if not blocks:
            print("No valid block generated!")
            return
        result_latent = torch.cat(blocks, dim=2)

    # Decode and save
    decoded = generator.decode_latent(result_latent)
    save_video_as_frames(decoded[0], args.output)


if __name__ == "__main__":
    main()
