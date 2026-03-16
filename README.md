# CTFlow

**CTFlow: Video-Inspired Latent Flow Matching for 3D CT Synthesis**
> ICCV 2025 Workshop on Vision-Language Models for 3D Understanding (VLM3D)

[[Paper]](https://openaccess.thecvf.com/content/ICCV2025W/VLM3D/papers/Wang_CTFlow_Video-Inspired_Latent_Flow_Matching_for_3D_CT_Synthesis_ICCVW_2025_paper.pdf) | [[Checkpoint (HuggingFace)]](https://huggingface.co/YOUR_HF_USERNAME/CTFlow)


Latent Video Flow Matching with Auto-Regressive Generation for 3D CT volumes.

This repository implements a Spatial-Temporal DiT (STDiT) trained via flow matching to generate 3D CT volumes block-by-block in an auto-regressive manner. The model is conditioned on text/CT report embeddings.

---

## Overview

**Training** — Flow matching on pre-encoded latent CT volumes. Each training sample is a pair of consecutive 16-frame latent blocks (current block as condition, next block as target), conditioned on a CT report embedding.

**Inference** — Auto-regressive generation: starting from a zero-padded initial block, the model iteratively generates the next block until a stop signal or maximum length is reached. Three inference modes are supported:
- `full-body` — generate the entire volume from scratch
- `gt-head` — use the ground-truth first block, then roll out auto-regressively
- `block-wise` — teacher-forcing mode for evaluation

---

## Repository Structure

```
CTFlow/
├── echosyn/
│   └── common/
│       ├── __init__.py          # Shared utilities (instantiation, training helpers, etc.)
│       ├── models.py            # DiffuserSTDiT and supporting architecture
│       ├── schedulers.py        # Learning rate schedulers
│       └── datasets.py          # LatentBlockDataset
├── lvfm/
│   ├── train.py                 # Training script (Latent Video Flow Matching)
│   └── configs/
│       ├── jiayi_lvdm_STDiT-S2_16f8_all.yaml   # STDiT-S (36M params)
│       ├── jiayi_lvfm_STDiT-B2_16f8_all.yaml   # STDiT-B (146M params)
│       └── jiayi_lvfm_STDiT-L2_16f8_all.yaml   # STDiT-L (512M params)
├── auto_regressive_generate/
│   ├── __init__.py              # LatentAutoregressiveGenerator class
│   └── main.py                  # Inference entry point
├── slurms/
│   ├── mnode_launcher_helma.sh  # SLURM multi-node training launcher
│   ├── trainer_helma.sh         # Per-node training worker
│   ├── submit_val_infer.sh      # SLURM inference launcher
│   └── infer_worker.sh          # Per-rank inference worker
├── logs/
├── README.md
└── requirements.txt
```

---

## Data Format

**Latent volumes** (`.pt`): Pre-encoded 3D CT volumes, shape `[C, T, H, W]` where `C=16` (FLUX VAE channels), `T` is the temporal dimension (number of slices), `H=W=32` (spatial latent resolution).

**Embeddings** (`.pt`): CT report text embeddings, shape `[N, D]` where `D=768`.

---

## Training

### Local (single node, 4 GPUs)

```bash
cd CTFlow

accelerate launch \
    --num_processes 4 \
    --multi_gpu \
    --mixed_precision bf16 \
    lvfm/train.py \
    --config lvfm/configs/jiayi_lvfm_STDiT-L2_16f8_all.yaml
```

### Multi-node on Helma (SLURM)

Edit `slurms/mnode_launcher_helma.sh` to set the desired config, then:

```bash
sbatch slurms/mnode_launcher_helma.sh
```

**Required environment variables** (set in `trainer_helma.sh`):
| Variable | Description |
|---|---|
| `LATTE_TRAIN_DATA_ROOT` | Directory of training latent `.pt` files |
| `LATTE_EMBEDDING_ROOT` | Directory of training embedding `.pt` files |
| `LATTE_VALID_DATA_ROOT` | Directory of validation latent `.pt` files |
| `LATTE_VALID_EMBEDDING_ROOT` | Directory of validation embedding `.pt` files |

---

## Inference

### Single sample

```bash
python auto_regressive_generate/main.py \
    --config /path/to/experiment/config.yaml \
    --ckpt   /path/to/checkpoint/denoiser_ema \
    --embedding /path/to/ct_embedding.pt \
    --output output_frames/ \
    --type full-body
```

### Batch inference on cluster (SLURM)

```bash
sbatch slurms/submit_val_infer.sh
```

This runs 64 parallel workers (16 nodes × 4 GPUs), each processing a partition of the validation set defined in `$PROJECT_DIR/parts/part_XX`.

---

## Model Configurations

| Config | Model | Params | Hidden | Depth | Heads |
|---|---|---|---|---|---|
| `jiayi_lvdm_STDiT-S2_16f8_all.yaml` | STDiT-S | 36M | 384 | 12 | 6 |
| `jiayi_lvfm_STDiT-B2_16f8_all.yaml` | STDiT-B | 146M | 768 | 12 | 12 |
| `jiayi_lvfm_STDiT-L2_16f8_all.yaml` | STDiT-L | 512M | 1024 | 24 | 16 |

All configs use:
- Input: 16 frames × 32×32 spatial latent
- `in_channels=32` (16 target + 16 condition)
- FLUX VAE (`AutoencoderKL`) for encoding/decoding
- AdamW optimizer with inverse square root LR decay

---

## Environment

See `requirements.txt` for Python dependencies.

The cluster setup uses a Singularity/Apptainer container (`tmi_container.sif`).

---

## Key Components

**`LatentAutoregressiveGenerator`** — Main inference class. Handles block-wise generation, stop-frame detection, and latent decoding.

**`DiffuserSTDiT`** — Spatial-Temporal DiT model. Extends a standard DiT with temporal attention and image conditioning (`cond_image`) via channel concatenation.

**`LatentBlockDataset`** — Loads `(cond_block, target_block, embedding)` triplets for training.

**`StepBasedLearningRateScheduleWithWarmup`** — Warmup + inverse square root decay scheduler.

---

## Pretrained Checkpoint

The pretrained STDiT-L2 checkpoint is available on HuggingFace:

👉 [https://huggingface.co/YOUR_HF_USERNAME/CTFlow](https://huggingface.co/YOUR_HF_USERNAME/CTFlow)

Download the checkpoint and set `INFER_CKPT` in `slurms/submit_val_infer.sh` accordingly.

---

## Citation

If you use this code or model, please cite:

```bibtex
@InProceedings{Wang_2025_ICCVW,
    author    = {Wang, Jiayi and Reynaud, Hadrien and Erick, Franciskus Xaverius and Kainz, Bernhard},
    title     = {CTFlow: Video-Inspired Latent Flow Matching for 3D CT Synthesis},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    year      = {2025},
}
