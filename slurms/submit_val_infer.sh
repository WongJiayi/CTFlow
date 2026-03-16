#!/bin/bash
#SBATCH --job-name=val_infer_multi
#SBATCH --output=logs/val_infer_%j.out
#SBATCH --error=logs/val_infer_%j.err
#SBATCH --partition=h100
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --mail-type=ALL

# ====================================================================
# Paths — edit these before submitting
# ====================================================================
PROJECT_DIR="$HOME/project/CTFlow-clean"
CONTAINER="/path/to/tmi_container.sif"
PARTS_DIR="$PROJECT_DIR/parts"

# Data paths (set DATA_ROOT to your storage mount point)
VAL_TAR="${DATA_ROOT}/embedding_special/embeddings_CT-RATE_valid_fixed_latents.tar"
GT_TAR="${DATA_ROOT}/encoded_output/CT-RATE_valid_fixed_latents.tar"
RAR_OUTPUT_DIR="${DATA_ROOT}/validation_infer_result"

# Inference config and checkpoint
INFER_CONFIG="${DATA_ROOT}/experiments/YOUR_EXPERIMENT/config.yaml"
INFER_CKPT="${DATA_ROOT}/experiments/YOUR_EXPERIMENT/checkpoint-XXXXXX/denoiser_ema"

# Export all vars to workers
export PROJECT_DIR CONTAINER PARTS_DIR VAL_TAR GT_TAR RAR_OUTPUT_DIR INFER_CONFIG INFER_CKPT

# Launch 64 parallel inference workers (16 nodes × 4 GPUs)
srun --ntasks=64 --gpus-per-task=1 --cpus-per-task=8 "$PROJECT_DIR/slurms/infer_worker.sh"

echo "All inference done. Results saved to: $RAR_OUTPUT_DIR"
