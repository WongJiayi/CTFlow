#!/bin/bash

RANK=$SLURM_PROCID
PART_FILE=$(printf "$PARTS_DIR/part_%02d" "$RANK")
SCRATCH_ROOT="/scratch/${USER}/infer_rank_${RANK}"
SCRATCH_EMBED="${SCRATCH_ROOT}/embeddings"
SCRATCH_OUTPUT="${SCRATCH_ROOT}/all_frames"
SCRATCH_LATENT="${SCRATCH_ROOT}/latent"
mkdir -p "$RAR_OUTPUT_DIR"

# Extract embeddings (once per rank)
if [ ! -f "$SCRATCH_EMBED/.untar_complete" ]; then
  mkdir -p "$SCRATCH_EMBED"
  tar -xf "$VAL_TAR" -C "$SCRATCH_EMBED"
  touch "$SCRATCH_EMBED/.untar_complete"
fi

# Extract GT latents (once per rank)
if [ ! -f "$SCRATCH_LATENT/.untar_complete" ]; then
  mkdir -p "$SCRATCH_LATENT"
  tar -xf "$GT_TAR" -C "$SCRATCH_LATENT" --strip-components=2
  touch "$SCRATCH_LATENT/.untar_complete"
fi

mkdir -p "$SCRATCH_OUTPUT"

# Run inference for each sample in this rank's partition
while read line; do
  EMB_PATH="${SCRATCH_EMBED}/${line}.pt"
  LATENT_PATH="${SCRATCH_LATENT}/${line}.pt"
  OUT_DIR="${SCRATCH_OUTPUT}/${line}"
  apptainer exec --nv \
    --env PYTHONPATH="${PROJECT_DIR}" \
    --bind "${PROJECT_DIR}:${PROJECT_DIR},${SCRATCH_ROOT}:${SCRATCH_ROOT}" \
    "$CONTAINER" \
    python "${PROJECT_DIR}/auto_regressive_generate/main.py" \
      --config "$INFER_CONFIG" \
      --ckpt "$INFER_CKPT" \
      --embedding "$EMB_PATH" \
      --gt-latent "$LATENT_PATH" \
      --output "$OUT_DIR" \
      --type "full-body"
done < "$PART_FILE"

# Pack this rank's outputs
TAR_NAME="rank_${RANK}_job_${SLURM_JOB_ID}.tar"
cd "$SCRATCH_OUTPUT"
tar -cf "${RAR_OUTPUT_DIR}/${TAR_NAME}" .
echo "RANK $RANK done: ${RAR_OUTPUT_DIR}/${TAR_NAME}"

# Uncomment to clean up scratch after packing:
# rm -rf "$SCRATCH_ROOT"
