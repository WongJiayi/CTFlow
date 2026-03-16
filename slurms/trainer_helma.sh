#!/usr/bin/env bash

# Variables inherited from mnode_launcher_helma.sh:
#   HOSTNAMES, MASTER_ADDR, MASTER_PORT, COUNT_NODE, NUM_GPU, SCRIPT, CONFIG, SIF_IMAGE, SLURM_JOB_ID

# Rank index
H=$(hostname | cut -d'.' -f1)
THEID=$(echo -e "$HOSTNAMES" | tr ' ' '\n' | nl -w1 -s' ' \
  | awk -v host="$H" '$2==host{print $1-1}')
INDEX=$THEID

MASTER_HOST=$(echo "$MASTER_ADDR" | cut -d'.' -f1)
if [ "$H" = "$MASTER_HOST" ]; then
    echo "This node ($H) is the master node."
else
    echo "This node ($H) is a worker node."
fi

# Proxy settings (cluster-specific, adjust or remove as needed)
export http_proxy="http://YOUR_PROXY:80"
export https_proxy="http://YOUR_PROXY:80"
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="$no_proxy"

# Change to the project directory
cd $HOME/project/CTFlow-clean
mkdir -p logs

# ====================================================================
# Data paths — set DATA_ROOT to your NVMe/storage mount point
# ====================================================================
export SCRATCH_ROOT=/scratch/${USER}/latte_train_${SLURM_JOB_ID}
export LATENT_TAR_DIR=${DATA_ROOT}/encoded_output
export EMBED_TAR_DIR=${DATA_ROOT}/embedding_special

# Tar list: one .tar per node (16 shards for 16 nodes)
LATENT_TARS=(
  CT-RATE_train_fixed_part_1_latents_balanced.tar
  CT-RATE_train_fixed_part_2_latents_balanced.tar
  CT-RATE_train_fixed_part_3_latents_balanced.tar
  CT-RATE_train_fixed_part_4_latents_balanced.tar
  CT-RATE_train_fixed_part_5_latents_balanced.tar
  CT-RATE_train_fixed_part_6_latents_balanced.tar
  CT-RATE_train_fixed_part_7_latents_balanced.tar
  CT-RATE_train_fixed_part_8_latents_balanced.tar
  CT-RATE_train_fixed_part_9_latents_balanced.tar
  CT-RATE_train_fixed_part_10_latents_balanced.tar
  CT-RATE_train_fixed_part_11_latents_balanced.tar
  CT-RATE_train_fixed_part_12_latents_balanced.tar
  CT-RATE_train_fixed_part_13_latents_balanced.tar
  CT-RATE_train_fixed_part_14_latents_balanced.tar
  CT-RATE_train_fixed_part_15_latents_balanced.tar
  CT-RATE_train_fixed_part_16_latents_balanced.tar
)
EMBED_TARS=(
  embeddings_CT-RATE_train_fixed_part_1_latents_balanced.tar
  embeddings_CT-RATE_train_fixed_part_2_latents_balanced.tar
  embeddings_CT-RATE_train_fixed_part_3_latents_balanced.tar
  embeddings_CT-RATE_train_fixed_part_4_latents_balanced.tar
  embeddings_CT-RATE_train_fixed_part_5_latents_balanced.tar
  embeddings_CT-RATE_train_fixed_part_6_latents_balanced.tar
  embeddings_CT-RATE_train_fixed_part_7_latents_balanced.tar
  embeddings_CT-RATE_train_fixed_part_8_latents_balanced.tar
  embeddings_CT-RATE_train_fixed_part_9_latents_balanced.tar
  embeddings_CT-RATE_train_fixed_part_10_latents_balanced.tar
  embeddings_CT-RATE_train_fixed_part_11_latents_balanced.tar
  embeddings_CT-RATE_train_fixed_part_12_latents_balanced.tar
  embeddings_CT-RATE_train_fixed_part_13_latents_balanced.tar
  embeddings_CT-RATE_train_fixed_part_14_latents_balanced.tar
  embeddings_CT-RATE_train_fixed_part_15_latents_balanced.tar
  embeddings_CT-RATE_train_fixed_part_16_latents_balanced.tar
)

mkdir -p "$SCRATCH_ROOT"

# Each node extracts its own shard
export SCRATCH_LATENTS=$SCRATCH_ROOT/latents_$INDEX
export SCRATCH_EMBEDS=$SCRATCH_ROOT/embeddings_$INDEX
mkdir -p "$SCRATCH_LATENTS" "$SCRATCH_EMBEDS" "$SCRATCH_ROOT/logs_train"

echo "[Node $INDEX] Extracting ${LATENT_TARS[$INDEX]} to $SCRATCH_LATENTS"
full_latent="$LATENT_TAR_DIR/${LATENT_TARS[$INDEX]}"
if [ -f "$full_latent" ]; then
  first_entry=$(tar -tf "$full_latent" | head -n1)
  strip_count=$(printf '%s' "$first_entry" | tr -cd '/' | wc -c)
  tar -xf "$full_latent" -C "$SCRATCH_LATENTS" --strip-components="$strip_count"
  echo "    Done: ${LATENT_TARS[$INDEX]}"
else
  echo "    ERROR: File not found: $full_latent"
fi

echo "[Node $INDEX] Extracting ${EMBED_TARS[$INDEX]} to $SCRATCH_EMBEDS"
full_embed="$EMBED_TAR_DIR/${EMBED_TARS[$INDEX]}"
if [ -f "$full_embed" ]; then
  first_entry=$(tar -tf "$full_embed" | head -n1)
  strip_count=$(printf '%s' "$first_entry" | tr -cd '/' | wc -c)
  tar -xf "$full_embed" -C "$SCRATCH_EMBEDS" --strip-components="$strip_count"
  echo "    Done: ${EMBED_TARS[$INDEX]}"
else
  echo "    ERROR: File not found: $full_embed"
fi

# Validation data
export SCRATCH_VALID_LATENTS=$SCRATCH_ROOT/valid_latents
export SCRATCH_VALID_EMBEDS=$SCRATCH_ROOT/valid_embeddings
mkdir -p "$SCRATCH_VALID_LATENTS" "$SCRATCH_VALID_EMBEDS"

VALID_LATENT_TAR=$LATENT_TAR_DIR/CT-RATE_valid_fixed_latents_v2.tar
echo "[Node $INDEX] Extracting $(basename "$VALID_LATENT_TAR") to $SCRATCH_VALID_LATENTS"
tar -xf "$VALID_LATENT_TAR" -C "$SCRATCH_VALID_LATENTS" --strip-components=2 --wildcards '*/extracted_volumes/*.pt'
echo "    Done: $(basename "$VALID_LATENT_TAR")"

VALID_EMBED_TAR=$EMBED_TAR_DIR/embeddings_CT-RATE_valid_fixed_latents.tar
echo "[Node $INDEX] Extracting $(basename "$VALID_EMBED_TAR") to $SCRATCH_VALID_EMBEDS"
tar -xf "$VALID_EMBED_TAR" -C "$SCRATCH_VALID_EMBEDS" --strip-components=1 --wildcards '*.pt'
echo "    Done: $(basename "$VALID_EMBED_TAR")"

echo "Valid latents: $(find "$SCRATCH_VALID_LATENTS" -type f -name "*.pt" | wc -l)"
echo "Valid embeddings: $(find "$SCRATCH_VALID_EMBEDS" -type f -name "*.pt" | wc -l)"

# Expose to config via envsubst
export LATTE_VALID_DATA_ROOT=$SCRATCH_VALID_LATENTS
export LATTE_VALID_EMBEDDING_ROOT=$SCRATCH_VALID_EMBEDS

# Generate per-node train split file
export LATTE_TRAIN_SPLIT_FILE=$SCRATCH_ROOT/train_$INDEX.txt
find "$SCRATCH_LATENTS" -type f -name "*.pt" \
  | sed "s|$SCRATCH_LATENTS/||; s|\.pt\$||" \
  | sort > "$LATTE_TRAIN_SPLIT_FILE"
echo "[Node $INDEX] Train file count: $(wc -l < "$LATTE_TRAIN_SPLIT_FILE")"

export LATTE_TRAIN_DATA_ROOT=$SCRATCH_LATENTS
export LATTE_EMBEDDING_ROOT=$SCRATCH_EMBEDS
export LATTE_TRAIN_SPLIT_FILE

# Substitute env vars into config
TMP_CONFIG=$SCRATCH_ROOT/config_${INDEX}.yaml
envsubst < $CONFIG > $TMP_CONFIG
CONFIG=$TMP_CONFIG
echo "Generated per-node config: $CONFIG"

# Launch training inside container
container_cmd="singularity exec --nv --pwd $PWD $SIF_IMAGE"
export PYTHONNOUSERSITE=1
export PYTHONPATH=$PWD:$PYTHONPATH
export ACCELERATE_DISABLE_RNG_SYNC=1

echo "Preparing accelerate command for node $INDEX"
if [ "$COUNT_NODE" -eq 1 ]; then
    accelerate_cmd="accelerate launch \
        --num_processes 4 \
        --multi_gpu \
        --num_machines 1 \
        --dynamo_backend inductor \
        --dynamo_use_dynamic \
        --mixed_precision bf16 \
        $SCRIPT \
        --config $CONFIG --no_wandb"
else
    accelerate_cmd="accelerate launch \
        --num_processes $((4 * COUNT_NODE)) \
        --num_machines $COUNT_NODE \
        --multi_gpu \
        --dynamo_backend inductor \
        --dynamo_use_dynamic \
        --num_cpu_threads_per_process 32 \
        --mixed_precision bf16 \
        --machine_rank $INDEX \
        --main_process_ip $MASTER_ADDR \
        --main_process_port $MASTER_PORT \
        $SCRIPT \
        --config $CONFIG"
fi

train_cmd="$container_cmd $accelerate_cmd"
echo "Launching training on node $INDEX:"
echo "$train_cmd"
$train_cmd