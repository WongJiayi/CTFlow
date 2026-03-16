#!/bin/bash -l
#
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=h100
#SBATCH --nodes=16
#SBATCH --time=20:00:00
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --mail-user=YOUR_EMAIL@example.com
#SBATCH --mail-type=ALL

unset SLURM_EXPORT_ENV

# Proxy settings (cluster-specific, adjust or remove as needed)
export http_proxy="http://YOUR_PROXY:80"
export https_proxy="http://YOUR_PROXY:80"
export no_proxy="localhost,127.0.0.1"
export NO_PROXY="$no_proxy"

# Change to the project directory
cd $HOME/project/CTFlow-clean
mkdir -p logs

# Singularity image path (used in trainer_helma.sh)
export SIF_IMAGE="/path/to/tmi_container.sif"
echo "Singularity set to: $SIF_IMAGE"

# NCCL & MPI tuning
export NCCL_DEBUG=info
export NCCL_PROTO=simple
export NCCL_SHARP_ENABLE=1
export NCCL_IB_HCA=mlx5_0

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export OMPI_MCA_mtl_base_verbose=1
export FI_LOG_LEVEL=1
export TORCH_LOGS="-dynamo"

# Cluster node info
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(echo $HOSTNAMES | awk '{print $1}').YOUR_CLUSTER_DOMAIN
export MASTER_PORT=12802
export COUNT_NODE=$(echo $HOSTNAMES | wc -w)
export NUM_GPU=$(nvidia-smi -L | wc -l)

echo "Launching on $COUNT_NODE nodes, total $NUM_GPU GPUs per node"
echo "Node list: $HOSTNAMES"
echo "MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT"

# Training script and config
export SCRIPT="lvfm/train.py"
export CONFIG="lvfm/configs/jiayi_lvfm_STDiT-L2_16f8_all.yaml"

# Sanity check
for file in "$SCRIPT" "$CONFIG"; do
  if [ ! -f "$file" ]; then
    echo "ERROR: $file not found. Abort."
    exit 1
  else
    echo "Found $file"
  fi
done

srun slurms/trainer_helma.sh