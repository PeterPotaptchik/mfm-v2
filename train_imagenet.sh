#!/bin/bash

#SBATCH --job-name=imagenet
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # Run a single task; torchrun will spawn GPU processes
#SBATCH --gpus-per-node=4           # Request 4 GPUs
#SBATCH --cpus-per-task=96          # Request all CPUs for the single task
#SBATCH --mem=300GB
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/new-outputs/imagenet/imagenet_train-%A.out
#SBATCH --error=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/new-outputs/imagenet/imagenet_train-%A.err
#SBATCH --mail-type=END,FAIL

# --- Environment Setup ---
set -euo pipefail  # Fail fast on errors

PROJECT_SCRATCH="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf"
cd "$PROJECT_SCRATCH"
source .venv/bin/activate

# --- Performance Tuning & Environment Variables ---

# 1. Dataloader Threading: Prevent CPU oversubscription
export OMP_NUM_THREADS=1
# Dynamically calculate the number of workers per GPU process
export NUM_DATA_WORKERS=$(( SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE ))

# 2. NCCL Settings for Optimal Single-Node Communication
export NCCL_DEBUG=WARN
export NCCL_P2P_DISABLE=0  # Enable Peer-to-Peer CUDA transfers
export NCCL_IB_DISABLE=1   # Disable InfiniBand for single-node; often faster over NVLink
export CUDA_DEVICE_MAX_CONNECTIONS=1

# 3. Rendezvous for torchrun (optional but good practice)
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=${MASTER_PORT:-29500}

echo "--- JOB CONFIGURATION ---"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Dataloader workers per GPU: $NUM_DATA_WORKERS"
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "--------------------------"

# model_checkpoint="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/2025-11-03/14-40-07/checkpoints/periodic-epoch=109-step=680000.ckpt"
# --- Run Training ---
srun --kill-on-bad-exit=1 --ntasks=1 torchrun --nproc_per_node="$SLURM_GPUS_PER_NODE" \
  --nnodes=1 \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  scripts/train.py model=unet_imagenet \
  dataset=imagenet_1k \
  ++lr.val=0.01 \
  ++lr.scheduler=cosine \
  ++lr.min_lr=0.005 \
  ++trainer.num_warmup_steps=200000 \
  ++trainer.num_train_steps=600000 \
  ++trainer.batch_size=15 \
  ++trainer.num_workers=24 \
  ++trainer.class_dropout_prob=0.5 \
  ++data_dir=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k \
  ++loss.explicit_v00_train=false \
  ++trainer.anneal_end_step=400000 \
  ++loss.distillation_type=lsd \
  ++trainer.accumulate_grad_batches=2 \
  ++compile=false \
  ++optimizer=RAdam \
  ++trainer.ema.decay=0.999
  # "resume_from_checkpoint=\"${model_checkpoint}\"" \
  
echo "Training finished."
