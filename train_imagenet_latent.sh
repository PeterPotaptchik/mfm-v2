#!/bin/bash

#SBATCH --job-name=imagenet
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # Run a single task; torchrun will spawn GPU processes
#SBATCH --gpus-per-node=4          # Request 4 GPUs
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

echo "--- JOB CONFIGURATION ---"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "GPUs: $SLURM_GPUS_PER_NODE"
echo "CPUs per Task: $SLURM_CPUS_PER_TASK"
echo "Dataloader workers per GPU: $NUM_DATA_WORKERS"
echo "--------------------------"

model_checkpoint="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/2025-11-18/21-09-48/checkpoints/periodic-epoch=26-step=450000.ckpt"
# model_checkpoint="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/2025-11-14/18-04-28/checkpoints/periodic-epoch=01-step=20000.ckpt"
# --- Run Training ---
srun --kill-on-bad-exit=1 --ntasks=1 torchrun --standalone --nproc_per_node="$SLURM_GPUS_PER_NODE" \
  --nnodes=1 \
  scripts/train.py model=unet_imagenet_small \
  dataset=imagenet_latent \
  ++lr.val=0.01 \
  ++lr.scheduler=cosine \
  ++lr.min_lr=0.001 \
  ++trainer.num_warmup_steps=450000 \
  ++trainer.num_train_steps=800000 \
  ++trainer.batch_size=20 \
  ++trainer.num_workers=20 \
  ++trainer.class_dropout_prob=0.5 \
  ++data_dir=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/cache/latents \
  ++loss.explicit_v00_train=false \
  ++trainer.anneal_end_step=600000 \
  ++loss.distillation_type=lsd \
  ++trainer.accumulate_grad_batches=3 \
  ++compile=false \
  ++optimizer=RAdam \
  ++trainer.ema.decay=0.995 \
  ++trainer.t_cond_warmup_steps=300000 \
  ++trainer.t_cond_0_rate=0.1 \
  ++trainer.t_cond_power=1.25 \
  ++sampling.every_n_steps=10000 \
  ++use_parametrization=False \
  "resume_from_checkpoint=\"${model_checkpoint}\"" \

  
echo "Training finished."
