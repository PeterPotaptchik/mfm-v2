#!/bin/bash

#SBATCH --job-name=cifar
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1                  # Run a single task; torchrun will spawn GPU processes
#SBATCH --gpus-per-node=4           # Request 4 GPUs
#SBATCH --cpus-per-task=48          # Request all CPUs for the single task
#SBATCH --mem=100GB
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/new-outputs/cifar/cifar_train-%A.out
#SBATCH --error=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/new-outputs/cifar/cifar_train-%A.err
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


# model_checkpoint="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/2025-10-25/20-29-21/checkpoints/epoch=942-step=88642.ckpt"
# model_checkpoint="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/2025-10-29/10-27-23/checkpoints/periodic-epoch=6170-step=580000.ckpt"
# model_checkpoint="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/2025-11-02/18-25-56/checkpoints/periodic-epoch=1276-step=120000.ckpt"
# --- Run Training ---
srun --kill-on-bad-exit=1 --ntasks=1 torchrun --nproc_per_node="$SLURM_GPUS_PER_NODE" \
  --nnodes=1 \
  --rdzv_id="$SLURM_JOB_ID" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
  scripts/train.py \
  model=unet_cifar_huge \
  ++lr.scheduler=cosine \
  ++trainer.num_warmup_steps=1200000 \
  ++trainer.num_train_steps=3000000 \
  ++trainer.num_no_posterior_steps=500000 \
  ++trainer.num_anneal_posterior_step=900000 \
  ++trainer.batch_size=120 \
  ++trainer.num_workers=24 \
  ++data_dir=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/data \
  ++loss.explicit_v00_train=false \
  ++trainer.anneal_end_step=2000000 \
  ++loss.distillation_type=lsd \
  ++trainer.accumulate_grad_batches=1 \
  ++compile=false \
  ++optimizer=RAdam \
  ++lr.val=0.01 \
  ++lr.min_lr=0.001 \
  ++trainer.ema.decay=0.999 \
  ++trainer.class_dropout_prob=0.3 \
  ++sampling.every_n_steps=5000
  # "resume_from_checkpoint=\"${model_checkpoint}\"" \

echo "Training finished."
