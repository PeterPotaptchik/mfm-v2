#!/bin/bash

#SBATCH --job-name=imagenet
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=22
#SBATCH --mem=100GB
#SBATCH --time=3-00:00:00
#SBATCH --output=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/new-outputs/imagenet/imagenet_train-%A.out
#SBATCH --error=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/new-outputs/imagenet/imagenet_train-%A.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_ROOT="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf"
cd "$PROJECT_ROOT"
source .venv/bin/activate
########################################
# 2. Basic distributed config from Slurm
########################################

# How many GPUs per node will torchrun use?
GPUS_PER_NODE=1   

# Total number of nodes from Slurm
export NNODES="${SLURM_NNODES}"

# Pick the first node in the allocation as MASTER_ADDR
HOST0=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Force IPv4 address
MASTER_ADDR=$(getent ahostsv4 "$HOST0" | awk '{print $1; exit}')
export MASTER_ADDR
export MASTER_PORT=$((10000 + SLURM_JOB_ID % 50000))  # also avoids EADDRINUSE
export GPUS_PER_NODE

# Network interface for distributed training
# export GLOO_SOCKET_IFNAME=bond0
# export NCCL_SOCKET_IFNAME=bond0

echo "MASTER_ADDR=${MASTER_ADDR}"
echo "MASTER_PORT=${MASTER_PORT}"
echo "NNODES=${NNODES}"
echo "GPUS_PER_NODE=${GPUS_PER_NODE}"

########################################
# 3. Launch: one torchrun per node via srun
########################################
# Slurm will run this command once per node (because ntasks-per-node=1).
# Each node gets its own SLURM_NODEID, which we use as --node_rank.

# export MODEL_CHECKPOINT="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/2025-11-21/01-35-32/checkpoints/periodic-epoch=06-step=120000.ckpt"
# export MODEL_CHECKPOINT="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/2025-11-21/18-34-27/checkpoints/periodic-epoch=01-step=20000.ckpt"
# export MODEL_CHECKPOINT="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/2025-11-22/12-14-22/checkpoints/periodic-epoch=00-step=5000.ckpt"
export MODEL_CHECKPOINT="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/outputs/2025-11-29/02-08-56/checkpoints/periodic-epoch=29-step=75000.ckpt"


# python REPA/generate.py \
#   --model SiT-XL/2 \
#   --num-fid-samples 50 \
#   --path-type=linear \
#   --encoder-depth=8 \
#   --projector-embed-dims=768 \
#   --per-proc-batch-size=64 \
#   --mode=sde \
#   --num-steps=250 \
#   --cfg-scale=1.8 \
#   --guidance-high=0.7


# python dmf/generate.py \
#   --model="DMFT-XL/2" \
#   --num-fid-samples=5 \
#   --dmf-depth=20 \
#   --per-proc-batch-size=64 \
#   --mode "euler" \
#   --num-steps 4 \
#   --shift 1.0 \
#   --ckpt="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/ckpt/dmf_xl_2_256.pt"


export HYDRA_FULL_ERROR=1
srun bash -c '
  echo "Starting on host $(hostname) with SLURM_NODEID=$SLURM_NODEID"

  # torchrun \
  #   --nnodes=$NNODES \
  #   --nproc_per_node=$GPUS_PER_NODE \
  #   --node_rank=$SLURM_NODEID \
  #   --master_addr=$MASTER_ADDR \
  #   --master_port=$MASTER_PORT \
    python scripts/train.py --config-name config_test \
    model=dit_imagenet_latent \
    dataset=imagenet_1k \
    ++model.input_size=32 \
    ++model.in_channels=4 \
    ++model.label_dim=1000 \
    ++trainer.devices="$GPUS_PER_NODE" \
    ++trainer.num_nodes="$NNODES" \
    ++lr.val=0.0001 \
    ++lr.scheduler=constant \
    ++lr.warmup_steps=1000 \
    ++trainer.num_warmup_steps=2000 \
    ++trainer.num_train_steps=500000 \
    ++trainer.batch_size=32 \
    ++trainer.num_workers=20 \
    ++trainer.class_dropout_prob=0.0 \
    ++data_dir=/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet_1k \
    ++loss.explicit_v00_train=false \
    ++trainer.anneal_end_step=5000 \
    ++loss.distillation_type=mf \
    ++trainer.accumulate_grad_batches=1 \
    ++compile=false \
    ++optimizer=RAdam \
    ++trainer.ema.decay=0.9995 \
    ++trainer.t_cond_warmup_steps=0 \
    ++trainer.t_cond_0_rate=0.1 \
    ++trainer.t_cond_power=1.25 \
    ++sampling.every_n_steps=500 \
    ++use_parametrization=False \
    ++trainer.checkpoint_every_n_steps=5000 \
    ++weight_decay=0.00 \
    "resume_from_checkpoint=\"$MODEL_CHECKPOINT\""
  '
    # "resume_from_checkpoint=\"$MODEL_CHECKPOINT\""

  #    # +init_from_sit=True

echo "Training finished."
