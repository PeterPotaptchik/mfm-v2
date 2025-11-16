#!/bin/bash

#SBATCH --job-name=encode_imagenet_latents
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=96
#SBATCH --mem=300GB
#SBATCH --time=1-00:00:00
#SBATCH --output=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/new-outputs/imagenet/encode_latents-%A.out
#SBATCH --error=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/new-outputs/imagenet/encode_latents-%A.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_ROOT="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf"
DATA_ROOT="/n/holylfs06/LABS/kempner_shared/Everyone/testbed/vision/imagenet21k_resized/raw"
AUTOENCODER="stabilityai/sd-vae-ft-mse"
OUTPUT_ROOT="${PROJECT_ROOT}/cache/latents"
PIXEL_RES=256
BATCH_SIZE=128        # per GPU process (H100-friendly)
NUM_WORKERS=$(( SLURM_CPUS_PER_TASK / SLURM_GPUS_PER_NODE ))
SHARD_SIZE=4096
DTYPE=fp16
SPLITS=(
  "imagenet21k_resized/imagenet21k_train"
  "imagenet21k_resized/imagenet21k_val"
)

cd "${PROJECT_ROOT}"
source .venv/bin/activate

export OMP_NUM_THREADS=1
# torchrun --standalone handles rendezvous internally; keep vars for logging only
export MASTER_ADDR=$(hostname -s)
export MASTER_PORT=${MASTER_PORT:-29501}

mkdir -p "${OUTPUT_ROOT}" "${PROJECT_ROOT}/new-outputs/imagenet"

echo "--- ENCODING CONFIGURATION ---"
echo "Job ID: ${SLURM_JOB_ID:-none}"
echo "GPUs: ${SLURM_GPUS_PER_NODE}"
echo "Per-rank batch size: ${BATCH_SIZE}"
echo "Dataloader workers per rank: ${NUM_WORKERS}"
echo "Pixel resolution: ${PIXEL_RES}"
echo "Output root: ${OUTPUT_ROOT}"
echo "-------------------------------"

for SPLIT in "${SPLITS[@]}"; do
  SANITIZED_ID=$(echo "${SPLIT}" | tr '/.' '__')
  echo "Encoding split: ${SPLIT}"
  srun --kill-on-bad-exit=1 --ntasks=1 torchrun --standalone --nproc_per_node="${SLURM_GPUS_PER_NODE}" \
    scripts/encode_imagenet_latents.py \
      --data-dir "${DATA_ROOT}" \
      --split "${SPLIT}" \
      --output-dir "${OUTPUT_ROOT}" \
      --autoencoder "${AUTOENCODER}" \
      --pixel-resolution "${PIXEL_RES}" \
      --batch-size "${BATCH_SIZE}" \
      --num-workers "${NUM_WORKERS}" \
      --dtype "${DTYPE}" \
      --shard-size "${SHARD_SIZE}" \
      --save-paths
  echo "Finished split: ${SPLIT}"
done

echo "All splits encoded."
