#!/bin/bash

#SBATCH --job-name=reshuffle_latents
#SBATCH --account=kempner_albergo_lab
#SBATCH --partition=kempner_h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120GB
#SBATCH --time=0-06:00:00
#SBATCH --output=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/new-outputs/imagenet/reshuffle-%A.out
#SBATCH --error=/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf/new-outputs/imagenet/reshuffle-%A.err
#SBATCH --mail-type=END,FAIL

set -euo pipefail

PROJECT_ROOT="/n/netscratch/albergo_lab/Lab/ppotaptchik/distributional-mf"
cd "$PROJECT_ROOT"
source .venv/bin/activate

export OMP_NUM_THREADS=1

INPUT_ROOT=${INPUT_ROOT:-cache/latents/imagenet21k_resized/imagenet21k_train}
OUTPUT_ROOT=${OUTPUT_ROOT:-cache/latents/imagenet21k_resized/imagenet21k_train_shuffled_v2}
CHUNK_SIZE=${CHUNK_SIZE:-4096}
GROUP_SIZE=${GROUP_SIZE:-64}
MAX_CACHE_SHARDS=${MAX_CACHE_SHARDS:-128}
SEED=${SEED:-123}
DROP_TAIL=${DROP_TAIL:-1}

echo "--- RESHUFFLE CONFIGURATION ---"
echo "Input root: $INPUT_ROOT"
echo "Output root: $OUTPUT_ROOT"
echo "Chunk size: $CHUNK_SIZE"
echo "Group size: $GROUP_SIZE"
echo "Max cache shards: $MAX_CACHE_SHARDS"
echo "Seed: $SEED"
echo "Drop tail: $DROP_TAIL"
echo "-------------------------------"

CMD=(
  python
  scripts/reshuffle_latent_shards.py
  --input-root "$INPUT_ROOT"
  --output-root "$OUTPUT_ROOT"
  --chunk-size "$CHUNK_SIZE"
  --group-size "$GROUP_SIZE"
  --max-cache-shards "$MAX_CACHE_SHARDS"
  --seed "$SEED"
  --overwrite
)

if [[ "$DROP_TAIL" == "1" ]]; then
  CMD+=(--drop-tail)
fi

srun --ntasks=1 "${CMD[@]}"

echo "Reshuffling complete."
