#!/bin/bash
#SBATCH --mail-user=adhithya.saravanan@stats.ox.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=/vols/bitbucket/saravanan/distributional-mf/slurm_outputs/slurm-%A_%a.out
#SBATCH --error=/vols/bitbucket/saravanan/distributional-mf/slurm_outputs/slurm-%A_%a.out

#SBATCH --job-name=sample_cifar
#SBATCH --cluster=swan
#SBATCH --partition=standard-rainml-gpu
#SBATCH --gres=gpu:Ampere_H100_80GB:1
#NOTSBATCH --nodelist=rainmlgpu01.cpu.stats.ox.ac.uk
#SBATCH --cpus-per-task=1
#SBATCH --time=15:00:00
#SBATCH --mem=32GB
#SBATCH --ntasks=1                
#SBATCH --array=0%1

source /data/localhost/not-backed-up/saravanan/miniconda3/bin/activate base
env_name="distcfm"
export PATH_TO_GIT="/vols/bitbucket/saravanan/distributional-mf"    
cd $PATH_TO_GIT
conda activate $env_name

job_commands=()
job_commands+=("python scripts/sample_openai_class.py ++conditional_score=dps ++samples_per_label=2048 ++sampling.batch_size=256 ++sampling.euler_steps=500")
command="${job_commands[$SLURM_ARRAY_TASK_ID]}"
eval $command