#!/bin/bash
#SBATCH --mail-user=adhithya.saravanan@stats.ox.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --output=/vols/bitbucket/saravanan/distributional-mf/slurm_outputs/slurm-%A_%a.out
#SBATCH --error=/vols/bitbucket/saravanan/distributional-mf/slurm_outputs/slurm-%A_%a.out

#SBATCH --job-name=baselines
#SBATCH --clusters=srf_gpu_01
#SBATCH --partition=high-bigbayes-test
#SBATCH --gres=gpu:Ampere_A40:1
#SBATCH --nodelist=zizgpu06.cpu.stats.ox.ac.uk
#SBATCH --cpus-per-task=1
#SBATCH --time=20:00:00
#SBATCH --mem=32GB
#SBATCH --ntasks=1                
#SBATCH --array=0%1

source /data/zizgpu06/not-backed-up/saravanan/miniforge3/bin/activate base
env_name="distcfm"
export PATH_TO_GIT="/vols/bitbucket/saravanan/distributional-mf/"    
cd $PATH_TO_GIT
conda activate $env_name

job_commands=()
job_commands+=("python scripts/train_denoiser.py model=unet_cifar_denoiser dataset=mnist ++model.learn_loss_weighting=false ++wandb.name=debug")

command=${job_commands[$SLURM_ARRAY_TASK_ID]}
eval $command
