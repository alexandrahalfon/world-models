#!/bin/bash
# GPU job: FID computation (exp3) via Inception network + PCA-KL divergence.
# 72 FID calls total (12 model-game pairs x 6 horizons) — GPU is required for practical speed.
#SBATCH --account=ds_ga_3001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --job-name=exp_gpu
#SBATCH --output=/scratch/%u/GenAIProject/results/logs/exp_gpu_%j.out
#SBATCH --requeue

source /scratch/$USER/GenAIProject/scripts/hpc_config.sh

module purge

singularity exec --nv \
    --overlay "$OVERLAY_TORCH:ro" \
    "$SINGULARITY_SIF" \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate env_torch
        cd $PROJECT
        echo '=== Experiment 3: Distributional divergence (FID + PCA-KL) ==='
        python experiments/exp3_distributional.py
    "
