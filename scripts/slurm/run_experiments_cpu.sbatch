#!/bin/bash
# CPU job: MSE + power-law fitting (exp1) and frame visualizations (exp2).
# FID is in run_experiments_gpu.sbatch — needs GPU for Inception network.
#SBATCH --account=ds_ga_3001-2026sp
#SBATCH --partition=n2c48m24
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --job-name=exp_cpu
#SBATCH --output=/scratch/%u/GenAIProject/results/logs/exp_cpu_%j.out
#SBATCH --requeue

source /scratch/$USER/GenAIProject/scripts/hpc_config.sh

module purge

singularity exec \
    --overlay "$OVERLAY_TORCH:ro" \
    "$SINGULARITY_SIF" \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate env_torch
        cd $PROJECT
        echo '=== Experiment 1: Error growth curves ==='
        python experiments/exp1_error_growth.py
        echo '=== Experiment 2: Failure mode visualization ==='
        python experiments/exp2_failure_modes.py
    "
