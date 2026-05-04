#!/bin/bash
#SBATCH --account=ds_ga_3001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --job-name=experiments
#SBATCH --output=/scratch/%u/GenAIProject/results/logs/experiments_%j.out
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
        echo '=== exp1 ==='
        python -u experiments/exp1_error_growth.py
        echo '=== exp2 ==='
        python -u experiments/exp2_failure_modes.py
        echo '=== exp3 ==='
        python -u experiments/exp3_distributional.py
        echo '=== generate_report ==='
        python -u scripts/generate_report.py
        echo '=== Done ==='
        echo 'Files in results/exp3:'
        ls -la $PROJECT/results/exp3
        echo 'Report:'
        ls -la $PROJECT/results/report_*.html | tail -3
    "
