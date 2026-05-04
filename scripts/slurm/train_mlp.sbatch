#!/bin/bash
#SBATCH --account=ds_ga_3001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --job-name=train_mlp
#SBATCH --output=/scratch/%u/GenAIProject/results/logs/train_mlp_%j.out
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
        python scripts/train_mlp.py --game Breakout Pong Boxing --n_steps 200000 --n_data 500000
    "
