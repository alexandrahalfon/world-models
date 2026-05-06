#!/bin/bash
# collect_actor_actions discards trajectories that terminate before K=50 and
# retries with a new seed. Pong games can end early under the IRIS actor, so
# the effective wall-time per survivor is higher than naive throughput suggests.
# 5h is a safe ceiling for 2000 survivors x 3 games on A100.
#SBATCH --account=ds_ga_3001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=05:00:00
#SBATCH --job-name=collect_actor
#SBATCH --output=/scratch/%u/GenAIProject/results/logs/collect_actor_%j.out

source /scratch/$USER/GenAIProject/scripts/hpc_config.sh
module purge

singularity exec --nv \
    --overlay "$OVERLAY_TORCH:ro" \
    "$SINGULARITY_SIF" \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate env_torch
        cd $PROJECT
        python -u scripts/collect_actor_actions.py
    "
