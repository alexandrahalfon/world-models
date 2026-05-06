#!/bin/bash
# Array job: DreamerV3 rollout per game (3 tasks).
# Runs the env_jax overlay directly from the compute node — Singularity does
# not support nested containers, so this CANNOT be launched from inside the
# env_torch shell. Use the main run_rollouts.sbatch for the torch models.
#
# Index mapping:
#   0: dreamerv3 / Breakout
#   1: dreamerv3 / Pong
#   2: dreamerv3 / Boxing
#
# DRY-RUN FIRST: run scripts/dreamerv3_rollout.py interactively with --n_traj 4
# --K 10 to confirm the installed dreamerv3 package's API matches the calls
# in that script (observe / imagine / decode / configs.atari). See CLAUDE.md.
#SBATCH --account=ds_ga_3001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=rollout_dv3
#SBATCH --output=/scratch/%u/GenAIProject/results/logs/rollout_dv3_%A_%a.out
#SBATCH --requeue

source /scratch/$USER/GenAIProject/scripts/hpc_config.sh

GAMES=(Breakout Pong Boxing)
GAME=${GAMES[$SLURM_ARRAY_TASK_ID]}

echo "DreamerV3 task $SLURM_ARRAY_TASK_ID: game=$GAME"

module purge

# Ground-truth frames need an env_torch import (gymnasium/ale_py + PIL); the
# JAX rollout subprocess only produces pred_frames. We split into two passes
# rather than nesting Singularity calls.

# Pass 1: produce dreamerv3_<GAME>.npz pred_frames inside env_jax.
singularity exec --nv \
    --overlay "$OVERLAY_JAX:ro" \
    "$SINGULARITY_SIF" \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate env_jax
        cd $PROJECT
        mkdir -p results/rollouts
        python -u scripts/dreamerv3_rollout.py \
            --checkpoint checkpoints/dreamerv3/ \
            --actions data/atari/${GAME}_actions.npz \
            --output results/rollouts/dreamerv3_${GAME}.npz \
            --game ${GAME} \
            --n_traj 2000 --K 50 --seed 42
    "

# Pass 2: re-open the cache in env_torch and append true_frames + valid_mask
# (the rollout pipeline expects both keys; the JAX script only writes pred_frames).
singularity exec --nv \
    --overlay "$OVERLAY_TORCH:ro" \
    "$SINGULARITY_SIF" \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate env_torch
        cd $PROJECT
        python - <<'PYEOF'
import sys, yaml, os
sys.path.insert(0, '.')
import numpy as np
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from src.rollout import _collect_true_frames, PIXEL_SIZE

with open('configs/experiment.yaml') as f:
    cfg = yaml.safe_load(f)

game = '${GAME}'
K = cfg['horizon_K']
n_traj = cfg['n_trajectories']
seed = cfg['seed']

cache_path = f'results/rollouts/dreamerv3_{game}.npz'
data = dict(np.load(cache_path))
pred_frames = data['pred_frames']

actions_data = np.load(f'data/atari/{game}_actions.npz')
actions = actions_data['actions'][:n_traj, :K]
saved_seeds = actions_data['seeds'][:n_traj] if 'seeds' in actions_data.files else None

true_frames = _collect_true_frames(game, actions, n_traj, K, seed, 'rgb',
                                   saved_seeds=saved_seeds)

# DreamerV3 + JAX subprocess does not have access to a live env, so all
# trajectories that survived action collection are assumed valid for K steps
# (matches the K-survivor contract of the new collect_actor_actions.py).
valid_mask = np.ones((n_traj, K), dtype=bool)

np.savez_compressed(cache_path,
                    pred_frames=pred_frames, true_frames=true_frames,
                    actions=actions, valid_mask=valid_mask)
print(f'Wrote dreamerv3 cache for {game}: pred={pred_frames.shape} true={true_frames.shape}')
PYEOF
    "
