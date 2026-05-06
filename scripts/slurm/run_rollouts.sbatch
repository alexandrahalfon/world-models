#!/bin/bash
# Array job: one task per torch model-game pair (9 total).
# Index mapping:
#   0-2:  mlp     / Breakout, Pong, Boxing
#   3-5:  iris    / Breakout, Pong, Boxing
#   6-8:  diamond / Breakout, Pong, Boxing
#
# DreamerV3 lives in the env_jax overlay and is dispatched by a separate
# sbatch (run_rollouts_dreamerv3.sbatch). Singularity does not support
# nesting an env_jax `singularity exec` *inside* a running env_torch
# container, so DreamerV3 must launch its overlay directly from the
# compute node.
#
# PILOT FIRST: run a single task interactively with n_traj=200 to validate
# k_star_multiplier before submitting this. See CLAUDE.md.
#SBATCH --account=ds_ga_3001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --array=0-8
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=rollout
#SBATCH --output=/scratch/%u/GenAIProject/results/logs/rollout_%A_%a.out
#SBATCH --requeue

source /scratch/$USER/GenAIProject/scripts/hpc_config.sh

MODELS=(mlp mlp mlp iris iris iris diamond diamond diamond)
GAMES=(Breakout Pong Boxing Breakout Pong Boxing Breakout Pong Boxing)

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
GAME=${GAMES[$SLURM_ARRAY_TASK_ID]}

echo "Task $SLURM_ARRAY_TASK_ID: model=$MODEL game=$GAME"

module purge

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
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from src.rollout import rollout

with open('configs/experiment.yaml') as f:
    cfg = yaml.safe_load(f)

model_name = '$MODEL'
game = '$GAME'
K = cfg['horizon_K']
n_traj = cfg['n_trajectories']
seed = cfg['seed']
actions_path = f'data/atari/{game}_actions.npz'

# Number of actions for the model architecture (must match the trained checkpoint)
probe_env = gym.make(f'ALE/{game}-v5', obs_type='rgb')
num_actions = int(probe_env.action_space.n)
probe_env.close()

if model_name == 'mlp':
    from src.models.mlp_baseline import MLPBaseline
    m = MLPBaseline()
    m.load(f'checkpoints/mlp/{game.lower()}.pt')
    obs_type = 'ram'
elif model_name == 'iris':
    from src.models.iris_wrapper import IRISWrapper
    m = IRISWrapper(f'checkpoints/iris/{game}.pt', num_actions=num_actions)
    m.load_checkpoint()
    obs_type = 'rgb'
elif model_name == 'diamond':
    from src.models.diamond_wrapper import DIAMONDWrapper
    m = DIAMONDWrapper(
        checkpoint_path=f'checkpoints/diamond/{game}.pt',
        agent_config_path='checkpoints/diamond/agent_config.yaml',
        num_actions=num_actions,
    )
    m.load_checkpoint()
    obs_type = 'rgb'
else:
    raise ValueError(f'Unknown model {model_name}')

result = rollout(
    model_name=model_name, model=m, game=game,
    n_trajectories=n_traj, K=K, seed=seed,
    actions_path=actions_path, obs_type=obs_type,
)
print(f'Done: pred_frames={result[\"pred_frames\"].shape}')
PYEOF
    "
