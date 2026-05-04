#!/bin/bash
#SBATCH --account=ds_ga_3001-2026sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --array=0-2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --job-name=rollout_iris
#SBATCH --output=/scratch/%u/GenAIProject/results/logs/rollout_iris_%A_%a.out
#SBATCH --requeue

source /scratch/$USER/GenAIProject/scripts/hpc_config.sh

GAMES=(Breakout Pong Boxing)
GAME=${GAMES[$SLURM_ARRAY_TASK_ID]}
echo "IRIS task $SLURM_ARRAY_TASK_ID: game=$GAME"

module purge

singularity exec --nv \
    --overlay "$OVERLAY_TORCH:ro" \
    "$SINGULARITY_SIF" \
    /bin/bash -c "
        source /ext3/env.sh
        conda activate env_torch
        cd $PROJECT
        python - <<'PYEOF'
import sys, yaml
sys.path.insert(0, '.')
import gymnasium as gym
import ale_py
gym.register_envs(ale_py)
from src.rollout import rollout

with open('configs/experiment.yaml') as f:
    cfg = yaml.safe_load(f)

game = '$GAME'
K = cfg['horizon_K']
n_traj = cfg['n_trajectories']
seed = cfg['seed']

probe_env = gym.make(f'ALE/{game}-v5', obs_type='rgb')
num_actions = int(probe_env.action_space.n); probe_env.close()

from src.models.iris_wrapper import IRISWrapper
m = IRISWrapper(f'checkpoints/iris/{game}.pt', num_actions=num_actions)
m.load_checkpoint()

result = rollout(
    model_name='iris', model=m, game=game,
    n_trajectories=n_traj, K=K, seed=seed,
    actions_path=f'data/atari/{game}_actions.npz', obs_type='rgb',
)
print(f'Done: pred_frames={result[\"pred_frames\"].shape}')
PYEOF
    "
