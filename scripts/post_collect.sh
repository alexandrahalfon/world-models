#!/bin/bash
cd /scratch/aih7261/GenAIProject

echo "=== Verify actor distributions ==="
python3 -c "
import numpy as np
for g in ['Breakout','Pong','Boxing']:
    a = np.load(f'data/atari/{g}_actions.npz')['actions']
    u, c = np.unique(a, return_counts=True)
    pct = (c / c.sum() * 100).round(1)
    print(f'{g}: shape={a.shape}, dist={dict(zip(u.tolist(), pct.tolist()))} %')
"

echo
echo "=== Clear stale random-action rollout caches ==="
rm -rf results/rollouts results/_pilot results/_pilot200
mkdir -p results/rollouts results/logs

echo
echo "=== Submit 9-task rollout array ==="
sbatch scripts/slurm/run_rollouts.sbatch
sleep 2
squeue -u $USER
