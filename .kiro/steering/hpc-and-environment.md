# HPC Environment and Execution

## Cluster

NYU Greene HPC cluster, accessed via `ssh greene.hpc.nyu.edu` then `ssh burst`.

## Two Singularity Overlays (Critical Constraint)

JAX and PyTorch conflict on CUDA versions — they MUST NEVER share an overlay. Singularity does NOT support nested containers.

| Overlay | Contents | Used For |
|---------|----------|----------|
| `env_torch.ext3` | PyTorch, IRIS, DIAMOND, MLP, all metrics/viz | Everything except DreamerV3 |
| `env_jax.ext3` | JAX, Flax, Optax, DreamerV3 repo | DreamerV3 rollout only |

Singularity image: `cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif`

Activation pattern inside Singularity:
```bash
singularity exec --nv \
    --overlay "$OVERLAY_TORCH:ro" \
    "$SINGULARITY_SIF" \
    /bin/bash -c "source /ext3/env.sh; conda activate env_torch; python ..."
```

## Central Config: `scripts/hpc_config.sh`

All SLURM scripts source this file. Key variables:
- `NETID="aih7261"`
- `SCRATCH="/scratch/$NETID"`
- `PROJECT="$SCRATCH/GenAIProject"`
- `SLURM_ACCOUNT="ds_ga_3001-2026sp"`
- `SINGULARITY_SIF`, `OVERLAY_TORCH`, `OVERLAY_JAX`

Edit `NETID` before running on a different account.

## SLURM Job Pipeline

### Step 1: Train MLP baseline
```bash
sbatch scripts/slurm/train_mlp.sbatch
```
- Partition: `c12m85-a100-1` (1 A100 GPU)
- Time: 14 hours
- Collects 500K transitions per game using IRIS actor (not random policy)
- Trains for 200K steps
- Uses two parallel envs (RGB for IRIS actor decisions, RAM for MLP training data)

### Step 2: Collect action sequences
```bash
sbatch scripts/slurm/collect_actor_actions.sbatch
```
- Time: 5 hours (Pong games end early under IRIS actor, requiring many retries)
- Uses IRIS's trained actor-critic to collect 2000 **K-survivor** trajectories × 50 steps per game
- Only keeps trajectories that survive all 50 steps without termination
- Saves actions + per-trajectory seeds to `data/atari/{game}_actions.npz`
- These actions are shared across ALL models for fair comparison

### Step 3a: Run rollouts — PyTorch models (array job)
```bash
sbatch scripts/slurm/run_rollouts.sbatch
```
- Array job: indices 0–8 (MLP, IRIS, DIAMOND × 3 games)
- Each task loads a model checkpoint, runs 2000 autoregressive rollouts of 50 steps
- Caches results to `results/rollouts/{model}_{game}.npz` (includes valid_mask)
- **NEVER delete rollout caches** — they are expensive to recompute

### Step 3b: Run rollouts — DreamerV3 (separate array job)
```bash
sbatch scripts/slurm/run_rollouts_dreamerv3.sbatch
```
- Array job: indices 0–2 (DreamerV3 × 3 games)
- Runs directly in env_jax overlay (NOT nested inside env_torch)
- Two-pass approach:
  - Pass 1 (env_jax): Run DreamerV3 imagination loop → pred_frames
  - Pass 2 (env_torch): Collect true_frames from ALE + append valid_mask
- DRY-RUN with `--n_traj 4 --K 10` first to verify API compatibility

### Step 4a: CPU experiments (exp1 + exp2)
```bash
sbatch scripts/slurm/run_experiments_cpu.sbatch
```
- Power-law fitting, MSE computation (with valid_mask), frame grid visualization

### Step 4b: GPU experiments (exp3)
```bash
sbatch scripts/slurm/run_experiments_gpu.sbatch
```
- FID computation requires Inception network on GPU
- PCA-KL divergence with stationary basis (k=1 true frames)
- Honors valid_mask — skips horizons with <100 live trajectories

## Important Operational Notes

1. **Pilot before full array**: Always validate k_star_multiplier with n_traj=200 on a single model/game before submitting the full array job. The multiplier must be re-validated after the rollout-termination fix.

2. **DreamerV3 cannot nest**: The wrapper detects `SINGULARITY_CONTAINER` / `APPTAINER_CONTAINER` env vars and refuses to run. Use the dedicated `run_rollouts_dreamerv3.sbatch` which launches env_jax directly from the compute node.

3. **DreamerV3 API verification**: Run `dreamerv3_rollout.py` with `--n_traj 4 --K 10` interactively first. The script checks for `dreamerv3.configs.atari`, `dreamerv3.Agent`, and `agent.world_model.{observe,imagine,decode}` and fails loudly with instructions if any are missing.

4. **Rollout caching**: `src/rollout.py` skips computation if the cache file exists. To regenerate, manually delete the specific `.npz` file.

5. **Resource allocation**: Rollouts need GPU (model inference). Experiments 1–2 are CPU-only. Experiment 3 needs GPU for Inception.

6. **Action collection is slow for Pong**: The IRIS actor frequently terminates Pong games before K=50, requiring many retry attempts. The 5h time limit accounts for this.

## Local Development

For local testing without HPC:
```bash
bash scripts/run_local.sh
```
This generates synthetic rollouts (64×64×3, with valid_mask) with known α values and runs exp1 + exp2 + report generation. Exp3 (FID) is skipped locally — too slow on CPU.
