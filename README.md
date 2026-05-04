# GenAI World Model Project

**DS-GA 3001 013 — Spring '26** | Alexandra Halfon & Mina Sha

How does prediction error accumulate over rollout horizon across generative world models of different architectures?

Compares MLP, DreamerV3, IRIS, and DIAMOND on Atari (Breakout, Pong, Boxing), measuring E_k ~ c·k^α over k=1–50 steps.

## Setup

```bash
# PyTorch env (IRIS, DIAMOND, MLP, metrics)
bash scripts/setup_env_torch.sh

# JAX env (DreamerV3 only — cannot share env with PyTorch on HPC)
bash scripts/setup_env_jax.sh

# Download pretrained checkpoints
bash scripts/download_checkpoints.sh
```

## Running experiments

```
# 1. Train MLP baseline
sbatch scripts/slurm/train_mlp.sbatch

# 2. Collect action sequences (uses IRIS/DIAMOND trained actor — not random policy)
#    Then run all model rollouts (array job: 12 tasks, one per model-game pair)
sbatch scripts/slurm/run_rollouts.sbatch

# 3. Compute MSE, power-law fit, and frame visualizations (CPU)
sbatch scripts/slurm/run_experiments_cpu.sbatch

# 4. Compute FID and PCA-KL divergence (GPU — Inception network)
sbatch scripts/slurm/run_experiments_gpu.sbatch

# Or run everything locally after rollout caches exist:
bash scripts/run_all.sh
```

## Important notes

- **Two conda environments required** — JAX and PyTorch conflict on NYU Greene. See `CLAUDE.md`.
- **Pilot k_star_multiplier before full array job** — validate with n_traj=200 first.
- **Never delete `results/rollouts/`** — rollout caches are expensive to recompute.
- MLP alpha (RAM space) is not comparable to pixel-space model alpha; see `CLAUDE.md`.

## Structure

```
src/models/     Model wrappers (mlp, iris, diamond, dreamerv3)
src/rollout.py  Fixed-action replay engine with caching
src/metrics.py  MSE, power-law fit, FID, PCA-KL
src/visualize.py  Plotting utilities
experiments/    exp1 (error growth), exp2 (failure modes), exp3 (distributional)
scripts/        Training, setup, SLURM job scripts
configs/        experiment.yaml, models.yaml
results/        Outputs (rollouts/, exp1/, exp2/, exp3/)
```
