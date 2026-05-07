# Conventions and Pitfalls

## Code Conventions

- **Python 3.10+** with type hints (PEP 604 union syntax `X | None`)
- **NumPy arrays** for all data interchange between modules
- **uint8 [0, 255]** for frame storage and model I/O boundaries
- **float32/float64** for internal computation (metrics, training)
- Frames are stored as `[n_traj, K, H, W, C]` (channels-last) in `.npz` caches
- All pixel-space frames are **64×64×3** (PIXEL_SIZE=64 in `src/rollout.py`)
- All random state uses `np.random.default_rng(seed)` for reproducibility
- Logging goes to both stdout and timestamped files in `results/logs/`
- Rollout caches include `valid_mask[n_traj, K]` alongside pred_frames, true_frames, actions

## Critical Pitfalls

### Never Delete Rollout Caches
`results/rollouts/{model}_{game}.npz` files are expensive to recompute (hours of GPU time per model-game pair). The rollout engine (`src/rollout.py`) automatically skips computation if the cache exists.

### Two Conda Environments Required — No Nesting
JAX and PyTorch conflict on CUDA versions on NYU Greene. They must live in separate Singularity overlays. Singularity does NOT support nested containers. DreamerV3 must be launched from its own sbatch (`run_rollouts_dreamerv3.sbatch`), not from inside the env_torch container.

### MLP Alpha Is Not Comparable
The MLP operates in 128-byte RAM space. Its error-growth exponent α cannot be directly compared to pixel-space models. The analysis separates them using the `space` field in config.

### MLP Must Be Trained on IRIS-Actor Data
The MLP is evaluated on rollouts using IRIS-actor action sequences. If trained on random-policy data (legacy `--random_policy` flag), there's a train/eval distribution mismatch that produces artificially high error. Default behavior now uses IRIS actor for training data collection.

### MLP Prediction Clamping
`mlp_baseline.py` clamps predictions to [0, 1] in float space before converting to uint8 and re-feeding. Without this, an unbounded prediction propagates through the recurrence and can saturate the network on a single bad step (root cause of the old MLP Pong α = −0.49).

### DIAMOND Conditioning Warm-Up (k_fit_min=6, DIAMOND only)
DIAMOND requires 4 conditioning frames. At reset, the initial observation is replicated 4× into the buffer. The buffer transitions are:
- k=1: buffer=[init, init, init, init]
- k=2: buffer=[init, init, init, pred1]
- k=3: buffer=[init, init, pred1, pred2]
- k=4: buffer=[init, pred1, pred2, pred3]
- k=5: buffer=[pred1, pred2, pred3, pred4] — first "clean" buffer, but predictions were generated from partly init-seeded buffers
- k=6: first step where the buffer is fully self-consistent

DIAMOND uses `k_fit_min=6`. Other models (IRIS, DreamerV3, MLP) use `k_fit_min=2` — they don't have this warm-up and their early-horizon data is valid. The config is now a per-model dict.

### FID Covariance Ridge = 1e-4
With N≈2000 and 2048-dim Inception activations, the empirical covariance is rank-deficient. `cov_ridge=1e-6` was insufficient (LinAlgWarning still fired). Bumped to 1e-4 — tiny relative to typical Fréchet distances, preserves relative ordering.

### PCA Basis Must Be Stationary (k=1 True Frames)
The PCA basis for `compute_pca_kl` is fit on true frames at k=1 only. Fitting on frames pooled across horizons (old behavior) bakes horizon-induced drift into the basis itself, producing non-monotone KL trajectories that are incomparable across k.

### K-Survivor Action Collection
`collect_actor_actions.py` only keeps trajectories that survive all K=50 steps without termination. Mid-rollout resets desynchronize the action/state pairing — after a reset, the saved actions correspond to a different game state than what the model will see during rollout replay. Per-trajectory seeds are saved alongside actions for exact env replay.

### DreamerV3: imagine() Not observe()
After step 0, `dreamerv3_rollout.py` uses `agent.world_model.imagine()` (pure latent dynamics) NOT `observe()`. Using observe() re-encodes a real observation at each step, which leaks ground truth into the prediction and invalidates the autoregressive protocol.

### DreamerV3 API Verification
The DreamerV3 package API may differ across versions. `dreamerv3_rollout.py` checks for required symbols (`configs.atari`, `Agent`, `world_model.observe/imagine/decode`) at startup and fails with clear instructions. Always dry-run with `--n_traj 4 --K 10` before submitting the full array job.

### Pilot Before Full SLURM Array
Always validate `k_star_multiplier` with a small pilot run (n_traj=200, single model/game) before submitting the full array job. The multiplier must be re-validated after the rollout-termination fix landed — prior results were tuned against buggy rollouts.

## File Naming Conventions

- Rollout caches: `results/rollouts/{model}_{game}.npz` (contains pred_frames, true_frames, actions, valid_mask)
- Action files: `data/atari/{game}_actions.npz` (contains actions, seeds)
- Error CSVs: `results/exp1/error_growth_{model}_{game}.csv`
- Divergence CSVs: `results/exp3/divergence_{model}_{game}.csv`
- Plots: `results/exp{N}/{type}_{game}.png` or `results/exp{N}/{type}_{model}_{game}.png`
- Logs: `results/logs/{source}_{timestamp}.log`
- Reports: `results/report_{YYYYMMDD_HHMMSS}.html`
- Checkpoints: `checkpoints/{model}/{game}.pt` (or directory for DreamerV3)

## Testing Locally

Use `scripts/run_local.sh` for smoke testing. It:
1. Generates synthetic rollouts at 64×64×3 with known α values and valid_mask (no GPU needed)
2. Runs exp1 (power-law fit) and exp2 (frame grids)
3. Skips exp3 (FID requires GPU)
4. Generates an HTML report

Synthetic rollouts use `scripts/generate_synthetic_rollouts.py` which injects controlled noise with known power-law exponents for validation. All synthetic valid_masks are True (no simulated terminations).
