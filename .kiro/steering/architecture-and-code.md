# Architecture and Code Structure

## Directory Layout

```
GenAIProject/
├── configs/
│   ├── experiment.yaml      # Central experiment parameters (games, K, n_traj, k_star_multiplier, models)
│   └── models.yaml          # Model-specific hyperparameters (dims, lr, train_steps, obs_shape)
├── src/
│   ├── models/
│   │   ├── mlp_baseline.py      # MLP world model (RAM space, clamped predictions, stateful interface)
│   │   ├── iris_wrapper.py      # IRIS wrapper (vendor/iris/, VQ-VAE + Transformer, KV-cache)
│   │   ├── diamond_wrapper.py   # DIAMOND wrapper (vendor/diamond/, diffusion, 4-frame deque)
│   │   └── dreamerv3_wrapper.py # DreamerV3 wrapper (refuses nested containers, points to sbatch)
│   ├── rollout.py           # Fixed-action replay engine with K-survivor collection, valid_mask, caching
│   ├── metrics.py           # MSE (with valid_mask), power-law fit, FID (cov_ridge=1e-4), PCA-KL
│   └── visualize.py         # Plotting utilities (error curves, frame grids, distributional)
├── experiments/
│   ├── exp1_error_growth.py     # Power-law fitting and error curves (honors valid_mask)
│   ├── exp2_failure_modes.py    # Frame visualization at k=1,10,50
│   └── exp3_distributional.py   # FID and PCA-KL (stationary PCA basis at k=1, honors valid_mask)
├── scripts/
│   ├── slurm/                   # SBATCH job scripts for HPC
│   │   ├── collect_actor_actions.sbatch  # 5h, K-survivor collection
│   │   ├── train_mlp.sbatch             # 14h, IRIS-actor training data
│   │   ├── run_rollouts.sbatch          # Array 0-8 (MLP, IRIS, DIAMOND × 3 games)
│   │   ├── run_rollouts_dreamerv3.sbatch # Array 0-2 (DreamerV3 × 3 games, separate overlay)
│   │   ├── run_experiments_cpu.sbatch    # exp1 + exp2
│   │   └── run_experiments_gpu.sbatch    # exp3 (FID needs GPU)
│   ├── collect_actor_actions.py # K-survivor action collection from IRIS actor
│   ├── train_mlp.py            # MLP training (IRIS-actor data by default, --random_policy for legacy)
│   ├── dreamerv3_rollout.py    # JAX-only DreamerV3 rollout (imagine() after step 0, API guards)
│   ├── generate_report.py      # HTML report generator (base64-embedded images)
│   ├── generate_synthetic_rollouts.py  # Fake 64×64 rollouts with valid_mask for local testing
│   ├── hpc_config.sh           # Central HPC paths (NETID, overlays, SIF)
│   ├── run_all.sh              # Full pipeline in Singularity
│   └── run_local.sh            # Local smoke test with synthetic data
├── data/atari/                  # Pre-collected action sequences + seeds (.npz)
├── results/
│   ├── rollouts/                # Cached model rollouts with valid_mask (expensive, never delete)
│   ├── exp1/                    # Error growth CSVs and plots
│   ├── exp2/                    # Frame comparison images
│   ├── exp3/                    # FID/KL CSVs and plots
│   ├── logs/                    # SLURM and local run logs
│   └── report_*.html           # Self-contained HTML reports
├── checkpoints/                 # Pretrained model weights (not in git)
└── vendor/                      # IRIS and DIAMOND source repos (not in git)
```

## Model Interface Contract

All models implement a stateful interface consumed by `src/rollout.py`:

```python
class ModelWrapper:
    def reset(self, obs_uint8: np.ndarray) -> np.ndarray:
        """Initialize from a real observation. Returns the model's reconstruction."""
        ...

    def step(self, action: int) -> np.ndarray:
        """Predict next observation given action. Autoregressive — uses internal state."""
        ...
```

- **MLP**: Stateless predict wrapped in stateful interface. Stores `_cur_obs`, chains predictions. Output clamped to [0,1] before re-feeding to prevent saturation.
- **IRIS**: Uses WorldModelEnv with KV-cache. `reset_from_initial_observations()` + `step()`.
- **DIAMOND**: Maintains a 4-frame sliding deque. Diffusion sampler conditions on the sliding window. 3 denoising steps per prediction.
- **DreamerV3**: Dispatched via separate sbatch (`run_rollouts_dreamerv3.sbatch`). The wrapper refuses to run inside a Singularity container (nested containers not supported). Uses `imagine()` after step 0 (not `observe()`, which would leak ground truth).

## Metrics Module (`src/metrics.py`)

| Function | Purpose |
|----------|---------|
| `per_step_mse(pred, true, valid_mask)` | MSE at each horizon k, masked to live trajectories only |
| `fit_power_law(E_k, k_star_multiplier, k_min)` | Log-log linear fit → α, c, k*, R² |
| `compute_fid(pred_k, true_k, cov_ridge=1e-4)` | Fréchet Inception Distance (ridge regularized) |
| `fit_reference_pca(frames, n_components)` | Fit shared PCA basis on stationary reference (k=1 true frames) |
| `compute_pca_kl(pred_k, true_k, pca)` | KL divergence in PCA-projected space |

## Visualization Module (`src/visualize.py`)

| Function | Output |
|----------|--------|
| `plot_error_curves()` | E_k vs k line plots per game (all models or pixel-only) |
| `plot_frame_grid()` | Side-by-side predicted frames at k=1,10,50 with visible row labels |
| `plot_distributional_divergence()` | FID + PCA-KL curves vs horizon |
| `plot_pca_scatter()` | 2D PCA scatter of true vs predicted frame distributions |

## Configuration

All experiment parameters live in `configs/experiment.yaml`:
- `horizon_K: 50` — maximum rollout steps
- `n_trajectories: 2000` — minimum for reliable FID
- `k_star_multiplier: 3` — must be re-validated after rollout-termination fix
- `k_fit_min: 6` — skip DIAMOND conditioning warm-up (k=1–5)
- `fid_horizons: [6, 10, 20, 30, 50]` — k<6 excluded (warm-up + unstable FID)
- `seed: 42` — reproducibility

All pixel models use `obs_shape: [64, 64, 3]` in `configs/models.yaml`.
