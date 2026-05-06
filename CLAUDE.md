# GenAI World Model Project

## Research Goal
Compare how prediction error accumulates over rollout horizons (k=1–50) across four world model architectures (MLP, DreamerV3, IRIS, DIAMOND) on Atari games (Breakout, Pong, Boxing).

## Critical Constraint: Two Singularity Overlays

NYU Greene uses **Singularity containers with conda overlay files** (not plain conda).
JAX and PyTorch also conflict on CUDA versions — they must never share an overlay.

| Overlay | Contents | Used for |
|---------|----------|----------|
| `env_torch.ext3` | PyTorch, IRIS, DIAMOND, MLP, all metrics/viz | Everything except DreamerV3 |
| `env_jax.ext3` | JAX, Flax, Optax, official DreamerV3 repo | DreamerV3 rollout only |

All paths are configured in `scripts/hpc_config.sh` — edit `NETID` and `SINGULARITY_SIF` before running anything.

Activation inside Singularity uses `source /ext3/env.sh`, not `conda activate` alone. Template:
```bash
singularity exec --nv \
    --overlay "$OVERLAY_TORCH:ro" \
    "$SINGULARITY_SIF" \
    /bin/bash -c "source /ext3/env.sh; conda activate env_torch; python ..."
```

`dreamerv3_wrapper.py` dispatches to `scripts/dreamerv3_rollout.py` via `singularity exec` with `OVERLAY_JAX`.
**Set `SINGULARITY_SIF` and `OVERLAY_JAX` env vars (via `source scripts/hpc_config.sh`) before running.**
**Test the subprocess interactively on a compute node before submitting SLURM array jobs.**

## MLP Baseline: RAM Space, Not Pixels

The MLP operates on Atari's 128-byte RAM state (not 84×84 pixel frames). A 3-layer hidden-256 MLP cannot learn anything useful in 28,224-dim pixel space. Because of this, the MLP's error-growth α is **not comparable** to pixel-space models (IRIS, DIAMOND, DreamerV3). The analysis notebook separates them using the `space` field in `configs/experiment.yaml`. MLP is a qualitative reference only.

## Fixed-Action Replay

All rollouts use a shared action sequence collected from the IRIS trained actor (not a random policy — random actions push models OOD from their training distribution and inflate error equally across all architectures, obscuring differences). Actions are saved to `data/atari/{game}_actions.npz` and reused across all models.

## Rollout Caching

`results/rollouts/{model}_{game}.npz` is expensive to recompute. **Never delete this directory.** `rollout.py` skips computation if the cache file already exists.

## Autoregressive Stepping

After step 0, models never receive real observations. They are stepped on their own predictions:
```python
model_obs = model.predict(model_obs, action)  # chained from t=1 onward
```
`true_frames` come from the real Atari env; `pred_frames` come from chained model output. This is intentional — it is the compounding error being measured.

## k* Definition

`k*` is the first step k where `E_k > mean(E_k[k_fit_min-1 : k_fit_min+4]) * k_star_multiplier`. Scale-invariant across RAM and pixel-space models. `k_star_multiplier` and `k_fit_min` live in `configs/experiment.yaml` and **must be validated with a pilot run before submitting full SLURM array jobs**.

`k_fit_min=6` skips DIAMOND's 4-step conditioning warm-up plus the residual k=5 transient (the buffer at k=5 holds 4 predictions, but those predictions were themselves generated from a partially init-seeded buffer).

## Pixel size

Pixel-space models (IRIS, DIAMOND, DreamerV3) all store frames at 64×64×3. `src/rollout.PIXEL_SIZE = 64`; the DreamerV3 JAX rollout decodes at the native encoder resolution (84×84) then resizes to 64×64 before saving. `configs/models.yaml` and `scripts/generate_synthetic_rollouts.py` must match.

## Rollout termination handling

`src/rollout.py` writes a `valid_mask[n_traj, K]` into the rollout cache. Trajectories that survive K steps without termination are marked True throughout; trajectories that terminate at step t have valid_mask set to False for k >= t. `per_step_mse` and exp3 honor this mask — they do *not* compute MSE/FID/KL between a frozen "last real frame" and the model's still-running prediction. Action collection (`scripts/collect_actor_actions.py` and `src/rollout.collect_actions`) discards trajectories that terminate before K steps and persists per-trajectory seeds alongside the actions, so the action sequence remains valid for the entire horizon when replayed.

## FID Notes

- Minimum 2,000 samples per distribution for reliable FID estimates.
- k<6 FID is excluded (warm-up + near-identical frames → unstable estimate with high variance). FID evaluation starts at k=6.
- FID uses the Inception network (GPU-accelerated) — use `run_experiments_gpu.sbatch`, not the CPU job.
- `compute_fid` adds `cov_ridge=1e-4` to both empirical covariances (1e-6 was insufficient — the LinAlgWarning still fired at N≈2000 in 2048-dim activation space).

## PCA-KL

Use `PCA(n_components=0.95)` in scikit-learn to dynamically retain 95% of variance. Do not hardcode `n_components=50` — on 64×64×3 pixel data that may explain only 20–30% of variance.

The PCA basis used by `compute_pca_kl` is fit on a **stationary reference**: true frames at k=1 across all trajectories. Fitting on true frames pooled across horizons (the prior behavior) bakes the very horizon-drift you're trying to measure into the basis itself, producing non-monotone KL trajectories.
