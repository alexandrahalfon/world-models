# Experiments and Metrics

## Experiment 1: Error Growth Curves and Power-Law Fitting (H1)

**File**: `experiments/exp1_error_growth.py`

**What it measures**: Per-step mean squared error E_k between model predictions and ground truth at each rollout horizon k=1..50, then fits a power law E_k ~ c · k^α.

**Procedure**:
1. Load cached rollout data (pred_frames, true_frames, valid_mask) for each model × game
2. Compute E_k = mean((pred - true)²) averaged over **live** trajectories only (valid_mask honored)
3. Fit log(E_k) = α·log(k) + log(c) via least squares on k=k_fit_min..K
4. Compute k* = first k where E_k > 3× baseline (mean of first 5 reliable steps starting at k_fit_min)
5. Report R² goodness-of-fit

**Key parameters**:
- `k_fit_min`: Per-model exclusion window (dict in config):
  - `diamond: 6` — skip 4-frame conditioning warm-up + k=5 transient
  - `iris: 2` — KV-cache initialized from real-obs tokens; only k=1 quirky
  - `dreamerv3: 2` — RSSM observed once at reset
  - `mlp: 2` — stateless
- `k_star_multiplier = 3`: Must be re-validated after the rollout-termination fix
- `valid_mask`: When present in cache, MSE only averages over trajectories that haven't terminated

**Outputs**:
- `results/exp1/error_growth_{model}_{game}.csv` — per-step MSE values
- `results/exp1/alpha_kstar_table.csv` — summary of α, c, k*, R² for all model-game pairs
- `results/exp1/error_curves_{game}.png` — all models plotted together
- `results/exp1/error_curves_pixel_{game}.png` — pixel-space models only (valid comparison)

**Why**: α quantifies the rate of error compounding. A model with lower α maintains prediction fidelity longer. k* gives a practical "reliability horizon."

---

## Experiment 2: Architecture-Specific Failure Modes (H2)

**File**: `experiments/exp2_failure_modes.py`

**What it measures**: Qualitative visual inspection of how each model's predictions degrade over time.

**Procedure**:
1. For each pixel-space model × game, select a representative trajectory (middle index, traj_idx=1000)
2. Render predicted frames at k=1, k=10, k=50 side-by-side with ground truth
3. Generate per-model strips and cross-model comparison grids

**Frame grid layout** (comparison_{game}.png):
- Row 1: Ground truth
- Row 2: IRIS
- Row 3: DIAMOND
- Columns: k=1, k=10, k=50

**Expected failure signatures**:
- **IRIS**: Blur and spectral artifacts from VQ-VAE quantization residuals compounding
- **DIAMOND**: Maintains visual coherence longer (diffusion denoising resists drift), but eventually diverges
- **DreamerV3**: Geometrically plausible but structurally incorrect (GRU state drift)
- **MLP**: Skipped (RAM space, no visual frames)

**Outputs**:
- `results/exp2/frames_{model}_{game}.png` — individual model frame strips
- `results/exp2/comparison_{game}.png` — all pixel models vs ground truth (row labels now visible)

---

## Experiment 3: Distributional Divergence (H3)

**File**: `experiments/exp3_distributional.py`

**What it measures**: How the distribution of predicted frames diverges from the true frame distribution as rollout horizon increases.

**Metrics**:
- **FID (Fréchet Inception Distance)**: Distributional distance in Inception-v3 feature space (2048-dim). Captures perceptual quality differences. Regularized with `cov_ridge=1e-4`.
- **PCA-KL Divergence**: Projects frames into a shared PCA basis, then computes KL divergence between binned marginal distributions per component.

**Procedure**:
1. For each pixel-space model × game, load rollout cache
2. Fit a shared PCA basis on **true frames at k=1** (stationary reference — NOT pooled across horizons)
3. At each horizon in `fid_horizons = [6, 10, 20, 30, 50]`:
   - Filter to live trajectories only (valid_mask; skip if <100 live)
   - Compute FID between pred_frames[:, k] and true_frames[:, k]
   - Compute PCA-KL between the same frame sets using the shared basis
4. Save CSV and plot FID + KL curves

**Key design decisions**:
- k<6 excluded from FID: warm-up + near-identical frames produce unstable estimates
- **Stationary PCA basis at k=1**: Fitting on pooled horizons (old behavior) baked drift into the basis itself, producing non-monotone KL. The k=1 basis captures the true-data manifold.
- Covariance ridge = 1e-4: 1e-6 was insufficient to suppress LinAlgWarning at N≈2000 in 2048-dim space
- Minimum 2000 samples per distribution for reliable FID
- valid_mask honored: only live trajectories contribute to FID/KL

**Outputs**:
- `results/exp3/divergence_{model}_{game}.csv` — FID and KL at each horizon
- `results/exp3/distributional_{game}.png` — dual-panel FID + KL plots
- `results/exp3/pca_scatter_{model}_{game}_k50.png` — 2D PCA scatter at k=50

---

## Metrics Reference

| Metric | Formula / Method | Interpretation |
|--------|-----------------|----------------|
| E_k (MSE) | mean((pred_k - true_k)²) over live trajectories and pixels | Average pixel-level prediction error at step k |
| α (alpha) | Slope of log(E_k) vs log(k), fit on k=k_fit_min..50 (per-model) | Rate of error compounding (higher = faster degradation) |
| c | Intercept of power-law fit | Initial error scale |
| k* | First k ≥ k_fit_min where E_k > 3× baseline | Practical reliability horizon |
| R² | 1 - SS_res/SS_tot on log-log fit | Goodness of power-law fit (1.0 = perfect) |
| FID | Fréchet distance between Inception activation Gaussians (ridge=1e-4) | Perceptual distributional distance |
| PCA-KL | Mean KL divergence across PCA components (basis fit at k=1) | Low-dimensional distributional divergence |

## Important Caveats

1. **MLP α is not comparable to pixel models**: MLP operates in 128-dim RAM space. Its α reflects error compounding in a fundamentally different representation.

2. **k_star_multiplier needs re-validation**: The prior value (3) was tuned against rollouts that contained the mid-rollout reset bug. After the K-survivor fix, error curves will be different and the multiplier may need adjustment.

3. **DreamerV3 uses `imagine()` not `observe()`**: After step 0, the rollout script uses pure latent imagination. Using `observe()` would re-encode a real observation at each step, leaking ground truth and invalidating the autoregressive protocol.
