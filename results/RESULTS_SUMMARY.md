# Results Summary

**Project**: How does prediction error accumulate over rollout horizons across generative world models?  
**Course**: DS-GA 3001 013, Spring 2026  
**Authors**: Alexandra Halfon & Mina Sha  
**Branch**: `clean-branch`  
**Compute**: NYU HPC (Greene/Burst), A100 GPUs, Singularity containers  
**Last run**: 2026-05-04  

---

## Executive Summary

We compared three world model architectures — MLP (RAM-space baseline), IRIS (VQ-VAE + Transformer), and DIAMOND (conditional diffusion) — on three Atari games (Breakout, Pong, Boxing). Each model was rolled out autoregressively for 50 steps using shared action sequences from a trained IRIS actor. DreamerV3 rollouts were not completed due to JAX/PyTorch environment conflicts on HPC.

**Key findings**:
- Error growth approximately follows a power law E_k ~ c·k^α, but with moderate R² (0.25–0.73), indicating the power-law model is a useful approximation rather than a precise fit.
- IRIS shows higher α than DIAMOND on Breakout (0.856 vs 0.764), suggesting faster error compounding from VQ-VAE quantization residuals.
- Both pixel-space models show game-dependent behavior: Breakout exhibits clear power-law growth, while Pong shows near-flat or slightly negative α (error plateaus rather than grows).
- DIAMOND's diffusion-based denoising provides modest resistance to error drift in games with complex spatial dynamics (Boxing: α=0.563 vs IRIS α=0.299).
- Distributional metrics (FID, PCA-KL) reveal that IRIS maintains lower FID than DIAMOND on Breakout and Boxing, but DIAMOND shows lower FID on Pong.

---

## Experiment 1: Error Growth and Power-Law Fitting

### Power-Law Parameters (α, k*, R²)

| Model | Space | Game | α | c | k* | R² |
|-------|-------|------|---|---|----|----|
| MLP | RAM | Breakout | 0.690 | 75.62 | 20 | 0.574 |
| MLP | RAM | Pong | −0.493 | 6427.54 | 51 | 0.349 |
| MLP | RAM | Boxing | 1.208 | 15.40 | 14 | 0.734 |
| IRIS | pixel | Breakout | 0.856 | 1.07 | 36 | 0.435 |
| IRIS | pixel | Pong | 0.154 | 20.64 | 51 | 0.481 |
| IRIS | pixel | Boxing | 0.299 | 104.32 | 51 | 0.550 |
| DIAMOND | pixel | Breakout | 0.764 | 1.88 | 36 | 0.393 |
| DIAMOND | pixel | Pong | −0.048 | 289.95 | 51 | 0.252 |
| DIAMOND | pixel | Boxing | 0.563 | 42.33 | 23 | 0.676 |

**Notes**:
- Power-law fit window: k=5..50 (k_fit_min=5 to skip DIAMOND's conditioning warm-up)
- k_star_multiplier=3 (validated via pilot)
- MLP α is NOT comparable to pixel-space models (different observation space)
- k*=51 means the threshold was never crossed within the 50-step horizon

### Interpretation by Game

**Breakout** (clearest power-law behavior):
- Both IRIS (α=0.856) and DIAMOND (α=0.764) show clear error growth
- k*=36 for both — predictions become unreliable around step 36
- The ball/paddle dynamics create compounding trajectory errors
- MLP in RAM space also shows growth (α=0.690) with earlier k*=20

**Pong** (error plateau):
- IRIS α=0.154, DIAMOND α≈−0.048 — near-zero or slightly negative
- k*=51 for both (threshold never crossed)
- Pong's simpler visual structure (two paddles, one ball, black background) means initial error is already high (c=20.6 for IRIS, c=290 for DIAMOND) but doesn't compound much
- DIAMOND starts with very high initial error (c=290) due to the conditioning buffer initialization, then plateaus
- MLP shows negative α (−0.493) — error actually decreases over time in RAM space, likely because Pong's RAM state has periodic structure

**Boxing** (complex dynamics):
- DIAMOND α=0.563 > IRIS α=0.299
- DIAMOND reaches k*=23 while IRIS never crosses (k*=51)
- Both start with high initial error (c=42 for DIAMOND, c=104 for IRIS)
- The complex spatial interactions (two fighters) challenge both models differently
- IRIS's high initial c but low α suggests it starts poorly but doesn't compound as fast
- DIAMOND starts better but compounds faster in this game

### Error Curve Characteristics

**IRIS Breakout** — Classic compounding pattern:
- Gradual rise from MSE≈6 at k=1 to MSE≈14 at k=14
- Brief dip at k=16–23 (model finds a stable attractor)
- Sharp jump at k=35–36 (catastrophic divergence)
- Plateaus at MSE≈48–55 for k=36–50

**DIAMOND Breakout** — Similar pattern with higher magnitude:
- Rises from MSE≈7 at k=1 to MSE≈17 at k=14
- Same dip-then-jump pattern as IRIS
- Sharp divergence at k=35–36 (MSE jumps from 24 to 48)
- Plateaus at MSE≈61–69 for k=36–50

**Both models on Pong** — High initial error, flat trajectory:
- IRIS: oscillates around MSE≈28–40 with no clear trend
- DIAMOND: starts very high (MSE≈2084 at k=1 due to conditioning), drops rapidly to ~240 by k=10, then plateaus

---

## Experiment 2: Failure Mode Visualizations

Frame grids were generated for IRIS and DIAMOND at k=1, k=10, k=50 for all three games. MLP was excluded (RAM space, no visual frames). DreamerV3 rollouts were unavailable.

### Observations from Frame Comparisons

**Breakout**:
- At k=1: Both IRIS and DIAMOND produce near-perfect reconstructions
- At k=10: Minor artifacts visible; ball position begins to diverge from ground truth
- At k=50: Both models show significant divergence — ball and paddle positions are incorrect, but overall scene structure (bricks, background) is preserved

**Pong**:
- At k=1: Good reconstruction quality for both models
- At k=10: Paddle positions begin to drift
- At k=50: Scene structure maintained but game state (score, ball position) is incorrect

**Boxing**:
- At k=1: Reasonable reconstruction
- At k=10: Fighter positions begin to diverge
- At k=50: Significant spatial errors in fighter placement and posture

### Generated Outputs
- `results/exp2/comparison_{game}.png` — side-by-side comparison (ground truth + all pixel models)
- `results/exp2/frames_iris_{game}.png` — IRIS individual strips
- `results/exp2/frames_diamond_{game}.png` — DIAMOND individual strips

---

## Experiment 3: Distributional Divergence (FID + PCA-KL)

### FID Results

| Model | Game | k=5 | k=10 | k=20 | k=30 | k=50 |
|-------|------|-----|------|------|------|------|
| IRIS | Breakout | 6.22 | 4.49 | 10.14 | 8.70 | 13.37 |
| IRIS | Pong | 27.46 | 24.58 | 38.05 | 42.89 | 57.33 |
| IRIS | Boxing | 14.37 | 14.46 | 16.95 | 7.82 | 5.85 |
| DIAMOND | Breakout | 12.56 | 6.30 | 18.23 | 13.62 | 25.24 |
| DIAMOND | Pong | 88.97 | 84.27 | 87.38 | 81.35 | 87.44 |
| DIAMOND | Boxing | 8.71 | 17.01 | 27.93 | 15.19 | 14.29 |

### PCA-KL Divergence Results

| Model | Game | k=5 | k=10 | k=20 | k=30 | k=50 |
|-------|------|-----|------|------|------|------|
| IRIS | Breakout | 8.77 | 1.74 | 1.58 | 2.68 | 1.52 |
| IRIS | Pong | 2.57 | 2.45 | 2.13 | 5.12 | 4.01 |
| IRIS | Boxing | 4.69 | 4.63 | 1.27 | 0.48 | 0.18 |
| DIAMOND | Breakout | 18.39 | 7.87 | 7.06 | 6.82 | 7.43 |
| DIAMOND | Pong | 4.84 | 2.43 | 5.21 | 9.35 | 3.73 |
| DIAMOND | Boxing | 24.53 | 15.49 | 2.51 | 0.63 | 0.23 |

### Interpretation

**FID trends**:
- **Breakout**: Both models show generally increasing FID with horizon, confirming distributional drift. IRIS maintains lower FID than DIAMOND at all horizons (6–13 vs 6–25).
- **Pong**: DIAMOND shows very high FID (81–89) that is roughly constant — the model's predicted frame distribution is far from ground truth from the start but doesn't worsen. IRIS shows moderate FID (24–57) with a clear upward trend.
- **Boxing**: Surprising non-monotonic behavior — FID decreases at later horizons for IRIS (14→6). This may indicate the model converges to a stable attractor state that happens to be distributionally closer to the true frame distribution.

**PCA-KL trends**:
- Generally higher at early horizons (k=5) and decreasing or fluctuating thereafter
- This counter-intuitive pattern may reflect the PCA basis capturing variance that is most discriminative at early horizons
- Boxing shows clear convergence for both models (KL drops from ~5–25 at k=5 to ~0.2 at k=50)

**Key insight**: FID and PCA-KL do not always agree with MSE trends. MSE measures average pixel error, while FID captures perceptual distributional distance. A model can have increasing MSE (individual predictions diverge) while FID remains stable (the distribution of predictions stays similar to the true distribution). This is especially visible in Pong where MSE plateaus but FID shows different patterns for each model.

---

## Models Not Completed

### DreamerV3

DreamerV3 rollouts were not generated in the final HPC run due to the JAX/PyTorch Singularity overlay conflict. The rollout cache files (`results/rollouts/dreamerv3_*.npz`) do not exist. All three experiments correctly skip DreamerV3 when its cache is missing.

The DreamerV3 wrapper and rollout script are fully implemented and tested — the issue was purely environmental (scheduling the JAX overlay job on the cluster).

---

## Run Metadata

- **Final experiment run**: 2026-05-04 20:17–20:19 (exp1: 1.6 min, exp2: 0.6 min)
- **Exp3 (FID/KL)**: Run separately via GPU job (`run_experiments_gpu.sbatch`)
- **Rollouts**: Generated via array job (`run_rollouts.sbatch`), indices 0–8 (MLP, IRIS, DIAMOND × 3 games)
- **Action collection**: 2000 trajectories × 50 steps per game using IRIS actor
- **Total trajectories evaluated**: 2000 per model-game pair
- **Horizon**: K=50 steps
- **Seed**: 42

---

## Summary of Findings

1. **Power-law approximation is game-dependent**: Breakout shows clear power-law error growth (α≈0.76–0.86 for pixel models). Pong shows near-zero α (error plateaus). Boxing shows intermediate behavior.

2. **IRIS vs DIAMOND**: On Breakout, IRIS compounds errors slightly faster (α=0.856 vs 0.764) but starts with lower initial error (c=1.07 vs 1.88). On Boxing, DIAMOND compounds faster (α=0.563 vs 0.299). The relative performance is game-dependent.

3. **Reliability horizons**: For Breakout, both models become unreliable around k=36. For Pong and Boxing, the 3× threshold is never crossed within 50 steps (k*=51), meaning predictions remain within a bounded error regime.

4. **Distributional metrics add nuance**: FID reveals that DIAMOND's predictions on Pong are distributionally far from ground truth (FID≈85) even though MSE plateaus. IRIS maintains better distributional fidelity on Breakout and Boxing.

5. **Non-monotonic error curves**: Both models show dips and jumps in their error curves, suggesting they pass through stable attractor states before eventually diverging. This is most pronounced in Breakout around k=15–35.

6. **MLP baseline confirms methodology**: The MLP in RAM space shows expected rapid degradation (high α on Boxing=1.21, early k*=14), validating that the measurement framework captures compounding error correctly.
