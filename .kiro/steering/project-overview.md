# Project Overview

## Research Question

How does prediction error accumulate over rollout horizons (k=1–50 autoregressive steps) across generative world models of different architectures?

## Context

This is a DS-GA 3001 013 (Spring 2026) research project by Alexandra Halfon and Mina Sha. The project compares four world model architectures on three Atari games, measuring how quickly each model's predictions diverge from ground truth when stepped autoregressively on its own outputs.

## Architectures Under Study

| Model | Type | Observation Space | Key Mechanism |
|-------|------|-------------------|---------------|
| **MLP** | Feedforward baseline | 128-byte RAM | 3-layer hidden-256 MLP predicting next RAM state |
| **IRIS** | Transformer + VQ-VAE | 64×64×3 pixel | Autoregressive transformer over discrete visual tokens |
| **DIAMOND** | Diffusion-based | 64×64×3 pixel | Conditional diffusion denoising with 4-frame context buffer |
| **DreamerV3** | RSSM (GRU latent) | 64×64×3 pixel (decoded from 84×84) | Recurrent state-space model with learned decoder |

## Games

- **Breakout** — paddle/ball dynamics, score counter
- **Pong** — two-player paddle game, simpler visual structure
- **Boxing** — two fighters, more complex spatial interactions

## Core Hypothesis

Error growth follows a power law: `E_k ~ c · k^α`, where α (the exponent) characterizes how fast each architecture compounds errors. Higher α means faster degradation. The scale-invariant critical horizon `k*` marks where predictions become unreliable.

## Key Design Decisions

1. **Fixed-action replay**: All models receive identical action sequences collected from IRIS's trained actor (not random policy). This ensures fair comparison — random actions push models OOD equally, obscuring architectural differences.

2. **K-survivor trajectories**: Only trajectories that survive all K=50 steps without termination are kept during action collection. Mid-rollout resets desynchronize the action/state pairing, making later actions effectively random.

3. **Autoregressive stepping**: After the initial observation, models never see real frames. They predict from their own previous predictions, measuring compounding error.

4. **MLP trained on IRIS-actor data**: The MLP is trained on transitions collected under the same IRIS actor policy used for evaluation. Training on random-policy data (legacy behavior) creates a train/eval distribution mismatch.

5. **MLP prediction clamping**: Predictions are clamped to [0, 1] before re-feeding to prevent one bad step from saturating the recurrence.

6. **MLP is qualitative only**: MLP operates in 128-dim RAM space; its α is not comparable to pixel-space models. It serves as a structural baseline.

7. **k_fit_min is per-model**: DIAMOND uses k_fit_min=6 (4-frame conditioning warm-up + k=5 transient). IRIS, DreamerV3, and MLP use k_fit_min=2 (only k=1 is a stateful-initial-step quirk). This avoids discarding valid early-horizon data from models that don't have DIAMOND's warm-up.

8. **k_star_multiplier=3**: Must be re-validated with a pilot run after the rollout-termination fix landed.

9. **Pixel size = 64×64×3**: All pixel-space models standardized to 64×64. DreamerV3 decodes at 84×84 then resizes to 64×64.

## Branch Structure

- `main` (formerly `clean-branch`) — production code used for all HPC runs
- Hosted at `github.com/alexandrahalfon/world-models`

All experiments run on NYU HPC cluster with GPU (A100) using Singularity containers.
