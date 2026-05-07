# Rollout Pipeline

## Overview

The rollout pipeline is the most computationally expensive part of the project. It generates the raw data (predicted vs true frames) that all three experiments consume.

## Pipeline Stages

```
1. Train MLP (IRIS-actor data)  →  checkpoints/mlp/{game}.pt
2. Collect K-survivor actions   →  data/atari/{game}_actions.npz (actions + seeds)
3. Run rollouts (torch models)  →  results/rollouts/{model}_{game}.npz
4. Run rollouts (DreamerV3)     →  results/rollouts/dreamerv3_{game}.npz
5. Run experiments              →  results/exp{1,2,3}/
6. Generate report              →  results/report_*.html
```

## Stage Details

### 1. Action Collection (`scripts/collect_actor_actions.py`)

- Loads IRIS checkpoint for each game
- Runs IRIS's trained actor-critic in the real ALE environment
- **K-survivor protocol**: Only keeps trajectories that survive all 50 steps without termination
  - Discards trajectories that terminate early and retries with a new seed
  - This is critical: mid-rollout resets desynchronize the action/state pairing
  - After a reset, saved actions correspond to a different game state than what models will see during rollout replay
  - Pong is particularly slow (games end early under IRIS actor, requiring many retries)
- Saves action arrays + per-trajectory seeds to `data/atari/{game}_actions.npz`
- Seeds enable exact env replay: rollout() resets the env with the same seed the actor saw

**Why not random actions?** Random actions push all models equally OOD, inflating error uniformly and obscuring architectural differences. Using a trained actor produces in-distribution action sequences.

**Why not random policy for MLP training?** The MLP is evaluated on IRIS-actor action sequences. Training on random-policy data creates a distribution mismatch where the MLP sees game states at evaluation time that it never trained on.

### 2. Rollout Execution (`src/rollout.py`)

For each model × game:
1. Load pre-collected action sequences + seeds from `data/atari/{game}_actions.npz`
2. For each trajectory:
   a. Reset the real ALE environment with the **saved per-trajectory seed** (exact replay)
   b. Reset the model with the initial observation
   c. For k=1..50:
      - Step the real environment with action[k] → true_frame[k]
      - Step the model with action[k] → pred_frame[k] (autoregressive on own output)
      - Track termination: if env terminates, mark valid_mask[traj, k:] = False
3. Save `{pred_frames, true_frames, actions, valid_mask}` to cache

**Autoregressive property**: After step 0, the model NEVER receives real observations. It predicts from its own previous predictions. This is the compounding error being measured.

**valid_mask**: Tracks which (trajectory, step) pairs have valid ground truth. After termination, the env emits the last real frame repeatedly — MSE against a frozen frame is meaningless. Downstream metrics honor this mask.

**Resolution**: All pixel models operate at 64×64×3 (PIXEL_SIZE=64). ALE returns 210×160×3, so ground truth is resized to 64×64 before storage.

**Caching**: If `results/rollouts/{model}_{game}.npz` exists, rollout is skipped entirely. Delete the file to force regeneration.

### 3. Model-Specific Rollout Behavior

**MLP** (`obs_type="ram"`):
- Environment returns 128-byte RAM state
- Model predicts next RAM state from (current_RAM, action)
- Predictions clamped to [0, 1] before re-feeding (prevents saturation)
- Chains predictions: model_obs = predict(model_obs, action)

**IRIS** (`obs_type="rgb"`):
- Uses WorldModelEnv with KV-cache for efficient autoregressive generation
- Tokenizes frames via VQ-VAE, predicts next tokens via transformer, decodes back
- Stateful: maintains transformer context across steps

**DIAMOND** (`obs_type="rgb"`):
- Maintains a 4-frame sliding window (deque)
- Each step: conditions diffusion model on last 4 frames + actions, denoises next frame
- 3 denoising steps per prediction (atari_100k default)
- Cold-start: initial obs replicated 4× into buffer at reset

**DreamerV3** (separate sbatch, two-pass):
- Pass 1 (env_jax): Encode initial obs via `observe()`, then `imagine()` for k=1..K
- Pass 2 (env_torch): Collect true_frames from ALE, append valid_mask
- Uses pure latent imagination after step 0 — never re-encodes predicted pixels
- Output resized from 84×84 to 64×64 to match PIXEL_SIZE

### 4. Data Shapes

| Model | pred_frames shape | true_frames shape | valid_mask shape |
|-------|-------------------|-------------------|------------------|
| MLP | [2000, 50, 128] | [2000, 50, 128] | [2000, 50] |
| IRIS | [2000, 50, 64, 64, 3] | [2000, 50, 64, 64, 3] | [2000, 50] |
| DIAMOND | [2000, 50, 64, 64, 3] | [2000, 50, 64, 64, 3] | [2000, 50] |
| DreamerV3 | [2000, 50, 64, 64, 3] | [2000, 50, 64, 64, 3] | [2000, 50] |

### 5. Seed Determinism

- Global seed: 42 (from `configs/experiment.yaml`)
- Per-trajectory seeds: saved during action collection, replayed during rollout
- Both real environment and model receive the same initial observation per trajectory
- Action sequences are pre-determined (loaded from file)

This ensures exact reproducibility: same seed → same env trajectory → same ground truth frames. The K-survivor protocol guarantees the action sequence remains valid for the full horizon.

### 6. Correctness Invariants

- Actions are collected under the IRIS actor, which is the same policy distribution used at evaluation time
- MLP is trained on IRIS-actor transitions (not random policy)
- All trajectories survive K=50 steps without termination (K-survivor guarantee)
- Per-trajectory seeds ensure rollout replays the exact env state the actor saw
- valid_mask handles any residual terminations during rollout replay
- DreamerV3 uses imagine() not observe() after step 0 (no ground-truth leakage)
- MLP predictions are clamped to [0, 1] before re-feeding (no saturation cascade)
