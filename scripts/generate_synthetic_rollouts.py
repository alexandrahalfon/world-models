#!/usr/bin/env python3
"""Generate synthetic rollout caches for local pipeline testing.

Creates fake .npz files in results/rollouts/ that mimic real model output shapes.
Each model gets a different injected alpha so exp1 produces realistic-looking curves.

  mlp:       alpha=1.8  shape [n_traj, K, 128]          (RAM space)
  iris:      alpha=1.4  shape [n_traj, K, 84, 84, 3]    (pixel)
  dreamerv3: alpha=1.1  shape [n_traj, K, 84, 84, 3]    (pixel)
  diamond:   alpha=0.7  shape [n_traj, K, 84, 84, 3]    (pixel)

The noise standard deviation at step k is sqrt(c) * k^(alpha/2), giving
  E_k = Var(pred - true) ≈ c * k^alpha  on the log-log scale.
"""
import os
import sys
import time
import numpy as np

ROLLOUT_DIR = "results/rollouts"
GAMES = ["Breakout", "Pong", "Boxing"]
N_TRAJ = 100
K = 50
SEED = 42

# Each model: injected alpha and noise scale c
MODELS = {
    "mlp":       {"alpha": 1.8, "c": 4.0, "space": "ram"},
    "iris":      {"alpha": 1.4, "c": 4.0, "space": "pixel"},
    "dreamerv3": {"alpha": 1.1, "c": 4.0, "space": "pixel"},
    "diamond":   {"alpha": 0.7, "c": 4.0, "space": "pixel"},
}


def generate_rollout(model_name: str, game: str, mcfg: dict, rng: np.random.Generator) -> str:
    alpha = mcfg["alpha"]
    c = mcfg["c"]
    obs_shape = (128,) if mcfg["space"] == "ram" else (84, 84, 3)

    out_path = os.path.join(ROLLOUT_DIR, f"{model_name}_{game}.npz")
    if os.path.exists(out_path):
        print(f"  [skip] {out_path} already exists.")
        return out_path

    true_frames = rng.integers(0, 256, size=(N_TRAJ, K) + obs_shape, dtype=np.uint8)
    pred_frames = true_frames.copy().astype(np.float64)

    ks = np.arange(1, K + 1)
    noise_stds = np.sqrt(c) * ks ** (alpha / 2.0)

    for k_idx in range(K):
        noise = rng.normal(0.0, noise_stds[k_idx], size=(N_TRAJ,) + obs_shape)
        pred_frames[:, k_idx, ...] += noise

    pred_frames = np.clip(pred_frames, 0, 255).astype(np.uint8)
    actions = rng.integers(0, 18, size=(N_TRAJ, K), dtype=np.int32)

    np.savez_compressed(out_path, pred_frames=pred_frames, true_frames=true_frames, actions=actions)
    return out_path


def main():
    t0 = time.time()
    os.makedirs(ROLLOUT_DIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    total = len(MODELS) * len(GAMES)
    done = 0

    for model_name, mcfg in MODELS.items():
        for game in GAMES:
            done += 1
            alpha = mcfg["alpha"]
            obs_shape = (128,) if mcfg["space"] == "ram" else (84, 84, 3)
            print(
                f"[{done}/{total}] {model_name}/{game}  alpha={alpha}"
                f"  shape=({N_TRAJ},{K},{','.join(str(d) for d in obs_shape)})",
                flush=True,
            )
            out_path = generate_rollout(model_name, game, mcfg, rng)
            print(f"  -> {out_path}", flush=True)

    elapsed = time.time() - t0
    print(f"\n[gen_synthetic] Done — {total} rollouts in {elapsed:.1f}s")
    print(f"[gen_synthetic] Output dir: {os.path.abspath(ROLLOUT_DIR)}")


if __name__ == "__main__":
    main()
