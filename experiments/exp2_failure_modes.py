"""Experiment 2: Architecture-specific failure mode visualization (H2).

For each model × game, samples representative trajectories and renders frames
at k=1, 10, 50 side-by-side for visual inspection of how each model degrades.

Expected observations:
  IRIS: blur and spectral artifacts (VQ-VAE quantization residuals compound)
  DIAMOND: maintains visual coherence longer (diffusion denoising resists drift)
  DreamerV3: geometrically plausible but structurally incorrect (GRU drift)
  MLP: degrades rapidly (RAM space — different failure character than pixel models)

Outputs:
  results/exp2/frames_{model}_{game}.png     (per-model frame strip)
  results/exp2/comparison_{game}.png         (all models side-by-side)
"""
import os
import sys
import time
import logging
from datetime import datetime

import yaml
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src import visualize

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "experiment.yaml")
ROLLOUT_DIR = "results/rollouts"
OUT_DIR = "results/exp2"

VIS_K = [0, 9, 49]   # 0-indexed: k=1, 10, 50
N_SAMPLE_TRAJ = 5    # representative trajectories to visualize

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"results/logs/exp2_{_ts}.log"),
    ],
)
log = logging.getLogger(__name__)


def main():
    t0 = time.time()
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    games = cfg["games"]
    model_cfgs = cfg["models"]

    for game in tqdm(games, desc="Exp2 games", unit="game", ncols=80):
        frames_by_model: dict[str, np.ndarray] = {}
        true_frames_by_model: dict[str, np.ndarray] = {}

        for model_name in tqdm(model_cfgs, desc=f"  {game} models", leave=False, ncols=80):
            cache_path = os.path.join(ROLLOUT_DIR, f"{model_name}_{game}.npz")
            if not os.path.exists(cache_path):
                log.warning("rollout cache missing for %s/%s", model_name, game)
                continue

            data = np.load(cache_path)
            pred_frames = data["pred_frames"]
            true_frames = data["true_frames"]

            space = model_cfgs[model_name]["space"]
            if space != "pixel":
                log.info("  [exp2] %s is RAM-space — skipping frame grid", model_name)
                continue

            # Pick a representative trajectory (middle index)
            traj_idx = len(pred_frames) // 2
            frames_by_model[model_name] = pred_frames[traj_idx]
            true_frames_by_model[model_name] = true_frames[traj_idx]
            log.info("  [exp2] loaded %s/%s  traj_idx=%d", model_name, game, traj_idx)

        if not frames_by_model:
            log.warning("[exp2] No pixel-space rollout caches for %s. Skipping.", game)
            continue

        visualize.plot_frame_grid(
            frames_by_model=frames_by_model,
            ks=VIS_K,
            game=game,
            true_frames=true_frames_by_model,
            save_path=os.path.join(OUT_DIR, f"comparison_{game}.png"),
        )

        for model_name, traj_frames in frames_by_model.items():
            visualize.plot_frame_grid(
                frames_by_model={model_name: traj_frames},
                ks=VIS_K,
                game=game,
                save_path=os.path.join(OUT_DIR, f"frames_{model_name}_{game}.png"),
            )

    elapsed = (time.time() - t0) / 60.0
    log.info("Exp 2 completed in %.1f min", elapsed)


if __name__ == "__main__":
    main()
