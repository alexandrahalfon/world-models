"""Experiment 1: Error growth curves and power-law fitting (H1).

For each model × game:
  1. Load cached rollout (or fail with instructions to run rollouts first)
  2. Compute per-step MSE E_k
  3. Fit power law E_k ~ c * k^alpha; derive scale-invariant k*
  4. Save CSV and plots

Outputs:
  results/exp1/error_growth_{model}_{game}.csv
  results/exp1/alpha_kstar_table.csv
  results/exp1/error_curves_{game}.png
  results/exp1/error_curves_pixel_{game}.png  (pixel-space models only)
"""
import os
import sys
import time
import logging
from datetime import datetime

import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import per_step_mse, fit_power_law
from src import visualize

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "experiment.yaml")
ROLLOUT_DIR = "results/rollouts"
OUT_DIR = "results/exp1"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"results/logs/exp1_{_ts}.log"),
    ],
)
log = logging.getLogger(__name__)


def main():
    t0 = time.time()
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    games = cfg["games"]
    K = cfg["horizon_K"]
    k_star_mult = cfg["k_star_multiplier"]
    k_fit_min = cfg.get("k_fit_min", 1)
    model_cfgs = cfg["models"]
    log.info("Power-law fit window: k=%d..%d (k_star_multiplier=%g)",
             k_fit_min, K, k_star_mult)

    summary_rows = []
    all_error_curves: dict[str, dict[str, np.ndarray]] = {m: {} for m in model_cfgs}

    pairs = [(m, g) for m in model_cfgs for g in games]
    pbar = tqdm(pairs, desc="Exp1 model×game", unit="pair", ncols=80)

    for model_name, game in pbar:
        pbar.set_description(f"Exp1 {model_name}/{game}")
        space = model_cfgs[model_name]["space"]
        cache_path = os.path.join(ROLLOUT_DIR, f"{model_name}_{game}.npz")

        if not os.path.exists(cache_path):
            log.warning("rollout cache missing for %s/%s: %s", model_name, game, cache_path)
            log.warning("  Run rollout.py first (or submit run_rollouts.sbatch).")
            continue

        data = np.load(cache_path)
        pred_frames = data["pred_frames"]
        true_frames = data["true_frames"]
        # valid_mask is written by the post-fix rollout pipeline; older caches
        # won't have it. When absent, fall back to all-valid (legacy behavior).
        valid_mask = data["valid_mask"] if "valid_mask" in data.files else None

        E_k = per_step_mse(pred_frames, true_frames, valid_mask=valid_mask)
        fit = fit_power_law(E_k, k_star_multiplier=k_star_mult, k_min=k_fit_min)

        df = pd.DataFrame({"k": np.arange(1, len(E_k) + 1), "mse": E_k})
        csv_path = os.path.join(OUT_DIR, f"error_growth_{model_name}_{game}.csv")
        df.to_csv(csv_path, index=False)

        all_error_curves[model_name][game] = E_k
        summary_rows.append({
            "model": model_name,
            "space": space,
            "game": game,
            "alpha": fit["alpha"],
            "c": fit["c"],
            "k_star": fit["k_star"],
            "fit_r2": fit["fit_r2"],
            "k_fit_min": fit["k_min"],
        })
        log.info(
            "%s/%s: alpha=%.3f  k*=%s  R2=%.3f  (fit on k>=%d)",
            model_name, game, fit["alpha"], fit["k_star"], fit["fit_r2"], fit["k_min"],
        )

    pbar.close()

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(OUT_DIR, "alpha_kstar_table.csv"), index=False)
        log.info("\n=== Alpha / k* table ===\n%s", summary_df.to_string(index=False))

        # Note: pixel-only alpha comparison is the valid scientific comparison.
        # MLP operates in RAM space and its alpha is NOT directly comparable.
        models_with_data = {m: v for m, v in all_error_curves.items() if v}
        for game in games:
            if any(game in v for v in models_with_data.values()):
                game_data = {m: {game: v[game]} for m, v in models_with_data.items() if game in v}
                visualize.plot_error_curves(
                    game_data,
                    save_path=os.path.join(OUT_DIR, f"error_curves_{game}.png"),
                )
                visualize.plot_error_curves(
                    game_data,
                    save_path=os.path.join(OUT_DIR, f"error_curves_pixel_{game}.png"),
                    pixel_only=True,
                )
    else:
        log.warning("No rollout caches found. Run rollouts first.")

    elapsed = (time.time() - t0) / 60.0
    log.info("Exp 1 completed in %.1f min", elapsed)


if __name__ == "__main__":
    main()
