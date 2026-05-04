"""Experiment 3: Distributional divergence over rollout horizon (H3).

Measures FID and PCA-KL divergence at each horizon in fid_horizons.
k=1 is excluded from evaluation — FID is unreliable at k=1 because predicted
and true frames are nearly identical, producing near-zero estimates with high
variance at N=2000. fid_horizons is read from configs/experiment.yaml.

Expected: distributional divergence grows faster than per-frame MSE; the gap
between FID and MSE is largest for IRIS (VQ-VAE quantization residuals).

Outputs:
  results/exp3/divergence_{model}_{game}.csv
  results/exp3/distributional_{game}.png
"""
import os
import sys
import yaml
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.metrics import compute_fid, compute_pca_kl
from src import visualize

CFG_PATH = os.path.join(os.path.dirname(__file__), "..", "configs", "experiment.yaml")
ROLLOUT_DIR = "results/rollouts"
OUT_DIR = "results/exp3"


def main():
    with open(CFG_PATH) as f:
        cfg = yaml.safe_load(f)

    games = cfg["games"]
    # fid_horizons comes from config — never hardcode here; k=1 is excluded in config
    fid_horizons: list[int] = cfg["fid_horizons"]
    model_cfgs = cfg["models"]

    os.makedirs(OUT_DIR, exist_ok=True)

    # {model: {game: [value per horizon]}}
    all_fid: dict[str, dict[str, list]] = {m: {} for m in model_cfgs}
    all_kl:  dict[str, dict[str, list]] = {m: {} for m in model_cfgs}

    for model_name, mcfg in model_cfgs.items():
        space = mcfg["space"]
        if space != "pixel":
            print(f"[exp3] Skipping {model_name} (RAM space — not comparable with pixel FID)")
            continue

        for game in games:
            cache_path = os.path.join(ROLLOUT_DIR, f"{model_name}_{game}.npz")
            if not os.path.exists(cache_path):
                print(f"[exp3] WARNING: rollout cache missing for {model_name}/{game}")
                continue

            data = np.load(cache_path)
            pred_frames = data["pred_frames"]   # [n_traj, K, 84, 84, 3]
            true_frames = data["true_frames"]

            fid_vals = []
            kl_vals = []

            for k in fid_horizons:
                k_idx = k - 1  # horizons are 1-indexed in config
                pred_k = pred_frames[:, k_idx]   # [n_traj, 84, 84, 3]
                true_k = true_frames[:, k_idx]

                fid_val = compute_fid(pred_k, true_k)
                kl_val = compute_pca_kl(pred_k, true_k)
                fid_vals.append(fid_val)
                kl_vals.append(kl_val)
                print(f"[exp3] {model_name}/{game} k={k}: FID={fid_val:.2f}, KL={kl_val:.4f}")

            all_fid[model_name][game] = fid_vals
            all_kl[model_name][game] = kl_vals

            df = pd.DataFrame({"k": fid_horizons, "fid": fid_vals, "kl_divergence": kl_vals})
            csv_path = os.path.join(OUT_DIR, f"divergence_{model_name}_{game}.csv")
            df.to_csv(csv_path, index=False)

    for game in games:
        if any(game in v for v in all_fid.values()):
            visualize.plot_distributional_divergence(
                fid_results=all_fid,
                kl_results=all_kl,
                horizons=fid_horizons,
                game=game,
                save_path=os.path.join(OUT_DIR, f"distributional_{game}.png"),
            )


if __name__ == "__main__":
    main()
