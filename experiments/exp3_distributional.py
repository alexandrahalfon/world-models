"""Experiment 3: Distributional divergence over rollout horizon (H3).

Measures FID and PCA-KL divergence at each horizon in fid_horizons.
k=1 is excluded from evaluation — FID is unreliable at k=1 because predicted
and true frames are nearly identical, producing near-zero estimates with high
variance at N=2000. fid_horizons is read from configs/experiment.yaml.

PCA-KL uses a *fixed* PCA basis fit once per (model, game) on true frames
pooled across the evaluated horizons, so KL values are comparable across k.
Refitting per horizon (the old behavior) makes the basis itself drift with k
and produced unstable, non-monotonic KL trajectories.

FID adds a tiny covariance ridge (cov_ridge=1e-6 default in compute_fid) to
suppress the "Matrix is singular" warnings caused by rank-deficient empirical
covariances at N≈2000 in 2048-dim Inception activation space.

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
from src.metrics import compute_fid, compute_pca_kl, fit_reference_pca
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
            pred_frames = data["pred_frames"]   # [n_traj, K, H, W, 3]
            true_frames = data["true_frames"]

            # Fit one shared PCA basis on true frames pooled across the evaluated
            # horizons. This keeps the projection space fixed across k so KL values
            # are directly comparable. Subsample if pooling is too large for memory.
            pooled_idx = [k - 1 for k in fid_horizons]
            pooled_true = true_frames[:, pooled_idx].reshape(
                -1, *true_frames.shape[2:]
            )
            max_pool_samples = 8000
            if len(pooled_true) > max_pool_samples:
                rng = np.random.default_rng(42)
                sel = rng.choice(len(pooled_true), max_pool_samples, replace=False)
                pooled_true = pooled_true[sel]
            pca_ref = fit_reference_pca(pooled_true, n_components=0.95)
            print(f"[exp3] {model_name}/{game}: PCA basis fit on {len(pooled_true)} "
                  f"pooled true frames -> {pca_ref.n_components_} dims "
                  f"({pca_ref.explained_variance_ratio_.sum()*100:.1f}% var)")

            fid_vals = []
            kl_vals = []

            for k in fid_horizons:
                k_idx = k - 1  # horizons are 1-indexed in config
                pred_k = pred_frames[:, k_idx]   # [n_traj, H, W, 3]
                true_k = true_frames[:, k_idx]

                fid_val = compute_fid(pred_k, true_k)
                kl_val = compute_pca_kl(pred_k, true_k, pca=pca_ref)
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
