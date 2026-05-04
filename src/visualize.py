"""Visualization utilities for world model experiment results."""
from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

sns.set_theme(style="whitegrid", palette="tab10")

MODEL_COLORS = {"mlp": "gray", "dreamerv3": "steelblue", "iris": "darkorange", "diamond": "seagreen"}
MODEL_LABELS = {"mlp": "MLP (RAM)", "dreamerv3": "DreamerV3", "iris": "IRIS", "diamond": "DIAMOND"}


def plot_error_curves(
    results: dict[str, dict[str, np.ndarray]],
    save_path: str,
    pixel_only: bool = False,
) -> None:
    """Plot E_k vs k for all models on all games.

    results: {model_name: {game: E_k_array}}
    pixel_only: if True, skip MLP (RAM space, not comparable)
    """
    games = list(next(iter(results.values())).keys())
    fig, axes = plt.subplots(1, len(games), figsize=(5 * len(games), 4), sharey=False)
    if len(games) == 1:
        axes = [axes]

    for ax, game in zip(axes, games):
        for model, game_results in results.items():
            if pixel_only and model == "mlp":
                continue
            E_k = game_results[game]
            ks = np.arange(1, len(E_k) + 1)
            ax.plot(ks, E_k, label=MODEL_LABELS.get(model, model), color=MODEL_COLORS.get(model))
        ax.set_title(game)
        ax.set_xlabel("Rollout step k")
        ax.set_ylabel("Mean squared error $E_k$")
        ax.legend()

    title = "Error growth curves (pixel-space models)" if pixel_only else "Error growth curves (all models)"
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[visualize] Saved {save_path}")


def plot_frame_grid(
    frames_by_model: dict[str, np.ndarray],
    ks: list[int],
    game: str,
    true_frames: dict[str, np.ndarray] | None = None,
    save_path: str = "frame_grid.png",
) -> None:
    """Side-by-side frame comparison at specified k values.

    frames_by_model: {model_name: frames [K, H, W, 3]}  (one representative trajectory)
    ks: list of k indices (0-indexed) to visualize
    true_frames: optional {model_name: frames [K, H, W, 3]} — shows ground truth in top row

    Pixel models in the rollout pipeline currently store 64x64x3 frames.
    """
    models = list(frames_by_model.keys())
    n_rows = len(models) + (1 if true_frames else 0)
    n_cols = len(ks)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
    axes = np.array(axes).reshape(n_rows, n_cols)

    row_labels = (["Ground truth"] if true_frames else []) + [MODEL_LABELS.get(m, m) for m in models]

    for row_idx, label in enumerate(row_labels):
        for col_idx, k in enumerate(ks):
            ax = axes[row_idx, col_idx]
            if label == "Ground truth":
                # Use first model's true frames (all identical)
                frame = list(true_frames.values())[0][k]
            else:
                model = models[row_idx - (1 if true_frames else 0)]
                frame = frames_by_model[model][k]
            ax.imshow(frame)
            ax.axis("off")
            if row_idx == 0:
                ax.set_title(f"k={k+1}", fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(label, fontsize=9, rotation=90, labelpad=40)

    fig.suptitle(f"{game} — Predicted frames at k = {[k+1 for k in ks]}", y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[visualize] Saved {save_path}")


def plot_distributional_divergence(
    fid_results: dict[str, dict[str, list]],
    kl_results: dict[str, dict[str, list]],
    horizons: list[int],
    game: str,
    save_path: str = "distributional.png",
) -> None:
    """Plot FID and PCA-KL divergence curves vs horizon k.

    fid_results, kl_results: {model_name: {game: [value_at_each_horizon]}}
    horizons: list of k values matching the lists above
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for model in fid_results:
        vals = fid_results[model].get(game, [])
        if not vals:  # skip models with no data for this game (e.g. mlp in RAM space, dreamerv3 missing)
            continue
        ax1.plot(horizons, vals, label=MODEL_LABELS.get(model, model), color=MODEL_COLORS.get(model), marker="o")
    ax1.set_title(f"{game} — FID vs rollout horizon")
    ax1.set_xlabel("Rollout step k")
    ax1.set_ylabel("FID")
    ax1.legend()

    for model in kl_results:
        vals = kl_results[model].get(game, [])
        if not vals:
            continue
        ax2.plot(horizons, vals, label=MODEL_LABELS.get(model, model), color=MODEL_COLORS.get(model), marker="o")
    ax2.set_title(f"{game} — PCA-KL divergence vs rollout horizon")
    ax2.set_xlabel("Rollout step k")
    ax2.set_ylabel("KL divergence")
    ax2.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[visualize] Saved {save_path}")


def plot_pca_scatter(
    pred_frames_k: np.ndarray,
    true_frames_k: np.ndarray,
    k: int,
    game: str,
    save_path: str,
) -> None:
    """Scatter plot of true vs predicted frames in 2D PCA space at horizon k.

    Useful for visualizing distributional drift: as k grows, predicted-frame
    cloud (red) should diverge from true-frame cloud (black).

    pred_frames_k, true_frames_k: [N, H, W, 3] uint8

    Note: this helper is currently not called from any experiment; the
    pca_scatter_*.png files in results/exp3/ were generated outside the pipeline.
    Wire it into experiments/exp3_distributional.py if those plots should be
    regenerated automatically.
    """
    from sklearn.decomposition import PCA
    N = min(len(pred_frames_k), len(true_frames_k))
    pred_flat = pred_frames_k[:N].reshape(N, -1).astype(np.float32) / 255.0
    true_flat = true_frames_k[:N].reshape(N, -1).astype(np.float32) / 255.0

    pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
    true_proj = pca.fit_transform(true_flat)
    pred_proj = pca.transform(pred_flat)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(true_proj[:, 0], true_proj[:, 1], s=8, alpha=0.4,
               label="True", color="black")
    ax.scatter(pred_proj[:, 0], pred_proj[:, 1], s=8, alpha=0.4,
               label="Predicted", color="tab:red")
    ax.set_title(f"{game} — frame distribution in PCA space at k={k}")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"[visualize] Saved {save_path}")
