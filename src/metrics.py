"""Metrics for measuring world model prediction error over rollout horizons.

All functions are pure (no side effects) and unit-testable.

Key design decisions:
- k* is scale-invariant: defined as the first k where E_k > mean(E_k[0:5]) * k_star_multiplier
  This works across RAM-space (MLP) and pixel-space models without normalization.
  The multiplier must be validated with a pilot run — see CLAUDE.md.
- Power-law fit is done on log-log scale to avoid numerical issues with curve_fit.
- PCA uses n_components=0.95 to retain 95% of variance dynamically; hardcoding
  n_components=50 would capture only ~20-30% on 84x84x3 data.
- FID requires a minimum of 2000 samples for reliable estimates.
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ── Per-step MSE ──────────────────────────────────────────────────────────────

def per_step_mse(pred_frames: np.ndarray, true_frames: np.ndarray) -> np.ndarray:
    """Compute mean squared error at each horizon step k.

    pred_frames, true_frames: [n_traj, K, ...obs_shape]
    Returns: E_k array of shape [K], averaged over trajectories and spatial dims.
    """
    assert pred_frames.shape == true_frames.shape
    diff = pred_frames.astype(np.float64) - true_frames.astype(np.float64)
    # Average over trajectories and all spatial dimensions
    sq = diff ** 2
    return sq.reshape(pred_frames.shape[0], pred_frames.shape[1], -1).mean(axis=(0, 2))


# ── Power-law fit ─────────────────────────────────────────────────────────────

def fit_power_law(E_k: np.ndarray, k_star_multiplier: float = 10.0) -> dict:
    """Fit E_k ~ c * k^alpha and compute scale-invariant k*.

    alpha is fitted on log-log scale: log(E_k) = alpha * log(k) + log(c)

    k* is the first k where E_k > mean(E_k[0:5]) * k_star_multiplier.
    This is scale-invariant across RAM-space and pixel-space models.

    Returns dict with keys: alpha, c, k_star (1-indexed), fit_quality (R^2)
    """
    K = len(E_k)
    ks = np.arange(1, K + 1, dtype=np.float64)

    # Avoid log(0)
    E_safe = np.maximum(E_k, 1e-12)

    def log_power_law(log_k, alpha, log_c):
        return alpha * log_k + log_c

    try:
        popt, _ = curve_fit(log_power_law, np.log(ks), np.log(E_safe), p0=[1.0, 0.0], maxfev=5000)
        alpha, log_c = popt
        c = np.exp(log_c)
        # R^2 on log scale
        log_E_pred = log_power_law(np.log(ks), alpha, log_c)
        ss_res = np.sum((np.log(E_safe) - log_E_pred) ** 2)
        ss_tot = np.sum((np.log(E_safe) - np.log(E_safe).mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    except RuntimeError:
        warnings.warn("Power-law curve_fit did not converge. Returning NaN.")
        alpha, c, r2 = float("nan"), float("nan"), float("nan")

    # Scale-invariant k*: first k where E_k > mean(E_k[0:5]) * multiplier
    baseline = float(np.mean(E_k[:5]))
    threshold = baseline * k_star_multiplier
    k_star_candidates = np.where(E_k > threshold)[0]
    k_star = int(k_star_candidates[0]) + 1 if len(k_star_candidates) > 0 else K + 1  # 1-indexed

    return {"alpha": float(alpha), "c": float(c), "k_star": k_star, "fit_r2": float(r2)}


# ── FID ───────────────────────────────────────────────────────────────────────

def compute_fid(pred_frames_k: np.ndarray, true_frames_k: np.ndarray) -> float:
    """Compute Fréchet Inception Distance between predicted and true frames at horizon k.

    pred_frames_k, true_frames_k: [N, 84, 84, 3] uint8 — at least 2000 samples.
    Uses pytorch-fid for Inception feature extraction (requires GPU for practical speed).
    """
    if len(pred_frames_k) < 2000:
        warnings.warn(f"FID computed with only {len(pred_frames_k)} samples; recommend >= 2000.")

    try:
        from pytorch_fid.fid_score import calculate_frechet_distance  # type: ignore[import]
        from pytorch_fid.inception import InceptionV3  # type: ignore[import]
        import torch
    except ImportError as e:
        raise ImportError("pytorch-fid not installed. Run: pip install pytorch-fid") from e

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(device)
    model.eval()

    def get_activations(frames: np.ndarray) -> np.ndarray:
        # frames: [N, H, W, C] uint8 → [N, C, H, W] float [0,1]
        t = torch.tensor(frames.transpose(0, 3, 1, 2), dtype=torch.float32).to(device) / 255.0
        acts = []
        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(t), batch_size):
                out = model(t[i:i+batch_size])[0]
                acts.append(out.squeeze(-1).squeeze(-1).cpu().numpy())
        return np.concatenate(acts, axis=0)

    act_pred = get_activations(pred_frames_k)
    act_true = get_activations(true_frames_k)

    mu_pred, sigma_pred = act_pred.mean(0), np.cov(act_pred, rowvar=False)
    mu_true, sigma_true = act_true.mean(0), np.cov(act_true, rowvar=False)

    return float(calculate_frechet_distance(mu_pred, sigma_pred, mu_true, sigma_true))


# ── PCA-projected KL divergence ───────────────────────────────────────────────

def compute_pca_kl(
    pred_frames_k: np.ndarray,
    true_frames_k: np.ndarray,
    n_components: float = 0.95,
    n_bins: int = 50,
) -> float:
    """Compute PCA-projected KL divergence between predicted and true frame distributions.

    Retains 95% of variance dynamically (n_components=0.95 in scikit-learn).
    Do NOT hardcode n_components=50 — on 84x84x3 pixel data that captures only ~20-30%.

    pred_frames_k, true_frames_k: [N, 84, 84, 3] uint8
    Returns: mean KL divergence across retained PCA dimensions.
    """
    # Flatten and normalize
    N = len(pred_frames_k)
    pred_flat = pred_frames_k.reshape(N, -1).astype(np.float32) / 255.0
    true_flat = true_frames_k.reshape(N, -1).astype(np.float32) / 255.0

    # Fit PCA on true frames, project both
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=42)
    true_proj = pca.fit_transform(true_flat)
    pred_proj = pca.transform(pred_flat)

    n_dims = true_proj.shape[1]
    kl_per_dim = []
    for d in range(n_dims):
        # Estimate densities via histogram; use true_proj range for both bins
        combined = np.concatenate([true_proj[:, d], pred_proj[:, d]])
        lo, hi = combined.min(), combined.max()
        if lo == hi:
            kl_per_dim.append(0.0)
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(true_proj[:, d], bins=bins, density=True)
        q, _ = np.histogram(pred_proj[:, d], bins=bins, density=True)
        # Smooth to avoid log(0)
        eps = 1e-10
        p = p + eps
        q = q + eps
        p /= p.sum()
        q /= q.sum()
        kl_per_dim.append(float(np.sum(p * np.log(p / q))))

    return float(np.mean(kl_per_dim))
