"""Metrics for measuring world model prediction error over rollout horizons.

All functions are pure (no side effects) and unit-testable.

Key design decisions:
- k* is scale-invariant: defined as the first k where E_k > mean(E_k[k_min..k_min+4])
  * k_star_multiplier. This works across RAM-space (MLP) and pixel-space models
  without normalization. The multiplier must be validated with a pilot run — see CLAUDE.md.
- Power-law fit is done on log-log scale to avoid numerical issues with curve_fit.
  k_min lets callers exclude an early conditioning warm-up regime: e.g. DIAMOND has
  4 conditioning frames seeded with the initial obs, so its k=1..4 predictions are
  on a fictional history and inflate MSE. Set k_min=5 to skip that regime.
- PCA-KL uses a *fixed* PCA basis across horizons so KL values are comparable as
  the true-frame distribution drifts with k. Fit the basis once via
  fit_reference_pca(reference_frames), then pass it to compute_pca_kl(...).
  Re-fitting PCA per horizon (the old behavior) makes per-k KL incomparable.
  Pool the reference from a *stationary* source (e.g. true frames at k=1 across
  all trajectories), NOT the per-horizon true frames pooled across k — the
  latter bakes the very horizon-drift you are trying to measure into the basis,
  producing non-monotone KL trajectories.
- FID is a Fréchet distance between Inception activation Gaussians. With N≈2000
  and 2048-dim activations the empirical covariance is rank-deficient, so we add
  a tiny ridge (eps*I) before computing the matrix square root.
"""
from __future__ import annotations
import warnings
import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ── Per-step MSE ──────────────────────────────────────────────────────────────

def per_step_mse(pred_frames: np.ndarray, true_frames: np.ndarray,
                 valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Compute mean squared error at each horizon step k.

    pred_frames, true_frames: [n_traj, K, ...obs_shape]
    valid_mask: optional [n_traj, K] bool array — True where the real env had
        not yet terminated. When provided, MSE at step k averages only over
        trajectories that were still live at that step. This avoids inflating
        E_k with comparisons to a frozen "last real frame" emitted after
        termination.
    Returns: E_k array of shape [K], averaged over (live) trajectories and spatial dims.
    """
    assert pred_frames.shape == true_frames.shape
    n_traj, K = pred_frames.shape[:2]
    diff = pred_frames.astype(np.float64) - true_frames.astype(np.float64)
    sq = diff.reshape(n_traj, K, -1).mean(axis=2)  # [n_traj, K] per-step squared error
    if valid_mask is None:
        return sq.mean(axis=0)
    assert valid_mask.shape == (n_traj, K)
    counts = valid_mask.sum(axis=0).astype(np.float64)
    counts = np.maximum(counts, 1.0)  # avoid divide-by-zero
    masked = np.where(valid_mask, sq, 0.0)
    return masked.sum(axis=0) / counts


# ── Power-law fit ─────────────────────────────────────────────────────────────

def fit_power_law(E_k: np.ndarray, k_star_multiplier: float = 10.0,
                  k_min: int = 1) -> dict:
    """Fit E_k ~ c * k^alpha and compute scale-invariant k*.

    alpha is fitted on log-log scale over k = k_min .. K:
        log(E_k) = alpha * log(k) + log(c)

    k_min: 1-indexed first horizon to include in the fit and the k* baseline.
    Use k_min=5 to skip DIAMOND's conditioning warm-up (see module docstring).

    k* is the first k >= k_min where E_k > baseline * k_star_multiplier, where
    baseline = mean(E_k[k_min-1 : k_min+4]) — the first 5 *reliable* steps.
    This stays scale-invariant across RAM-space and pixel-space models.

    Returns dict with keys: alpha, c, k_star (1-indexed), fit_r2, k_min.
    """
    K = len(E_k)
    if not 1 <= k_min <= K - 1:
        raise ValueError(f"k_min must be in [1, K-1]; got k_min={k_min}, K={K}")

    ks_full = np.arange(1, K + 1, dtype=np.float64)
    E_safe = np.maximum(E_k, 1e-12)

    # Fit only on the in-range slice
    fit_slice = slice(k_min - 1, K)
    ks_fit = ks_full[fit_slice]
    E_fit = E_safe[fit_slice]

    def log_power_law(log_k, alpha, log_c):
        return alpha * log_k + log_c

    try:
        popt, _ = curve_fit(log_power_law, np.log(ks_fit), np.log(E_fit),
                            p0=[1.0, 0.0], maxfev=5000)
        alpha, log_c = popt
        c = np.exp(log_c)
        log_E_pred = log_power_law(np.log(ks_fit), alpha, log_c)
        ss_res = np.sum((np.log(E_fit) - log_E_pred) ** 2)
        ss_tot = np.sum((np.log(E_fit) - np.log(E_fit).mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    except RuntimeError:
        warnings.warn("Power-law curve_fit did not converge. Returning NaN.")
        alpha, c, r2 = float("nan"), float("nan"), float("nan")

    # Baseline = first 5 reliable steps starting at k_min (or fewer if K is small)
    base_end = min(k_min - 1 + 5, K)
    baseline = float(np.mean(E_k[k_min - 1 : base_end]))
    threshold = baseline * k_star_multiplier

    # Search for k* only at or after k_min — earlier steps are not in the fit regime
    search_region = E_k[k_min - 1 :]
    above = np.where(search_region > threshold)[0]
    k_star = int(above[0]) + k_min if len(above) > 0 else K + 1  # 1-indexed

    return {"alpha": float(alpha), "c": float(c), "k_star": k_star,
            "fit_r2": float(r2), "k_min": int(k_min)}


# ── FID ───────────────────────────────────────────────────────────────────────

def compute_fid(pred_frames_k: np.ndarray, true_frames_k: np.ndarray,
                cov_ridge: float = 1e-4) -> float:
    """Compute Fréchet Inception Distance between predicted and true frames at horizon k.

    pred_frames_k, true_frames_k: [N, H, W, C] uint8 (any H,W — pytorch-fid resizes
    internally to 299x299 for InceptionV3). Recommend N >= 2000.

    cov_ridge: small diagonal added to both covariance matrices before computing
    the Fréchet distance. With N≈2000 and 2048-dim Inception activations the
    empirical covariance is rank-deficient, which makes the matrix square root
    in calculate_frechet_distance numerically unstable (LinAlgWarning: "Matrix is
    singular"). The default 1e-4 was needed to suppress that warning at this
    sample size; 1e-6 was insufficient. The shift is tiny relative to the
    typical Fréchet distance and preserves relative ordering.
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

    if cov_ridge > 0:
        d = sigma_pred.shape[0]
        eye = np.eye(d, dtype=sigma_pred.dtype)
        sigma_pred = sigma_pred + cov_ridge * eye
        sigma_true = sigma_true + cov_ridge * eye

    return float(calculate_frechet_distance(mu_pred, sigma_pred, mu_true, sigma_true))


# ── PCA-projected KL divergence ───────────────────────────────────────────────

def fit_reference_pca(reference_frames: np.ndarray, n_components: float = 0.95) -> PCA:
    """Fit a PCA basis on a stationary reference set (e.g. true frames pooled across
    horizons) so KL values from compute_pca_kl are comparable across k.

    reference_frames: [N, ...obs_shape] uint8. Flattened and scaled to [0, 1].
    """
    N = len(reference_frames)
    flat = reference_frames.reshape(N, -1).astype(np.float32) / 255.0
    pca = PCA(n_components=n_components, svd_solver="auto", random_state=42)
    pca.fit(flat)
    return pca


def compute_pca_kl(
    pred_frames_k: np.ndarray,
    true_frames_k: np.ndarray,
    pca: PCA | None = None,
    n_components: float = 0.95,
    n_bins: int = 50,
) -> float:
    """Compute PCA-projected KL divergence between predicted and true frame distributions.

    pca: pre-fit basis from fit_reference_pca(). REQUIRED for cross-horizon
    comparability — otherwise the basis itself shifts with k, and KL values
    can't be compared across horizons. If None, falls back to fitting on
    true_frames_k (legacy behavior; produces per-k incomparable values).

    n_components: only used when pca is None.

    Returns: mean KL divergence across retained PCA dimensions.
    """
    N = len(pred_frames_k)
    pred_flat = pred_frames_k.reshape(N, -1).astype(np.float32) / 255.0
    true_flat = true_frames_k.reshape(N, -1).astype(np.float32) / 255.0

    if pca is None:
        warnings.warn(
            "compute_pca_kl called without a pre-fit PCA basis; per-horizon KL values "
            "are not comparable across k. Pass pca=fit_reference_pca(true_frames_pooled)."
        )
        pca = PCA(n_components=n_components, svd_solver="auto", random_state=42)
        true_proj = pca.fit_transform(true_flat)
    else:
        true_proj = pca.transform(true_flat)
    pred_proj = pca.transform(pred_flat)

    n_dims = true_proj.shape[1]
    kl_per_dim = []
    for d in range(n_dims):
        combined = np.concatenate([true_proj[:, d], pred_proj[:, d]])
        lo, hi = combined.min(), combined.max()
        if lo == hi:
            kl_per_dim.append(0.0)
            continue
        bins = np.linspace(lo, hi, n_bins + 1)
        p, _ = np.histogram(true_proj[:, d], bins=bins, density=True)
        q, _ = np.histogram(pred_proj[:, d], bins=bins, density=True)
        eps = 1e-10
        p = p + eps
        q = q + eps
        p /= p.sum()
        q /= q.sum()
        kl_per_dim.append(float(np.sum(p * np.log(p / q))))

    return float(np.mean(kl_per_dim))
