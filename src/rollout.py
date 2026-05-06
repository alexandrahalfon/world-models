"""Central rollout engine with fixed-action replay and result caching.

Fixed-action protocol:
- Actions are collected once and saved per game; both real env and model are stepped
  with identical action sequences from identical seeds.
- After step 0, models are stepped AUTOREGRESSIVELY on their own predictions via
  model.reset(initial_obs) then model.step(action) per step.

Pixel models (IRIS, DIAMOND, DreamerV3) operate at 64x64x3 RGB. ALE returns
210x160x3 RGB, so the ground-truth frames are resized to 64x64 before storing
(both true_frames and pred_frames must be at the same resolution for MSE/FID/KL).

Caching: results/rollouts/{model}_{game}.npz is expensive to recompute; never delete.
"""
from __future__ import annotations
import os
from typing import Any
import numpy as np
import gymnasium as gym
import yaml

import ale_py
gym.register_envs(ale_py)

PIXEL_SIZE = 64


def _resize_uint8(frame: np.ndarray, size: int = PIXEL_SIZE) -> np.ndarray:
    if frame.shape[:2] == (size, size):
        return frame
    from PIL import Image
    return np.array(Image.fromarray(frame).resize((size, size), Image.BILINEAR))


def _load_config() -> dict:
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "configs", "experiment.yaml")
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def collect_actions(
    game: str,
    n_trajectories: int,
    K: int,
    actor,
    seed: int = 42,
    save_path: str | None = None,
    device: str = "cuda",
) -> np.ndarray:
    """Collect action sequences that survive K steps without termination.

    A trajectory is *only* recorded if it lasts the full K steps. This guarantees
    the (env_state, action) pairing remains valid for the rest of the pipeline:
    rollout() can replay these actions against a freshly reset env and the model
    will see the same state at the same action it was conditioned on.

    Mid-rollout resets (the previous behavior) silently desynchronize the
    trajectory's state from the action sequence, which makes the actions
    effectively random from that point on and inflates rollout MSE non-monotonically.
    """
    import torch  # noqa: F401
    env = gym.make(f"ALE/{game}-v5", obs_type="rgb", render_mode=None)
    rng = np.random.default_rng(seed)
    actions = np.zeros((n_trajectories, K), dtype=np.int32)
    seeds = np.zeros((n_trajectories,), dtype=np.int64)

    traj_idx = 0
    attempts = 0
    max_attempts = 50 * n_trajectories
    while traj_idx < n_trajectories:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Could not collect {n_trajectories} K={K} trajectories that survive "
                f"without termination after {attempts} attempts. K may be too long."
            )
        seed_i = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=seed_i)
        traj_actions = np.zeros((K,), dtype=np.int32)
        survived = True
        for k in range(K):
            action = actor.act(obs)
            traj_actions[k] = int(action)
            obs, _, terminated, truncated, _ = env.step(int(action))
            if terminated or truncated:
                survived = False
                break
        if survived:
            actions[traj_idx] = traj_actions
            seeds[traj_idx] = seed_i
            traj_idx += 1

    env.close()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        # Persist the seeds alongside actions so downstream rollouts can replay
        # the *same* env trajectory the actor saw during action collection.
        np.savez_compressed(save_path, actions=actions, seeds=seeds)
    return actions


def rollout(
    model_name: str,
    model: Any,
    game: str,
    n_trajectories: int,
    K: int,
    seed: int = 42,
    actions_path: str | None = None,
    cache_dir: str = "results/rollouts",
    obs_type: str = "rgb",
) -> dict:
    """Run fixed-action rollout for a model and game, with caching.

    obs_type:
        'rgb'   pixel models (IRIS, DIAMOND, DreamerV3) — frames stored as 64x64x3 uint8
        'ram'   MLP baseline — frames stored as (128,) uint8

    The model must implement reset(initial_obs) -> obs and step(action) -> obs.
    """
    cache_path = os.path.join(cache_dir, f"{model_name}_{game}.npz")
    if os.path.exists(cache_path):
        print(f"[rollout] Loading cached results from {cache_path}")
        data = np.load(cache_path)
        return {"pred_frames": data["pred_frames"],
                "true_frames": data["true_frames"],
                "actions": data["actions"]}

    if not actions_path or not os.path.exists(actions_path):
        raise FileNotFoundError(
            f"Action file not found at {actions_path}. "
            "Collect actions first using collect_actions()."
        )
    actions_data = np.load(actions_path)
    actions = actions_data["actions"]
    assert actions.shape[0] >= n_trajectories and actions.shape[1] >= K, (
        f"Action file shape {actions.shape} insufficient for n_traj={n_trajectories}, K={K}"
    )
    actions = actions[:n_trajectories, :K]
    # Per-trajectory seeds (when available) reproduce the exact env state sequence
    # the actor saw during action collection. Falls back to the legacy
    # (unsynchronized) RNG draw if the file pre-dates the seeds field.
    saved_seeds = actions_data["seeds"][:n_trajectories] if "seeds" in actions_data.files else None

    # DreamerV3 path: dispatch to env_jax subprocess
    if model_name == "dreamerv3":
        output_path = cache_path
        pred_frames = model.run_rollout(
            game=game, actions_path=actions_path, output_path=output_path,
            n_traj=n_trajectories, K=K, seed=seed,
        )
        true_frames = _collect_true_frames(game, actions, n_trajectories, K, seed, "rgb",
                                           saved_seeds=saved_seeds)
        os.makedirs(cache_dir, exist_ok=True)
        np.savez_compressed(cache_path, pred_frames=pred_frames,
                            true_frames=true_frames, actions=actions)
        return {"pred_frames": pred_frames, "true_frames": true_frames, "actions": actions}

    # In-process PyTorch models (MLP, IRIS, DIAMOND)
    if obs_type == "ram":
        obs_shape = (128,)
        env_obs_type = "ram"
    else:
        obs_shape = (PIXEL_SIZE, PIXEL_SIZE, 3)
        env_obs_type = "rgb"

    pred_frames = np.zeros((n_trajectories, K, *obs_shape), dtype=np.uint8)
    true_frames = np.zeros((n_trajectories, K, *obs_shape), dtype=np.uint8)
    # valid_mask[i, k] = True iff the real env had not yet terminated by step k.
    # MSE/FID/KL are computed over the masked region only — see metrics.per_step_mse.
    valid_mask = np.zeros((n_trajectories, K), dtype=bool)

    env = gym.make(f"ALE/{game}-v5", obs_type=env_obs_type, render_mode=None)
    rng = np.random.default_rng(seed)

    for traj_idx in range(n_trajectories):
        seed_i = int(saved_seeds[traj_idx]) if saved_seeds is not None else int(rng.integers(0, 2**31))
        real_obs, _ = env.reset(seed=seed_i)
        if obs_type != "ram":
            real_obs_stored = _resize_uint8(real_obs, PIXEL_SIZE)
        else:
            real_obs_stored = real_obs

        # Initialize model from the (possibly-resized) initial observation
        _ = model.reset(real_obs_stored)

        terminated_step = K  # k where env terminated; everything after is invalid
        for k in range(K):
            action = int(actions[traj_idx, k])

            if k < terminated_step:
                real_obs, _, terminated, truncated, _ = env.step(action)
                if obs_type != "ram":
                    real_obs_stored = _resize_uint8(real_obs, PIXEL_SIZE)
                else:
                    real_obs_stored = real_obs
                if terminated or truncated:
                    terminated_step = k + 1  # step k+1 onward invalid

            model_obs = model.step(action)

            true_frames[traj_idx, k] = real_obs_stored
            pred_frames[traj_idx, k] = model_obs
            valid_mask[traj_idx, k] = (k < terminated_step)

    env.close()

    os.makedirs(cache_dir, exist_ok=True)
    np.savez_compressed(cache_path, pred_frames=pred_frames,
                        true_frames=true_frames, actions=actions, valid_mask=valid_mask)
    print(f"[rollout] Saved {model_name}_{game} rollout to {cache_path} "
          f"(survival rate at k=K-1: {valid_mask[:, -1].mean():.2%})")
    return {"pred_frames": pred_frames, "true_frames": true_frames,
            "actions": actions, "valid_mask": valid_mask}


def _collect_true_frames(game: str, actions: np.ndarray, n_trajectories: int,
                         K: int, seed: int, obs_type: str,
                         saved_seeds: np.ndarray | None = None) -> np.ndarray:
    if obs_type == "ram":
        obs_shape = (128,)
        env_obs_type = "ram"
    else:
        obs_shape = (PIXEL_SIZE, PIXEL_SIZE, 3)
        env_obs_type = "rgb"
    true_frames = np.zeros((n_trajectories, K, *obs_shape), dtype=np.uint8)
    env = gym.make(f"ALE/{game}-v5", obs_type=env_obs_type, render_mode=None)
    rng = np.random.default_rng(seed)

    for traj_idx in range(n_trajectories):
        seed_i = int(saved_seeds[traj_idx]) if saved_seeds is not None else int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=seed_i)
        last_obs = obs if obs_type == "ram" else _resize_uint8(obs, PIXEL_SIZE)
        for k in range(K):
            action = int(actions[traj_idx, k])
            obs, _, terminated, truncated, _ = env.step(action)
            if not (terminated or truncated):
                last_obs = obs if obs_type == "ram" else _resize_uint8(obs, PIXEL_SIZE)
            # On termination we keep emitting last_obs; downstream metrics
            # should mask via the rollout cache's valid_mask.
            true_frames[traj_idx, k] = last_obs

    env.close()
    return true_frames
