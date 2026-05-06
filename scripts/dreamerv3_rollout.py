"""DreamerV3 rollout script — runs in env_jax only.

Called by src/models/dreamerv3_wrapper.py via subprocess.
Loads the official JAX DreamerV3 checkpoint, runs autoregressive rollout using
pre-saved action sequences, decodes latent states to pixel frames at 64x64x3,
and saves results.

IMPORTANT: the DreamerV3 imagination loop must use the latent dynamics
(`world_model.imagine` style), NOT `world_model.observe`, after step 0.
`observe` re-encodes a real observation each step and so leaks ground truth
into the prediction — invalidating the autoregressive-rollout protocol.

Output frames are 64x64x3 to match PIXEL_SIZE in src/rollout.py.

Usage:
    source ~/.bashrc && conda activate env_jax
    python scripts/dreamerv3_rollout.py \
        --checkpoint checkpoints/dreamerv3/ \
        --actions data/atari/Breakout_actions.npz \
        --output results/rollouts/dreamerv3_Breakout.npz \
        --game Breakout --n_traj 2000 --K 50 --seed 42
"""
import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--actions", required=True, help="Path to pre-saved actions .npz")
    p.add_argument("--output", required=True, help="Output .npz path for pred_frames")
    p.add_argument("--game", required=True)
    p.add_argument("--n_traj", type=int, required=True)
    p.add_argument("--K", type=int, required=True)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


PIXEL_SIZE = 64


def main():
    args = parse_args()

    # These imports are intentionally deferred — this script only runs in env_jax
    import jax  # noqa: F401
    import jax.numpy as jnp

    try:
        import dreamerv3  # type: ignore[import]
    except ImportError as e:
        raise ImportError("dreamerv3 not installed. Activate env_jax.") from e

    # The method names below (configs.atari, Agent, world_model.observe,
    # world_model.imagine, world_model.decode) match the patterns documented
    # in the DreamerV3 reference implementation, but the installed package
    # version may differ. Fail loudly with a clear message so the user knows
    # exactly which symbol to fix instead of getting an opaque AttributeError
    # mid-rollout. Run this script with --n_traj 4 --K 10 first to verify.
    _required = [
        ("dreamerv3.configs.atari",  hasattr(getattr(dreamerv3, "configs", object()), "atari")),
        ("dreamerv3.Agent",          hasattr(dreamerv3, "Agent")),
    ]
    _missing = [name for name, ok in _required if not ok]
    if _missing:
        raise AttributeError(
            "Installed dreamerv3 package does not expose: "
            + ", ".join(_missing)
            + ". Update scripts/dreamerv3_rollout.py to match the installed "
              "API (see the DreamerV3 README for the equivalent symbols). "
              "DRY-RUN this script with --n_traj 4 --K 10 before submitting "
              "run_rollouts_dreamerv3.sbatch."
        )

    actions_data = np.load(args.actions)
    actions = actions_data["actions"]  # [n_traj, K]
    saved_seeds = actions_data["seeds"] if "seeds" in actions_data.files else None
    assert actions.shape[0] >= args.n_traj and actions.shape[1] >= args.K

    # Load checkpoint and initialize model.
    # The exact API depends on the installed dreamerv3 version. The two-stage
    # protocol below — encode the initial obs once, then imagine forward using
    # only the latent and the action — is the *correct* one for this study;
    # observe() at every step would leak ground truth and invalidate alpha.
    config = dreamerv3.configs.atari.update({"logdir": args.checkpoint})
    agent = dreamerv3.Agent(config)
    agent.load(args.checkpoint)

    wm = getattr(agent, "world_model", None)
    if wm is None or not all(hasattr(wm, n) for n in ("observe", "imagine", "decode")):
        raise AttributeError(
            "agent.world_model is missing one of: observe, imagine, decode. "
            "The installed dreamerv3 package likely renames these (e.g. "
            "obs_step / img_step / decoder). Update scripts/dreamerv3_rollout.py "
            "before resubmitting run_rollouts_dreamerv3.sbatch."
        )

    pred_frames = np.zeros((args.n_traj, args.K, PIXEL_SIZE, PIXEL_SIZE, 3), dtype=np.uint8)

    import gymnasium as gym
    env = gym.make(f"ALE/{args.game}-v5", obs_type="rgb", render_mode=None)

    rng = np.random.default_rng(args.seed)
    for traj_idx in range(args.n_traj):
        seed_i = int(saved_seeds[traj_idx]) if saved_seeds is not None else int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=seed_i)
        obs_in = _resize_obs(obs, 84)  # DreamerV3 encoder operates at 84x84

        state = agent.initial_state(batch_size=1)
        # Step 0: encode the real initial observation to get the first latent.
        obs_jax = jnp.array(obs_in)[None]
        first_action = jnp.array([int(actions[traj_idx, 0])])
        latent, state = agent.world_model.observe(obs_jax, first_action, state)

        for k in range(args.K):
            action = int(actions[traj_idx, k])
            if k > 0:
                # Pure imagination from latent — never re-encode predicted pixels.
                act_jax = jnp.array([action])
                latent, state = agent.world_model.imagine(latent, act_jax, state)
            decoded = agent.world_model.decode(latent)  # in [0, 1] or [0, 255]
            frame = np.array(decoded[0])
            if frame.dtype != np.uint8:
                frame = (frame * 255.0).clip(0, 255).astype(np.uint8) if frame.max() <= 1.5 \
                    else frame.clip(0, 255).astype(np.uint8)
            pred_frames[traj_idx, k] = _resize_obs(frame, PIXEL_SIZE)

    env.close()

    np.savez_compressed(args.output, pred_frames=pred_frames)
    print(f"Saved pred_frames {pred_frames.shape} to {args.output}")


def _resize_obs(obs: np.ndarray, size: int) -> np.ndarray:
    if obs.shape[:2] == (size, size):
        return obs
    from PIL import Image  # type: ignore[import]
    return np.array(Image.fromarray(obs).resize((size, size), Image.BILINEAR))


if __name__ == "__main__":
    main()
