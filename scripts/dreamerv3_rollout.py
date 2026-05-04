"""DreamerV3 rollout script — runs in env_jax only.

Called by src/models/dreamerv3_wrapper.py via subprocess.
Loads the official JAX DreamerV3 checkpoint, runs autoregressive rollout using
pre-saved action sequences, decodes latent states to pixel frames, and saves results.

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


def main():
    args = parse_args()

    # These imports are intentionally deferred — this script only runs in env_jax
    import jax
    import jax.numpy as jnp

    try:
        import dreamerv3  # type: ignore[import]
    except ImportError as e:
        raise ImportError("dreamerv3 not installed. Activate env_jax.") from e

    actions_data = np.load(args.actions)
    actions = actions_data["actions"]  # [n_traj, K]
    assert actions.shape[0] >= args.n_traj and actions.shape[1] >= args.K

    # Load checkpoint and initialize model
    # NOTE: The exact dreamerv3 API depends on the repo version.
    # Update these calls to match the installed dreamerv3 checkpoint loading API.
    config = dreamerv3.configs.atari.update({"logdir": args.checkpoint})
    agent = dreamerv3.Agent(config)
    agent.load(args.checkpoint)

    pred_frames = np.zeros((args.n_traj, args.K, 84, 84, 3), dtype=np.uint8)

    # Roll out autoregressively using saved action sequences
    # Model is stepped on its own decoded predictions after step 0
    import gymnasium as gym
    env = gym.make(f"ALE/{args.game}-v5", obs_type="rgb", render_mode=None)

    rng = np.random.default_rng(args.seed)
    for traj_idx in range(args.n_traj):
        seed_i = int(rng.integers(0, 2**31))
        obs, _ = env.reset(seed=seed_i)
        obs = _resize_obs(obs)

        state = agent.initial_state(batch_size=1)

        for k in range(args.K):
            action = int(actions[traj_idx, k])
            # Step model autoregressively on its own previous prediction
            obs_jax = jnp.array(obs)[None]  # [1, 84, 84, 3]
            act_jax = jnp.array([action])
            next_obs_latent, state = agent.world_model.observe(obs_jax, act_jax, state)
            # Decode latent state to pixel frame — required before saving
            next_obs_decoded = agent.world_model.decode(next_obs_latent)
            obs = np.array(next_obs_decoded[0]).clip(0, 255).astype(np.uint8)
            pred_frames[traj_idx, k] = obs

    env.close()

    np.savez_compressed(args.output, pred_frames=pred_frames)
    print(f"Saved pred_frames {pred_frames.shape} to {args.output}")


def _resize_obs(obs: np.ndarray) -> np.ndarray:
    """Resize Atari observation to 84x84x3 if needed."""
    if obs.shape == (84, 84, 3):
        return obs
    from PIL import Image  # type: ignore[import]
    return np.array(Image.fromarray(obs).resize((84, 84), Image.BILINEAR))


if __name__ == "__main__":
    main()
