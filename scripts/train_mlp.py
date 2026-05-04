"""Train the MLP baseline on Atari RAM observations.

Collects training data by running a random policy in the Gymnasium ALE environment,
then trains the MLP to predict next RAM state from (current RAM state, action).

The MLP operates in 128-byte RAM space (not pixels) — see CLAUDE.md for why.
After training, saves the checkpoint and the action sequences used (via IRIS actor)
for consistent rollout evaluation across all models.

Usage:
    python scripts/train_mlp.py --game Breakout --n_steps 200000
    python scripts/train_mlp.py --game Pong --game Boxing --n_steps 200000
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.mlp_baseline import MLPBaseline


def collect_training_data(game: str, n_steps: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect (ram_state, action, next_ram_state) tuples using a random policy."""
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)
    env = gym.make(f"ALE/{game}-v5", obs_type="ram", render_mode=None)
    env.reset(seed=seed)

    ram_states, actions, next_ram_states = [], [], []
    obs, _ = env.reset()
    for _ in range(n_steps):
        action = env.action_space.sample()
        next_obs, _, terminated, truncated, _ = env.step(action)
        ram_states.append(obs.copy())
        actions.append(action)
        next_ram_states.append(next_obs.copy())
        obs = next_obs
        if terminated or truncated:
            obs, _ = env.reset()

    env.close()
    return (
        np.array(ram_states, dtype=np.uint8),
        np.array(actions, dtype=np.int32),
        np.array(next_ram_states, dtype=np.uint8),
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--game", nargs="+", default=["Breakout", "Pong", "Boxing"])
    p.add_argument("--n_steps", type=int, default=200_000, help="Training steps (not data collection steps)")
    p.add_argument("--n_data", type=int, default=500_000, help="Data collection steps")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("checkpoints/mlp", exist_ok=True)

    for game in args.game:
        print(f"\n=== Training MLP for {game} ===")
        print(f"  Collecting {args.n_data} transitions...")
        ram_states, acts, next_ram_states = collect_training_data(game, args.n_data, seed=args.seed)
        print(f"  Dataset: {len(ram_states)} transitions")

        mlp = MLPBaseline(device=args.device)
        print(f"  Training for {args.n_steps} steps...")
        losses = mlp.train(ram_states, acts, next_ram_states, n_steps=args.n_steps)

        ckpt_path = f"checkpoints/mlp/{game.lower()}.pt"
        mlp.save(ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}")
        print(f"  Final loss: {np.mean(losses[-1000:]):.6f}")


if __name__ == "__main__":
    main()
