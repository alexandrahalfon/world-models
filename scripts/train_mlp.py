"""Train the MLP baseline on Atari RAM observations.

Collects training data using the *same* IRIS-actor policy that produces the
shared action sequences in data/atari/{game}_actions.npz. Training under
random actions (the previous behavior) creates a train/eval distribution
mismatch: at rollout time the MLP sees actor-distribution actions, which
exercise game states it never trained on, producing runaway error.

The MLP operates in 128-byte RAM space (not pixels) — see CLAUDE.md for why.

Usage:
    python scripts/train_mlp.py --game Breakout --n_steps 200000
    python scripts/train_mlp.py --game Pong Boxing --n_steps 200000
"""
import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.models.mlp_baseline import MLPBaseline


def collect_training_data(game: str, n_steps: int, seed: int = 42,
                          use_actor: bool = True,
                          iris_checkpoint_dir: str = "checkpoints/iris"
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect (ram_state, action, next_ram_state) using the IRIS actor (default)
    or a random policy (use_actor=False).

    The IRIS actor needs RGB observations to choose actions, but the MLP
    trains on RAM. We run two parallel envs with synchronized seeds: one
    yields RGB for the actor, one yields RAM for the dataset.
    """
    import gymnasium as gym
    import ale_py
    gym.register_envs(ale_py)

    if use_actor:
        from src.models.iris_wrapper import IRISWrapper
        rgb_env = gym.make(f"ALE/{game}-v5", obs_type="rgb", render_mode=None)
        ram_env = gym.make(f"ALE/{game}-v5", obs_type="ram", render_mode=None)
        n_actions = int(rgb_env.action_space.n)
        ckpt = os.path.join(iris_checkpoint_dir, f"{game}.pt")
        iris = IRISWrapper(ckpt, num_actions=n_actions)
        iris.load_checkpoint()

        rng = np.random.default_rng(seed)
        seed_i = int(rng.integers(0, 2**31))
        rgb_obs, _ = rgb_env.reset(seed=seed_i)
        ram_obs, _ = ram_env.reset(seed=seed_i)
        iris.reset_actor()

        ram_states, actions, next_ram_states = [], [], []
        for _ in range(n_steps):
            action = iris.act(rgb_obs, should_sample=True)
            rgb_obs_next, _, term_r, trunc_r, _ = rgb_env.step(action)
            ram_obs_next, _, term_m, trunc_m, _ = ram_env.step(action)
            ram_states.append(ram_obs.copy())
            actions.append(action)
            next_ram_states.append(ram_obs_next.copy())
            rgb_obs = rgb_obs_next
            ram_obs = ram_obs_next
            if term_r or trunc_r or term_m or trunc_m:
                seed_i = int(rng.integers(0, 2**31))
                rgb_obs, _ = rgb_env.reset(seed=seed_i)
                ram_obs, _ = ram_env.reset(seed=seed_i)
                iris.reset_actor()

        rgb_env.close()
        ram_env.close()
    else:
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
    p.add_argument("--random_policy", action="store_true",
                   help="Collect training data with a random policy (legacy). "
                        "Default is the IRIS actor used for action sequences.")
    p.add_argument("--iris_checkpoint_dir", default="checkpoints/iris")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs("checkpoints/mlp", exist_ok=True)

    for game in args.game:
        print(f"\n=== Training MLP for {game} ===")
        policy = "random" if args.random_policy else "IRIS-actor"
        print(f"  Collecting {args.n_data} transitions ({policy} policy)...")
        ram_states, acts, next_ram_states = collect_training_data(
            game, args.n_data, seed=args.seed,
            use_actor=not args.random_policy,
            iris_checkpoint_dir=args.iris_checkpoint_dir,
        )
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
