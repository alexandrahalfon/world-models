"""Collect a shared action sequence per game using IRIS's trained actor.

Per CLAUDE.md: actions come from a trained actor (not random). All four world
models then read from data/atari/{game}_actions.npz.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import gymnasium as gym
import ale_py
from tqdm import tqdm
gym.register_envs(ale_py)
from src.models.iris_wrapper import IRISWrapper


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--games", nargs="+", default=["Breakout", "Pong", "Boxing"])
    p.add_argument("--n_traj", type=int, default=2000)
    p.add_argument("--K", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--checkpoint_dir", default="checkpoints/iris")
    p.add_argument("--out_dir", default="data/atari")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for game in args.games:
        env = gym.make(f"ALE/{game}-v5", obs_type="rgb", render_mode=None)
        n_actions = int(env.action_space.n)
        print(f"\n=== {game} (n_actions={n_actions}, n_traj={args.n_traj}, K={args.K}) ===", flush=True)
        ckpt = os.path.join(args.checkpoint_dir, f"{game}.pt")
        iris = IRISWrapper(ckpt, num_actions=n_actions); iris.load_checkpoint()
        print(f"IRIS loaded from {ckpt}", flush=True)
        actions = np.zeros((args.n_traj, args.K), dtype=np.int32)
        pbar = tqdm(range(args.n_traj), desc=f"{game}", unit="traj", ncols=100, mininterval=2.0)
        for t in pbar:
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
            iris.reset_actor()
            for k in range(args.K):
                a = iris.act(obs, temperature=args.temperature, should_sample=True)
                actions[t, k] = a
                obs, _, terminated, truncated, _ = env.step(a)
                if terminated or truncated:
                    obs, _ = env.reset(seed=int(rng.integers(0, 2**31)))
                    iris.reset_actor()
        pbar.close()
        env.close()
        out = os.path.join(args.out_dir, f"{game}_actions.npz")
        np.savez_compressed(out, actions=actions)
        unique, counts = np.unique(actions, return_counts=True)
        print(f"Saved {out} {actions.shape}; dist: {dict(zip(unique.tolist(), counts.tolist()))}", flush=True)


if __name__ == "__main__":
    main()
