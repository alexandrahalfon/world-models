"""DreamerV3 wrapper that dispatches to env_jax via Singularity subprocess.

Does NOT import JAX. All JAX/DreamerV3 code lives in scripts/dreamerv3_rollout.py
which runs inside the env_jax Singularity overlay (separate from env_torch).

JAX and PyTorch conflict on CUDA versions on NYU Greene — they must never share an env.
See CLAUDE.md for the full constraint explanation.

Required environment variables (set in scripts/hpc_config.sh):
    SINGULARITY_SIF      path to the .sif image
    OVERLAY_JAX          path to the env_jax .ext3 overlay file
"""
from __future__ import annotations
import subprocess
import os
import numpy as np


class DreamerV3Wrapper:
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def run_rollout(
        self,
        game: str,
        actions_path: str,
        output_path: str,
        n_traj: int,
        K: int,
        seed: int = 42,
    ) -> np.ndarray:
        """Run DreamerV3 rollout inside the env_jax Singularity overlay and return pred_frames.

        Saves pred_frames to output_path (.npz) and returns the array.
        DreamerV3 operates in latent space internally; dreamerv3_rollout.py
        decodes latents back to pixel frames before saving.

        Uses Singularity exec with the env_jax overlay. Reads SINGULARITY_SIF and
        OVERLAY_JAX from environment (set via scripts/hpc_config.sh).
        Test interactively on a compute node before submitting SLURM array jobs.
        """
        sif = os.environ.get("SINGULARITY_SIF", "")
        overlay_jax = os.environ.get("OVERLAY_JAX", "")
        if not sif or not overlay_jax:
            raise EnvironmentError(
                "SINGULARITY_SIF and OVERLAY_JAX must be set. "
                "Run: source scripts/hpc_config.sh"
            )

        script = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "dreamerv3_rollout.py")
        script = os.path.abspath(script)

        inner_cmd = (
            f"source /ext3/env.sh && conda activate env_jax && "
            f"python {script} "
            f"--checkpoint {self.checkpoint_path} "
            f"--actions {actions_path} "
            f"--output {output_path} "
            f"--game {game} "
            f"--n_traj {n_traj} "
            f"--K {K} "
            f"--seed {seed}"
        )

        cmd = f'singularity exec --nv --overlay "{overlay_jax}:ro" "{sif}" /bin/bash -c "{inner_cmd}"'

        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"dreamerv3_rollout.py failed (returncode={result.returncode}):\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )

        data = np.load(output_path)
        pred_frames = data["pred_frames"]

        # DreamerV3 must output decoded pixel frames, not latents
        assert pred_frames.shape[-3:] == (84, 84, 3), (
            f"Expected decoded pixel frames [n_traj, K, 84, 84, 3], got shape {pred_frames.shape}. "
            "dreamerv3_rollout.py must decode latent states before saving."
        )
        return pred_frames
