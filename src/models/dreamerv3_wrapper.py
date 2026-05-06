"""DreamerV3 wrapper.

Does NOT import JAX. All JAX/DreamerV3 code lives in scripts/dreamerv3_rollout.py
which runs inside the env_jax Singularity overlay (separate from env_torch).

JAX and PyTorch conflict on CUDA versions on NYU Greene — they must never share
an env. See CLAUDE.md.

This wrapper is kept for in-process callers only (e.g. when env_torch is NOT
already inside Singularity, such as a login node). The supported production
entry point is `scripts/slurm/run_rollouts_dreamerv3.sbatch`, which launches
the env_jax overlay directly from the compute node — Singularity does not
support nested containers, so calling run_rollout() from *inside* a running
env_torch overlay (for example from `run_rollouts.sbatch`) will fail.

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
        # Refuse to run nested. Singularity sets SINGULARITY_CONTAINER inside
        # the running container; if it's set, the inner `singularity exec`
        # below will fail with a permission error. Use
        # scripts/slurm/run_rollouts_dreamerv3.sbatch instead.
        if os.environ.get("SINGULARITY_CONTAINER") or os.environ.get("APPTAINER_CONTAINER"):
            raise EnvironmentError(
                "DreamerV3Wrapper.run_rollout() detected it is running inside "
                "a Singularity container. Nested containers are not supported. "
                "Submit scripts/slurm/run_rollouts_dreamerv3.sbatch from the "
                "compute node instead."
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

        # DreamerV3 must output decoded pixel frames at 64x64x3 to match the
        # PIXEL_SIZE used by IRIS, DIAMOND, and rollout._collect_true_frames.
        assert pred_frames.shape[-3:] == (64, 64, 3), (
            f"Expected decoded pixel frames [n_traj, K, 64, 64, 3], got shape {pred_frames.shape}. "
            "dreamerv3_rollout.py must decode latent states and resize to 64x64."
        )
        return pred_frames
