#!/usr/bin/env bash
# Central HPC configuration for NYU torch cluster.
# All sbatch scripts and setup scripts source this file.

export NETID="aih7261"
export SCRATCH="/scratch/$NETID"
export PROJECT="$SCRATCH/GenAIProject"

# CUDA 12.6 image on torch cluster
export SINGULARITY_SIF="/share/apps/images/cuda12.6.3-cudnn9.5.1-ubuntu22.04.5.sif"

# Overlay template locations
export OVERLAY_TEMPLATE_15G="/share/apps/overlay-fs-ext3/overlay-15GB-500K.ext3.gz"
export OVERLAY_TEMPLATE_10G="/share/apps/overlay-fs-ext3/overlay-10GB-400K.ext3.gz"

# Your overlay files (created by setup_env_torch.sh / setup_env_jax.sh)
export OVERLAY_TORCH="$SCRATCH/overlays/env_torch.ext3"
export OVERLAY_JAX="$SCRATCH/overlays/env_jax.ext3"

export SLURM_ACCOUNT="ds_ga_3001-2026sp"

# Use apptainer (singularity-ce module has broken libsubid on this cluster)
