#!/usr/bin/env bash
# Create env_jax overlay — used ONLY by scripts/dreamerv3_rollout.py.
# Run once on the torch login node.
set -e

source "$(dirname "$0")/hpc_config.sh"
export PATH=/share/apps/apptainer/bin:$PATH

mkdir -p "$SCRATCH/overlays"

if [ -f "$OVERLAY_JAX" ]; then
    echo "Overlay already exists at $OVERLAY_JAX. Delete it first to rebuild."
    exit 1
fi

echo "Creating overlay file..."
cp "$OVERLAY_TEMPLATE_10G" "$OVERLAY_JAX.gz"
gunzip "$OVERLAY_JAX.gz"
echo "Overlay ready at $OVERLAY_JAX"

echo "Installing env_jax (this takes ~10 min)..."
singularity exec --fakeroot \
    --overlay "$OVERLAY_JAX" \
    "$SINGULARITY_SIF" \
    /bin/bash -c "
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p /ext3/miniconda3
        rm /tmp/miniconda.sh

        cat > /ext3/env.sh << 'ENVEOF'
#!/bin/bash
unset -f which
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
export PYTHONPATH=/ext3/miniconda3/bin:$PATH
ENVEOF

        source /ext3/env.sh
        conda create -n env_jax python=3.11 -y
        conda activate env_jax

        pip install 'jax[cuda12_pip]>=0.4.25' flax optax numpy
        pip install gymnasium[atari] ale-py pillow
        pip install git+https://github.com/danijar/dreamerv3.git

        echo 'env_jax installation complete'
    "

echo "Done. Test with:"
echo "  singularity exec --nv --overlay $OVERLAY_JAX:ro $SINGULARITY_SIF /bin/bash -c \"source /ext3/env.sh; conda activate env_jax; python -c 'import jax; print(jax.devices())'\""
