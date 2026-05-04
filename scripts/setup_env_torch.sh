#!/usr/bin/env bash
# Create env_torch overlay and install PyTorch, IRIS, DIAMOND, metrics inside it.
# Run once on the torch login node.
set -e

source "$(dirname "$0")/hpc_config.sh"
export PATH=/share/apps/apptainer/bin:$PATH

mkdir -p "$SCRATCH/overlays"

if [ -f "$OVERLAY_TORCH" ]; then
    echo "Overlay already exists at $OVERLAY_TORCH. Delete it first to rebuild."
    exit 1
fi

# Copy and unzip the 15GB overlay template
echo "Creating overlay file..."
cp "$OVERLAY_TEMPLATE_10G" "$OVERLAY_TORCH.gz"
gunzip "$OVERLAY_TORCH.gz"
echo "Overlay ready at $OVERLAY_TORCH"

# Install everything inside the overlay
echo "Installing env_torch (this takes ~20 min)..."
singularity exec --fakeroot \
    --overlay "$OVERLAY_TORCH" \
    "$SINGULARITY_SIF" \
    /bin/bash -c "
        # Install Miniconda into the overlay at /ext3/miniconda3
        wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
        bash /tmp/miniconda.sh -b -p /ext3/miniconda3
        rm /tmp/miniconda.sh

        # Write env.sh — must include unset -f which to fix rbash restrictions
        cat > /ext3/env.sh << 'ENVEOF'
#!/bin/bash
unset -f which
source /ext3/miniconda3/etc/profile.d/conda.sh
export PATH=/ext3/miniconda3/bin:$PATH
export PYTHONPATH=/ext3/miniconda3/bin:$PATH
ENVEOF

        source /ext3/env.sh
        conda create -n env_torch python=3.11 -y
        conda activate env_torch

        pip install torch>=2.2.0 torchvision>=0.17.0
        pip install 'gymnasium[atari]>=0.29.0' ale-py autorom[accept-rom-license]
        pip install scipy scikit-learn pytorch-fid
        pip install matplotlib seaborn pandas numpy pyyaml tqdm ipykernel

        AutoROM --accept-license

        # Install IRIS and DIAMOND from source
        pip install git+https://github.com/eloialonso/iris.git
        pip install git+https://github.com/eloialonso/diamond.git

        echo 'env_torch installation complete'
    "

echo "Done. Test with:"
echo "  singularity exec --nv --overlay $OVERLAY_TORCH:ro $SINGULARITY_SIF /bin/bash -c \"source /ext3/env.sh; conda activate env_torch; python -c 'import torch; print(torch.__version__)'\""
