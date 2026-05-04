#!/usr/bin/env bash
# Download pretrained checkpoints for IRIS and DIAMOND.
# DreamerV3 checkpoints must be downloaded separately from the official repo.
# MLP is trained from scratch via scripts/train_mlp.py.
set -e

CKPT_DIR="checkpoints"

echo "=== IRIS checkpoints ==="
# Official IRIS checkpoints from ICLR 2023 paper
# Replace these URLs with the actual checkpoint links from https://github.com/eloialonso/iris
# iris_ckpt_url="https://..."  # TODO: fill in after checking IRIS repo
# wget -P "$CKPT_DIR/iris/" "$iris_ckpt_url"
echo "TODO: Download IRIS checkpoint from https://github.com/eloialonso/iris (see Releases)"

echo "=== DIAMOND checkpoints ==="
# Official DIAMOND checkpoints from NeurIPS 2024 paper
# Replace with actual links from https://github.com/eloialonso/diamond
# diamond_ckpt_url="https://..."  # TODO: fill in after checking DIAMOND repo
# wget -P "$CKPT_DIR/diamond/" "$diamond_ckpt_url"
echo "TODO: Download DIAMOND checkpoint from https://github.com/eloialonso/diamond (see Releases)"

echo "=== DreamerV3 checkpoints ==="
# Official DreamerV3 checkpoints (JAX format) from https://github.com/danijar/dreamerv3
# dreamerv3_ckpt_url="https://..."  # TODO: fill in after checking DreamerV3 repo
# wget -P "$CKPT_DIR/dreamerv3/" "$dreamerv3_ckpt_url"
echo "TODO: Download DreamerV3 checkpoint from https://github.com/danijar/dreamerv3"

echo ""
echo "Place each checkpoint in its subdirectory under checkpoints/ and update configs/experiment.yaml paths."
