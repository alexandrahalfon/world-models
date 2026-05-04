#!/usr/bin/env bash
# Run all post-rollout experiment scripts sequentially inside the Singularity container.
# Assumes rollout caches already exist in results/rollouts/.
# Fails immediately if any script errors (set -e).
set -e

source "$(dirname "$0")/hpc_config.sh"

run_in_torch() {
    singularity exec --nv \
        --overlay "$OVERLAY_TORCH:ro" \
        "$SINGULARITY_SIF" \
        /bin/bash -c "source /ext3/env.sh; conda activate env_torch; cd $PROJECT; $1"
}

echo "=== Experiment 1: Error growth curves ==="
run_in_torch "python experiments/exp1_error_growth.py"

echo "=== Experiment 2: Failure mode visualization ==="
run_in_torch "python experiments/exp2_failure_modes.py"

echo "=== Experiment 3: Distributional divergence ==="
run_in_torch "python experiments/exp3_distributional.py"

echo "=== All experiments complete. Results in $PROJECT/results/ ==="
