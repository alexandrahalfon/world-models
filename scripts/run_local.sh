#!/usr/bin/env bash
# Run the full local pipeline using synthetic rollout data.
# All output is captured to a timestamped log file.
# FID (exp3) is skipped locally — Inception network is too slow on CPU.
set -e

cd "$(dirname "$0")/.."

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG="results/logs/local_run_${TIMESTAMP}.log"
mkdir -p results/logs results/rollouts results/exp1 results/exp2

echo "============================================================"
echo "  GenAI World Model — Local Smoke Test"
echo "  Started: $(date)"
echo "  Log:     $LOG"
echo "============================================================"

run_step() {
    local step="$1"
    local cmd="$2"
    echo ""
    echo "──────────────────────────────────────────────────────────"
    echo "  STEP $step"
    echo "  CMD:  $cmd"
    echo "  TIME: $(date +%H:%M:%S)"
    echo "──────────────────────────────────────────────────────────"
    eval "$cmd"
}

{
    echo "Local run started at $(date)"

    run_step "1/4  Generate synthetic rollouts" \
        "python scripts/generate_synthetic_rollouts.py"

    run_step "2/4  Exp 1 — Error growth curves + power-law fit" \
        "python experiments/exp1_error_growth.py"

    run_step "3/4  Exp 2 — Failure mode frame grids" \
        "python experiments/exp2_failure_modes.py"

    run_step "4/4  Generate HTML report" \
        "python scripts/generate_report.py"

    echo ""
    echo "============================================================"
    echo "  ALL STEPS COMPLETE"
    echo "  Finished: $(date)"
    echo "  Log:      $LOG"
    echo ""
    REPORT=$(ls -t results/report_*.html 2>/dev/null | head -1)
    if [ -n "$REPORT" ]; then
        echo "  Report:   $(pwd)/$REPORT"
        echo "  Open with:  open $REPORT"
    fi
    echo "============================================================"

} 2>&1 | tee "$LOG"
