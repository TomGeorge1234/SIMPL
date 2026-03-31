#!/bin/bash
#SBATCH --job-name=simpl-sweep-hd
#SBATCH --array=0-49
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00

# ── Sweep name (change this for each run) ────────────────────────────
SWEEP_NAME=${SWEEP_NAME:-"default_hd"}

# ── Hyperparameter grid ──────────────────────────────────────────────
KERNEL_BANDWIDTHS=(0.2 0.3 0.4 0.5)
SPEED_PRIORS=(50 75 100 150 200)
BEHAVIOR_PRIORS=(0.1 0.5 1.0 2.0 10.0)
BIN_SIZE=$(python3 -c "import math; print(2 * math.pi / 100)")

N_KB=${#KERNEL_BANDWIDTHS[@]}
N_SP=${#SPEED_PRIORS[@]}
N_BP=${#BEHAVIOR_PRIORS[@]}

# Map flat array index -> (kb, sp, bp) indices
IDX=$SLURM_ARRAY_TASK_ID
KB_IDX=$(( IDX / (N_SP * N_BP) ))
SP_IDX=$(( (IDX / N_BP) % N_SP ))
BP_IDX=$(( IDX % N_BP ))

KB=${KERNEL_BANDWIDTHS[$KB_IDX]}
SP=${SPEED_PRIORS[$SP_IDX]}
BP=${BEHAVIOR_PRIORS[$BP_IDX]}
BS=$BIN_SIZE

# ── Run ──────────────────────────────────────────────────────────────
SWEEP_DIR="$SLURM_SUBMIT_DIR"
OUT_DIR="$SWEEP_DIR/outputs/$SWEEP_NAME"
mkdir -p "$OUT_DIR/results" "$OUT_DIR/logs"

if [ "$IDX" -eq 0 ]; then
    N_TOTAL=$((N_KB * N_SP * N_BP))
    echo "════════════════════════════════════════════════════════"
    echo "Sweep: $SWEEP_NAME ($N_TOTAL jobs) [head direction cells]"
    echo "  kernel_bandwidth: ${KERNEL_BANDWIDTHS[*]}"
    echo "  speed_prior:      ${SPEED_PRIORS[*]}"
    echo "  behavior_prior:   ${BEHAVIOR_PRIORS[*]}"
    echo "  bin_size:         $BIN_SIZE (fixed)"
    echo "  output:           $OUT_DIR"
    echo "════════════════════════════════════════════════════════"
fi

python "$SWEEP_DIR/sweep_worker.py" \
    --dataset headdirectioncells \
    --sweep-name "$SWEEP_NAME" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --output-dir "$OUT_DIR/results" \
    --kernel-bandwidth "$KB" \
    --speed-prior "$SP" \
    --bin-size "$BS" \
    --behavior-prior "$BP" \
    2>&1 | tee "$OUT_DIR/logs/task_${SLURM_ARRAY_TASK_ID}.log"
