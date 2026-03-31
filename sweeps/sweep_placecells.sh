#!/bin/bash
#SBATCH --job-name=simpl-sweep-pc
#SBATCH --array=0-49
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/dev/null

# ── Sweep name (change this for each run) ────────────────────────────
SWEEP_NAME=${SWEEP_NAME:-"default_pc"}

# ── Hyperparameter grid ──────────────────────────────────────────────
KERNEL_BANDWIDTHS=(0.05 0.075 0.1 0.15 0.2)
SPEED_PRIORS=(0.4 0.6 0.8 1.0 1.5)
ENV_PADS=(0.0 0.1)
BIN_SIZE=0.05

N_KB=${#KERNEL_BANDWIDTHS[@]}
N_SP=${#SPEED_PRIORS[@]}
N_EP=${#ENV_PADS[@]}

# Map flat array index -> (kb, sp, ep) indices
IDX=$SLURM_ARRAY_TASK_ID
KB_IDX=$(( IDX / (N_SP * N_EP) ))
SP_IDX=$(( (IDX / N_EP) % N_SP ))
EP_IDX=$(( IDX % N_EP ))

KB=${KERNEL_BANDWIDTHS[$KB_IDX]}
SP=${SPEED_PRIORS[$SP_IDX]}
EP=${ENV_PADS[$EP_IDX]}
BS=$BIN_SIZE

# ── Run ──────────────────────────────────────────────────────────────
SWEEP_DIR="$SLURM_SUBMIT_DIR"
OUT_DIR="$SWEEP_DIR/outputs/$SWEEP_NAME"
mkdir -p "$OUT_DIR/results" "$OUT_DIR/logs"

if [ "$IDX" -eq 0 ]; then
    N_TOTAL=$((N_KB * N_SP * N_EP))
    echo "════════════════════════════════════════════════════════"
    echo "Sweep: $SWEEP_NAME ($N_TOTAL jobs) [place cells]"
    echo "  kernel_bandwidth: ${KERNEL_BANDWIDTHS[*]}"
    echo "  speed_prior:      ${SPEED_PRIORS[*]}"
    echo "  env_pad:          ${ENV_PADS[*]}"
    echo "  bin_size:         $BIN_SIZE (fixed)"
    echo "  output:           $OUT_DIR"
    echo "════════════════════════════════════════════════════════"
fi

python "$SWEEP_DIR/sweep_worker.py" \
    --dataset placecells \
    --sweep-name "$SWEEP_NAME" \
    --task-id "$SLURM_ARRAY_TASK_ID" \
    --output-dir "$OUT_DIR/results" \
    --kernel-bandwidth "$KB" \
    --speed-prior "$SP" \
    --bin-size "$BS" \
    --env-pad "$EP" \
    2>&1 | tee "$OUT_DIR/logs/task_${SLURM_ARRAY_TASK_ID}.log"
