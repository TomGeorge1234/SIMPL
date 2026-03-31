#!/bin/bash
#SBATCH --job-name=simpl-sweep-pc
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8G
#SBATCH --time=24:00:00
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
N_TOTAL=$((N_KB * N_SP * N_EP))

# ── Setup ────────────────────────────────────────────────────────────
SWEEP_DIR="$SLURM_SUBMIT_DIR"
OUT_DIR="$SWEEP_DIR/outputs/$SWEEP_NAME"
mkdir -p "$OUT_DIR/results" "$OUT_DIR/logs"

echo "════════════════════════════════════════════════════════"
echo "Sweep: $SWEEP_NAME ($N_TOTAL combos) [place cells]"
echo "  kernel_bandwidth: ${KERNEL_BANDWIDTHS[*]}"
echo "  speed_prior:      ${SPEED_PRIORS[*]}"
echo "  env_pad:          ${ENV_PADS[*]}"
echo "  bin_size:         $BIN_SIZE (fixed)"
echo "  output:           $OUT_DIR"
echo "════════════════════════════════════════════════════════"

# ── Launch all combos ────────────────────────────────────────────────
for (( IDX=0; IDX<N_TOTAL; IDX++ )); do
    KB_IDX=$(( IDX / (N_SP * N_EP) ))
    SP_IDX=$(( (IDX / N_EP) % N_SP ))
    EP_IDX=$(( IDX % N_EP ))

    KB=${KERNEL_BANDWIDTHS[$KB_IDX]}
    SP=${SPEED_PRIORS[$SP_IDX]}
    EP=${ENV_PADS[$EP_IDX]}

    srun --ntasks=1 --exclusive \
        python "$SWEEP_DIR/sweep_worker.py" \
            --dataset placecells \
            --sweep-name "$SWEEP_NAME" \
            --task-id "$IDX" \
            --output-dir "$OUT_DIR/results" \
            --kernel-bandwidth "$KB" \
            --speed-prior "$SP" \
            --bin-size "$BIN_SIZE" \
            --env-pad "$EP" \
            > "$OUT_DIR/logs/task_${IDX}.log" 2>&1 &
done

wait
echo "All $N_TOTAL tasks complete."
