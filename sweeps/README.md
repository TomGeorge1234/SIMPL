# Hyperparameter sweeps

Hyperparameter sweeps for SIMPL, selecting the configuration that maximises validation bits-per-spike.

## Datasets

### Place cells
Sweeps over `kernel_bandwidth`, `speed_prior`, and `env_pad` (fixed `bin_size`).

```bash
cd sweeps
SWEEP_NAME=pc_v1 sbatch sweep_placecells.sh
python sweep_aggregate.py pc_v1
```

### Head direction cells
Sweeps over `kernel_bandwidth`, `speed_prior`, and `behavior_prior` (fixed `bin_size`, `is_1D_angular=True`).

```bash
cd sweeps
SWEEP_NAME=hd_v1 sbatch sweep_headdirectioncells.sh
python sweep_aggregate.py hd_v1
```

## Files

- `sweep_worker.py` — Shared worker script. Loads data, fits one SIMPL model, and saves metrics to `.npz`. Handles both place cells and head direction cells via `--dataset`.
- `sweep_placecells.sh` — SLURM array job for place cells. Defines the hyperparameter grid and calls `sweep_worker.py`.
- `sweep_headdirectioncells.sh` — SLURM array job for head direction cells.
- `sweep_aggregate.py` — Collects `.npz` results for any sweep and prints the best configurations. Auto-detects which hyperparameters were saved.

## Output structure

```
sweeps/
  outputs/
    <sweep_name>/
      results/    # One .npz per task with params + validation bits-per-spike
      logs/       # Per-task log files
```

## Customising the grid

Edit the arrays at the top of the relevant `.sh` script and update `--array=0-N` to match the total number of combinations (product of array lengths minus 1).

## Running on a different cluster

The scripts avoid hardcoded paths and should work on any SLURM cluster. Before submitting, you may need to:

1. **Set your partition** — add `#SBATCH --partition=<your-partition>` to the `.sh` script, or pass it at submit time: `sbatch --partition=<your-partition> sweep_placecells.sh`.
2. **Activate your environment** — `simpl` must be importable by the Python on your `PATH`. If you use a venv or conda env, add the activation command (e.g. `source /path/to/venv/bin/activate`) before the `python` call in the `.sh` script.
3. **Adjust resources** — the defaults (`--mem=8G`, `--cpus-per-task=2`, `--time=02:00:00`) are sized for CPU-only runs. Increase memory for smaller `bin_size` values. If running on GPU, add `#SBATCH --gres=gpu:1`.
4. **SLURM logs** — by default logs go to the submission directory. To redirect, add `#SBATCH --output=<path>/slurm-%A_%a.out` (absolute path recommended).
