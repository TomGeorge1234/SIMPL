# Hyperparameter sweeps

Hyperparameter sweeps for SIMPL, selecting the configuration that maximises validation bits-per-spike.

## Usage

Submit from the `sweeps/` directory:

```bash
cd sweeps
SWEEP_NAME=pc_v1 sbatch sweep_placecells.sh
SWEEP_NAME=hd_v1 sbatch sweep_headdirectioncells.sh
```

Each script submits a **single SLURM job** that launches all hyperparameter combos as parallel `srun` tasks. SLURM allocates resources for `--ntasks` tasks and runs them as CPUs become available — no array job limits to worry about.

After all jobs complete, aggregate results:

```bash
python sweep_aggregate.py pc_v1
python sweep_aggregate.py hd_v1
```

## Datasets

### Place cells
Sweeps over `kernel_bandwidth`, `speed_prior`, and `env_pad` (fixed `bin_size`).

### Head direction cells
Sweeps over `kernel_bandwidth`, `speed_prior`, and `behavior_prior` (fixed `bin_size`, `is_1D_angular=True`).

## Files

- `sweep_worker.py` — Shared worker script. Loads data, fits one SIMPL model, and saves metrics to `.npz`. Handles both datasets via `--dataset`.
- `sweep_placecells.sh` — SLURM job for place cells.
- `sweep_headdirectioncells.sh` — SLURM job for head direction cells.
- `sweep_aggregate.py` — Collects `.npz` results for any sweep and prints the best configurations. Auto-detects which hyperparameters were saved.

## Output structure

```
sweeps/
  outputs/
    <sweep_name>/
      results/    # One .npz per task with params + validation bits-per-spike
      logs/       # Per-task log files (task_0.log, task_1.log, ...)
```

## Customising the grid

Edit the arrays at the top of the relevant `.sh` script. Update `#SBATCH --ntasks` to match the total number of combinations (product of array lengths).

## Running on a different cluster

The scripts avoid hardcoded paths and should work on any SLURM cluster. Before submitting, you may need to:

1. **Set your partition** — add `#SBATCH --partition=<your-partition>` to the `.sh` script, or pass it at submit time: `sbatch --partition=<your-partition> sweep_placecells.sh`.
2. **Activate your environment** — `simpl` must be importable by the Python on your `PATH`. If you use a venv or conda env, add the activation command (e.g. `source /path/to/venv/bin/activate`) before the `python` call in the `.sh` script.
3. **Adjust resources** — the defaults (`--mem-per-cpu=4G`, `--cpus-per-task=2`, `--time=24:00:00`) are sized for CPU-only runs. Increase memory for smaller `bin_size` values. If running on GPU, add `#SBATCH --gres=gpu:1`.
