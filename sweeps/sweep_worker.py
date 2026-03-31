"""Generic sweep worker for SIMPL.

Fits a single SIMPL model with the given hyperparameters and saves results.
Called by dataset-specific sbatch scripts (e.g. sweep_placecells.sh).

Usage:
    python sweep_worker.py --dataset placecells --sweep-name v1 --task-id 0 \
        --kernel-bandwidth 0.1 --speed-prior 0.8 --bin-size 0.05 --env-pad 0.0

    python sweep_worker.py --dataset headdirectioncells --sweep-name v1 --task-id 0 \
        --kernel-bandwidth 0.3 --speed-prior 100 --bin-size 0.0628 --behavior-prior 1.0
"""

import argparse
import os

import numpy as np
from simpl import SIMPL, load_demo_data
from simpl.utils import find_time_jumps

# ── CLI ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, required=True,
                    choices=["placecells", "headdirectioncells"])
parser.add_argument("--sweep-name", type=str, required=True)
parser.add_argument("--task-id", type=int, required=True)
parser.add_argument("--output-dir", type=str, required=True,
                    help="Directory to write result .npz files")
parser.add_argument("--kernel-bandwidth", type=float, required=True)
parser.add_argument("--speed-prior", type=float, required=True)
parser.add_argument("--bin-size", type=float, required=True)
parser.add_argument("--env-pad", type=float, default=0.0)
parser.add_argument("--behavior-prior", type=float, default=None)
parser.add_argument("--n-iterations", type=int, default=10)
args = parser.parse_args()

print(f"Task {args.task_id} [{args.dataset}]: " + ", ".join(
    f"{k}={v}" for k, v in vars(args).items()
    if k not in ("dataset", "sweep_name", "task_id", "n_iterations", "output_dir") and v is not None
))

# ── Load data ────────────────────────────────────────────────────────
DATASETS = {
    "placecells": "placecells_tanni2022.npz",
    "headdirectioncells": "headdirectioncells_vollan2025.npz",
}

data = load_demo_data(DATASETS[args.dataset])
Y_all = data["Y"]
Xb_all = data["Xb"]
time_all = data["time"]

dt = float(time_all[1] - time_all[0])
N_test = int(60 / dt)
N_train = len(time_all) - N_test

Y = Y_all[:N_train]
Xb = Xb_all[:N_train]
time = time_all[:N_train]

# Trial boundaries (HD cells have time jumps between trials)
trial_boundaries = None
if args.dataset == "headdirectioncells":
    jump_indices = find_time_jumps(time_all)
    trial_boundaries_all = np.concatenate([[0], jump_indices + 1])
    trial_boundaries = trial_boundaries_all[trial_boundaries_all < N_train]

# ── Fit ──────────────────────────────────────────────────────────────
model_kwargs = dict(
    kernel_bandwidth=args.kernel_bandwidth,
    speed_prior=args.speed_prior,
    bin_size=args.bin_size,
    env_pad=args.env_pad,
)
if args.dataset == "headdirectioncells":
    model_kwargs["is_1D_angular"] = True
if args.behavior_prior is not None:
    model_kwargs["behavior_prior"] = args.behavior_prior

model = SIMPL(**model_kwargs)

fit_kwargs = dict(n_iterations=args.n_iterations)
if trial_boundaries is not None:
    fit_kwargs["trial_boundaries"] = trial_boundaries

model.fit(Y, Xb, time, **fit_kwargs)

bps_val = float(model.loglikelihoods_.bits_per_spike_val.max().values)
best_iter = int(model.loglikelihoods_.bits_per_spike_val.argmax().values)
print(f"  -> val bits/spike = {bps_val:.4f} (best at iteration {best_iter})")

# ── Save ─────────────────────────────────────────────────────────────
out_dir = args.output_dir
os.makedirs(out_dir, exist_ok=True)

tag = f"{args.task_id:04d}"

# Save all non-None hyperparameters
save_dict = dict(
    dataset=args.dataset,
    kernel_bandwidth=args.kernel_bandwidth,
    speed_prior=args.speed_prior,
    bin_size=args.bin_size,
    env_pad=args.env_pad,
    val_bits_per_spike=bps_val,
    best_iteration=best_iter,
)
if args.behavior_prior is not None:
    save_dict["behavior_prior"] = args.behavior_prior

np.savez(os.path.join(out_dir, f"{tag}.npz"), **save_dict)

print(f"Saved to {out_dir}/{tag}.npz")
