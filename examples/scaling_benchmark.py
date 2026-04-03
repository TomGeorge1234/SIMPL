"""Benchmark SIMPL fit time vs dataset size for CPU and GPU.

Generates synthetic data with 100 neurons at dt=0.01s for increasing recording
durations, fits for 5 iterations on both CPU and GPU, and saves results to a
CSV for plotting.

Usage
-----
    python examples/scaling_benchmark.py
    python examples/scaling_benchmark.py --minutes 1 2 5 10 20 30
    python examples/scaling_benchmark.py --output results.csv
    python examples/scaling_benchmark.py --plot              # skip benchmark, just plot from CSV

The output CSV has columns: minutes, timepoints, device, fit_time_s, total_spikes, duration_s, spike_array_mb
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

# Parse args before any JAX import so we can set env vars
parser = argparse.ArgumentParser(description="SIMPL scaling benchmark: dataset size vs fit time")
parser.add_argument(
    "--minutes",
    type=float,
    nargs="+",
    default=[1, 2, 5, 10, 20, 30],
    help="List of recording durations in minutes to benchmark",
)
parser.add_argument("--neurons", type=int, default=100, help="Number of neurons")
parser.add_argument("--n_iterations", type=int, default=5, help="Number of EM iterations per fit")
parser.add_argument("--bin_size", type=float, default=0.02, help="Spatial bin size")
parser.add_argument("--output", type=str, default="scaling_benchmark.csv", help="Output CSV path")
parser.add_argument(
    "--devices",
    type=str,
    nargs="+",
    default=["cpu", "gpu"],
    help="Devices to benchmark",
)
parser.add_argument("--plot", action="store_true", help="Skip benchmark, just plot from existing CSV")
args = parser.parse_args()

import jax  # noqa: E402
import numpy as np  # noqa: E402

from simpl import SIMPL  # noqa: E402

DT = 0.02  # 20 ms time bins (50 Hz)


def generate_synthetic_data(T: int, N: int, D: int = 2, dt: float = DT, seed: int = 42):
    """Generate synthetic 2D random-walk trajectory with Poisson spike counts."""
    rng = np.random.default_rng(seed)
    velocity = 0.02 * rng.standard_normal((T, D))
    trajectory = np.cumsum(velocity, axis=0)
    trajectory = trajectory - trajectory.min(axis=0)
    trajectory = trajectory / (trajectory.max(axis=0) + 1e-6)
    centers = rng.uniform(0.1, 0.9, size=(N, D))
    widths = rng.uniform(0.05, 0.15, size=(N,))
    diff = trajectory[:, None, :] - centers[None, :, :]
    rates = np.exp(-0.5 * np.sum(diff**2, axis=2) / widths[None, :] ** 2) * 0.5
    spikes = rng.poisson(rates).astype(float)
    time_arr = np.arange(T) * dt
    return trajectory, spikes, time_arr


import os
import subprocess
import tempfile

# Each (minutes, device) pair runs in a fresh subprocess so JIT compilation
# is always included in the timing — no cached compilations between runs.
_RUNNER = """\
import os, sys, time
import numpy as np

minutes = float(sys.argv[1])
neurons = int(sys.argv[2])
n_iterations = int(sys.argv[3])
bin_size = float(sys.argv[4])
device = sys.argv[5]
dt = 0.02

T = int(minutes * 60 / dt)
rng = np.random.default_rng(42)
velocity = 0.02 * rng.standard_normal((T, 2))
traj = np.cumsum(velocity, axis=0)
traj = traj - traj.min(axis=0)
traj = traj / (traj.max(axis=0) + 1e-6)
centers = rng.uniform(0.1, 0.9, size=(neurons, 2))
widths = rng.uniform(0.05, 0.15, size=(neurons,))
diff = traj[:, None, :] - centers[None, :, :]
rates = np.exp(-0.5 * np.sum(diff**2, axis=2) / widths[None, :] ** 2) * 0.5
spikes = rng.poisson(rates).astype(float)
time_arr = np.arange(T) * dt

from simpl import SIMPL
import jax
use_gpu = (device == "gpu")
model = SIMPL(kernel_bandwidth=0.04, speed_prior=0.3, bin_size=bin_size, env_pad=0.05, use_gpu=use_gpu)
t0 = time.perf_counter()
model.fit(spikes, traj, time_arr, n_iterations=n_iterations, verbose=False)
jax.block_until_ready(model.X_)
jax.block_until_ready(model.F_)
elapsed = time.perf_counter() - t0
total_spikes = int(spikes.sum())
spike_mb = spikes.nbytes / 1e6
duration_s = float(time_arr[-1] - time_arr[0])
print(f"OK {elapsed:.6f} {total_spikes} {duration_s:.2f} {spike_mb:.1f}")
"""


def time_fit(minutes, neurons, n_iterations, bin_size, device):
    """Fit SIMPL in a fresh subprocess so JIT is always included in timing."""
    script_path = Path(tempfile.gettempdir()) / "_simpl_scaling_runner.py"
    script_path.write_text(_RUNNER)

    env = os.environ.copy()
    if device == "cpu":
        env["JAX_PLATFORMS"] = "cpu"
        env["CUDA_VISIBLE_DEVICES"] = ""
    else:
        env.pop("JAX_PLATFORMS", None)

    proc = subprocess.run(
        [sys.executable, str(script_path),
         str(minutes), str(neurons), str(n_iterations), str(bin_size), device],
        capture_output=True, text=True, env=env, timeout=3600,
    )

    for line in reversed(proc.stdout.strip().split("\n")):
        if line.startswith("OK"):
            parts = line.split()
            return float(parts[1]), int(parts[2]), float(parts[3]), float(parts[4])

    stderr_tail = proc.stderr.strip().split("\n")[-1][:80] if proc.stderr else "no output"
    raise RuntimeError(f"Subprocess failed: {stderr_tail}")


_DEVICE_STYLE = {
    "cpu": {"label": "CPU", "linestyle": "-", "color": "#ff595e"},
    "gpu": {"label": "CUDA GPU", "linestyle": "-", "color": "#1982c4"},
    "gpu_metal": {"label": "GPU (Macbook Pro M4, jax_metal)", "linestyle": "--", "color": "#1982c4"},
}


def plot_results(csv_path: str | Path):
    """Load CSV and plot recording duration vs fit time, grouped by device."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for device, group in df.groupby("device"):
        group = group.sort_values("minutes")
        style = _DEVICE_STYLE.get(device, {"label": device.upper(), "linestyle": "-", "color": None})
        ax.plot(
            group["minutes"],
            group["fit_time_s"],
            "o",
            linestyle=style["linestyle"],
            color=style["color"],
            label=style["label"],
            linewidth=2,
            markersize=6,
        )
    ax.set_xlabel("Recording duration (minutes)")
    ax.set_ylabel("Fit time (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = str(Path(csv_path).with_suffix(".png"))
    fig.savefig(out_path, dpi=150)
    print(f"Plot saved to {out_path}")
    plt.show()


def main():
    if args.plot:
        csv_path = Path(args.output)
        if not csv_path.exists():
            print(f"CSV not found: {csv_path}. Run the benchmark first (without --plot).")
            sys.exit(1)
        plot_results(csv_path)
        return

    gpu_available = False
    try:
        gpu_available = any(d.platform in ("gpu", "METAL") for d in jax.devices())
    except Exception:
        pass

    devices = []
    for d in args.devices:
        if d == "gpu" and not gpu_available:
            print("Skipping GPU (not available)")
            continue
        devices.append(d)

    if not devices:
        print("No devices to benchmark.")
        sys.exit(1)

    print("=" * 60)
    print("SIMPL Scaling Benchmark")
    print("=" * 60)
    print(f"Neurons      : {args.neurons}")
    print(f"dt           : {DT}s")
    print(f"Iterations   : {args.n_iterations}")
    print(f"Bin size     : {args.bin_size}")
    print(f"Minutes      : {args.minutes}")
    print(f"Devices      : {devices}")
    print()

    results = []

    for minutes in args.minutes:
        T = int(minutes * 60 / DT)

        for device in devices:
            print(f"{minutes:>5.1f}min (T={T:>7d}), device={device:>3s} ... ", end="", flush=True)
            try:
                elapsed, total_spikes, duration_s, spike_array_mb = time_fit(
                    minutes, args.neurons, args.n_iterations, args.bin_size, device
                )
                print(f"{elapsed:.2f}s")
                results.append(
                    {
                        "minutes": minutes,
                        "timepoints": T,
                        "device": device,
                        "fit_time_s": elapsed,
                        "total_spikes": total_spikes,
                        "duration_s": duration_s,
                        "spike_array_mb": spike_array_mb,
                    }
                )
            except Exception as e:
                print(f"FAILED ({e})")
                results.append(
                    {
                        "minutes": minutes,
                        "timepoints": T,
                        "device": device,
                        "fit_time_s": float("nan"),
                        "total_spikes": 0,
                        "duration_s": 0,
                        "spike_array_mb": 0,
                    }
                )

    # Write CSV
    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        fieldnames = ["minutes", "timepoints", "device", "fit_time_s", "total_spikes", "duration_s", "spike_array_mb"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Results saved to {output_path}")
    print()
    print(f"{'minutes':>7s}  {'timepoints':>10s}  {'device':>6s}  {'fit_s':>8s}  {'spikes':>10s}  {'spike_mb':>8s}")
    print("-" * 58)
    for row in results:
        print(
            f"{row['minutes']:>7.1f}  {row['timepoints']:>10d}  {row['device']:>6s}"
            f"  {row['fit_time_s']:>8.2f}  {row['total_spikes']:>10d}  {row['spike_array_mb']:>8.1f}"
        )


if __name__ == "__main__":
    main()
