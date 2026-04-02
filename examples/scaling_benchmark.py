"""Benchmark SIMPL fit time vs dataset size for CPU and GPU.

Generates synthetic data with 100 neurons and increasing numbers of time bins,
fits for 5 iterations on both CPU and GPU, and saves results to a CSV for
plotting.

Usage
-----
    python examples/scaling_benchmark.py
    python examples/scaling_benchmark.py --timepoints 1000 2000 4000 8000 16000
    python examples/scaling_benchmark.py --output results.csv
    python examples/scaling_benchmark.py --plot              # skip benchmark, just plot from CSV

The output CSV has columns: timepoints, device, fit_time_s, total_spikes, duration_s, spike_array_mb
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Parse args before any JAX import so we can set env vars
parser = argparse.ArgumentParser(description="SIMPL scaling benchmark: dataset size vs fit time")
parser.add_argument(
    "--timepoints",
    type=int,
    nargs="+",
    default=[500, 1000, 2000, 4000, 8000, 16000],
    help="List of dataset sizes (number of time bins) to benchmark",
)
parser.add_argument("--neurons", type=int, default=100, help="Number of neurons")
parser.add_argument("--n_iterations", type=int, default=5, help="Number of EM iterations per fit")
parser.add_argument("--bin_size", type=float, default=0.03, help="Spatial bin size")
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
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from simpl import SIMPL  # noqa: E402


def generate_synthetic_data(T: int, N: int, D: int = 2, dt: float = 0.1, seed: int = 42):
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


def time_fit(Y, Xb, time_arr, use_gpu, n_iterations, bin_size):
    """Fit SIMPL and return wall-clock time (excluding warm-up)."""
    model = SIMPL(
        kernel_bandwidth=0.04,
        speed_prior=0.3,
        bin_size=bin_size,
        env_pad=0.05,
        use_gpu=use_gpu,
    )
    # Warm-up: 1 iteration to trigger JIT compilation
    model.fit(Y, Xb, time_arr, n_iterations=1, verbose=False)
    jax.block_until_ready(model.X_)
    jax.block_until_ready(model.F_)

    # Timed run
    t0 = time.perf_counter()
    model.fit(Y, Xb, time_arr, n_iterations=n_iterations, verbose=False)
    jax.block_until_ready(model.X_)
    jax.block_until_ready(model.F_)
    elapsed = time.perf_counter() - t0
    return elapsed


def plot_results(csv_path: str | Path):
    """Load CSV and plot dataset size vs fit time, grouped by device."""
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for device, group in df.groupby("device"):
        group = group.sort_values("timepoints")
        ax.plot(group["timepoints"], group["fit_time_s"], "o-", label=device.upper(), linewidth=2, markersize=6)
    ax.set_xlabel("Dataset size (time bins)")
    ax.set_ylabel(f"Fit time (s)")
    n_iter = df.get("n_iterations", [None]).iloc[0]
    title = "SIMPL scaling: dataset size vs fit time"
    if "duration_s" in df.columns:
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        tick_locs = group["timepoints"].values
        tick_labels = [f"{d:.0f}" for d in group["duration_s"].values]
        ax2.set_xticks(tick_locs)
        ax2.set_xticklabels(tick_labels)
        ax2.set_xlabel("Recording duration (s)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(Path(csv_path).with_suffix(".png"), dpi=150)
    print(f"Plot saved to {Path(csv_path).with_suffix('.png')}")
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
        gpu_available = any(d.platform == "gpu" for d in jax.devices())
    except Exception:
        pass

    devices = []
    for d in args.devices:
        if d == "gpu" and not gpu_available:
            print(f"Skipping GPU (not available)")
            continue
        devices.append(d)

    if not devices:
        print("No devices to benchmark.")
        sys.exit(1)

    print("=" * 60)
    print("SIMPL Scaling Benchmark")
    print("=" * 60)
    print(f"Neurons      : {args.neurons}")
    print(f"Iterations   : {args.n_iterations}")
    print(f"Bin size     : {args.bin_size}")
    print(f"Timepoints   : {args.timepoints}")
    print(f"Devices      : {devices}")
    print()

    results = []

    for T in args.timepoints:
        Xb, Y, time_arr = generate_synthetic_data(T=T, N=args.neurons)
        total_spikes = int(Y.sum())
        duration_s = float(time_arr[-1] - time_arr[0])
        spike_array_mb = Y.nbytes / 1e6

        for device in devices:
            use_gpu = device == "gpu"
            print(f"T={T:>6d}, device={device:>3s} ... ", end="", flush=True)
            try:
                elapsed = time_fit(Y, Xb, time_arr, use_gpu, args.n_iterations, args.bin_size)
                print(f"{elapsed:.2f}s")
                results.append({
                    "timepoints": T,
                    "device": device,
                    "fit_time_s": elapsed,
                    "total_spikes": total_spikes,
                    "duration_s": duration_s,
                    "spike_array_mb": spike_array_mb,
                })
            except Exception as e:
                print(f"FAILED ({e})")
                results.append({
                    "timepoints": T,
                    "device": device,
                    "fit_time_s": float("nan"),
                    "total_spikes": total_spikes,
                    "duration_s": duration_s,
                    "spike_array_mb": spike_array_mb,
                })

    # Write CSV
    output_path = Path(args.output)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["timepoints", "device", "fit_time_s", "total_spikes", "duration_s", "spike_array_mb"])
        writer.writeheader()
        writer.writerows(results)

    print()
    print(f"Results saved to {output_path}")
    print()
    print(f"{'timepoints':>10s}  {'device':>6s}  {'fit_time_s':>10s}  {'total_spikes':>12s}  {'duration_s':>10s}  {'spike_mb':>8s}")
    print("-" * 68)
    for row in results:
        print(
            f"{row['timepoints']:>10d}  {row['device']:>6s}  {row['fit_time_s']:>10.2f}"
            f"  {row['total_spikes']:>12d}  {row['duration_s']:>10.1f}  {row['spike_array_mb']:>8.1f}"
        )


if __name__ == "__main__":
    main()

