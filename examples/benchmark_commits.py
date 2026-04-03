"""Benchmark SIMPL fit time across key performance PRs.

Hand-written benchmark code per commit to handle API changes.
Results appended to CSV as they complete.

Usage:
    python local/benchmark_commits.py
    python local/benchmark_commits.py --output local/commit_benchmarks.csv
    python local/benchmark_commits.py --n_timepoints 20000 --n_neurons 100
"""

import argparse
import csv
import os
import subprocess
import sys
import textwrap
from pathlib import Path

parser = argparse.ArgumentParser(description="Benchmark SIMPL across key PRs")
parser.add_argument("--output", type=str, default="local/commit_benchmarks.csv", help="Output CSV path")
parser.add_argument("--n_iterations", type=int, default=5, help="Number of EM iterations")
parser.add_argument("--n_neurons", type=int, default=100, help="Number of neurons")
parser.add_argument("--minutes", type=float, default=30, help="Recording duration in minutes")
parser.add_argument("--last", type=int, default=None, help="Only run the last N commits")
args = parser.parse_args()

REPO_ROOT = Path(__file__).resolve().parent.parent

# ── Hand-written benchmark snippets per commit ──
# Each is a Python code string. Variables available: Y, Xb, time_arr, N (iterations)

# Pre-sklearn era: SIMPL(data=xr.Dataset, environment=Environment)
# then model.train_N_epochs(N, verbose=False)
_PRE_SKLEARN = textwrap.dedent("""\
    import xarray as xr
    from simpl.environment import Environment
    from simpl.simpl import SIMPL
    env = Environment(X=Xb, pad=0.05, bin_size=0.02, verbose=False)
    ds = xr.Dataset({"Y": (["time", "neuron"], Y), "Xb": (["time", "dim"], Xb)},
                     coords={"time": time_arr})
    ds.attrs["trial_slices"] = [slice(0, len(time_arr))]
    model = SIMPL(data=ds, environment=env, kernel_bandwidth=0.04, speed_prior=0.3)
    model.train_N_epochs(N, verbose=False)
""")

# sklearn era with n_epochs (no use_gpu)
_SKLEARN_EPOCHS = textwrap.dedent("""\
    from simpl import SIMPL
    model = SIMPL(kernel_bandwidth=0.04, speed_prior=0.3, bin_size=0.02, env_pad=0.05)
    model.fit(Y, Xb, time_arr, n_epochs=N, verbose=False)
""")

# sklearn era with n_epochs + use_gpu
_SKLEARN_EPOCHS_GPU = textwrap.dedent("""\
    from simpl import SIMPL
    model = SIMPL(kernel_bandwidth=0.04, speed_prior=0.3, bin_size=0.02, env_pad=0.05, use_gpu=False)
    model.fit(Y, Xb, time_arr, n_epochs=N, verbose=False)
""")

# Modern era: n_iterations + use_gpu
_MODERN = textwrap.dedent("""\
    from simpl import SIMPL
    model = SIMPL(kernel_bandwidth=0.04, speed_prior=0.3, bin_size=0.02, env_pad=0.05, use_gpu=False)
    model.fit(Y, Xb, time_arr, n_iterations=N, verbose=False)
""")

COMMITS = [
    # ── Pre-sklearn era ──
    {"sha": "1d197cd", "message": "PR#19 package-refactor",      "code": _PRE_SKLEARN},
    # ── sklearn era (n_epochs, no use_gpu) ──
    {"sha": "5802dc8", "message": "PR#37 sklearn-refactor",      "code": _SKLEARN_EPOCHS},
    {"sha": "6a245ea", "message": "PR#42 FX-memory-fix",         "code": _SKLEARN_EPOCHS},
    {"sha": "f910570", "message": "PR#46 colab-data",            "code": _SKLEARN_EPOCHS},
    {"sha": "f2ea4f5", "message": "PR#65 single-pass-kalman",    "code": _SKLEARN_EPOCHS},
    # ── sklearn era (n_epochs + use_gpu) ──
    {"sha": "5391136", "message": "PR#72 gpu_speedup",           "code": _SKLEARN_EPOCHS_GPU},
    # ── Modern era (n_iterations) ──
    {"sha": "7219c16", "message": "PR#74 n_iterations rename",   "code": _MODERN},
    {"sha": "b7f172e", "message": "PR#75 speed_up",              "code": _MODERN},
    {"sha": "db7601e", "message": "PR#82 batch_decoding",        "code": _MODERN},
    {"sha": "96d9b18", "message": "PR#79 main HEAD",             "code": _MODERN},
    {"sha": "17ce414", "message": "PR#86 jax-metal (Metal GPU)", "device": "gpu", "code": textwrap.dedent("""\
        from simpl import SIMPL
        model = SIMPL(kernel_bandwidth=0.04, speed_prior=0.3, bin_size=0.02, env_pad=0.05, use_gpu=True)
        model.fit(Y, Xb, time_arr, n_iterations=N, verbose=False)
    """)},
]

if args.last is not None:
    COMMITS = COMMITS[-args.last:]

# ── Runner template ──
# Wraps the per-commit code with data loading, timing, and error handling
# Uses __DATA_PATH__ style placeholders to avoid brace conflicts
RUNNER_TEMPLATE = """\
import os
import sys
import time
import numpy as np

def main():
    # Generate synthetic data (same as scaling_benchmark.py)
    dt = 0.02
    T = int(__MINUTES__ * 60 / dt)
    N_neurons = __N_NEURONS__
    rng = np.random.default_rng(42)
    velocity = 0.02 * rng.standard_normal((T, 2))
    trajectory = np.cumsum(velocity, axis=0)
    trajectory = trajectory - trajectory.min(axis=0)
    trajectory = trajectory / (trajectory.max(axis=0) + 1e-6)
    centers = rng.uniform(0.1, 0.9, size=(N_neurons, 2))
    widths = rng.uniform(0.05, 0.15, size=(N_neurons,))
    diff = trajectory[:, None, :] - centers[None, :, :]
    rates = np.exp(-0.5 * np.sum(diff**2, axis=2) / widths[None, :] ** 2) * 0.5
    Y = rng.poisson(rates).astype(float)
    Xb = trajectory
    time_arr = np.arange(T) * dt
    N = __N_ITERATIONS__

    t0 = time.perf_counter()
__CODE__
    import jax
    device = jax.default_backend()
    if hasattr(model, "X_"):
        jax.block_until_ready(model.X_)
    if hasattr(model, "F_"):
        jax.block_until_ready(model.F_)
    elapsed = time.perf_counter() - t0
    print(f"OK {elapsed:.4f} {device}")

try:
    main()
except Exception as e:
    err = str(e).replace(chr(10), " ")[:100]
    print(f"FAIL {err}")
"""

# ── Save current branch ──
result = subprocess.run(
    ["git", "rev-parse", "--abbrev-ref", "HEAD"],
    capture_output=True, text=True, cwd=REPO_ROOT,
)
original_ref = result.stdout.strip()
if original_ref == "HEAD":
    result = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    original_ref = result.stdout.strip()

# ── Prepare CSV ──
csv_path = Path(args.output)
csv_path.parent.mkdir(parents=True, exist_ok=True)
T = int(args.minutes * 60 / 0.02)
fieldnames = ["commit", "message", "n_iterations", "n_neurons", "minutes", "timepoints", "fit_time_s", "status"]
write_header = not csv_path.exists() or csv_path.stat().st_size == 0

print("=" * 60)
print("SIMPL Commit Benchmark")
print("=" * 60)
print(f"Commits    : {len(COMMITS)}")
print(f"Iterations : {args.n_iterations}")
print(f"Data       : {args.minutes} min (T={T}), {args.n_neurons} neurons, dt=0.02s")
print(f"Output     : {csv_path}")
print()

try:
    for i, commit in enumerate(COMMITS):
        sha, msg = commit["sha"], commit["message"]
        label = f"[{i+1}/{len(COMMITS)}] {sha} {msg:30s}"
        print(f"{label:.<55s}", end=" ", flush=True)

        # Checkout (force to avoid conflicts with local changes)
        checkout = subprocess.run(
            ["git", "checkout", sha, "--force", "--quiet"],
            cwd=REPO_ROOT, capture_output=True, text=True,
        )
        if checkout.returncode != 0:
            print(f"CHECKOUT FAIL ({checkout.stderr.strip()[:50]})")
            continue
        # Clear __pycache__ so Python doesn't use stale .pyc from a different commit
        subprocess.run(
            ["find", str(REPO_ROOT / "src"), "-type", "d", "-name", "__pycache__", "-exec", "rm", "-rf", "{}", "+"],
            capture_output=True, cwd=REPO_ROOT,
        )
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet", "--no-deps"],
            capture_output=True, cwd=REPO_ROOT,
        )

        # Build runner script with indented commit-specific code
        indented_code = textwrap.indent(commit["code"], "    ")
        script = (
            RUNNER_TEMPLATE
            .replace("__MINUTES__", str(args.minutes))
            .replace("__N_NEURONS__", str(args.n_neurons))
            .replace("__N_ITERATIONS__", str(args.n_iterations))
            .replace("__CODE__", indented_code)
        )

        # Set device: default to CPU unless commit specifies "gpu"
        env = os.environ.copy()
        use_device = commit.get("device", "cpu")
        if use_device == "cpu":
            env["JAX_PLATFORMS"] = "cpu"
            env["CUDA_VISIBLE_DEVICES"] = ""
        else:
            env.pop("JAX_PLATFORMS", None)

        try:
            # Write script to temp file outside the repo (git checkout can't touch it)
            import tempfile
            script_path = Path(tempfile.gettempdir()) / "_simpl_bench_runner.py"
            script_path.write_text(script)

            proc = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True, text=True, cwd=REPO_ROOT,
                env=env, timeout=300,
            )

            # Parse last line of stdout (earlier lines may be verbose output from old commits)
            all_lines = proc.stdout.strip().split("\n") if proc.stdout.strip() else []
            # Find the OK/FAIL line (may not be last if there's trailing output)
            stdout = ""
            for line in reversed(all_lines):
                if line.strip().startswith("OK") or line.strip().startswith("FAIL"):
                    stdout = line.strip()
                    break
            if stdout.startswith("OK"):
                parts = stdout.split()
                elapsed = float(parts[1])
                device = parts[2] if len(parts) > 2 else "?"
                status = "ok"
                print(f"{elapsed:.2f}s [{device}]")
            elif stdout.startswith("FAIL"):
                elapsed = float("nan")
                status = "fail"
                err = stdout[5:55]
                print(f"FAIL ({err})")
            else:
                elapsed = float("nan")
                status = "error"
                # Show useful error info (filter JAX/Metal noise)
                noise = ("W0", "I0", "WARNING", "Platform", "Metal device", "systemMemory", "maxCache", "XLA service", "StreamExecutor", "Using Simple", "XLA backend", "MetalClient")
                stderr_lines = [l for l in proc.stderr.strip().split("\n") if not l.strip().startswith(noise)] if proc.stderr else []
                stderr_msg = stderr_lines[-1][:70] if stderr_lines else f"rc={proc.returncode}"
                stdout_msg = proc.stdout.strip()[-70:] if proc.stdout.strip() else ""
                print(f"ERROR (stderr: {stderr_msg}) (stdout: {stdout_msg})")

        except subprocess.TimeoutExpired:
            elapsed = float("nan")
            status = "timeout"
            print("TIMEOUT")

        # Append to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
                write_header = False
            writer.writerow({
                "commit": sha,
                "message": msg,
                "n_iterations": args.n_iterations,
                "n_neurons": args.n_neurons,
                "minutes": args.minutes,
                "timepoints": T,
                "fit_time_s": elapsed,
                "status": status,
            })

except KeyboardInterrupt:
    print("\n\nInterrupted — partial results saved.")

finally:
    subprocess.run(
        ["git", "checkout", original_ref, "--force", "--quiet"],
        cwd=REPO_ROOT, capture_output=True,
    )
    # Reinstall current branch
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", ".", "--quiet", "--no-deps"],
        cwd=REPO_ROOT, capture_output=True,
    )
    print(f"\nDone. Results in {csv_path}")
    print(f"Returned to: {original_ref}")
