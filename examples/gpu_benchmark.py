"""GPU vs CPU benchmark for SIMPL.

Toggle the device via --device flag or SIMPL_DEVICE env var:

    # Force CPU
    python examples/gpu_benchmark.py --device cpu

    # Force GPU (default if available)
    python examples/gpu_benchmark.py --device gpu

Vary data size to see how the speedup scales:

    python examples/gpu_benchmark.py --device gpu --T 20000 --N 300

Run a per-component breakdown to see where time is spent:

    python examples/gpu_benchmark.py --breakdown
"""

import argparse
import os
import time

# ── Device selection (must happen before any JAX import) ──────────────────────
parser = argparse.ArgumentParser(description="SIMPL GPU benchmark")
parser.add_argument("--device", type=str, default=None, help="Force 'cpu' or 'gpu'. Env var SIMPL_DEVICE also works.")
parser.add_argument("--T", type=int, default=3000, help="Number of timesteps")
parser.add_argument("--N", type=int, default=100, help="Number of neurons")
parser.add_argument("--n_iterations", type=int, default=3, help="Number of EM iterations")
parser.add_argument("--bin_size", type=float, default=0.03, help="Spatial bin size")
parser.add_argument("--breakdown", action="store_true", help="Run per-component timing breakdown")
args = parser.parse_args()

device = (args.device or os.environ.get("SIMPL_DEVICE", "gpu")).lower()
if device == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
# ──────────────────────────────────────────────────────────────────────────────

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from simpl import SIMPL  # noqa: E402
from simpl.kalman import KalmanFilter  # noqa: E402
from simpl.kde import gaussian_kernel, kde, poisson_log_likelihood  # noqa: E402
from simpl.utils import fit_gaussian  # noqa: E402


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
    rates = np.exp(-0.5 * np.sum(diff**2, axis=2) / widths[None, :] ** 2)
    rates *= 0.5

    spikes = rng.poisson(rates).astype(float)
    time_arr = np.arange(T) * dt

    return trajectory, spikes, time_arr


def timed(fn, label, sync_result=True):
    """Run fn(), block until done (important for GPU), return (result, elapsed)."""
    t0 = time.perf_counter()
    result = fn()
    if sync_result and result is not None:
        jax.block_until_ready(result)
    elapsed = time.perf_counter() - t0
    print(f"  {label:40s} {elapsed:.3f}s")
    return result, elapsed


def run_breakdown(Xb, Y, time_arr):
    """Time each computational component individually."""
    T, N = Y.shape
    D = Xb.shape[1]

    print("\n" + "-" * 60)
    print("PER-COMPONENT BREAKDOWN")
    print("-" * 60)

    # Setup: build environment bins
    model = SIMPL(kernel_bandwidth=0.04, speed_prior=0.3, bin_size=args.bin_size, env_pad=0.05)
    model.fit(Y, Xb, time_arr, n_iterations=0, verbose=False)
    bins = model.xF_
    N_bins = bins.shape[0]
    print(f"  Grid: {N_bins} bins, {T} timesteps, {N} neurons, {D}D")
    print()

    Xb_jax = jnp.array(Xb)
    Y_jax = jnp.array(Y)
    mask = jnp.ones_like(Y_jax, dtype=bool)

    # ── 1. KDE (M-step core) ──
    # Warm up
    F = kde(bins, Xb_jax, Y_jax, gaussian_kernel, 0.04, mask=mask)
    jax.block_until_ready(F)

    F, t_kde = timed(
        lambda: kde(bins, Xb_jax, Y_jax, gaussian_kernel, 0.04, mask=mask), "KDE (receptive field fitting)"
    )

    # ── 2. Poisson log-likelihood ──
    _ = poisson_log_likelihood(Y_jax, F, mask=mask)
    jax.block_until_ready(_)

    ll_maps, t_ll = timed(lambda: poisson_log_likelihood(Y_jax, F, mask=mask), "Poisson log-likelihood maps")

    # ── 3. Gaussian fitting (batched, JIT-compiled) ──
    probs = jnp.exp(ll_maps)
    _ = fit_gaussian(bins, probs)
    jax.block_until_ready(_[0])

    (mu_l, mode_l, sigma_l), t_gauss = timed(lambda: fit_gaussian(bins, probs), "Gaussian fitting (batched)")

    # ── 4. Kalman filter ──
    force_cpu = jax.default_backend() != "cpu"
    kf = KalmanFilter(dim_Z=D, dim_Y=D, dim_U=D, batch_size=T, force_cpu=force_cpu)
    kf.F = jnp.eye(D)
    kf.Q = jnp.eye(D) * 0.001
    kf.H = jnp.eye(D)
    kf.mu0 = mode_l[0]
    kf.sigma0 = sigma_l[0]

    # Warm up
    mu_f, sigma_f = kf.filter(Y=mode_l, U=Xb_jax, R=sigma_l)
    jax.block_until_ready(mu_f)

    (mu_f, sigma_f), t_filter = timed(lambda: kf.filter(Y=mode_l, U=Xb_jax, R=sigma_l), "Kalman filter")

    # ── 5. Kalman smoother ──
    _ = kf.smooth(mu_f, sigma_f)
    jax.block_until_ready(_[0])

    _, t_smooth = timed(lambda: kf.smooth(mu_f, sigma_f), "Kalman smoother")

    print()
    total = t_kde + t_ll + t_gauss + t_filter + t_smooth
    print(f"  {'TOTAL (components)':40s} {total:.3f}s")
    print()
    print("  Proportions:")
    for name, t in [
        ("KDE", t_kde),
        ("Poisson LL", t_ll),
        ("Gaussian fit", t_gauss),
        ("Kalman filter", t_filter),
        ("Kalman smoother", t_smooth),
    ]:
        print(f"    {name:30s} {t / total * 100:5.1f}%")


def main():
    print("=" * 60)
    print("SIMPL GPU Benchmark")
    print("=" * 60)

    backend = jax.default_backend()
    devices = jax.devices()
    print(f"JAX backend : {backend}")
    print(f"JAX devices : {devices}")
    print(f"Data size   : T={args.T}, N_neurons={args.N}, bin_size={args.bin_size}")
    print(f"Iterations  : {args.n_iterations}")
    print()

    # Generate data
    print("Generating synthetic data...", end=" ", flush=True)
    Xb, Y, time_arr = generate_synthetic_data(T=args.T, N=args.N)
    print("done.")

    # Build model
    model = SIMPL(
        kernel_bandwidth=0.04,
        speed_prior=0.3,
        bin_size=args.bin_size,
        env_pad=0.05,
    )

    # Warm-up (includes JIT compilation)
    print("\nWarm-up run (includes JIT compilation)...")
    t0 = time.perf_counter()
    model.fit(Y, Xb, time_arr, n_iterations=1, verbose=False)
    jax.block_until_ready(model.X_)
    t_warmup = time.perf_counter() - t0
    print(f"  Warm-up: {t_warmup:.2f}s (includes JIT compilation)")

    # Timed run
    print(f"\nTimed run ({args.n_iterations} iterations)...")
    t0 = time.perf_counter()
    model.fit(Y, Xb, time_arr, n_iterations=args.n_iterations, verbose=False)
    jax.block_until_ready(model.X_)
    t_fit = time.perf_counter() - t0

    per_iteration = t_fit / args.n_iterations
    print(f"  Total   : {t_fit:.2f}s")
    print(f"  Per iteration: {per_iteration:.2f}s")

    # Predict
    print("\nPredict (decode new spikes)...")
    t0 = time.perf_counter()
    X_pred = model.predict(Y)
    jax.block_until_ready(X_pred)
    t_pred = time.perf_counter() - t0
    print(f"  Predict : {t_pred:.2f}s")

    print("\n" + "=" * 60)
    print(f"SUMMARY  [{backend.upper()}]")
    print(f"  Warm-up (1 iteration + JIT) : {t_warmup:.2f}s")
    print(f"  Fit ({args.n_iterations} iterations)         : {t_fit:.2f}s  ({per_iteration:.2f}s/iteration)")
    print(f"  Predict                 : {t_pred:.2f}s")
    print("=" * 60)

    # Optional per-component breakdown
    if args.breakdown:
        run_breakdown(Xb, Y, time_arr)


if __name__ == "__main__":
    main()
