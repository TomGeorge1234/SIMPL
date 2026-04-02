"""Benchmark SIMPL fit speed across decode_batch_size values.

This benchmark runs full ``SIMPL.fit(..., n_iterations=1)`` for several
``decode_batch_size`` values, reports steady-state runtime after a warm-up fit,
and checks that the fitted outputs are identical to the unbatched baseline.

Example
-------
python tests/benchmarks/compare_decode_batch_size.py --timepoints 8000 --neurons 20
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax

from simpl import SIMPL, load_demo_data


OUTPUT_KEYS = ("X", "mu_l", "mode_l", "sigma_l", "mu_f", "sigma_f", "mu_s", "sigma_s")


def parse_batch_sizes(raw: str) -> list[int | None]:
    batch_sizes = []
    for item in raw.split(","):
        item = item.strip().lower()
        if item in {"none", "full", "unbatched"}:
            batch_sizes.append(None)
        else:
            batch_sizes.append(int(item))
    if None not in batch_sizes:
        batch_sizes.insert(0, None)
    return batch_sizes


def capture_outputs(model: SIMPL) -> dict[str, np.ndarray]:
    outputs = {key: np.asarray(model.E_[key]) for key in OUTPUT_KEYS}
    outputs["F"] = np.asarray(model.F_)
    return outputs


def assert_identical(reference: dict[str, np.ndarray], current: dict[str, np.ndarray], batch_size: int | None) -> None:
    label = "None" if batch_size is None else str(batch_size)
    for key, ref in reference.items():
        cur = current[key]
        if not np.allclose(ref, cur, rtol=0.0, atol=0.0, equal_nan=True):
            diff = np.nanmax(np.abs(ref - cur))
            raise AssertionError(f"{key} changed for decode_batch_size={label}. max_abs_diff={diff}")


def run_fit(
    Y: np.ndarray,
    Xb: np.ndarray,
    time_arr: np.ndarray,
    decode_batch_size: int | None,
    use_gpu: bool | str,
) -> tuple[SIMPL, float]:
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        model = SIMPL(decode_batch_size=decode_batch_size, random_seed=0, use_gpu=use_gpu)
        start = time.perf_counter()
        model.fit(Y=Y, Xb=Xb, time=time_arr, n_iterations=1, verbose=False)
        jax.block_until_ready(model.X_)
        jax.block_until_ready(model.F_)
        elapsed = time.perf_counter() - start
    return model, elapsed


def benchmark_batch_sizes(
    Y: np.ndarray,
    Xb: np.ndarray,
    time_arr: np.ndarray,
    batch_sizes: list[int | None],
    repeats: int,
    use_gpu: bool | str,
) -> list[dict[str, float | int | None]]:
    results = []
    reference_outputs = None
    baseline_mean = None

    for batch_size in batch_sizes:
        warmup_model, warmup_time = run_fit(Y, Xb, time_arr, batch_size, use_gpu)
        assert warmup_model.iteration_ == 1
        current_outputs = capture_outputs(warmup_model)
        if reference_outputs is None:
            reference_outputs = current_outputs
        else:
            assert_identical(reference_outputs, current_outputs, batch_size)

        timings = []
        for _ in range(repeats):
            _, elapsed = run_fit(Y, Xb, time_arr, batch_size, use_gpu)
            timings.append(elapsed)

        mean_time = float(np.mean(timings))
        std_time = float(np.std(timings))
        if baseline_mean is None:
            baseline_mean = mean_time

        results.append(
            {
                "decode_batch_size": batch_size,
                "warmup_s": warmup_time,
                "mean_s": mean_time,
                "std_s": std_time,
                "relative_to_full": mean_time / baseline_mean,
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark SIMPL decode_batch_size values")
    parser.add_argument("--timepoints", type=int, default=8000, help="Number of time bins to use from demo data")
    parser.add_argument("--neurons", type=int, default=20, help="Number of neurons to use from demo data")
    parser.add_argument("--repeats", type=int, default=3, help="Number of timed fits per batch size after warm-up")
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="none,4096,2048,1024,512,256",
        help="Comma-separated decode_batch_size values. Use 'none' for unbatched decode.",
    )
    parser.add_argument(
        "--use-gpu",
        choices=("if_available", "true", "false"),
        default="if_available",
        help="Pass through to SIMPL(use_gpu=...).",
    )
    args = parser.parse_args()

    data = load_demo_data("gridcells_synthetic.npz")
    Y = data["Y"][: args.timepoints, : args.neurons]
    Xb = data["Xb"][: args.timepoints]
    time_arr = data["time"][: args.timepoints]
    batch_sizes = parse_batch_sizes(args.batch_sizes)
    use_gpu: bool | str
    if args.use_gpu == "true":
        use_gpu = True
    elif args.use_gpu == "false":
        use_gpu = False
    else:
        use_gpu = "if_available"

    print("=" * 72)
    print("SIMPL decode_batch_size benchmark (fit with n_iterations=1)")
    print("=" * 72)
    print(f"timepoints   : {Y.shape[0]}")
    print(f"neurons      : {Y.shape[1]}")
    print(f"repeats      : {args.repeats}")
    print(f"batch_sizes  : {['None' if x is None else x for x in batch_sizes]}")
    print()

    results = benchmark_batch_sizes(
        Y=Y,
        Xb=Xb,
        time_arr=time_arr,
        batch_sizes=batch_sizes,
        repeats=args.repeats,
        use_gpu=use_gpu,
    )

    print(f"{'decode_batch_size':>18} {'warmup_s':>10} {'mean_s':>10} {'std_s':>10} {'rel_full':>10}")
    for row in results:
        label = "None" if row["decode_batch_size"] is None else str(row["decode_batch_size"])
        print(
            f"{label:>18} "
            f"{row['warmup_s']:10.4f} "
            f"{row['mean_s']:10.4f} "
            f"{row['std_s']:10.4f} "
            f"{row['relative_to_full']:10.3f}"
        )

    print()
    print("All compared outputs matched the unbatched baseline exactly.")


if __name__ == "__main__":
    main()
