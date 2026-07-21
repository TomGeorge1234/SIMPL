"""Reproduce the Chowdhury et al. hand-reaching demo dataset.

The source is the training asset from DANDI 000127, version 0.220113.0359.
The separate held-out test asset is excluded, but the complete continuous
training timeline is retained, including active reaches, passive bump trials,
unsuccessful trials, and intertrial periods.

The historical pipeline depends on ``nlb_tools``. Its package metadata pins an
obsolete pandas release, so clone https://github.com/neurallatents/nlb_tools
and pass the checkout using ``--nlb-tools-root`` instead of installing it.

Example
-------
python scripts/data_preparation/preprocess_hand_reaching.py \
    --raw-nwb /path/to/sub-Han_desc-train_behavior+ecephys.nwb \
    --nlb-tools-root /path/to/nlb_tools \
    --target-bin-ms 10

The target bin size defaults to 50 ms. The default output name includes the
selected resolution, for example
``data/somatosensory_chowdhury2020_reprocessed_50ms.npz``.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal

if TYPE_CHECKING:
    import pandas as pd

DEFAULT_TARGET_BIN_MS = 50
EXPECTED_SOURCE_SHA256 = "639b1a03ab813f96fec76681110a1634942476badc0fd4c93d406c24701f5db2"
SOURCE_DANDISET = "000127/0.220113.0359"
SOURCE_ASSET = "sub-Han_desc-train_behavior+ecephys.nwb"


def repository_root() -> Path:
    """Find the repository root independently of the working directory."""
    script_path = Path(__file__).resolve()
    root = next((path for path in script_path.parents if (path / "pyproject.toml").is_file()), None)
    if root is None:
        raise FileNotFoundError(f"Could not find the SIMPL repository root above {script_path}")
    return root


def parse_args() -> argparse.Namespace:
    root = repository_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-nwb",
        type=Path,
        required=True,
        help=f"Path to the {SOURCE_ASSET} DANDI asset.",
    )
    parser.add_argument(
        "--nlb-tools-root",
        type=Path,
        required=True,
        help="Path to a checkout of https://github.com/neurallatents/nlb_tools.",
    )
    parser.add_argument(
        "--target-bin-ms",
        type=int,
        default=DEFAULT_TARGET_BIN_MS,
        help=f"Output bin size in milliseconds (default: {DEFAULT_TARGET_BIN_MS}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output NPZ path. Defaults to data/..._reprocessed_<bin>ms.npz.",
    )
    parser.set_defaults(repository_root=root)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_nwb_dataset(raw_nwb: Path, nlb_tools_root: Path):
    interface = nlb_tools_root / "nlb_tools/nwb_interface.py"
    if not interface.is_file():
        raise FileNotFoundError(f"nlb_tools checkout not found at {nlb_tools_root}")

    sys.path.insert(0, str(nlb_tools_root.resolve()))
    module = importlib.import_module("nlb_tools.nwb_interface")

    # The exact filename is verified before the historical '*train' selection.
    if raw_nwb.name != SOURCE_ASSET:
        raise ValueError(f"Expected source asset {SOURCE_ASSET!r}, got {raw_nwb.name!r}")
    return module.NWBDataset(raw_nwb.parent, "*train", split_heldout=False)


def rebin_spikes(frame: pd.DataFrame, factor: int) -> np.ndarray:
    """Sum spike counts while preserving NLB's original missing-bin behavior."""
    values = frame.to_numpy(copy=True)
    first_sample_nan = np.isnan(values[::factor])
    remainder = values.shape[0] % factor
    main = values[:-remainder] if remainder else values
    rebinned = np.nan_to_num(main).reshape(len(main) // factor, factor, -1).sum(axis=1)
    if remainder:
        rebinned = np.vstack([rebinned, np.nan_to_num(values[-remainder:]).sum(axis=0)])
    rebinned[first_sample_nan] = np.nan
    return rebinned


def decimate_continuous(frame: pd.DataFrame, factor: int) -> np.ndarray:
    """Interpolate, FIR-filter, and decimate a continuous NLB signal."""
    interpolated = frame.apply(lambda column: column.interpolate(limit_direction="both"))
    values = signal.decimate(interpolated, factor, axis=0, n=500, ftype="fir")
    missing_at_output = frame.iloc[::factor].isna().to_numpy()
    values[missing_at_output] = np.nan
    return values


def percentile_normalize(
    values: np.ndarray, lower: float = 0.5, upper: float = 99.5
) -> tuple[np.ndarray, np.ndarray]:
    """Map each coordinate's percentile interval to [0, 1] and clip."""
    result = np.empty_like(values, dtype=np.float32)
    bounds = []
    for column in range(values.shape[1]):
        low, high = np.percentile(values[:, column], [lower, upper])
        if high <= low:
            raise ValueError(f"Invalid percentile bounds for column {column}: {(low, high)}")
        result[:, column] = np.clip((values[:, column] - low) / (high - low), 0, 1)
        bounds.append((low, high))
    return result, np.asarray(bounds)


def timedeltas_to_bins(series: pd.Series, time: np.ndarray, missing: int = -1) -> np.ndarray:
    """Map trial timestamps to the nearest float32 time index."""
    seconds = series.dt.total_seconds().to_numpy(dtype=np.float32)
    return np.array(
        [missing if np.isnan(value) else np.argmin(np.abs(time - value)) for value in seconds],
        dtype=np.int32,
    )


def reconstruct(raw_nwb: Path, nlb_tools_root: Path, target_bin_ms: int) -> dict[str, np.ndarray]:
    raw_nwb = raw_nwb.expanduser().resolve()
    if not raw_nwb.is_file():
        raise FileNotFoundError(raw_nwb)

    source_sha256 = sha256_file(raw_nwb)
    if source_sha256 != EXPECTED_SOURCE_SHA256:
        raise ValueError(
            f"Source SHA-256 mismatch: expected {EXPECTED_SOURCE_SHA256}, got {source_sha256}"
        )
    print(f"Verified source SHA-256: {source_sha256}")

    dataset = load_nwb_dataset(raw_nwb, nlb_tools_root.expanduser().resolve())
    if target_bin_ms <= 0:
        raise ValueError(f"Target bin size must be positive, got {target_bin_ms}")
    factor = target_bin_ms // dataset.bin_width
    if target_bin_ms % dataset.bin_width:
        raise ValueError("Target bin must be an integer multiple of the source bin")
    sample_index = dataset.data.index[::factor]

    spikes = rebin_spikes(dataset.data["spikes"], factor)
    position = decimate_continuous(dataset.data["hand_pos"], factor)
    velocity = decimate_continuous(dataset.data["hand_vel"], factor)

    y = np.nan_to_num(spikes, nan=0.0).astype(np.float32)
    position_raw = np.nan_to_num(position, nan=0.0)
    velocity_raw = np.nan_to_num(velocity, nan=0.0)
    time = np.array([value.total_seconds() for value in sample_index], dtype=np.float32)

    # Retain the continuous training timeline, including passive trials and
    # intertrial periods. Only the separate held-out NWB asset is excluded.
    if y.ndim != 2:
        raise ValueError(f"Expected a 2D spike-count array, got Y shape {y.shape}")
    if position_raw.shape != (len(y), 2):
        raise ValueError(f"Unexpected position shape: {position_raw.shape}")
    if velocity_raw.shape != (len(y), 2):
        raise ValueError(f"Unexpected velocity shape: {velocity_raw.shape}")
    time_tolerance = max(2e-6, target_bin_ms / 1000 * 0.03)
    if not np.allclose(np.diff(time), target_bin_ms / 1000, atol=time_tolerance):
        raise ValueError(f"The resampled time axis is not uniformly {target_bin_ms} ms")

    xb, position_bounds = percentile_normalize(position_raw)
    xb_vel, velocity_bounds = percentile_normalize(velocity_raw)

    trials = dataset.trial_info
    trial_start = timedeltas_to_bins(trials["start_time"], time)
    trial_stop = timedeltas_to_bins(trials["end_time"], time)
    trial_move_onset = timedeltas_to_bins(trials["move_onset_time"], time)
    trial_cond_dir = trials["cond_dir"].fillna(-1).to_numpy(dtype=np.float32)
    trial_result = trials["result"].to_numpy(dtype=str)
    trial_is_bump = trials["ctr_hold_bump"].to_numpy(dtype=bool)

    active_trials = int((~trial_is_bump).sum())
    bump_trials = int(trial_is_bump.sum())

    print(f"Y shape: {y.shape}")
    print(f"Duration: {time[-1]:.1f} s; dt: {np.median(np.diff(time)):.3f} s")
    print(f"Trials: {active_trials} active, {bump_trials} passive bump")

    data = {
        "Y": y,
        "Xb": xb,
        "Xb_vel": xb_vel,
        "time": time,
        "dim": np.array(["x", "y"]),
        "neuron": np.arange(y.shape[1]),
        "trial_start": trial_start,
        "trial_stop": trial_stop,
        "trial_move_onset": trial_move_onset,
        "trial_cond_dir": trial_cond_dir,
        "trial_result": trial_result,
        "trial_is_bump": trial_is_bump,
        "source_dandiset": np.array(SOURCE_DANDISET),
        "source_asset": np.array(raw_nwb.name),
        "source_sha256": np.array(source_sha256),
        "target_bin_ms": np.array(target_bin_ms),
        "normalization_percentiles": np.array([0.5, 99.5]),
        "position_percentile_bounds": position_bounds,
        "velocity_percentile_bounds": velocity_bounds,
    }
    return data


def main() -> None:
    args = parse_args()
    data = reconstruct(args.raw_nwb, args.nlb_tools_root, args.target_bin_ms)

    output = args.output
    if output is None:
        output = args.repository_root / (
            f"data/somatosensory_chowdhury2020_reprocessed_{args.target_bin_ms}ms.npz"
        )
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **data)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
