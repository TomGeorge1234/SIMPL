"""SIMPL — Scalable and hassle-free optimisation of neural representations from behaviour.

This package implements the SIMPL algorithm from the ICLR 2025 paper. The main
entry point is the ``SIMPL`` class, which follows the
scikit-learn ``fit`` / ``predict`` API pattern.

Convenience utilities re-exported at the package level:

* ``accumulate_spikes`` — aggregate spikes over a sliding window
* ``coarsen_dt`` — re-bin data at a coarser time resolution
* ``load_demo_data`` — load the bundled grid-cell demo dataset
* ``load_results`` — reload a saved ``xr.Dataset`` from disk

To load a fitted model from saved results, use ``SIMPL.load(path)``.
"""

import os as _os

# Suppress verbose XLA compiler logs (xtile_compiler fusion messages, etc.)
_os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
_os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")

from .simpl import SIMPL
from .utils import accumulate_spikes, coarsen_dt, load_demo_data, load_results, train_test_split

#: Maximum number of float32 elements in the largest intermediate array per batch.
#: Used by KDE, likelihood, and Kalman filter to auto-size batches.
#: Default 64_000_000 ≈ 256 MB peak memory (64M × 4 bytes).
MAX_BATCH_ELEMENTS = 64_000_000

__all__ = ["SIMPL", "accumulate_spikes", "coarsen_dt", "load_demo_data", "load_results", "train_test_split"]
