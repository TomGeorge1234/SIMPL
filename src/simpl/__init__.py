"""SIMPL — Scalable and hassle-free optimisation of neural representations from behaviour.

This package implements the SIMPL algorithm from the ICLR 2025 paper. The main
entry point is the ``SIMPL`` class, which follows the
scikit-learn ``fit`` / ``predict`` API pattern.

Convenience utilities re-exported at the package level:

* ``accumulate_spikes`` — aggregate spikes over a sliding window
* ``coarsen_dt`` — re-bin data at a coarser time resolution
* ``load_demo_data`` — load the bundled grid-cell demo dataset
* ``load_results`` — reload a saved ``xr.Dataset`` from disk
* ``rehydrate_model`` — reconstruct a ``SIMPL`` instance from saved results
"""

from .simpl import SIMPL
from .utils import accumulate_spikes, coarsen_dt, load_demo_data, load_results, rehydrate_model

__all__ = ["SIMPL", "accumulate_spikes", "coarsen_dt", "load_demo_data", "load_results", "rehydrate_model"]
