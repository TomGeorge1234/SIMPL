"""SIMPL — Scalable and hassle-free optimisation of neural representations from behaviour.

This package implements the SIMPL algorithm from the ICLR 2025 paper. The main
entry point is the :class:`~simpl.simpl.SIMPL` class, which follows the
scikit-learn ``fit`` / ``predict`` API pattern.

Convenience utilities re-exported at the package level:

* :func:`~simpl.utils.accumulate_spikes` — aggregate spikes over a sliding window
* :func:`~simpl.utils.coarsen_dt` — re-bin data at a coarser time resolution
* :func:`~simpl.utils.load_demo_data` — load the bundled grid-cell demo dataset
* :func:`~simpl.utils.load_results` — reload a saved ``xr.Dataset`` from disk
"""

from .simpl import SIMPL
from .utils import accumulate_spikes, coarsen_dt, load_demo_data, load_results

__all__ = ["SIMPL", "accumulate_spikes", "coarsen_dt", "load_demo_data", "load_results"]
