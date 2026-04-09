# Release Notes — v0.9.0

First release under the `simpl-neuro` PyPI name. Import remains `from simpl import SIMPL`.

## Highlights

- **Batched decoding** — likelihood computation is now batched to avoid large memory intermediates, with automatic batch sizing based on available memory. Enables fitting much longer recordings without OOM.
- **Apple Silicon GPU support** — SIMPL now supports JAX Metal for Apple GPUs via `pip install "simpl-neuro[metal]"`. Experimental, but functional for most operations.
- **Model save/load** — rehydrate fitted SIMPL models from saved `.nc` results via `model.load("results.nc")` for resumed training, prediction, or plotting.
- **`simpl demo` CLI** — new terminal command downloads the demo notebook into the current directory and optionally launches Jupyter.
- **Mutual information metric** — spatial information (SI) and bits-per-spike (BPS) equivalence clarified; mutual information now tracked across iterations.

## Breaking changes

- **`env_pad` default changed from `0.1` to `0.0`** — previously the environment was padded by default, which could cause unexpected receptive field edge effects. Set `env_pad=0.1` explicitly to restore old behavior.
- **`environment` parameter renamed to `env`** in `SIMPL.__init__()`.
- **`F_` shape corrected** — `model.F_` is now returned in the documented shape `(N_neurons, *env_dims)`.

## New features

- **Batched likelihood computation** — `kde.py` wraps Poisson log-likelihood and Gaussian fitting in a batched loop, controlled by a global memory budget parameter. Avoids retaining large `logPYXF_maps` by default.
- **`load_demo_data()` improvements** — downloads from GitHub releases with cache, `force_download` option, and informative messages about data source.
- **`_setup_device()` method** — explicit GPU/CPU device selection with suppressed GPU warnings when not needed.
- **Refactored log-likelihood calculation** — cleaner definitions of LL, BPS, MI, and SI with simpler `_get_ML_likelihood` helper functions.
- **Trial boundary support** — `trial_boundaries` parameter prevents the Kalman smoother from blending across recording discontinuities.
- **`simpl demo` CLI command** — `pip install "simpl-neuro[demos]" && simpl demo` to get started.

## Improvements

- Regularisation added to `sigma_predict` with NaN smoothing warning to improve numerical stability.
- Removed dangerous NaN-masking that could silently hide issues.
- Refactored imports for cleaner module structure.
- Removed dead/unused utility functions.
- Removed deprecation warnings for cleaned-up API.
- Inline printing refactored for cleaner training progress output.
- Plotting improvements and new `plot_all_metrics()` method.

## Docs & demos

- README overhauled with Key Features, API quick start, plotting examples, and GPU benchmarks.
- Documentation site (`docs/`) now includes all content from README via `include-markdown` — single source of truth.
- Demo notebook revamped with four datasets: synthetic grid cells, real place cells, head direction cells, and motor cortex hand reaching.
- Hand-reaching demo split into separate example (removed from main demo for clarity).

## Infrastructure

- Package renamed to `simpl-neuro` on PyPI (import unchanged: `from simpl import SIMPL`).
- Added `nbstripout` to dev dependencies.
- Scaling benchmark script and figures added.
- Test coverage expanded with new fixtures and GPU-skip markers.
