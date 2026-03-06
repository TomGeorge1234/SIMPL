"""Plotting functions for SIMPL results.

All functions accept an ``xr.Dataset`` (the ``results_`` attribute of a fitted
:class:`~simpl.simpl.SIMPL` model) and return matplotlib ``Axes`` so users can
customise further.  ``matplotlib`` is imported lazily inside each function so
the rest of the package stays lightweight.
"""

from __future__ import annotations

import importlib.resources
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.colors import LinearSegmentedColormap

# ── Style sheet ──────────────────────────────────────────────────────────────
_STYLE_PATH = str(importlib.resources.files("simpl").joinpath("simpl.mplstyle"))
plt.style.use(_STYLE_PATH)


# ── Register "flare" colormap (seaborn's flare, bundled to avoid the dependency) ──
def _register_cmap(name, stops):
    if name not in matplotlib.colormaps:
        cmap = LinearSegmentedColormap.from_list(name, stops, N=256)
        matplotlib.colormaps.register(cmap, name=name)
        matplotlib.colormaps.register(cmap.reversed(), name=f"{name}_r")


_register_cmap(
    "flare",
    [
        (0.0000, (0.9291, 0.6888, 0.5041)),
        (0.0902, (0.9212, 0.6018, 0.4505)),
        (0.1804, (0.9104, 0.5134, 0.3993)),
        (0.2706, (0.8926, 0.4238, 0.3653)),
        (0.3608, (0.8595, 0.3391, 0.3630)),
        (0.4510, (0.8019, 0.2755, 0.3893)),
        (0.5451, (0.7184, 0.2410, 0.4186)),
        (0.6353, (0.6333, 0.2182, 0.4356)),
        (0.7255, (0.5495, 0.1956, 0.4423)),
        (0.8157, (0.4645, 0.1772, 0.4349)),
        (0.9059, (0.3793, 0.1605, 0.4127)),
        (1.0000, (0.2941, 0.1372, 0.3844)),
    ],
)

_register_cmap(
    "crest",
    [
        (0.0000, (0.6468, 0.8029, 0.5659)),
        (0.0902, (0.5466, 0.7550, 0.5693)),
        (0.1804, (0.4471, 0.7078, 0.5664)),
        (0.2706, (0.3619, 0.6574, 0.5654)),
        (0.3608, (0.2932, 0.6045, 0.5626)),
        (0.4510, (0.2341, 0.5509, 0.5570)),
        (0.5451, (0.1731, 0.4954, 0.5504)),
        (0.6353, (0.1238, 0.4412, 0.5437)),
        (0.7255, (0.1123, 0.3839, 0.5323)),
        (0.8157, (0.1327, 0.3234, 0.5122)),
        (0.9059, (0.1586, 0.2605, 0.4801)),
        (1.0000, (0.1736, 0.1908, 0.4455)),
    ],
)

# ── Default colormaps ────────────────────────────────────────────────────────
EPOCH_CMAP = "crest"
FIELD_CMAP = "inferno"

# ── Default figure width (inches) ───────────────────────────────────────────
FIG_WIDTH = 10


# ── Low-level helpers ────────────────────────────────────────────────────────


def outset_axes(ax, offset_mm: float = 2) -> None:
    """Outset bottom/left spines by *offset_mm* mm."""
    ax.spines["bottom"].set_position(("outward", offset_mm * 72 / 25.4))
    ax.spines["left"].set_position(("outward", offset_mm * 72 / 25.4))


def _epoch_color(epoch: int, last_epoch: int, cmap: str = EPOCH_CMAP):
    """Return a colour for *epoch* from *cmap* scaled to [0, last_epoch]."""
    cm = matplotlib.colormaps[cmap]
    if last_epoch == 0:
        return cm(1.0)
    return cm(epoch / last_epoch)


def _non_negative_epochs(results: xr.Dataset) -> np.ndarray:
    """Return epoch values >= 0."""
    epochs = results.epoch.values
    return epochs[epochs >= 0]


def _last_non_negative_epoch(results: xr.Dataset) -> int:
    return int(_non_negative_epochs(results)[-1])


def _baseline_epochs(results: xr.Dataset) -> np.ndarray:
    """Return epoch values < 0 (baselines)."""
    epochs = results.epoch.values
    return epochs[epochs < 0]


# ── Public plot functions ────────────────────────────────────────────────────


def plot_fitting_summary(
    results: xr.Dataset,
    show_neurons: bool = True,
    cmap: str | None = None,
    **plot_kwargs,
) -> np.ndarray:
    """Two-panel summary: log-likelihood (left) and spatial information (right).

    Parameters
    ----------
    results : xr.Dataset
        The ``results_`` Dataset from a fitted SIMPL model.
    show_neurons : bool
        Show individual neuron dots for per-neuron metrics (spatial information).
    cmap : str
        Colormap for epoch colouring.
    **plot_kwargs
        Forwarded to the main scatter calls.

    Returns
    -------
    axes : np.ndarray of Axes, shape (2,)
    """
    cmap = cmap or EPOCH_CMAP
    epochs = _non_negative_epochs(results)
    last_epoch = int(epochs[-1])

    fig, axes = plt.subplots(1, 2, figsize=(0.7 * FIG_WIDTH, 0.7 * FIG_WIDTH * 0.35), layout="constrained")
    ax_ll, ax_si = axes

    ll_train, ll_test, si_means = [], [], []
    for e in epochs:
        c = _epoch_color(e, last_epoch, cmap)
        ll_train.append(float(results.logPYXF.sel(epoch=e)))
        ll_test.append(float(results.logPYXF_test.sel(epoch=e)))
        ax_ll.scatter(e, ll_train[-1], color=c, zorder=5, **plot_kwargs)
        ax_ll.scatter(e, ll_test[-1], color="w", edgecolors=c, linewidth=2, zorder=5, **plot_kwargs)

        si = results.spatial_information.sel(epoch=e).values
        if show_neurons:
            jitter = np.random.default_rng(int(e)).uniform(-0.15, 0.15, size=len(si))
            ax_si.scatter(e + jitter, si, color=c, alpha=0.15, s=5, linewidths=0)
        si_means.append(float(np.mean(si)))
        ax_si.scatter(e, si_means[-1], color=c, s=60, zorder=5, edgecolors="k", linewidths=0.5)

    # connecting lines
    for i in range(len(epochs) - 1):
        c = _epoch_color(epochs[i + 1], last_epoch, cmap)
        ax_ll.plot(epochs[i : i + 2], ll_train[i : i + 2], color=c, lw=0.8, zorder=3)
        ax_ll.plot(epochs[i : i + 2], ll_test[i : i + 2], color=c, lw=0.8, zorder=3)
        ax_si.plot(epochs[i : i + 2], si_means[i : i + 2], color=c, lw=0.8, zorder=3)

    # baseline: only epoch -1 ("best model")
    if -1 in results.epoch.values:
        if "logPYXF" in results:
            ax_ll.axhline(float(results.logPYXF.sel(epoch=-1)), color="k", ls="--", lw=0.8)
        if "spatial_information" in results:
            ax_si.axhline(float(results.spatial_information.sel(epoch=-1).mean()), color="k", ls="--", lw=0.8)

    ax_ll.set(xlabel="Epoch", ylabel="Log likelihood")
    ax_si.set(xlabel="Epoch", ylabel="Spatial information (bits/s)")
    for ax in axes:
        outset_axes(ax)
        ax.spines["bottom"].set_bounds(0, int(epochs[-1]))
    return axes


def _plot_trajectory_panel(
    t: np.ndarray,
    traces: list[tuple[np.ndarray, dict]],
    Xt: np.ndarray | None,
    dim_names: list[str],
    title: str | None = None,
    **plot_kwargs,
) -> np.ndarray:
    """Shared implementation for trajectory plots (one subplot per spatial dim).

    Parameters
    ----------
    t : array, shape (T,)
        Time axis.
    traces : list of (X, style_dict)
        Each entry is ``(positions_array_TxD, dict(color=..., alpha=..., label=...))``.
    Xt : array or None, shape (T, D)
        Ground truth positions, plotted as ``"k--"`` if provided.
    dim_names : list of str
        Spatial dimension names (e.g. ``["x", "y"]``).
    title : str or None
        Optional ``fig.suptitle``.
    **plot_kwargs
        Forwarded to ``ax.plot``.
    """
    D = len(dim_names)
    fig, axes = plt.subplots(
        D, 1, figsize=(FIG_WIDTH, FIG_WIDTH * 0.2 * D), sharex=True, squeeze=False, layout="constrained"
    )
    axes = axes[:, 0]

    for i, d in enumerate(dim_names):
        ax = axes[i]
        for X, style in traces:
            ax.plot(t, X[:, i], **style, **plot_kwargs)
        if Xt is not None:
            ax.plot(t, Xt[:, i], "k--", lw=1, label="Ground truth")
        ax.set_ylabel(f"{d}-position (m)")
        outset_axes(ax)
        if i == 0:
            ax.legend(loc="upper right")

    axes[-1].set_xlabel("Time (s)")
    for ax in axes:
        ax.set_xlim(np.floor(t[0]), np.ceil(t[-1]))
    if title:
        fig.suptitle(title)
    return axes


def plot_latent_trajectory(
    results: xr.Dataset,
    time_range: tuple[float, float] | None = None,
    epoch: int | tuple[int, ...] | None = None,
    include_behavior: bool = True,
    include_ground_truth: bool = True,
    cmap: str | None = None,
    **plot_kwargs,
) -> np.ndarray:
    """Plot decoded latent trajectory (one subplot per spatial dimension).

    Parameters
    ----------
    results : xr.Dataset
    time_range : tuple, optional
        ``(t_start, t_end)`` in seconds.  Default: first 120 s.
    epoch : int or tuple of ints, optional
        Which epoch(s) to show.  A single int plots one epoch; a tuple
        plots multiple.  Default: all non-negative epochs.
    include_behavior : bool
        Show the behavioral initialisation (epoch 0) alongside.
    include_ground_truth : bool
        Show ``Xt`` as ``"k--"`` if present.
    cmap : str
    **plot_kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    axes : np.ndarray of Axes, shape (D,)
    """
    cmap = cmap or EPOCH_CMAP
    if epoch is None:
        epochs_to_plot = tuple(_non_negative_epochs(results))
    elif isinstance(epoch, int):
        epochs_to_plot = (epoch,)
    else:
        epochs_to_plot = tuple(epoch)

    if time_range is None:
        t0 = float(results.time.values[0])
        time_range = (t0, t0 + 120)

    dim_names = list(results.dim.values)
    tslice = slice(*time_range)
    t = results.time.sel(time=tslice).values
    last_epoch = _last_non_negative_epoch(results)

    traces = []
    if include_behavior and 0 not in epochs_to_plot:
        traces.append(
            (
                results.X.sel(epoch=0, time=tslice).values,
                dict(color=_epoch_color(0, last_epoch, cmap), alpha=0.8, label="Behavior (epoch 0)"),
            )
        )
    for ep in epochs_to_plot:
        label = f"Epoch {ep} (behavior)" if ep == 0 else f"Epoch {ep}"
        traces.append(
            (
                results.X.sel(epoch=ep, time=tslice).values,
                dict(color=_epoch_color(ep, last_epoch, cmap), alpha=0.8, label=label),
            )
        )

    Xt = results.Xt.sel(time=tslice).values if (include_ground_truth and "Xt" in results) else None
    return _plot_trajectory_panel(t, traces, Xt, dim_names, **plot_kwargs)


def plot_prediction(
    prediction_results: xr.Dataset,
    Xb: np.ndarray | None = None,
    Xt: np.ndarray | None = None,
    time_range: tuple[float, float] | None = None,
    cmap: str | None = None,
    **plot_kwargs,
) -> np.ndarray:
    """Plot predicted trajectory from ``predict()``.

    Parameters
    ----------
    prediction_results : xr.Dataset
        The ``prediction_results_`` Dataset from :meth:`SIMPL.predict`.
    Xb : np.ndarray, optional
        Behavioral positions for the prediction window, shape ``(T, D)``.
    Xt : np.ndarray, optional
        Ground truth positions for the prediction window, shape ``(T, D)``.
    time_range : tuple, optional
        ``(t_start, t_end)`` in seconds.  Default: full prediction range.
    cmap : str
    **plot_kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    axes : np.ndarray of Axes, shape (D,)
    """
    cmap = cmap or EPOCH_CMAP
    dim_names = list(prediction_results.dim.values)

    T = len(prediction_results.time)
    if Xb is not None:
        assert Xb.shape[0] == T, f"Xb length {Xb.shape[0]} != prediction_results time length {T}"
    if Xt is not None:
        assert Xt.shape[0] == T, f"Xt length {Xt.shape[0]} != prediction_results time length {T}"

    if time_range is not None:
        tslice = slice(*time_range)
        mask = (prediction_results.time.values >= time_range[0]) & (prediction_results.time.values <= time_range[1])
    else:
        tslice = slice(None)
        mask = slice(None)

    t = prediction_results.time.sel(time=tslice).values
    traces = []
    if Xb is not None:
        traces.append((Xb[mask], dict(color=_epoch_color(0, 1, cmap), alpha=0.8, label="Behavior")))
    traces.append(
        (
            prediction_results.mu_s.sel(time=tslice).values,
            dict(color=_epoch_color(1, 1, cmap), alpha=0.8, label="Predicted"),
        )
    )

    Xt_sliced = Xt[mask] if Xt is not None else None
    return _plot_trajectory_panel(t, traces, Xt_sliced, dim_names, title="Prediction on held-out data", **plot_kwargs)


def plot_receptive_fields(
    results: xr.Dataset,
    extent: tuple | None = None,
    epoch: int | tuple[int, ...] | None = None,
    neurons: list[int] | np.ndarray | None = None,
    include_behavior: bool = True,
    include_baselines: bool = False,
    ncols: int = 4,
    cmap: str | None = None,
    **plot_kwargs,
) -> np.ndarray:
    """Plot receptive fields for selected neurons.

    Parameters
    ----------
    results : xr.Dataset
    extent : tuple, optional
        Matplotlib extent ``(xmin, xmax, ymin, ymax, ...)``.  Used for 2-D imshow.
    epoch : int or tuple of int, optional
        Which epoch(s) to show.  ``None`` shows the first (0) and last
        non-negative epochs.  An ``int`` shows a single epoch; a ``tuple``
        shows multiple.
    neurons : array-like, optional
        Subset of neuron indices.  Default: all neurons.
    include_behavior : bool
        Show epoch-0 (behavior) fields alongside.  Only adds a column when
        epoch 0 is not already in the requested epochs.
    include_baselines : bool
        Show ground-truth fields (``Ft``) if present.
    ncols : int
        Maximum number of neuron-columns in the grid.
    cmap : str
    **plot_kwargs
        Forwarded to ``imshow`` (2-D) or ``plot`` (1-D).

    Returns
    -------
    axes : np.ndarray of Axes
    """
    cmap = cmap or FIELD_CMAP
    dim_names = list(results.dim.values)
    D = len(dim_names)
    if D > 2:
        raise ValueError(f"plot_receptive_fields only supports 1-D and 2-D environments, got {D}-D.")

    # Resolve epoch(s) to a tuple
    if epoch is None:
        last = _last_non_negative_epoch(results)
        epochs = (0, last) if last != 0 else (0,)
    elif isinstance(epoch, int):
        epochs = (epoch,)
    else:
        epochs = tuple(epoch)

    if neurons is None:
        neurons = results.neuron.values
    neurons = np.asarray(neurons)

    if len(neurons) > 50:
        warnings.warn(f"Plotting {len(neurons)} neurons — this may be slow.", stacklevel=2)

    # Resolve baseline source: prefer Ft, fall back to F at epoch -1
    has_baselines = False
    baseline_label = None
    if include_baselines:
        if "Ft" in results:
            has_baselines = True
            baseline_label = "GT"
        elif -1 in results.epoch.values:
            has_baselines = True
            baseline_label = "Best"

    # Build column labels per neuron
    col_labels = []
    # Behavior column if requested and not already in epochs
    show_beh_col = include_behavior and 0 not in epochs
    if show_beh_col:
        col_labels.append("Beh")
    for ep in epochs:
        col_labels.append(f"Ep {ep}" if ep != 0 else "Ep 0 (behavior)")
    if has_baselines:
        col_labels.append(baseline_label)
    n_cols_per_neuron = len(col_labels)

    # Layout: neurons along rows, with a spacer column between neuron groups
    n_neurons = len(neurons)
    n_neuron_cols = min(n_neurons, ncols)
    n_neuron_rows = int(np.ceil(n_neurons / n_neuron_cols))

    # Each neuron group gets n_cols_per_neuron + 1 spacer, except the last group
    total_cols = n_neuron_cols * n_cols_per_neuron + (n_neuron_cols - 1)
    total_rows = n_neuron_rows

    # Width ratios: data columns are 1, spacer columns are 0.3
    width_ratios = []
    for g in range(n_neuron_cols):
        width_ratios.extend([1] * n_cols_per_neuron)
        if g < n_neuron_cols - 1:
            width_ratios.append(0.3)

    fig, axes = plt.subplots(
        total_rows,
        total_cols,
        figsize=(FIG_WIDTH / 6 * n_neuron_cols * n_cols_per_neuron, FIG_WIDTH / 6 * total_rows),
        squeeze=False,
        gridspec_kw={"width_ratios": width_ratios},
    )

    # Pre-compute the starting column index for each neuron group (skipping spacers)
    group_col_starts = []
    for g in range(n_neuron_cols):
        group_col_starts.append(g * (n_cols_per_neuron + 1))

    # Track which axes are used for plotting
    used_axes = set()

    # build extent for imshow
    if D == 2 and extent is not None:
        ext = [extent[0], extent[1], extent[2], extent[3]]
    elif D == 2:
        ext = [
            float(results[dim_names[0]].values[0]),
            float(results[dim_names[0]].values[-1]),
            float(results[dim_names[1]].values[0]),
            float(results[dim_names[1]].values[-1]),
        ]
    else:
        ext = None

    imkw = dict(cmap=cmap, origin="lower", aspect="equal", **plot_kwargs)
    if ext is not None:
        imkw["extent"] = ext

    for idx, n in enumerate(neurons):
        row = idx // n_neuron_cols
        group = idx % n_neuron_cols
        col_base = group_col_starts[group]

        col_offset = 0

        # behavior column (only when 0 not in epochs)
        if show_beh_col:
            ax = axes[row, col_base + col_offset]
            used_axes.add((row, col_base + col_offset))
            F_beh = results.F.sel(epoch=0, neuron=n)
            if D == 2:
                ax.imshow(F_beh.values.T, **imkw)
            else:
                ax.plot(results[dim_names[0]].values, F_beh.values, **plot_kwargs)
            if row == 0:
                ax.set_title("Beh", fontsize=8)
            col_offset += 1

        # epoch columns
        for ep in epochs:
            ax = axes[row, col_base + col_offset]
            used_axes.add((row, col_base + col_offset))
            F_ep = results.F.sel(epoch=ep, neuron=n)
            if D == 2:
                ax.imshow(F_ep.values.T, **imkw)
            else:
                ax.plot(results[dim_names[0]].values, F_ep.values, **plot_kwargs)
            if row == 0:
                label = f"Ep {ep}" if ep != 0 else "Ep 0 (behavior)"
                ax.set_title(label, fontsize=8)
            col_offset += 1

        # baseline column (Ft if available, else F at epoch -1)
        if has_baselines:
            ax = axes[row, col_base + col_offset]
            used_axes.add((row, col_base + col_offset))
            if "Ft" in results:
                F_base = results.Ft.sel(neuron=n)
            else:
                F_base = results.F.sel(epoch=-1, neuron=n)
            if D == 2:
                ax.imshow(F_base.values.T, **imkw)
            else:
                ax.plot(results[dim_names[0]].values, F_base.values, **plot_kwargs)
            if row == 0:
                ax.set_title(baseline_label, fontsize=8)

        # label
        axes[row, col_base].set_ylabel(f"N{n}", fontsize=7, rotation=0, labelpad=15)

    # Clean up: remove ticks from used axes, turn off unused/spacer axes entirely
    for r in range(total_rows):
        for c in range(total_cols):
            ax = axes[r, c]
            if (r, c) in used_axes:
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.axis("off")

    fig.tight_layout()
    return axes


def plot_all_metrics(
    results: xr.Dataset,
    show_neurons: bool = True,
    cmap: str | None = None,
    ncols: int = 3,
    **plot_kwargs,
) -> np.ndarray:
    """Auto-discover and plot all per-epoch metrics.

    Parameters
    ----------
    results : xr.Dataset
    show_neurons : bool
        Show individual neuron dots for per-neuron metrics.
    cmap : str
    ncols : int
        Number of columns in the grid.
    **plot_kwargs
        Forwarded to line/scatter calls.

    Returns
    -------
    axes : np.ndarray of Axes
    """
    cmap = cmap or EPOCH_CMAP
    epochs = _non_negative_epochs(results)
    last_epoch = int(epochs[-1])
    baselines = _baseline_epochs(results)

    # discover metric variables: anything with epoch dim and only neuron/place_field remaining
    metric_names = []
    for var_name in results.data_vars:
        da = results[var_name]
        if "epoch" not in da.dims:
            continue
        other_dims = set(da.dims) - {"epoch"}
        if other_dims <= {"neuron", "place_field"}:
            metric_names.append(var_name)

    n_metrics = len(metric_names)
    if n_metrics == 0:
        warnings.warn("No metrics found to plot.", stacklevel=2)
        return np.array([])

    nrows = int(np.ceil(n_metrics / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.5 * nrows), squeeze=False, layout="constrained")

    for i, var_name in enumerate(metric_names):
        ax = axes.flat[i]
        da = results[var_name]
        other_dims = [d for d in da.dims if d != "epoch"]
        attrs = da.attrs
        ylabel = attrs.get("axis_title", attrs.get("axis title", var_name))

        is_scalar = len(other_dims) == 0
        has_place_field = "place_field" in other_dims

        if is_scalar:
            # line plot
            vals = [float(da.sel(epoch=e)) for e in epochs]
            for j in range(len(epochs)):
                c = _epoch_color(epochs[j], last_epoch, cmap)
                ax.scatter(epochs[j], vals[j], color=c, zorder=5, **plot_kwargs)
            for j in range(len(epochs) - 1):
                c = _epoch_color(epochs[j + 1], last_epoch, cmap)
                ax.plot(epochs[j : j + 2], vals[j : j + 2], color=c, lw=0.8, zorder=3)
            # baseline: only epoch -1 ("best model")
            if -1 in baselines:
                ax.axhline(float(da.sel(epoch=-1)), color="k", ls="--", lw=0.8)
        else:
            # per-neuron (possibly mean over place_field first)
            means = []
            for e in epochs:
                c = _epoch_color(e, last_epoch, cmap)
                vals_e = da.sel(epoch=e)
                if has_place_field:
                    vals_e = vals_e.mean(dim="place_field", skipna=True)
                v = vals_e.values
                if show_neurons:
                    jitter = np.random.default_rng(int(e)).uniform(-0.15, 0.15, size=len(v))
                    ax.scatter(e + jitter, v, color=c, alpha=0.15, s=5, linewidths=0)
                if np.all(np.isnan(v)):
                    means.append(np.nan)
                else:
                    means.append(float(np.nanmean(v)))
                ax.scatter(e, means[-1], color=c, s=60, zorder=5, edgecolors="k", linewidths=0.5)
            for j in range(len(epochs) - 1):
                c = _epoch_color(epochs[j + 1], last_epoch, cmap)
                ax.plot(epochs[j : j + 2], means[j : j + 2], color=c, lw=0.8, zorder=3)

        ax.set(xlabel="Epoch", ylabel=ylabel)
        outset_axes(ax)
        ax.spines["bottom"].set_bounds(0, int(epochs[-1]))

    # hide unused axes
    for j in range(n_metrics, len(axes.flat)):
        axes.flat[j].set_visible(False)

    return axes
