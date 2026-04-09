"""Plotting functions for SIMPL results.

All functions accept an ``xr.Dataset`` (the ``results_`` attribute of a fitted
``SIMPL`` model) and return matplotlib ``Axes`` so users can
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
ITERATION_CMAP = "crest"
FIELD_CMAP = "inferno"

# ── Default figure width (inches) ───────────────────────────────────────────
FIG_WIDTH = 10


# ── Low-level helpers ────────────────────────────────────────────────────────


def outset_axes(ax, offset_mm: float = 2) -> None:
    """Outset bottom/left spines by *offset_mm* mm."""
    ax.spines["bottom"].set_position(("outward", offset_mm * 72 / 25.4))
    ax.spines["left"].set_position(("outward", offset_mm * 72 / 25.4))


def _resolve_iterations(
    iterations: int | tuple[int, ...] | None,
    results: xr.Dataset,
    default: tuple[int, ...] | None = (0, -1),
) -> tuple[int, ...]:
    """Normalise an *iterations* argument to a tuple of concrete iteration values.

    Supports negative indexing: ``-1`` maps to the last non-negative iteration,
    ``-2`` to the second-last, etc.  Pass ``default=None`` to default to all
    non-negative iterations.
    """
    nn = _non_negative_iterations(results)
    if iterations is None:
        raw = tuple(int(e) for e in nn) if default is None else default
    elif isinstance(iterations, int):
        raw = (iterations,)
    else:
        raw = tuple(iterations)
    resolved = []
    for e in raw:
        if e < 0:
            idx = e  # -1 → last, -2 → second-last, etc.
            resolved.append(int(nn[idx]))
        else:
            resolved.append(int(e))
    # deduplicate while preserving order
    seen = set()
    unique = []
    for e in resolved:
        if e not in seen:
            seen.add(e)
            unique.append(e)
    return tuple(unique)


def _iteration_color(iteration: int, last_iteration: int):
    """Return a colour for *iteration* from ITERATION_CMAP scaled to [0, last_iteration]."""
    cm = matplotlib.colormaps[ITERATION_CMAP]
    if last_iteration == 0:
        return cm(1.0)
    return cm(iteration / last_iteration)


def _non_negative_iterations(results: xr.Dataset) -> np.ndarray:
    """Return iteration values >= 0."""
    iterations = results.iteration.values
    return iterations[iterations >= 0]


def _last_non_negative_iteration(results: xr.Dataset) -> int:
    return int(_non_negative_iterations(results)[-1])


def _baseline_iterations(results: xr.Dataset) -> np.ndarray:
    """Return iteration values < 0 (baselines)."""
    iterations = results.iteration.values
    return iterations[iterations < 0]


# ── Public plot functions ────────────────────────────────────────────────────


def plot_fitting_summary(
    results: xr.Dataset,
    show_neurons: bool = True,
    **plot_kwargs,
) -> np.ndarray:
    """Two-panel summary: bits per spike (left) and mutual information (right).

    Parameters
    ----------
    results : xr.Dataset
        The ``results_`` Dataset from a fitted SIMPL model.
    show_neurons : bool
        Show individual neuron dots for per-neuron metrics (mutual information).
    **plot_kwargs
        Forwarded to the main scatter calls.

    Returns
    -------
    axes : np.ndarray of Axes, shape (2,)
    """
    iterations = _non_negative_iterations(results)
    last_iteration = int(iterations[-1])

    fig, axes = plt.subplots(1, 2, figsize=(0.7 * FIG_WIDTH, 0.7 * FIG_WIDTH * 0.35), layout="constrained")
    ax_bps, ax_mi = axes

    bps_train, bps_val, mi_means = [], [], []
    for e in iterations:
        c = _iteration_color(e, last_iteration)
        bps_train.append(float(results.bits_per_spike.sel(iteration=e)))
        bps_val.append(float(results.bits_per_spike_val.sel(iteration=e)))
        ax_bps.scatter(e, bps_train[-1], color=c, zorder=5, **plot_kwargs)
        ax_bps.scatter(e, bps_val[-1], color=c, marker="o", facecolors="none", linewidth=1.5, zorder=5, **plot_kwargs)

        mi = results.mutual_information.sel(iteration=e).values
        if show_neurons:
            jitter = np.random.default_rng(int(e)).uniform(-0.15, 0.15, size=len(mi))
            ax_mi.scatter(e + jitter, mi, color=c, alpha=0.15, s=5, linewidths=0)
        mi_means.append(float(np.mean(mi)))
        ax_mi.scatter(e, mi_means[-1], color=c, s=60, zorder=5, linewidths=0)

    # connecting lines
    for i in range(len(iterations) - 1):
        c = _iteration_color(iterations[i + 1], last_iteration)
        ax_bps.plot(iterations[i : i + 2], bps_train[i : i + 2], color=c, lw=0.8, zorder=3)
        ax_bps.plot(iterations[i : i + 2], bps_val[i : i + 2], color=c, lw=0.8, ls="--", zorder=3)
        ax_mi.plot(iterations[i : i + 2], mi_means[i : i + 2], color=c, lw=0.8, zorder=3)

    # baseline: only iteration -1 ("best model")
    if -1 in results.iteration.values:
        if "bits_per_spike" in results:
            y_gt = float(results.bits_per_spike.sel(iteration=-1))
            ax_bps.axhline(y_gt, color="k", ls="--", lw=0.8)
            ax_bps.text(
                0.0,
                y_gt,
                " ground truth",
                va="bottom",
                ha="left",
                fontsize="x-small",
                color="k",
                transform=ax_bps.get_yaxis_transform(),
            )
        if "mutual_information" in results:
            y_gt_mi = float(results.mutual_information.sel(iteration=-1).mean())
            ax_mi.axhline(y_gt_mi, color="k", ls="--", lw=0.8)
            ax_mi.text(
                0.0,
                y_gt_mi,
                " ground truth",
                va="bottom",
                ha="left",
                fontsize="x-small",
                color="k",
                transform=ax_mi.get_yaxis_transform(),
            )

    # # ML baseline
    # if "mode_l" in results and 1 in results.iteration.values:
    #     from simpl.utils import get_ML_loglikelihoods

    #     ml = get_ML_loglikelihoods(results)
    #     y_ml = ml["bits_per_spike"]
    #     ax_bps.axhline(y_ml, color="k", ls=":", lw=0.8)
    #     ax_bps.text(
    #         0.0,
    #         y_ml,
    #         " naive ML",
    #         va="bottom",
    #         ha="left",
    #         fontsize="x-small",
    #         color="k",
    #         transform=ax_bps.get_yaxis_transform(),
    #     )

    # legend on first panel
    ax_bps.plot([], [], color="gray", lw=0.8, label="train")
    ax_bps.plot([], [], color="gray", lw=0.8, ls="--", label="val")
    ax_bps.legend(fontsize="small", frameon=False)

    ax_bps.set(xlabel="Iteration", ylabel="Bits per spike")
    ax_mi.set(xlabel="Iteration", ylabel="Mutual information (bits/s)")
    for ax in axes:
        outset_axes(ax)
        ax.spines["bottom"].set_bounds(0, int(iterations[-1]))
    return axes


def _break_angular_wraps(t: np.ndarray, x: np.ndarray, threshold: float = np.pi) -> tuple[np.ndarray, np.ndarray]:
    """Handle angular wraps so the line goes through the boundary rather than jumping across.

    At each wrap (jump > ``threshold``), the line is extended to the boundary
    (pi or -pi), a NaN breaks the line, and the continuation starts from the
    opposite boundary. This gives the visual effect of the trajectory leaving
    at +/-pi and re-entering at -/+pi.
    """
    jumps = np.where(np.abs(np.diff(x)) > threshold)[0]
    if len(jumps) == 0:
        return t, x

    t_list = []
    x_list = []
    prev = 0
    for j in jumps:
        # Include points up to and including index j
        t_list.append(t[prev : j + 1])
        x_list.append(x[prev : j + 1])

        # Interpolate the time at which the wrap crosses the boundary
        x_before, x_after = x[j], x[j + 1]
        t_before, t_after = t[j], t[j + 1]
        if x_before > 0:
            # Wrapping from near +pi down to near -pi
            boundary_exit = np.pi
            boundary_enter = -np.pi
        else:
            # Wrapping from near -pi up to near +pi
            boundary_exit = -np.pi
            boundary_enter = np.pi

        # Linear interpolation to find crossing time
        # x_before -> boundary_exit across the wrap
        dist_exit = boundary_exit - x_before
        dist_enter = x_after - boundary_enter
        total_dist = abs(dist_exit) + abs(dist_enter)
        if total_dist > 0:
            frac = abs(dist_exit) / total_dist
        else:
            frac = 0.5
        t_cross = t_before + frac * (t_after - t_before)

        # Extend line to boundary, NaN break, then continue from opposite boundary
        t_list.append(np.array([t_cross, np.nan, t_cross]))
        x_list.append(np.array([boundary_exit, np.nan, boundary_enter]))

        prev = j + 1

    # Append remaining data
    t_list.append(t[prev:])
    x_list.append(x[prev:])

    return np.concatenate(t_list), np.concatenate(x_list)


def _plot_trajectory_panel(
    t: np.ndarray,
    traces: list[tuple[np.ndarray, dict]],
    Xt: np.ndarray | None,
    dim_names: list[str],
    title: str | None = None,
    trial_boundary_times: np.ndarray | None = None,
    is_1D_angular: bool = False,
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
    trial_boundary_times : array or None
        Time values at trial boundaries (excluding the first trial).
        A shaded band is drawn at each boundary to indicate the gap.
    is_1D_angular : bool
        If True, insert NaN at angular wraps (jumps > 3pi/2) so that
        matplotlib does not draw a line across the plot.
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
        if trial_boundary_times is not None:
            for tb in trial_boundary_times:
                ax.axvspan(tb[0], tb[1], color="0.85", zorder=3)
        for X, style in traces:
            if is_1D_angular:
                t_plot, x_plot = _break_angular_wraps(t, X[:, i])
            else:
                t_plot, x_plot = t, X[:, i]
            ax.plot(t_plot, x_plot, **style, **plot_kwargs)
        if Xt is not None:
            if is_1D_angular:
                t_plot, xt_plot = _break_angular_wraps(t, Xt[:, i])
            else:
                t_plot, xt_plot = t, Xt[:, i]
            ax.plot(t_plot, xt_plot, "k--", lw=1, label="Ground truth")
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
    iterations: int | tuple[int, ...] | None = None,
    include_ground_truth: bool = True,
    **plot_kwargs,
) -> np.ndarray:
    """Plot decoded latent trajectory (one subplot per spatial dimension).

    Parameters
    ----------
    results : xr.Dataset
    time_range : tuple, optional
        ``(t_start, t_end)`` in seconds.  Default: first 120 s.
    iterations : int or tuple of ints, optional
        Which iteration(s) to show.  Negative values index from the end of the
        non-negative iterations (``-1`` = last iteration).  Default: all iterations.
    include_ground_truth : bool
        Show ``Xt`` as ``"k--"`` if present.
    **plot_kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    axes : np.ndarray of Axes, shape (D,)
    """
    iterations_to_plot = _resolve_iterations(iterations, results, default=None)

    if time_range is None:
        t0 = float(results.time.values[0])
        time_range = (t0, t0 + 120)

    dim_names = list(results.dim.values)
    tslice = slice(*time_range)
    t = results.time.sel(time=tslice).values
    last_iteration = _last_non_negative_iteration(results)

    traces = []
    for ep in iterations_to_plot:
        label = f"Iteration {ep} (behavior)" if ep == 0 else f"Iteration {ep}"
        traces.append(
            (
                results.X.sel(iteration=ep, time=tslice).values,
                dict(color=_iteration_color(ep, last_iteration), alpha=0.8, label=label),
            )
        )

    Xt = results.Xt.sel(time=tslice).values if (include_ground_truth and "Xt" in results) else None

    # Extract trial boundary times (shaded bands between end of one trial and start of next)
    trial_boundary_times = None
    tb_indices = results.attrs.get("trial_boundaries", None)
    if tb_indices is not None and len(tb_indices) > 1:
        all_t = results.time.values
        pairs = []
        for b in tb_indices[1:]:
            t_end = all_t[b - 1]  # last timestep of previous trial
            t_start = all_t[b]  # first timestep of next trial
            if t_start >= time_range[0] and t_end <= time_range[1]:
                pairs.append((t_end, t_start))
        if pairs:
            trial_boundary_times = np.array(pairs)

    is_angular = bool(results.attrs.get("is_1D_angular", 0))
    return _plot_trajectory_panel(
        t, traces, Xt, dim_names, trial_boundary_times=trial_boundary_times, is_1D_angular=is_angular, **plot_kwargs
    )


def plot_prediction(
    prediction_results: xr.Dataset,
    Xb: np.ndarray | None = None,
    Xt: np.ndarray | None = None,
    time_range: tuple[float, float] | None = None,
    **plot_kwargs,
) -> np.ndarray:
    """Plot predicted trajectory from ``predict()``.

    Parameters
    ----------
    prediction_results : xr.Dataset
        The ``prediction_results_`` Dataset from ``SIMPL.predict``.
    Xb : np.ndarray, optional
        Behavioral positions for the prediction window, shape ``(T, D)``.
    Xt : np.ndarray, optional
        Ground truth positions for the prediction window, shape ``(T, D)``.
    time_range : tuple, optional
        ``(t_start, t_end)`` in seconds.  Default: full prediction range.
    **plot_kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    axes : np.ndarray of Axes, shape (D,)
    """
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
        traces.append((Xb[mask], dict(color=_iteration_color(0, 1), alpha=0.8, label="Behavior")))
    traces.append(
        (
            prediction_results.mu_s.sel(time=tslice).values,
            dict(color=_iteration_color(1, 1), alpha=0.8, label="Predicted"),
        )
    )

    Xt_sliced = Xt[mask] if Xt is not None else None

    # Extract trial boundary times if available
    trial_boundary_times = None
    tb_indices = prediction_results.attrs.get("trial_boundaries", None)
    if tb_indices is not None and len(tb_indices) > 1:
        all_t = prediction_results.time.values
        t0 = t[0] if len(t) > 0 else -np.inf
        t1 = t[-1] if len(t) > 0 else np.inf
        pairs = []
        for b in tb_indices[1:]:
            t_end = all_t[b - 1]
            t_start = all_t[b]
            if t_start >= t0 and t_end <= t1:
                pairs.append((t_end, t_start))
        if pairs:
            trial_boundary_times = np.array(pairs)

    is_angular = bool(prediction_results.attrs.get("is_1D_angular", 0))
    return _plot_trajectory_panel(
        t,
        traces,
        Xt_sliced,
        dim_names,
        title="Prediction on held-out data",
        trial_boundary_times=trial_boundary_times,
        is_1D_angular=is_angular,
        **plot_kwargs,
    )


def plot_receptive_fields(
    results: xr.Dataset,
    extent: tuple | None = None,
    iterations: int | tuple[int, ...] | None = None,
    neurons: list[int] | np.ndarray | None = None,
    include_baselines: bool = False,
    sort_by_spatial_information: bool = False,
    ncols: int = 4,
    **plot_kwargs,
) -> np.ndarray:
    """Plot receptive fields for selected neurons.

    Parameters
    ----------
    results : xr.Dataset
    extent : tuple, optional
        Matplotlib extent ``(xmin, xmax, ymin, ymax, ...)``.  Used for 2-D imshow.
    iterations : int or tuple of int, optional
        Which iteration(s) to show.  Negative values index from the end of the
        non-negative iterations (``-1`` = last iteration).  Default: ``(0, -1)``
        (behavior and final iteration).
    neurons : array-like, optional
        Subset of neuron indices.  Default: all neurons.
    include_baselines : bool
        Show ground-truth fields (``Ft``) if present.
    sort_by_spatial_information : bool
        If ``True``, reorder neurons so that the most spatially informative
        appear first (uses the last training iteration).
    ncols : int
        Maximum number of neuron-columns in the grid.
    **plot_kwargs
        Forwarded to ``imshow`` (2-D) or ``plot`` (1-D).

    Returns
    -------
    axes : np.ndarray of Axes
    """
    dim_names = list(results.dim.values)
    D = len(dim_names)
    if D > 2:
        raise ValueError(f"plot_receptive_fields only supports 1-D and 2-D environments, got {D}-D.")

    iterations = _resolve_iterations(iterations, results)

    if neurons is None:
        neurons = results.neuron.values
    neurons = np.asarray(neurons)

    if sort_by_spatial_information:
        neurons = _sort_neurons_by_si(results, neurons)

    if len(neurons) > 50:
        warnings.warn(f"Plotting {len(neurons)} neurons — this may be slow.", stacklevel=2)

    # Resolve baseline source: prefer Ft, fall back to F at iteration -1
    has_baselines = False
    baseline_label = None
    if include_baselines:
        if "Ft" in results:
            has_baselines = True
            baseline_label = "GT"
        elif -1 in results.iteration.values:
            has_baselines = True
            baseline_label = "Best"

    # Build column labels per neuron
    col_labels = []
    for ep in iterations:
        col_labels.append(f"It {ep}" if ep != 0 else "It 0 (behavior)")
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

    is_polar = bool(results.attrs.get("is_1D_angular", 0)) and D == 1

    if is_polar:
        return _plot_receptive_fields_polar(
            results,
            iterations,
            neurons,
            has_baselines,
            baseline_label,
            dim_names,
            n_cols_per_neuron,
            n_neuron_cols,
            n_neuron_rows,
            total_cols,
            total_rows,
            col_labels,
            **plot_kwargs,
        )

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

    imkw = dict(cmap=FIELD_CMAP, origin="lower", aspect="equal", **plot_kwargs)
    if ext is not None:
        imkw["extent"] = ext

    for idx, n in enumerate(neurons):
        row = idx // n_neuron_cols
        group = idx % n_neuron_cols
        col_base = group_col_starts[group]

        col_offset = 0

        # iteration columns
        for ep in iterations:
            ax = axes[row, col_base + col_offset]
            used_axes.add((row, col_base + col_offset))
            F_ep = results.F.sel(iteration=ep, neuron=n)
            if D == 2:
                ax.imshow(F_ep.values.T, **imkw)
            else:
                ax.plot(results[dim_names[0]].values, F_ep.values, **plot_kwargs)
            if row == 0:
                label = f"It {ep}" if ep != 0 else "It 0 (behavior)"
                ax.set_title(label, fontsize=8)
            col_offset += 1

        # baseline column (Ft if available, else F at iteration -1)
        if has_baselines:
            ax = axes[row, col_base + col_offset]
            used_axes.add((row, col_base + col_offset))
            if "Ft" in results:
                F_base = results.Ft.sel(neuron=n)
            else:
                F_base = results.F.sel(iteration=-1, neuron=n)
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


def _plot_receptive_fields_polar(
    results,
    iterations,
    neurons,
    has_baselines,
    baseline_label,
    dim_names,
    n_cols_per_neuron,
    n_neuron_cols,
    n_neuron_rows,
    total_cols,
    total_rows,
    col_labels,
    **plot_kwargs,
):
    """Polar plot variant of ``plot_receptive_fields`` for 1-D angular data."""
    from matplotlib.gridspec import GridSpec

    last_iteration = int(_non_negative_iterations(results)[-1])

    # Width ratios: data columns are 1, spacer columns are 0.3
    width_ratios = []
    for g in range(n_neuron_cols):
        width_ratios.extend([1] * n_cols_per_neuron)
        if g < n_neuron_cols - 1:
            width_ratios.append(0.3)

    fig = plt.figure(figsize=(FIG_WIDTH / 4 * n_neuron_cols * n_cols_per_neuron, FIG_WIDTH / 4 * total_rows))
    gs = GridSpec(total_rows, total_cols, figure=fig, width_ratios=width_ratios)

    axes = np.empty((total_rows, total_cols), dtype=object)

    group_col_starts = []
    for g in range(n_neuron_cols):
        group_col_starts.append(g * (n_cols_per_neuron + 1))

    used_axes = set()
    theta = results[dim_names[0]].values

    # Close the polar curve by appending the first point
    theta_closed = np.concatenate([theta, [theta[0]]])

    def _style_polar_ax(ax, rmax):
        """Apply consistent styling to a polar axis."""
        ax.set_ylim(0, rmax)
        ax.set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2])
        ax.set_xticklabels(["0", "π/2", "π", "3π/2"], fontsize=6)
        # Place ~4 evenly spaced radial gridlines including rmax
        n_ticks = 4
        rticks = np.linspace(0, rmax, n_ticks + 1)[1:]  # exclude 0
        ax.set_yticks(rticks)
        ax.set_yticklabels([""] * (len(rticks) - 1) + [f"{rmax:.2f}"], fontsize=5)
        ax.tick_params(pad=1)

    for idx, n in enumerate(neurons):
        row = idx // n_neuron_cols
        group = idx % n_neuron_cols
        col_base = group_col_starts[group]

        # Pre-compute max firing rate across all iterations (and baseline) for this neuron
        rmax = 0.0
        for ep in iterations:
            F_ep = results.F.sel(iteration=ep, neuron=n).values
            rmax = max(rmax, float(np.nanmax(F_ep)))
        if has_baselines:
            if "Ft" in results:
                rmax = max(rmax, float(np.nanmax(results.Ft.sel(neuron=n).values)))
            elif -1 in results.iteration.values:
                rmax = max(rmax, float(np.nanmax(results.F.sel(iteration=-1, neuron=n).values)))
        if rmax == 0:
            rmax = 1.0

        col_offset = 0

        for ep in iterations:
            pos = (row, col_base + col_offset)
            ax = fig.add_subplot(gs[pos[0], pos[1]], projection="polar")
            axes[pos[0], pos[1]] = ax
            used_axes.add(pos)
            color = _iteration_color(ep, last_iteration)
            F_ep = results.F.sel(iteration=ep, neuron=n).values
            r_closed = np.concatenate([F_ep, [F_ep[0]]])
            ax.plot(theta_closed, r_closed, color=color, **plot_kwargs)
            ax.fill(theta_closed, r_closed, alpha=0.3, color=color)
            _style_polar_ax(ax, rmax)
            if row == 0:
                label = f"It {ep}" if ep != 0 else "It 0 (behavior)"
                ax.set_title(label, fontsize=8, pad=12)
            col_offset += 1

        if has_baselines:
            pos = (row, col_base + col_offset)
            ax = fig.add_subplot(gs[pos[0], pos[1]], projection="polar")
            axes[pos[0], pos[1]] = ax
            used_axes.add(pos)
            if "Ft" in results:
                F_base = results.Ft.sel(neuron=n).values
            else:
                F_base = results.F.sel(iteration=-1, neuron=n).values
            r_closed = np.concatenate([F_base, [F_base[0]]])
            ax.plot(theta_closed, r_closed, color="grey", **plot_kwargs)
            ax.fill(theta_closed, r_closed, alpha=0.3, color="grey")
            _style_polar_ax(ax, rmax)
            if row == 0:
                ax.set_title(baseline_label, fontsize=8, pad=12)

    # Turn off unused/spacer cells
    for r in range(total_rows):
        for c in range(total_cols):
            if (r, c) not in used_axes:
                ax = fig.add_subplot(gs[r, c])
                axes[r, c] = ax
                ax.axis("off")

    fig.tight_layout()
    return axes


def _sort_neurons_by_si(results: xr.Dataset, neurons: np.ndarray) -> np.ndarray:
    """Return *neurons* sorted by spatial information (highest first).

    Uses the last non-negative iteration.  Falls back to the original order
    if ``spatial_information`` is not present.
    """
    if "spatial_information" not in results:
        warnings.warn("spatial_information not found in results — neurons left unsorted.", stacklevel=3)
        return neurons
    last_it = _last_non_negative_iteration(results)
    si = results.spatial_information.sel(iteration=last_it, neuron=neurons).values
    return neurons[np.argsort(si)[::-1]]


def plot_spikes(
    results: xr.Dataset,
    time_range: tuple[float, float] | None = None,
    neurons: list[int] | np.ndarray | None = None,
    sort_by_spatial_information: bool = False,
    cmap: str = "Greys",
    **plot_kwargs,
) -> matplotlib.axes.Axes:
    """Visualise spike counts as a heatmap (time × neurons).

    Parameters
    ----------
    results : xr.Dataset
        The ``results_`` Dataset from a fitted SIMPL model.
    time_range : tuple, optional
        ``(t_start, t_end)`` in seconds.  Default: first 120 s.
    neurons : array-like, optional
        Subset of neuron indices to display.  Default: all neurons.
    sort_by_spatial_information : bool
        If ``True``, reorder neurons so that the most spatially informative
        appear at the top of the heatmap (uses the last training iteration).
    cmap : str
        Colormap for ``imshow``.  Default: ``"Greys"``.
    **plot_kwargs
        Forwarded to ``ax.imshow``.

    Returns
    -------
    ax : matplotlib Axes
    """
    if "Y" not in results:
        raise ValueError("results Dataset does not contain 'Y' (spike counts).")

    Y = results.Y
    t = results.time.values

    # Default to first 120 s (same as plot_latent_trajectory)
    if time_range is None:
        t0 = float(t[0])
        time_range = (t0, t0 + 120)

    tslice = slice(*time_range)
    Y = Y.sel(time=tslice)
    t = Y.time.values

    # Neuron subset
    if neurons is not None:
        neurons = np.asarray(neurons)
    else:
        neurons = results.neuron.values

    if sort_by_spatial_information:
        neurons = _sort_neurons_by_si(results, neurons)

    Y = Y.sel(neuron=neurons)
    data = np.array(Y.values)  # (T, N)

    fig, ax = plt.subplots(
        1,
        1,
        figsize=(FIG_WIDTH, FIG_WIDTH * 0.35),
        layout="constrained",
    )

    extent = [float(t[0]), float(t[-1]), -0.5, data.shape[1] - 0.5]
    imkw = dict(
        cmap=cmap,
        aspect="auto",
        interpolation="none",
        origin="lower",
        extent=extent,
    )
    imkw.update(plot_kwargs)
    im = ax.imshow(data.T, **imkw)
    fig.colorbar(im, ax=ax, label="Spike count", shrink=0.8, pad=0.02)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Neuron")
    outset_axes(ax)

    return ax


def plot_all_metrics(
    results: xr.Dataset,
    show_neurons: bool = True,
    ncols: int = 3,
    **plot_kwargs,
) -> np.ndarray:
    """Auto-discover and plot all per-iteration metrics.

    Parameters
    ----------
    results : xr.Dataset
    show_neurons : bool
        Show individual neuron dots for per-neuron metrics.
    ncols : int
        Number of columns in the grid.
    **plot_kwargs
        Forwarded to line/scatter calls.

    Returns
    -------
    axes : np.ndarray of Axes
    """
    iterations = _non_negative_iterations(results)
    last_iteration = int(iterations[-1])
    baselines = _baseline_iterations(results)

    # discover metric variables: anything with iteration dim and only neuron/place_field remaining
    # skip _val variants — they are plotted alongside their train counterpart
    metric_names = []
    for var_name in results.data_vars:
        da = results[var_name]
        if "iteration" not in da.dims:
            continue
        other_dims = set(da.dims) - {"iteration"}
        if other_dims <= {"neuron", "place_field"}:
            if var_name.endswith("_val") and var_name[:-4] in results.data_vars:
                continue  # will be plotted with the train variant
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
        other_dims = [d for d in da.dims if d != "iteration"]
        attrs = da.attrs
        ylabel = attrs.get("axis_title", attrs.get("axis title", var_name))

        is_scalar = len(other_dims) == 0
        has_place_field = "place_field" in other_dims

        # Only plot iterations that have data for this variable (skip all-NaN iterations)
        var_iterations = []
        for e in iterations:
            vals_e = da.sel(iteration=e)
            if not np.all(np.isnan(vals_e.values)):
                var_iterations.append(e)
        var_iterations = np.array(var_iterations) if var_iterations else iterations

        if is_scalar:
            # line plot
            val_name = f"{var_name}_val"
            has_val = val_name in results.data_vars
            vals = [float(da.sel(iteration=e)) for e in var_iterations]
            vals_v = [float(results[val_name].sel(iteration=e)) for e in var_iterations] if has_val else None
            for j in range(len(var_iterations)):
                c = _iteration_color(var_iterations[j], last_iteration)
                ax.scatter(var_iterations[j], vals[j], color=c, zorder=5, **plot_kwargs)
                if has_val:
                    ax.scatter(
                        var_iterations[j],
                        vals_v[j],
                        color=c,
                        marker="o",
                        facecolors="none",
                        linewidth=1.5,
                        zorder=5,
                        **plot_kwargs,
                    )
            for j in range(len(var_iterations) - 1):
                c = _iteration_color(var_iterations[j + 1], last_iteration)
                ax.plot(var_iterations[j : j + 2], vals[j : j + 2], color=c, lw=0.8, zorder=3)
                if has_val:
                    ax.plot(var_iterations[j : j + 2], vals_v[j : j + 2], color=c, lw=0.8, ls="--", zorder=3)
            # baseline: only iteration -1 ("best model")
            if -1 in baselines:
                y_gt = float(da.sel(iteration=-1))
                ax.axhline(y_gt, color="k", ls="--", lw=0.8)
                ax.text(
                    0.0,
                    y_gt,
                    " ground truth",
                    va="bottom",
                    ha="left",
                    fontsize="x-small",
                    color="k",
                    transform=ax.get_yaxis_transform(),
                )
            # ML baseline for LL / bps panels
            # if var_name in ("bits_per_spike", "logPYXF") and "mode_l" in results and 1 in results.iteration.values:
                # from simpl.utils import get_ML_loglikelihoods

                # ml = get_ML_loglikelihoods(results)
                # ml_key = f"{var_name}_val"
                # if ml_key in ml:
                #     y_ml = ml[ml_key]
                #     ax.axhline(y_ml, color="k", ls=":", lw=0.8)
                #     ax.text(
                #         0.0,
                #         y_ml,
                #         " naive ML",
                #         va="bottom",
                #         ha="left",
                #         fontsize="x-small",
                #         color="k",
                #         transform=ax.get_yaxis_transform(),
                #     )
        else:
            # per-neuron (possibly mean over place_field first)
            means = []
            for e in var_iterations:
                c = _iteration_color(e, last_iteration)
                vals_e = da.sel(iteration=e)
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
                ax.scatter(e, means[-1], color=c, s=60, zorder=5, linewidths=0)
            for j in range(len(var_iterations) - 1):
                c = _iteration_color(var_iterations[j + 1], last_iteration)
                ax.plot(var_iterations[j : j + 2], means[j : j + 2], color=c, lw=0.8, zorder=3)

        ax.set(xlabel="Iteration", ylabel=ylabel, xlim=(-0.5, int(iterations[-1]) + 0.5))
        outset_axes(ax)
        ax.spines["bottom"].set_bounds(0, int(iterations[-1]))

    # legend on first panel showing train/val distinction
    first_ax = axes.flat[0]
    first_ax.plot([], [], color="gray", lw=0.8, label="train")
    first_ax.plot([], [], color="gray", lw=0.8, ls="--", label="val")
    first_ax.legend(fontsize="small", frameon=False)

    # hide unused axes
    for j in range(n_metrics, len(axes.flat)):
        axes.flat[j].set_visible(False)

    return axes
