"""Diagnostic functions for checking data quality."""

import jax.numpy as jnp
import numpy as np
import xarray as xr

from simpl.environment import Environment
from simpl.kde import gaussian_kernel, kde
from simpl.utils import calculate_spatial_information


def binning_diagnostic(
    data: xr.Dataset,
    bin_size: float = 0.02,
    kernel_bandwidth: float = 0.02,
) -> tuple[float, float]:
    """Check whether the time binning is appropriate for the data.

    Returns two numbers:
        (i)  fraction of time bins containing spikes from >= 2 distinct neurons
        (ii) average spatial information per time bin (bits)

    Parameters
    ----------
    data : xr.Dataset
        Dataset as returned by ``prepare_data``, must contain ``Y``, ``Xb``,
        and ``time``.
    bin_size : float
        Spatial bin size for the KDE place-field estimate.
    kernel_bandwidth : float
        KDE kernel bandwidth.

    Returns
    -------
    frac_multi_cell : float
        Fraction of time bins with spikes from 2 or more neurons.
    mean_info_per_bin : float
        Average spatial information per time bin (bits).
    """
    Y = np.array(data.Y.values)  # (T, N)
    Xb = data.Xb.values
    dt = float(data.time.values[1] - data.time.values[0])

    # (i) fraction of bins with spikes from >= 2 cells
    neurons_active = (Y > 0).sum(axis=1)  # number of distinct neurons firing per bin
    frac_multi_cell = float(np.mean(neurons_active >= 2))

    # (ii) spatial information per time bin
    env = Environment(X=Xb, pad=0.0, bin_size=bin_size, verbose=False)
    bins = jnp.array(env.flattened_discretised_coords)
    X = jnp.array(Xb)
    spikes = jnp.array(Y)

    F, PX = kde(
        bins=bins,
        trajectory=X,
        spikes=spikes,
        kernel=gaussian_kernel,
        kernel_bandwidth=kernel_bandwidth,
        return_position_density=True,
    )

    # Spatial information per neuron (bits/s) and per-bin information
    r = F / dt
    spatial_info = calculate_spatial_information(r, PX)  # bits/s per neuron

    # Information per time bin: I_t = sum_n y_{t,n} * (I_n / r_mean_n) where I_n/r_mean = bits/spike
    r_mean = jnp.sum(r * PX[None, :], axis=1)
    bits_per_spike = spatial_info / (r_mean + 1e-10)  # bits/spike per neuron
    I_t = np.array(spikes @ bits_per_spike)
    mean_info_per_bin = float(np.mean(I_t))

    info_rate = mean_info_per_bin / dt

    print("SIMPL works best when there are multiple active neurons ")
    print(f"Fraction of bins with >= 2 active neurons: {frac_multi_cell:.1%}")
    print(f"Average spatial information per bin:        {mean_info_per_bin:.1f} bits per bin ({info_rate:.1f} bits/s)")

    return frac_multi_cell, mean_info_per_bin
