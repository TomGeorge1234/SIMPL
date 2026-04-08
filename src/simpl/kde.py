"""Kernel density estimation (KDE) of neural firing-rate maps and Poisson log-likelihood.

The M-step of the SIMPL EM loop estimates each neuron's receptive field using
KDE: spikes are smoothed with a Gaussian kernel over the spatial grid, and
normalised by occupancy, yielding a firing-rate map in units of spikes per
time bin.

For 1-D angular environments (``is_1D_angular=True``), the specialised
``kde_angular`` function convolves with a wrapped Gaussian kernel via FFT so
that the estimate is seamless across the \\([-\\pi, \\pi)\\) boundary.

The Poisson log-likelihood functions (``poisson_log_likelihood_maps`` and
``poisson_log_likelihood``) evaluate how well the estimated
receptive fields explain the observed spike counts and are used during the
E-step to construct likelihood maps over position space."""

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
from jax import vmap

from simpl.utils import _bin_indices_minuspi_pi, _circular_conv_fft_1d

__all__ = [
    "decode_observations",
    "gaussian_kernel",
    "kde",
    "kde_angular",
    "poisson_log_likelihood_maps",
    "poisson_log_likelihood",
    "get_ll_and_bps_splits",
]


def _log_factorial_stirling(spikes: jax.Array) -> jax.Array:
    """Stirling's approximation of log(spikes!), with manual correction for 0! = 1."""
    spikes_ = jnp.where(spikes == 0, 1, spikes)
    return jnp.log(jnp.sqrt(2 * jnp.pi)) + (spikes_ + 0.5) * jnp.log(spikes_) - spikes_


def gaussian_kernel(
    x1: jax.Array,
    x2: jax.Array,
    bandwidth: float,
) -> jax.Array:
    """Evaluates the Gaussian kernel between two points \\(x_1\\) and \\(x_2\\):

    $$K(x_1, x_2) = \\frac{1}{\\sqrt{(2\\pi)^D |\\Sigma|}}
    \\exp\\!\\left(-\\frac{1}{2}(x_1 - x_2)^\\top \\Sigma^{-1} (x_1 - x_2)\\right)$$

    where \\(\\Sigma = \\sigma^2 I\\) is the isotropic covariance with bandwidth \\(\\sigma\\).

    Parameters
    ----------
    x1: (D,) jax.Array
        The first position
    x2: (D,) jax.Array
        The second position
    bandwidth: float
        The bandwidth of the kernel

    Returns
    -------
    kernel: float
        The probability density at x
    """
    diff = x1 - x2
    D = x1.shape[0]
    norm_const = 1.0 / ((2.0 * jnp.pi) ** (D / 2.0) * bandwidth**D)
    return norm_const * jnp.exp(-0.5 * jnp.sum(diff**2) / bandwidth**2)


def kde(
    bins: jax.Array,
    trajectory: jax.Array,
    spikes: jax.Array,
    kernel: Callable = gaussian_kernel,
    kernel_bandwidth: float = 0.01,
    mask: jax.Array | None = None,
    batch_size: int | None = None,
    return_position_density: bool = False,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """
    Performs KDE to estimate the expected number of spikes each neuron will fire
    at each position in `bins` given past `trajectory` and `spikes` data. This
    estimate is an expected-spike-count-per-timebin, in order to get firing rate
    in Hz, divide this by dt.

    Kernel Density Estimation goes as follows (the denominator corrects for
    non-uniform position density):

    $$\\mu(x) = \\frac{\\sum_{t_s \\in \\text{spike times}} K(x, x(t_s))}{\\sum_{t} K(x, x(t))} = \\frac{K_s}{K_x}$$

    In practice this is computed in log-space as
    \\(\\mu(x) = \\exp[\\log(K_s) - \\log(K_x)]\\).

    Optionally, a boolean mask same shape as spikes can be passed to ignore
    certain spikes. This restricts the KDE calculation to only the spikes
    where mask is True.

    Parameters
    ----------
    bins : jax.Array, shape (N_bins, D,)
        The position bins at which to estimate the firing rate
    trajectory : jax.Array, shape (T, D)
        The position of the agent at each time step
    spikes : jax.Array, shape (T, N_neurons)
        The spike counts of the neuron at each time step (integer array, can be > 1)
    kernel : function
        The kernel function to use for density estimation. See `kernels.py` for signature and examples.
    kernel_bandwidth : float
        The bandwidth of the kernel
    mask : jax.Array, shape (T, N_neurons), optional
        A boolean mask to apply to the spikes. If None, no mask is applied. Default is None.
    batch_size : int or None
        The time axis is split into batches of this size to avoid memory
        errors, each batch is then processed in series. If None (default),
        chosen adaptively to target ~64 MB peak for the kernel matrix.
    return_position_density : bool
        If True, this function also returns the position density (the denominator of the KDE) at each bin.


    Returns
    -------
    kernel_density_estimate : jax.Array, shape (N_neurons, N_bins)
    position_density : jax.Array, shape (N_bins,) (optional)
        Normalised position density (sums to 1 over bins), independent of neuron masks.
    """
    assert bins.ndim == 2
    assert trajectory.ndim == 2
    assert spikes.ndim == 2

    N_neurons = spikes.shape[1]
    N_bins = bins.shape[0]
    T = trajectory.shape[0]

    if batch_size is None:
        from simpl import MAX_BATCH_ELEMENTS

        batch_size = max(256, MAX_BATCH_ELEMENTS // N_bins)
    batch_size = min(batch_size, T)

    # If not passed make a trivial mask (all True)
    if mask is None:
        mask = jnp.ones_like(spikes, dtype=bool)
    # vmap the kernel K(x,mu,sigma) so it takes in a vector of positions and a vector of means
    kernel_fn = partial(kernel, bandwidth=kernel_bandwidth)
    vmapped_kernel = vmap(vmap(kernel_fn, in_axes=(0, None)), in_axes=(None, 0))

    spike_density = jnp.zeros((N_bins, N_neurons))
    position_density_internal = jnp.zeros(
        (N_bins, N_neurons)
    )  # Seperate position density for neuron-specific masks, used to calculate KDE estimates
    position_density = jnp.zeros(
        (N_bins,)
    )  # Mask agnostic density, just "where has the animal been", optionally returned for downstream calculations.

    N_batchs = int(jnp.ceil(T / batch_size))
    for i in range(N_batchs):
        start = i * batch_size
        end = min((i + 1) * batch_size, T)
        # Get the batch of trajectory, spikes and mask
        trajectory_batch = trajectory[start:end]
        spikes_batch = spikes[start:end]
        mask_batch = mask[start:end]

        # Pairwise kernel values for each trajectory-bin position pair. The bulk of the computation is done here.
        kernel_values = vmapped_kernel(trajectory_batch, bins)
        # Calculate normalisation position density (the +epsilon means unvisited
        # positions should approach 0 density and avoid nans)
        position_density_internal_batch = kernel_values @ mask_batch + 1e-6
        # Mask-free position density for return
        position_density += kernel_values.sum(axis=1)
        # Calculate spike density, replace nans from no-spikes with 0
        spike_density_batch = kernel_values @ (mask_batch * spikes_batch)
        spike_density_batch = jnp.where(jnp.isnan(spike_density_batch), 0, spike_density_batch)

        # Add these to the running total
        spike_density += spike_density_batch
        position_density_internal += position_density_internal_batch

    # calculate kde at each bin position
    kernel_density_estimate = jnp.exp(jnp.log(spike_density) - jnp.log(position_density_internal)).T

    if return_position_density:
        # Normalise position density to a valid PDF
        position_density = position_density / position_density.sum()
        return kernel_density_estimate, position_density
    else:
        return kernel_density_estimate


def poisson_log_likelihood(
    spikes: jax.Array,
    rates: jax.Array,
) -> jax.Array:
    """Per-element Poisson log-likelihood of spike counts given predicted rates.

    The Poisson probability of observing \\(X\\) spikes given mean rate \\(\\mu\\) is:

    $$P(X \\mid \\mu) = \\frac{\\mu^X \\, e^{-\\mu}}{X!}$$

    so the log-likelihood is:

    $$\\log P(X \\mid \\mu) = X \\log \\mu - \\mu - \\log(X!)$$

    where \\(\\log(X!)\\) is computed via Stirling's approximation (manually
    correcting for when \\(X = 0\\)):

    $$\\log(X!) \\approx \\log\\sqrt{2\\pi} + (X + 0.5)\\log X - X$$

    Accepts arrays of any shape; ``spikes`` and ``rates`` must share the same
    shape. Returns an array of that same shape.

    Parameters
    ----------
    spikes : jax.Array
        Observed spike counts.
    rates : jax.Array
        Predicted firing rates (expected spikes per time bin). Same shape as ``spikes``.

    Returns
    -------
    log_likelihood : jax.Array
        Per-element Poisson log-likelihood. Same shape as inputs.
    """
    assert spikes.shape == rates.shape, f"spikes {spikes.shape} and rates {rates.shape} must have the same shape"

    log_spikecount_factorial = _log_factorial_stirling(spikes)

    return (spikes * jnp.log(rates + 1e-3)) - rates - log_spikecount_factorial


def poisson_log_likelihood_maps(
    spikes: jax.Array,
    mean_rate: jax.Array,
    mask: jax.Array | None = None,
) -> jax.Array:
    """Version of ``poisson_log_likelihood`` optimised to broadcast over the
    spatial binning dimension and sum over neurons.

    Used in the E-step to build likelihood maps: for each time step, evaluates
    the Poisson log-likelihood at every spatial bin simultaneously via a matrix
    multiply ``(T, N) @ (N, B) → (T, B)``.

    Parameters
    ----------
    spikes : jax.Array, shape (T, N_neurons)
        Observed spike counts.
    mean_rate : jax.Array, shape (N_neurons, N_bins)
        Receptive fields: expected spike count per time bin at each spatial bin.
    mask : jax.Array, shape (T, N_neurons), optional
        Boolean spike mask. If None, all neurons are used.

    Returns
    -------
    log_likelihood : jax.Array, shape (T, N_bins)
        Log-likelihood summed over neurons at each spatial bin.
    """
    if mask is None:
        mask = jnp.ones_like(spikes, dtype=bool)

    log_spikecount_factorial = _log_factorial_stirling(spikes)

    return (
        (mask * spikes) @ jnp.log(mean_rate + 1e-3)
        - mask @ mean_rate
        - jnp.sum(log_spikecount_factorial * mask, axis=1)[:, None]
    )


def get_ll_and_bps_splits(
    Y: jax.Array,
    FX: jax.Array,
    mask: jax.Array,
) -> dict:
    """Compute train/val log-likelihoods and bits-per-spike from spikes and predicted rates.

    Parameters
    ----------
    Y : jax.Array, shape (T, N_neurons)
        Observed spike counts.
    FX : jax.Array, shape (T, N_neurons)
        Predicted firing rates (expected spikes per time bin).
    mask : jax.Array, shape (T, N_neurons)
        Boolean training mask. True = train, False = validation.

    Returns
    -------
    dict
        Keys: ``logPYXF``, ``logPYXF_val``, ``bits_per_spike``, ``bits_per_spike_val``.
    """
    val_mask = ~mask
    ll = poisson_log_likelihood(Y, FX)

    ll_train = (ll * mask).sum()
    ll_val = (ll * val_mask).sum()

    mean_FX = (Y * mask).sum(axis=0, keepdims=True) / mask.sum(axis=0, keepdims=True)
    mean_FX = jnp.broadcast_to(mean_FX, FX.shape)
    ll_baseline = poisson_log_likelihood(Y, mean_FX)

    LLs = {
        "logPYXF": ll_train / mask.sum(),
        "logPYXF_val": ll_val / val_mask.sum(),
    }
    for m, suffix, ll_model in [(mask, "", ll_train), (val_mask, "_val", ll_val)]:
        ll_base = (ll_baseline * m).sum()
        n_spikes = (Y * m).sum()
        LLs[f"bits_per_spike{suffix}"] = jnp.where(n_spikes > 0, (ll_model - ll_base) / (n_spikes * jnp.log(2.0)), 0.0)

    return {k: float(v) for k, v in LLs.items()}


def decode_observations(
    xF: jax.Array,
    spikes: jax.Array,
    mean_rate: jax.Array,
    mask: jax.Array,
    batch_size: int | None = None,
    return_log_maps: bool = False,
) -> tuple:
    """Compute Poisson likelihood maps, fit Gaussian observations, and flag silent bins.

    This combines ``poisson_log_likelihood_maps`` and ``fit_gaussian`` in a single
    batched pipeline so that the full ``(T, N_bins)`` likelihood tensor is never
    materialised at once, keeping peak memory low for long sessions.

    Parameters
    ----------
    xF : jax.Array, shape (N_bins, D)
        Spatial bin centres.
    spikes : jax.Array, shape (T, N_neurons)
        Spike counts.
    mean_rate : jax.Array, shape (N_neurons, N_bins)
        Receptive fields (expected spike counts per bin per time step).
    mask : jax.Array, shape (T, N_neurons)
        Boolean mask (True = use neuron at this time step).
    batch_size : int or None
        Number of time bins per batch. If None (default), chosen adaptively
        to target ~64 MB peak memory for the likelihood tensor.
    return_log_maps : bool
        If True, also return the full ``(T, N_bins)`` log-likelihood maps.

    Returns
    -------
    mu_l, mode_l, sigma_l : jax.Array
        Gaussian observation parameters fitted from the likelihood.
    no_spikes : jax.Array, shape (T,)
        Boolean, True where total (masked) spike count is zero.
    logPYXF_maps : jax.Array, shape (T, N_bins)
        Only returned when ``return_log_maps`` is True.
    """
    from simpl.utils import fit_gaussian  # local to avoid circular import

    T = spikes.shape[0]
    N_bins = xF.shape[0]
    if batch_size is None:
        from simpl import MAX_BATCH_ELEMENTS

        batch_size = max(256, MAX_BATCH_ELEMENTS // N_bins)
    batch_size = min(batch_size, T)

    @partial(jax.jit, static_argnames=("_return_log_maps",))
    def _process_batch(xF, spikes_batch, mean_rate, mask_batch, _return_log_maps=False):
        log_maps = poisson_log_likelihood_maps(spikes_batch, mean_rate, mask=mask_batch)
        log_maps = log_maps - jnp.max(log_maps, axis=1)[:, None]  # shift max to 0 to avoid NaNs in exp
        mu, mode, sigma = fit_gaussian(xF, jnp.exp(log_maps))
        no_spk = jnp.sum(spikes_batch * mask_batch, axis=1) == 0
        if _return_log_maps:
            return mu, mode, sigma, no_spk, log_maps
        return mu, mode, sigma, no_spk

    mu_batches, mode_batches, sigma_batches, no_spike_batches = [], [], [], []
    log_map_batches = [] if return_log_maps else None

    for start in range(0, T, batch_size):
        end = min(start + batch_size, T)
        result = _process_batch(
            xF,
            spikes[start:end],
            mean_rate,
            mask[start:end],
            _return_log_maps=return_log_maps,
        )
        if return_log_maps:
            mu_b, mode_b, sigma_b, no_spk_b, log_maps_b = result
            log_map_batches.append(log_maps_b)
        else:
            mu_b, mode_b, sigma_b, no_spk_b = result
        mu_batches.append(mu_b)
        mode_batches.append(mode_b)
        sigma_batches.append(sigma_b)
        no_spike_batches.append(no_spk_b)

    mu_l = jnp.concatenate(mu_batches, axis=0)
    mode_l = jnp.concatenate(mode_batches, axis=0)
    sigma_l = jnp.concatenate(sigma_batches, axis=0)
    no_spikes = jnp.concatenate(no_spike_batches, axis=0)

    if return_log_maps:
        return mu_l, mode_l, sigma_l, no_spikes, jnp.concatenate(log_map_batches, axis=0)
    return mu_l, mode_l, sigma_l, no_spikes


@partial(jax.jit, static_argnames=("kernel", "return_position_density"))
def kde_angular(
    bins: jax.Array,  # (N_bins,) bin centers in [-pi, pi)
    trajectory: jax.Array,  # (T,) angles in radians
    spikes: jax.Array,  # (T, N_neurons) spike counts
    kernel=None,  # unused placeholder
    kernel_bandwidth: float = 0.3,  # std dev of smoothing kernel in radians
    mask: jax.Array | None = None,  # (T, N_neurons) boolean
    return_position_density: bool = False,
    eps: float = 1e-6,
) -> jax.Array | tuple[jax.Array, jax.Array]:
    """
    Circular KDE for angular data. Estimates expected spike count per timebin
    at each angular bin (divide by dt for Hz). See `kde()` for the linear
    equivalent.

    $$\\mu(\\theta) = \\frac{K_s}{K_x}$$

    where \\(K_s\\) and \\(K_x\\) are the kernel-smoothed spike count and
    occupancy histograms, respectively.

    Unlike `kde()`, which evaluates all pairwise kernel values, this function
    first histograms the data then smooths via FFT-based circular convolution
    with a von Mises kernel:

    $$\\text{smooth}(x) = \\mathcal{F}^{-1}\\!\\big[\\mathcal{F}(\\text{histogram})
    \\cdot \\mathcal{F}(\\text{von Mises kernel})\\big]$$

    This is \\(O(N_{\\text{bins}} \\log N_{\\text{bins}})\\) rather than
    \\(O(N_{\\text{bins}} \\cdot T)\\).

    Both `bins` and `trajectory` must be in radians in \\([-\\pi, \\pi)\\). Bins
    must be uniformly spaced. `kernel_bandwidth` is in radians and is converted
    internally to von Mises concentration \\(\\kappa = 1 / \\sigma^2\\)
    (accurate for \\(\\kappa > 2\\), i.e. small bandwidth \\(\\sigma < {\\sim}0.7\\) rad).

    Parameters
    ----------
    bins : jax.Array, shape (N_bins,) or (N_bins, 1)
        Angle bin centres in radians, uniformly spaced in [-pi, pi).
    trajectory : jax.Array, shape (T,) or (T, 1)
        Angular position of the agent at each time step, in radians in [-pi, pi).
    spikes : jax.Array, shape (T, N_neurons)
        Spike counts at each time step (integer array, can be > 1).
    kernel : None
        Unused, kept for API consistency with `kde()`.
    kernel_bandwidth : float
        Std dev of smoothing kernel in radians. Larger = smoother.
        Converted to von Mises kappa = 1 / kernel_bandwidth^2.
    mask : jax.Array, shape (T, N_neurons), optional
        Boolean mask for spikes. Default is None (no masking).
    return_position_density : bool
        If True, also returns normalised position density. Default is False.
    eps : float
        Small constant to avoid division by zero. Default is 1e-6.

    Returns
    -------
    kernel_density_estimate : jax.Array, shape (N_neurons, N_bins)
    position_density : jax.Array, shape (N_bins,) (optional)
        Normalised position density (sums to 1), independent of neuron masks.
    """
    assert bins.ndim == 1 or (bins.ndim == 2 and bins.shape[1] == 1), "bins should be shape (N_bins,) or (N_bins, 1)."
    assert trajectory.ndim == 1 or (trajectory.ndim == 2 and trajectory.shape[1] == 1), (
        "trajectory should be shape (T,) or (T, 1). kde_angular only supports 1D circular data."
    )

    bins = jnp.asarray(bins).flatten()
    trajectory = jnp.asarray(trajectory).flatten()
    spikes = jnp.asarray(spikes)

    n_bins = bins.shape[0]
    assert n_bins % 2 == 0, "n_bins should be even for FFT-based circular convolution."
    T = trajectory.shape[0]
    n_neurons = spikes.shape[1]

    if mask is None:
        mask = jnp.ones((T, n_neurons), dtype=bool)
    mask_f = mask.astype(jnp.float32)

    # 1) bin indices consistent with [-pi, pi)
    idx = _bin_indices_minuspi_pi(trajectory, n_bins)  # (T,)

    # 2) von Mises kernel over offsets in [-pi, pi)
    # Build on symmetric grid => delta_theta=0 sits at index n_bins//2
    dtheta = jnp.linspace(-jnp.pi, jnp.pi, n_bins, endpoint=False)
    # Convert bandwidth (std in radians) to von Mises concentration.
    # kappa ~ 1/sigma^2 is a good approximation for kappa > 2 (sigma < ~0.7 rad).
    kappa = 1.0 / (kernel_bandwidth**2)
    vm = jnp.exp(kappa * jnp.cos(dtheta))
    vm = vm / jnp.sum(vm)

    # Align for FFT: put delta_theta=0 at index 0
    vm = jnp.roll(vm, -n_bins // 2)

    # 3) histogram per neuron using bincount (vmap over neurons)
    def hist_for_neuron(weights_t: jax.Array) -> jax.Array:
        return jnp.bincount(idx, weights=weights_t, length=n_bins)

    # occupancy: mask only (per-neuron, used internally for KDE denominator)
    pos_hist = vmap(hist_for_neuron, in_axes=1, out_axes=0)(mask_f)  # (N, B)

    # mask-free position histogram for return
    pos_hist_total = hist_for_neuron(jnp.ones(T, dtype=jnp.float32))  # (B,)

    # spikes: spikes * mask
    spike_w = spikes.astype(jnp.float32) * mask_f  # (T, N)
    spike_hist = vmap(hist_for_neuron, in_axes=1, out_axes=0)(spike_w)  # (N, B)

    # 4) smooth via circular convolution
    pos_smooth = vmap(_circular_conv_fft_1d, in_axes=(0, None), out_axes=0)(pos_hist, vm)
    spike_smooth = vmap(_circular_conv_fft_1d, in_axes=(0, None), out_axes=0)(spike_hist, vm)
    pos_smooth_total = _circular_conv_fft_1d(pos_hist_total, vm)  # (B,)

    # 5) ratio
    kde_result = spike_smooth / (pos_smooth + eps)

    if return_position_density:
        position_density = pos_smooth_total / pos_smooth_total.sum()
        return kde_result, position_density
    return kde_result
