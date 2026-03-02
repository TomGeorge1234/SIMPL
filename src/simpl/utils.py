"""Utility functions: Gaussian PDF helpers, Gaussian fitting, CCA alignment,
data preparation, I/O, and circular/angular helpers."""

import jax
import jax.numpy as jnp
import numpy as np
import scipy
import skimage.measure
import sklearn.cross_decomposition
import xarray as xr
from jax import random

_TAU = 2 * jnp.pi


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian helpers
# ──────────────────────────────────────────────────────────────────────────────


def gaussian_pdf(
    x: jax.Array,
    mu: jax.Array,
    sigma: jax.Array,
) -> jax.Array:
    """Calculates the gaussian pdf of a multivariate normal distribution of mean mu and covariance sigma at x

    Parameters
    ----------

    x: (D,) array
        The position at which to evaluate the pdf
    mu: (D,) array
        The mean of the distribution
    sigma: (D, D) array
        The covariance of the distribution

    Returns
    -------
    pdf: float
        The probability density at x
    """
    assert x.ndim == 1
    assert mu.ndim == 1
    assert sigma.ndim == 2
    assert x.shape[0] == mu.shape[0]
    assert x.shape[0] == sigma.shape[0]
    assert sigma.shape[0] == sigma.shape[1]

    x = x - mu
    norm_const = gaussian_norm_const(sigma)
    return norm_const * jnp.exp(-0.5 * jnp.sum(x @ jnp.linalg.inv(sigma) * x, axis=-1))


def log_gaussian_pdf(
    x: jax.Array,
    mu: jax.Array,
    sigma: jax.Array,
) -> jax.Array:
    """Calculates the log of the gaussian pdf of a multivariate normal distribution of mean mu and covariance sigma at x

    Parameters
    ----------
    x: (D,) array
        The position at which to evaluate the pdf
    mu: (D,) array
        The mean of the distribution
    sigma: (D, D) array
        The covariance of the distribution

    Returns
    -------
    log_pdf: float
        The log probability density at x
    """
    assert x.ndim == 1
    assert mu.ndim == 1
    assert sigma.ndim == 2
    assert x.shape[0] == mu.shape[0]
    assert x.shape[0] == sigma.shape[0]
    assert sigma.shape[0] == sigma.shape[1]

    x = x - mu
    norm_const = gaussian_norm_const(sigma)
    return jnp.log(norm_const) - 0.5 * jnp.sum(x @ jnp.linalg.inv(sigma) * x)


def gaussian_norm_const(sigma: jax.Array) -> jax.Array:
    """Calculates the normalizing constant of a multivariate normal distribution with covariance sigma

    Parameters
    ----------
    sigma: jnp.ndarray, shape (D, D)
        The covariance matrix of the distribution

    Returns
    -------
    norm_const: jnp.ndarray, shape (1,)
        The normalizing constant
    """
    assert sigma.ndim == 2
    D = sigma.shape[0]
    return 1 / jnp.sqrt((2 * jnp.pi) ** D * jnp.linalg.det(sigma))


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian fitting
# ──────────────────────────────────────────────────────────────────────────────


def fit_gaussian(x: jax.Array, likelihood: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fits a multivariate-Gaussian to the likelihood function P(spikes | x) in x-space.

    Parameters
    ----------
    x : jnp.ndarray, shape (N_bins,D)
        The position bins in which the likelihood is calculated
    likelihood : jnp.ndarray, shape (N_bins,)
        The combined likelihood (not log-likelihood) of the neurons firing at each position bin

    Returns
    -------
    mu : jnp.ndarray, shape (D,)
        The mean of the Gaussian
    mode : jnp.ndarray, shape (D,)
        The mode of the Gaussian
    cov : jnp.ndarray, shape (D, D)
        The covariance of the Gaussian
    """
    assert x.ndim == 2
    assert likelihood.ndim == 1
    assert x.shape[0] == likelihood.shape[0]

    mu = (x.T @ likelihood) / likelihood.sum()
    mode = x[jnp.argmax(likelihood)]
    cov = ((x - mu) * likelihood[:, None]).T @ (x - mu) / likelihood.sum()
    return mu, mode, cov


def fit_gaussian_vmap(x: jax.Array, likelihoods: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fits a multivariate-Gaussian to each row of a batch of likelihood arrays.

    This is the vmapped version of ``fit_gaussian``: it accepts likelihoods of
    shape ``(T, N_bins)`` and returns batched means, modes, and covariances.

    Parameters
    ----------
    x : jnp.ndarray, shape (N_bins, D)
        The position bins (shared across all time steps).
    likelihoods : jnp.ndarray, shape (T, N_bins)
        Likelihood values at each bin for each time step.

    Returns
    -------
    means : jnp.ndarray, shape (T, D)
        The mean of each fitted Gaussian.
    modes : jnp.ndarray, shape (T, D)
        The mode of each fitted Gaussian.
    covariances : jnp.ndarray, shape (T, D, D)
        The covariance of each fitted Gaussian.
    """
    return jax.vmap(fit_gaussian, in_axes=(None, 0))(x, likelihoods)


def gaussian_sample(key: jax.Array, mu: jax.Array, sigma: jax.Array) -> jax.Array:
    """Samples from a multivariate normal distribution with mean mu and covariance sigma.

    Parameters
    ----------
    key : PRNGKey
        The random key
    mu : jnp.ndarray, shape (D,)
        The mean of the distribution
    sigma : jnp.ndarray, shape (D, D)
        The covariance of the distribution

    Returns
    -------
    sample : jnp.ndarray, shape (D,)
        The sample
    """
    assert mu.ndim == 1
    assert sigma.ndim == 2
    sample = random.multivariate_normal(key, mu, sigma)
    return sample


# ──────────────────────────────────────────────────────────────────────────────
# Circular / angular helpers (ported from kalmax)
# ──────────────────────────────────────────────────────────────────────────────


def _wrap_minuspi_pi(theta: jax.Array) -> jax.Array:
    """Wrap angles to [-pi, pi).

    Parameters
    ----------
    theta : jnp.ndarray
        Angles in radians (any range)

    Returns
    -------
    jnp.ndarray
        Angles wrapped to [-pi, pi)
    """
    return jnp.mod(theta + jnp.pi, _TAU) - jnp.pi


def _bin_indices_minuspi_pi(theta: jax.Array, n_bins: int) -> jax.Array:
    """Map theta in radians to integer bin indices [0, n_bins).

    Maps angles to bin indices where bin 0 corresponds to [-pi, -pi + delta).

    Parameters
    ----------
    theta : jnp.ndarray
        Angles in radians (any range)
    n_bins : int
        Number of bins

    Returns
    -------
    jnp.ndarray
        Integer bin indices in [0, n_bins)
    """
    theta = _wrap_minuspi_pi(theta)
    u = (theta + jnp.pi) * (n_bins / _TAU)  # in [0, n_bins)
    idx = jnp.floor(u).astype(jnp.int32)
    # guard against theta == pi mapping to n_bins (shouldn't happen for [-pi,pi) but safe)
    return jnp.clip(idx, 0, n_bins - 1)


def _circular_conv_fft_1d(x: jax.Array, k: jax.Array) -> jax.Array:
    """Circular convolution via FFT for 1D arrays.

    Parameters
    ----------
    x : jnp.ndarray
        Input array of length N
    k : jnp.ndarray
        Kernel array of length N

    Returns
    -------
    jnp.ndarray
        Circular convolution of x and k, same length as input
    """
    return jnp.fft.ifft(jnp.fft.fft(x) * jnp.fft.fft(k)).real


# ──────────────────────────────────────────────────────────────────────────────
# Statistical / analysis helpers
# ──────────────────────────────────────────────────────────────────────────────


def coefficient_of_determination(
    X: jax.Array,
    Y: jax.Array,
) -> jax.Array:
    """Calculates the coefficient of determination (R^2) between X and Y.

    This reflects the proportion of the variance in Y that is predictable from X.

    Parameters
    ----------
    X : jnp.ndarray, shape (N, D)
        The predicted latent positions
    Y : jnp.ndarray, shape (N, D)
        The true latent positions"""
    assert X.shape == Y.shape, "The predicted and true latent positions must have the same shape."
    SST = jnp.sum((Y - jnp.mean(Y, axis=0)) ** 2)
    SSR = jnp.sum((Y - X) ** 2)
    R2 = 1 - SSR / SST
    return R2


def cca(X: jax.Array, Y: jax.Array) -> tuple[np.ndarray, np.ndarray]:
    """Uses canonical correlation between X and Y (the "target") to establish the best linear mapping from X to Y.

    Parameters
    ----------
    X : jnp.ndarray, shape (N, D)
        The inputs
    Y : jnp.ndarray, shape (N, D)
        The targets

    Returns:
    -------
    coef : jnp.ndarray, shape (D, D)
        The coefficients of the linear mapping from X to Y such that Y ~= Y_pred = X @ coef.T + intercept
    intercept : jnp.ndarray, shape (D,)
        The intercept of the linear mapping from X to Y such that Y ~= Y_pred = X @ coef.T + intercept
    """
    assert X.shape == Y.shape, "The predicted and true latent positions must have the same shape."
    D = X.shape[1]

    cca = sklearn.cross_decomposition.CCA(n_components=D)
    cca.fit(X, Y)
    coef = cca.coef_  # / cca._x_std # this randomly changed at some point
    intercept = cca.intercept_ - cca._x_mean @ coef.T
    return coef, intercept


def correlation_at_lag(X1: jax.Array, X2: jax.Array, lag: int) -> jax.Array:
    """Calculates the correlation between X1 and X2[lag:].

    If X is D-dimensional, calculates the average correlation across dimensions.

    Parameters
    ----------

    X1 : jnp.ndarray, shape (T, D)
        The first time series - remains fixed
    X2 : jnp.ndarray, shape (T, D)
        The second time series
    lag : int
        The lag to calculate the correlation at

    Returns
    -------
    float
        The average correlation across dimensions
    """
    T, D = X1.shape
    if lag >= 0:
        X2 = X2[lag:, :]
        X1 = X1[: T - lag]
    else:
        lag = -lag
        X1 = X1[lag:, :]
        X2 = X2[: T - lag]
    return jnp.mean(jnp.diag(jnp.corrcoef(X1.T, X2.T)[D:, :D]))


# ──────────────────────────────────────────────────────────────────────────────
# Data preparation and I/O
# ──────────────────────────────────────────────────────────────────────────────


def accumulate_spikes(Y: np.ndarray, window: int) -> np.ndarray:
    """Causal rolling sum of spikes over a backward-looking window.

    Each time bin accumulates spikes from the current and previous
    ``window - 1`` bins. This is equivalent to smoothing the spikes with
    a causal rectangular kernel.

    .. warning::

        This changes the interpretation of the estimated receptive fields.
        Since each bin now contains on average ``window`` times more spikes,
        the fitted firing rates (and therefore ``F``) will be approximately
        ``window`` times higher than the true single-bin rates. The receptive
        field *shapes* are unaffected, but their *amplitudes* should not be
        interpreted as physical firing rates.

    Parameters
    ----------
    Y : np.ndarray, shape (T, N_neurons)
        Spike counts.
    window : int
        Number of bins to sum over (looking backwards). For example,
        ``window=5`` sums the current bin and the 4 preceding bins.

    Returns
    -------
    Y_accumulated : np.ndarray, shape (T, N_neurons)
        Spike counts after causal rolling sum.
    """
    Y_out = np.zeros_like(Y)
    for i in range(Y.shape[0]):
        start = max(0, i - window + 1)
        Y_out[i] = Y[start : i + 1].sum(axis=0)
    return Y_out


def coarsen_dt(
    Y: np.ndarray,
    Xb: np.ndarray,
    time: np.ndarray,
    dt_multiplier: int,
    Xt: np.ndarray | None = None,
) -> tuple:
    """Coarsen data by averaging over groups of ``dt_multiplier`` time bins.

    Spikes are summed (not averaged) so that spike counts remain integers.
    Positions and time are averaged.

    Parameters
    ----------
    Y : np.ndarray, shape (T, N_neurons)
        Spike counts.
    Xb : np.ndarray, shape (T, D)
        Behavioural positions.
    time : np.ndarray, shape (T,)
        Time stamps.
    dt_multiplier : int
        Factor by which to coarsen the data.
    Xt : np.ndarray or None, shape (T, D), optional
        Ground truth positions.

    Returns
    -------
    Y_coarse : np.ndarray
        Coarsened spike counts (summed).
    Xb_coarse : np.ndarray
        Coarsened behavioural positions (averaged).
    time_coarse : np.ndarray
        Coarsened time stamps (averaged).
    Xt_coarse : np.ndarray (only if Xt was provided)
        Coarsened ground truth positions (averaged).
    """
    T = Y.shape[0]
    T_new = T // dt_multiplier
    T_trim = T_new * dt_multiplier

    Y_coarse = Y[:T_trim].reshape(T_new, dt_multiplier, -1).sum(axis=1)
    Xb_coarse = Xb[:T_trim].reshape(T_new, dt_multiplier, -1).mean(axis=1)
    time_coarse = time[:T_trim].reshape(T_new, dt_multiplier).mean(axis=1)

    if Xt is not None:
        Xt_coarse = Xt[:T_trim].reshape(T_new, dt_multiplier, -1).mean(axis=1)
        return Y_coarse, Xb_coarse, time_coarse, Xt_coarse

    return Y_coarse, Xb_coarse, time_coarse


def create_speckled_mask(
    size: tuple[int, int],
    sparsity: float = 0.1,
    block_size: int = 10,
    random_seed: int = 0,
) -> jax.Array:
    """
    TODO : Rewrite this in JAX
    Creates a boolean mask of size `size`. This mask is all True except along each column randomly
    there are contiguous blocks of False of length `block_size`. Overall ~`sparsity`
    of the mask is False. For example, if sparsity is 0.3, block size is 3 and size is
    (4, 15), a valid mask would be:

    [[T, T, T, T, T, T, T, T, F, F, F, T, F, F, F, T, T, T, T, T],
    [T, T, F, F, F, T, T, T, T, T, T, T, T, T, T, T, F, F, F, T],
    [T, T, T, T, T, T, T, T, T, F, F, F, T, T, F, F, F, T, T, T],
    [F, F, F, T, T, T, T, T, T, T, T, T, F, F, F, T, T, T, T, T]]

    Parameters
    ----------
    size : tuple of int
        The dimensions of the mask to create.
    sparsity : float
        The fraction of the mask that should be False.
    block_size : int
        The size of the contiguous False blocks.

    Returns
    -------
    mask : np.ndarray
        A boolean mask with the specified properties.
    """
    mask = np.ones(size, dtype=bool)
    num_blocks_per_row = int(sparsity * size[0] / block_size)
    np.random.seed(random_seed)
    for row in range(size[1]):
        for block in range(num_blocks_per_row):
            # Randomly choose starting positions within the bounds
            start_idx = np.random.randint(0, size[0] - block_size)
            end_idx = min(start_idx + block_size, size[0])
            mask[start_idx:end_idx, row] = False
    return jnp.array(mask)


def load_datafile(name: str = "gridcelldata.npz") -> np.lib.npyio.NpzFile:
    # Use pkg_resources.files to get a pathlib.Path object
    import importlib.resources as pkg_resources

    path = pkg_resources.files("simpl").joinpath("data/" + name)
    data_npz = np.load(path)
    return data_npz


def print_data_summary(data: xr.Dataset) -> None:
    """Print a concise summary of the dataset."""
    Y = data.Y.values
    Xb = data.Xb.values
    time = data.time.values
    dt = float(time[1] - time[0])
    T = len(time)
    N_neurons = Y.shape[1]
    D = Xb.shape[1]
    duration = float(time[-1] - time[0]) + dt

    # Per-neuron firing rates
    fr_per_neuron = Y.sum(axis=0) / (T * dt)  # (N_neurons,) Hz
    fr_min = float(np.min(fr_per_neuron))
    fr_q25 = float(np.percentile(fr_per_neuron, 25))
    fr_median = float(np.median(fr_per_neuron))
    fr_q75 = float(np.percentile(fr_per_neuron, 75))
    fr_max = float(np.max(fr_per_neuron))
    fr_mean = float(np.mean(fr_per_neuron))

    # Speed and step size
    step_size = np.linalg.norm(np.diff(Xb, axis=0), axis=1)
    speed = step_size / dt
    mean_speed = float(np.median(speed))
    mean_step = float(np.mean(step_size))

    # Active neurons per bin
    active = (Y > 0).sum(axis=1)
    frac_0 = float(np.mean(active == 0))
    frac_1 = float(np.mean(active == 1))
    frac_2 = float(np.mean(active == 2))
    frac_3plus = float(np.mean(active >= 3))
    max_bar = 30  # max bar width in characters
    max_frac = max(frac_0, frac_1, frac_2, frac_3plus, 1e-10)

    def bar(frac):
        return "=" * max(1, int(frac / max_frac * max_bar))

    # Number of trials
    n_trials = len(data.trial_slices.values)

    print("DATA SUMMARY:")
    print(f"  Neurons:    {N_neurons}")
    print(f"  Dimensions: {D}")
    print(f"  dt:         {dt:.4f} s")
    print(f"  Duration:   {duration:.1f} s ({T} bins)")
    print(f"  Trials:     {n_trials}")
    print(
        f"  Neuron firing rate (Hz): mean {fr_mean:.2f}, "
        f"min {fr_min:.2f}, Q1 {fr_q25:.2f}, "
        f"median {fr_median:.2f}, Q3 {fr_q75:.2f}, max {fr_max:.2f}"
    )
    print(f"  Median speed:     {mean_speed:.3f} m/s")
    print(f"  Mean distance travelled per dt: {mean_step:.4f} m (may guide your choice of dx)")
    print("  Simultaneously active neurons per bin:")
    print(f"    0  {bar(frac_0)} {frac_0:.0%}")
    print(f"    1  {bar(frac_1)} {frac_1:.0%}")
    print(f"    2  {bar(frac_2)} {frac_2:.0%}")
    print(f"    3+ {bar(frac_3plus)} {frac_3plus:.0%}")


def save_results_to_netcdf(results: xr.Dataset, path: str) -> None:
    """
    Save results to a file.
    To make the data netCDF safe, some variables need to be converted.

    """
    results["spike_mask"] = results["spike_mask"].astype("int32")
    # Convert boolean 'reshape' attrs to int (netCDF4 doesn't support bool attrs)
    for var in results.data_vars:
        if "reshape" in results[var].attrs:
            results[var].attrs["reshape"] = int(results[var].attrs["reshape"])
    # Convert trial_slices (list of slice objects) to a serializable 1D array
    # Format: [start0, stop0, start1, stop1, ...] with -1 for None
    if "trial_slices" in results.attrs:
        slices = results.attrs["trial_slices"]
        flat = []
        for s in slices:
            flat.append(s.start if s.start is not None else -1)
            flat.append(s.stop if s.stop is not None else -1)
        results.attrs["trial_slices"] = np.array(flat, dtype=np.int64)
    results.to_netcdf(path)


def load_results(path: str) -> xr.Dataset:
    """
    Load results from a saved file.
    Some variables need to be converted back to their original form.

    See below issues for detail.
    https://github.com/TomGeorge1234/SIMPL/issues/5
    https://github.com/TomGeorge1234/SIMPL/issues/8

    Parameters
    ----------
    path : str
        Path to the saved file.

    Returns
    -------
    xr.Dataset
        Results.
    """
    results = xr.open_dataset(path)
    # Convert int 'reshape' attrs back to bool
    for var in results.data_vars:
        if "reshape" in results[var].attrs:
            results[var].attrs["reshape"] = bool(results[var].attrs["reshape"])

    results["spike_mask"] = results["spike_mask"].astype("bool")

    # Convert trial_slices back from flat 1D array to list of slice objects
    if "trial_slices" in results.attrs:
        arr = results.attrs["trial_slices"]
        results.attrs["trial_slices"] = [
            slice(int(arr[i]) if arr[i] != -1 else None, int(arr[i + 1]) if arr[i + 1] != -1 else None)
            for i in range(0, len(arr), 2)
        ]

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Place-field analysis
# ──────────────────────────────────────────────────────────────────────────────


def analyse_place_fields(
    F: jax.Array,
    N_neurons: int,
    N_PFmax: int,
    D: int,
    xF_shape: tuple,
    xF: jax.Array,
    dt: float,
    bin_size: float,
    n_bins: int,
) -> dict:
    """Analyse tuning curves and return information about place fields.

    Terminology: "field" is the *whole* tuning curve.  "place field" (pf) is the
    portion of the whole tuning curve identified as a particular place field.

    Parameters
    ----------
    F : jnp.ndarray, shape (N_neurons, N_bins)
        The estimated place fields.
    N_neurons : int
        Number of neurons.
    N_PFmax : int
        Maximum number of place fields per neuron (for fixed-shape arrays).
    D : int
        Dimensionality of the latent space.
    xF_shape : tuple
        Shape of the discretised environment grid (e.g. ``(nx, ny)``).
    xF : jnp.ndarray, shape (N_bins, D)
        Flattened discretised environment coordinates.
    dt : float
        Time-step size (seconds).
    bin_size : float
        Spatial bin size of the environment.
    n_bins : int
        Total number of spatial bins.

    Returns
    -------
    dict
        Place-field results dictionary with keys such as
        ``place_field_count``, ``place_field_size``, etc.
    """

    # Initialise arrays
    pf_count = np.zeros((N_neurons))
    pf_size = np.nan * np.ones((N_neurons, N_PFmax))
    pf_position = np.nan * np.ones((N_neurons, N_PFmax, D))
    pf_covariance = np.nan * np.ones((N_neurons, N_PFmax, D, D))
    pf_maxfr = np.nan * np.zeros((N_neurons, N_PFmax))
    pf_edges = np.zeros((N_neurons, *xF_shape))
    pf_roundness = np.nan * np.zeros((N_neurons, N_PFmax))

    # Reshape the fields
    F_fields = F.reshape(N_neurons, *xF_shape)  # reshape F into fields

    # Threshold the fields
    F_1Hz = jnp.where(F_fields > 1.0 * dt, 1, 0)  # threshold at 1Hz

    # Total environment size
    volume_element = bin_size**D
    env_size = n_bins * volume_element

    # For each cell in turn, analyse the place fields
    for n in range(N_neurons):
        # Finds contiguous field areas, O/False is considered background and labelled "0".
        # Doesn't count diagonal pixel-connections as connections
        field = F_fields[n]
        field_thresh = F_1Hz[n]
        putative_pfs, putative_pfs_count = scipy.ndimage.label(field_thresh)
        n_pfs = 0  # some of which won't meet out criteria so we use our own counter
        combined_pf_mask = np.zeros_like(field)
        for f in range(1, min(N_PFmax, putative_pfs_count + 1)):
            pf_mask = jnp.where(putative_pfs == f, 1, 0)
            pf = jnp.where(putative_pfs == f, field, 0)
            # Check the field isn't too large
            size = pf_mask.sum() * volume_element
            if size > (1 / 2) * env_size:
                continue
            # Check max firing rate is over 2Hz
            maxfr = jnp.max(pf)
            if maxfr < 2.0 * dt:
                continue
            # Assuming it's passed these, it's a legit field. Now fit a Gaussian.
            perimeter = bin_size * skimage.measure.perimeter(pf_mask)
            perimeter_dilated = bin_size * skimage.measure.perimeter(scipy.ndimage.binary_dilation(pf_mask))
            perimeter = (perimeter + perimeter_dilated) / 2
            roundness = 4 * np.pi * size / perimeter**2
            combined_pf_mask += pf_mask
            mu, mode, cov = fit_gaussian(xF, pf.flatten())
            pf_size[n, n_pfs] = size
            pf_position[n, n_pfs] = mu
            pf_covariance[n, n_pfs] = cov
            pf_maxfr[n, n_pfs] = maxfr
            pf_roundness[n, n_pfs] = roundness
            n_pfs += 1
        # pad combined_pf_mask with zeros
        is_pf = combined_pf_mask > 0
        pf_edges[n] = scipy.ndimage.binary_dilation(is_pf) ^ is_pf
        pf_count[n] = n_pfs

    place_field_results = {
        "place_field_count": jnp.array(pf_count),
        "place_field_size": jnp.array(pf_size),
        "place_field_position": jnp.array(pf_position),
        "place_field_covariance": jnp.array(pf_covariance),
        "place_field_max_firing_rate": jnp.array(pf_maxfr),
        "place_field_roundness": jnp.array(pf_roundness),
        "place_field_outlines": jnp.array(pf_edges),
    }

    return place_field_results


def calculate_spatial_information(
    r: jax.Array,
    PX: jax.Array,
) -> jax.Array:
    """Calculate Skaggs spatial information per neuron (bits/s).

    Parameters
    ----------
    r : jax.Array (N_neurons, N_bins)
        Firing rate maps in Hz (spikes per second, not per bin).
    PX : jax.Array (N_bins,)
        Occupancy probability over spatial bins (sums to 1).

    Returns
    -------
    spatial_info : jax.Array (N_neurons,)
        Spatial information per neuron in bits/s.
    """
    r_mean = jnp.sum(r * PX[None, :], axis=1)  # mean firing rate (N_neurons,) Hz
    eps = 1e-10
    ratio = r / (r_mean[:, None] + eps)
    spatial_info = jnp.sum((r * jnp.log2(ratio + eps)) * PX[None, :], axis=1)
    return spatial_info
