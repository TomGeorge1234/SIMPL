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


def coarsen_dt(dataset: xr.Dataset, dt_multiplier: int) -> xr.Dataset:
    """Takes the dataset and reinterpolates the data onto a new time grid dt_new.

    Parameters
    ----------
    dataset : xr.Dataset
        The dataset to be coarsened
    dt_multiplier : int
        The factor by which to coarsen the data

    Returns
    -------
    dataset : xr.Dataset
        The coarsened dataset"""
    dataset = dataset.coarsen(dim={"time": dt_multiplier}).mean()
    dataset["X"] = dataset["X"] * dt_multiplier
    return dataset


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


def prepare_data(
    Y: np.ndarray,
    Xb: np.ndarray,
    time: np.ndarray,
    dims: np.ndarray | None = None,
    neurons: np.ndarray | None = None,
    Xt: np.ndarray | None = None,
    Ft: np.ndarray | None = None,
    Ft_coords_dict: dict | None = None,
    trial_boundaries: np.ndarray | None = None,
) -> xr.Dataset:
    """
    Prepare data for simpl model fitting.

    Parameters
    ----------
    Y : np.ndarray
        Spike data, shape (T, N), where T is number of time points and N is number of neurons.
    Xb : np.ndarray
        Initialisation for the latent variables, shape (T, D), where D is number of dimensions.
    time : np.ndarray
        Time stamps for each time point, shape (T,).
    dims : np.ndarray, optional
        Dimension names, shape (D,).
    neurons : np.ndarray, optional
        Neuron IDs, shape (N,). If not provided, neuron IDs are assumed to be [0, 1, ..., N-1].
    Xt : np.ndarray, optional
        Ground truth latent variables, shape (T, D).
    Ft : np.ndarray, optional
        Tuning curves, shape (N, *Ft_coords_dict.values()).
    Ft_coords_dict : dict, optional
        Dictionary of coordinates for the tuning curves. For example if D=2,
        Ft_coords_dict = {'x': xbins, 'y': ybins} where xbins and ybins are
        the coordinates for the centres of the tuning curve bins.
    trial_boundaries : np.ndarray, optional
        Array of indices where trials start. If provided, each trial will be
        processed independently by the Kalman filter. Shape should be
        (N_trials,) with first element typically 0. For example,
        [0, 1000, 2000] means trials are [0:1000, 1000:2000, 2000:end].
        If None, all data is treated as a single continuous trial, so
        trial_boundaries will be set to [0].

    Returns
    -------
    xr.Dataset
        Data for simpl model fitting. If trial_boundaries was provided,
        stores validated boundaries in trial_boundaries and trial_slices
        for SIMPL to use.
    """

    assert Y.shape[0] == Xb.shape[0]
    assert Y.shape[0] == len(time)
    if Xt is not None:
        assert Y.shape[0] == Xt.shape[0]

    T = Y.shape[0]

    if neurons is None:
        neurons = np.arange(Y.shape[1])
    if dims is None:
        dims = np.arange(Xb.shape[1])

    Y = xr.DataArray(Y, dims=["time", "neuron"], coords={"time": time, "neuron": neurons})
    Xb = xr.DataArray(Xb, dims=["time", "dim"], coords={"time": time, "dim": dims})
    if Xt is not None:
        Xt = xr.DataArray(Xt, dims=["time", "dim"], coords={"time": time, "dim": dims})
    if Ft is not None:
        Ft = xr.DataArray(Ft, dims=["neuron", *Ft_coords_dict.keys()], coords={"neuron": neurons, **Ft_coords_dict})

    data = xr.Dataset({"Y": Y, "Xb": Xb})
    if Xt is not None:
        data["Xt"] = Xt
    if Ft is not None:
        data["Ft"] = Ft

    # Trial boundary handling: validate and store for SIMPL
    if trial_boundaries is None:
        trial_boundaries = [0]  # This makes the whole data a single "trial"

    trial_boundaries = np.array(trial_boundaries)
    assert trial_boundaries[0] == 0, "First trial boundary must be 0"
    assert trial_boundaries[-1] < T, "Last trial boundary must be < T"
    assert np.all(np.diff(trial_boundaries) > 0), "Trial boundaries must be strictly increasing"
    data["trial_boundaries"] = trial_boundaries

    trial_slices = [slice(trial_boundaries[i], trial_boundaries[i + 1]) for i in range(len(trial_boundaries) - 1)]
    trial_slices.append(slice(trial_boundaries[-1], T))
    data["trial_slices"] = trial_slices

    return data


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
