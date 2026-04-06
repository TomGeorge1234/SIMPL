"""Utility functions used throughout SIMPL and available for downstream analysis.

Gaussian Helpers
    ``gaussian_pdf``, ``log_gaussian_pdf``, ``gaussian_norm_const``

Gaussian Fitting
    ``fit_gaussian``, ``fit_gaussian_legacy``, ``fit_gaussian_vmap``, ``gaussian_sample``

Statistical and Analysis Helpers
    ``coefficient_of_determination``, ``cca``, ``cca_angular``,
    ``correlation_at_lag``

Data Preparation
    ``accumulate_spikes``, ``coarsen_dt``, ``create_speckled_mask``

Data I/O
    ``load_demo_data``, ``print_data_summary``, ``save_results_to_netcdf``,
    ``load_results``, ``rehydrate_model``

Place-Field Analysis
    ``get_field_peaks``, ``analyse_place_fields``,
    ``calculate_spatial_information``, ``calculate_mutual_information``
"""

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
    """Calculates the multivariate Gaussian PDF at x.

    $$\\mathcal{N}(x \\mid \\mu, \\Sigma) = \\frac{1}{\\sqrt{(2\\pi)^D |\\Sigma|}}
    \\exp\\!\\left(-\\frac{1}{2}(x - \\mu)^\\top \\Sigma^{-1} (x - \\mu)\\right)$$

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
    """Calculates the log of the multivariate Gaussian PDF at x.

    $$\\log \\mathcal{N}(x \\mid \\mu, \\Sigma) = -\\frac{D}{2}\\log(2\\pi)
    - \\frac{1}{2}\\log|\\Sigma|
    - \\frac{1}{2}(x - \\mu)^\\top \\Sigma^{-1} (x - \\mu)$$

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
    """Calculates the normalizing constant of a multivariate normal distribution with covariance sigma.

    $$Z = \\frac{1}{\\sqrt{(2\\pi)^D |\\Sigma|}}$$

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


def fit_gaussian_legacy(x: jax.Array, likelihood: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fits a multivariate-Gaussian to the likelihood function P(spikes | x) in x-space.

    Computes the weighted mean and covariance:

    $$\\mu = \\frac{\\sum_i x_i \\, p_i}{\\sum_i p_i}, \\qquad
    \\Sigma = \\frac{\\sum_i (x_i - \\mu)(x_i - \\mu)^\\top p_i}{\\sum_i p_i}$$

    where \\(p_i\\) is the likelihood weight at bin \\(i\\).

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


@jax.jit
def fit_gaussian(x: jax.Array, likelihoods: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fits a multivariate-Gaussian to each of T likelihood distributions over spatial bins.

    For each timestep, computes the weighted mean, mode, and covariance of the
    spatial bin coordinates ``x`` under the likelihood weights:

    $$\\mu_t = \\frac{\\sum_i x_i \\, p_{t,i}}{\\sum_i p_{t,i}}, \\qquad
    \\Sigma_t = \\mathbb{E}_t[x x^\\top] - \\mu_t \\mu_t^\\top$$

    The covariance uses the identity ``Cov = E[xx^T] - mu mu^T``.  This lets us
    precompute ``x x^T`` once as a small ``(N_bins, D, D)`` array and contract it
    with the ``(T, N_bins)`` likelihoods via a single einsum, rather than
    materialising a ``(T, N_bins, D)`` intermediate as the naive formula would.

    The function is JIT-compiled so the XLA computation is traced once and
    reused on subsequent calls.

    Parameters
    ----------
    x : jnp.ndarray, shape (N_bins, D)
        The position bins (shared across all time steps).
    likelihoods : jnp.ndarray, shape (T, N_bins)
        Likelihood values (not log) at each bin for each time step.

    Returns
    -------
    means : jnp.ndarray, shape (T, D)
        The weighted mean position at each time step.
    modes : jnp.ndarray, shape (T, D)
        The bin coordinate with the highest likelihood at each time step.
    covariances : jnp.ndarray, shape (T, D, D)
        The weighted covariance at each time step.
    """
    sums = likelihoods.sum(axis=1)  # (T,)

    # Mean: weighted average via matmul
    mu = (likelihoods @ x) / sums[:, None]  # (T, D)

    # Mode: position of max likelihood
    mode = x[jnp.argmax(likelihoods, axis=1)]  # (T, D)

    # Covariance: E[xx^T] - mu mu^T (avoids (T, N_bins, D) intermediate)
    x_outer = x[:, :, None] * x[:, None, :]  # (N_bins, D, D)
    E_xxT = jnp.einsum("tb,bij->tij", likelihoods, x_outer) / sums[:, None, None]  # (T, D, D)
    cov = E_xxT - mu[:, :, None] * mu[:, None, :]  # (T, D, D)

    return mu, mode, cov


def fit_gaussian_vmap(x: jax.Array, likelihoods: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Fits a multivariate-Gaussian to each row of a batch of likelihood arrays.

    This is the vmapped version of ``fit_gaussian``: it accepts likelihoods of
    shape ``(T, N_bins)`` and returns batched means, modes, and covariances.

    .. deprecated::
        Use :func:`fit_gaussian` instead for better performance.

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
    return jax.vmap(fit_gaussian_legacy, in_axes=(None, 0))(x, likelihoods)


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
    """Calculates the coefficient of determination (\\(R^2\\)) between X and Y.

    This reflects the proportion of the variance in Y that is predictable from X.

    $$R^2 = 1 - \\frac{SS_{\\textrm{res}}}{SS_{\\textrm{tot}}}
    = 1 - \\frac{\\sum_i (Y_i - X_i)^2}{\\sum_i (Y_i - \\bar{Y})^2}$$

    Parameters
    ----------
    X : jnp.ndarray, shape (N, D)
        The predicted latent positions
    Y : jnp.ndarray, shape (N, D)
        The true latent positions

    Returns
    -------
    R2 : jax.Array, scalar
        The coefficient of determination.  1.0 indicates a perfect prediction;
        0.0 indicates the model explains no more variance than the mean of *Y*;
        negative values indicate worse-than-mean predictions."""
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


def cca_angular(
    X: jax.Array,
    Y: jax.Array,
    n_angles: int = 360,
) -> tuple[jax.Array, jax.Array]:
    """Align 1D circular trajectories by a pure rotation (no scaling).

    Searches rotation angles in [-pi, pi) and returns the angle that minimises
    mean squared wrapped angular error. Unlike ``cca``, this only performs a rotation
    (no shift or scaling), which is the correct transform for angular data.

    Parameters
    ----------
    X : jnp.ndarray, shape (N, 1) or (N,)
        Source trajectory in radians.
    Y : jnp.ndarray, shape (N, 1) or (N,)
        Target trajectory in radians.
    n_angles : int, optional
        Number of candidate angles in [-pi, pi), by default 360.

    Returns
    -------
    best_angle : jnp.ndarray, shape ()
        Rotation angle (radians) that minimises circular error.
    best_error : jnp.ndarray, shape ()
        Minimum mean squared wrapped angular error.
    """
    X = jnp.asarray(X).reshape(-1)
    Y = jnp.asarray(Y).reshape(-1)
    assert X.shape == Y.shape, "The predicted and target circular trajectories must have the same shape."

    angles = jnp.linspace(-jnp.pi, jnp.pi, n_angles, endpoint=False)
    diffs = _wrap_minuspi_pi(X[:, None] + angles[None, :] - Y[:, None])
    errs = jnp.mean(diffs**2, axis=0)
    idx = jnp.argmin(errs)
    best_angle = angles[idx]
    return best_angle, errs[idx]


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

    !!! warning

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
        Behavioral positions.
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
        Coarsened behavioral positions (averaged).
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


def train_test_split(
    *arrays: np.ndarray,
    test_seconds: float = 60.0,
    dt: float | None = None,
) -> tuple:
    """Split arrays into train and test sets by holding out the last N seconds.

    If ``dt`` is not provided, it is inferred from the last 1D array in
    ``arrays`` (assumed to be a time-stamps array).

    Parameters
    ----------
    *arrays : np.ndarray
        Arrays to split, all with the same first dimension (T).
    test_seconds : float
        Duration of test set in seconds. Default 60.
    dt : float or None
        Time bin size in seconds. If None, inferred from the last 1D array.

    Returns
    -------
    splits : tuple
        ``(train_1, test_1, train_2, test_2, ...)`` — alternating train/test
        for each input array.

    Examples
    --------
    >>> Y, Y_test, Xb, Xb_test, time, time_test = train_test_split(
    ...     Y_all, Xb_all, time_all, test_seconds=60
    ... )
    """
    if dt is None:
        # Infer dt from the last 1D array (assumed to be timestamps)
        for arr in reversed(arrays):
            if arr.ndim == 1:
                dt = float(arr[1] - arr[0])
                break
    if dt is None:
        raise ValueError("Could not infer dt. Pass dt= explicitly or include a 1D time array.")

    T = arrays[0].shape[0]
    N_test = int(test_seconds / dt)
    N_train = T - N_test

    result = []
    for arr in arrays:
        result.append(arr[:N_train])
        result.append(arr[N_train:])
    return tuple(result)


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
    if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"size must be a pair of positive integers, got {size}")
    if not 0 <= sparsity <= 1:
        raise ValueError(f"sparsity must be between 0 and 1 (inclusive), got {sparsity}")
    if block_size < 0:
        raise ValueError(f"block_size cannot be negative, got {block_size}")
    if block_size == 0:
        return jnp.ones(size, dtype=bool)
    if block_size >= size[0]:
        raise ValueError(
            f"block_size must be smaller than the time dimension so the mask leaves training data, got {block_size}"
        )

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


def find_time_jumps(
    time: np.ndarray,
    threshold_multiplier: float = 2.0,
) -> np.ndarray:
    """Find indices where the time step jumps significantly.

    Parameters
    ----------
    time : np.ndarray, shape (T,)
        Time stamps.
    threshold_multiplier : float, optional
        A time step is considered a jump if it exceeds
        ``threshold_multiplier * median(dt)``.  Default: 2.0.

    Returns
    -------
    jump_indices : np.ndarray
        Indices where jumps were detected (the last index before each gap).
    """
    dt = np.diff(time)
    median_dt = np.median(dt)
    return np.where(dt > threshold_multiplier * median_dt)[0]


_AVAILABLE_DEMO_DATA = [
    "gridcells_synthetic.npz",
    "placecells_tanni2022.npz",
    "headdirectioncells_vollan2025.npz",
    "somatosensory_chowdhury2020.npz",
]


def load_demo_data(
    name: str = "gridcells_synthetic.npz",
    directory: str | None = None,
    force_download: bool = False,
) -> np.lib.npyio.NpzFile:
    """Load a demo data file, downloading from GitHub releases if not cached.

    Resolution order (skipped when *force_download* is ``True``):

    1. **User-specified directory** — if *directory* is given, look for
       ``<directory>/<name>`` first.
    2. **Local source tree** — ``examples/data/`` relative to the package root
       (available in editable / development installs).
    3. **User cache** — ``~/.simpl/data/``.
    4. **Download** — fetched from the latest GitHub release and saved to the
       user cache for next time.

    Parameters
    ----------
    name : str
        Filename to load (e.g. ``"gridcells_synthetic.npz"``).
    directory : str or None
        Optional directory to search for *name* before the default locations.
    force_download : bool
        If ``True``, skip local/cache lookups and always download from GitHub,
        overwriting any cached copy.

    Returns
    -------
    np.lib.npyio.NpzFile
        The loaded ``.npz`` archive.

    Raises
    ------
    ValueError
        If *name* is not one of the available demo data files.
    """
    from pathlib import Path

    # 1. Check user-specified directory
    if not force_download and directory is not None:
        dir_path = Path(directory) / name
        if dir_path.is_file():
            print(f"Loaded {name} from user directory: {dir_path}")
            return np.load(dir_path)

    # 2. Check local source tree (editable installs)
    if not force_download:
        local_path = Path(__file__).resolve().parent.parent.parent / "examples" / "data" / name
        if local_path.is_file():
            print(f"Loaded {name} from local source tree: {local_path}")
            return np.load(local_path)

    # 3. Check user cache
    cache_dir = Path.home() / ".simpl" / "data"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_path = cache_dir / name

    if not force_download and cached_path.exists():
        print(f"Loaded {name} from cache: {cached_path}")
        return np.load(cached_path)

    # File not found locally — check it's a known release asset before attempting download
    if name not in _AVAILABLE_DEMO_DATA:
        available = ", ".join(f'"{f}"' for f in _AVAILABLE_DEMO_DATA)
        raise FileNotFoundError(
            f'Could not find "{name}" locally and it is not a known release asset. Available for download: {available}'
        )

    # 4. Download from GitHub releases
    import json
    import os
    import sys
    import urllib.request

    def _reporthook(block_num, block_size, total_size):
        if total_size > 0:
            downloaded = block_num * block_size
            pct = min(100, downloaded * 100 // total_size)
            mb_done = downloaded / 1_000_000
            mb_total = total_size / 1_000_000
            print(f"\r  {pct:3d}% ({mb_done:.1f}/{mb_total:.1f} MB)", end="", file=sys.stderr)

    api_url = "https://api.github.com/repos/TomGeorge1234/SIMPL/releases"
    headers = {"Accept": "application/vnd.github+json"}
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(api_url, headers=headers)
    with urllib.request.urlopen(req) as resp:
        releases = json.loads(resp.read())

    download_url = None
    for release in releases:
        for asset in release.get("assets", []):
            if asset["name"] == name:
                download_url = asset["browser_download_url"]
                break
        if download_url:
            break

    if download_url is None:
        raise FileNotFoundError(f'Could not find "{name}" in any GitHub release at {api_url}')

    print(f"Downloading {name} from {download_url} ...", file=sys.stderr)
    try:
        urllib.request.urlretrieve(download_url, cached_path, reporthook=_reporthook)
    except Exception:
        cached_path.unlink(missing_ok=True)
        raise
    print(file=sys.stderr)  # newline after progress
    print(f"Loaded {name} from GitHub (saved to cache: {cached_path})")

    return np.load(cached_path)


def print_data_summary(data: xr.Dataset) -> None:
    """Print a concise summary of an ``xr.Dataset`` loaded via ``load_demo_data``.

    Prints the number of neurons, time bins, dimensionality, recording
    duration, time-bin width, firing-rate statistics (min / Q25 / median /
    Q75 / max / mean), median speed, mean step size, and the fraction of
    time bins with 0, 1, or 2+ simultaneously active neurons.

    Parameters
    ----------
    data : xr.Dataset
        Dataset containing at least ``Y`` (spike counts), ``Xb``
        (behavioural positions), and a ``time`` coordinate."""
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
    n_trials = len(data.attrs.get("trial_boundaries", [0]))

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
    """Save a SIMPL results ``xr.Dataset`` to a netCDF file.

    Before writing, the function performs several type conversions required by
    the netCDF4 format: boolean arrays (e.g. ``spike_mask``) are cast to
    ``int32``, boolean ``attrs`` are cast to ``int``, and ``trial_slices``
    (a list of Python ``slice`` objects) is serialised to a flat ``int64``
    array.  Use ``load_results`` to reload and automatically reverse
    these conversions.

    Parameters
    ----------
    results : xr.Dataset
        The results dataset (typically ``model.results_``).
    path : str
        Destination file path (e.g. ``'results.nc'``)."""
    results_to_save = results.copy(deep=True)
    if "spike_mask" in results_to_save:
        results_to_save["spike_mask"] = results_to_save["spike_mask"].astype("int32")
    # Convert boolean 'reshape' attrs to int (netCDF4 doesn't support bool attrs)
    for var in results_to_save.data_vars:
        if "reshape" in results_to_save[var].attrs:
            results_to_save[var].attrs["reshape"] = int(results_to_save[var].attrs["reshape"])
    results_to_save.to_netcdf(path)


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
    results = xr.load_dataset(path)
    # Convert int 'reshape' attrs back to bool
    for var in results.data_vars:
        if "reshape" in results[var].attrs:
            results[var].attrs["reshape"] = bool(results[var].attrs["reshape"])

    if "spike_mask" in results:
        results["spike_mask"] = results["spike_mask"].astype("bool")

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Results loading helpers
# ──────────────────────────────────────────────────────────────────────────────


def last_training_iteration(results: xr.Dataset) -> int:
    """Return the last non-negative iteration label in *results*."""
    iterations = np.asarray(results.coords["iteration"].values, dtype=int)
    nonneg = iterations[iterations >= 0]
    return int(nonneg.max()) if len(nonneg) > 0 else int(iterations.max())


def loglikelihoods_from_results(results: xr.Dataset) -> xr.Dataset:
    """Extract log-likelihood variables from *results* as a standalone Dataset."""
    ll_vars = [v for v in ("logPYXF", "logPYXF_val", "bits_per_spike", "bits_per_spike_val") if v in results]
    if not ll_vars:
        return xr.Dataset(coords={"iteration": results.coords["iteration"]})
    return results[ll_vars].copy(deep=True)


def restore_E_step_state(results: xr.Dataset, iteration: int, device, T: int, D: int) -> dict:
    """Restore E-step state dict from *results* at the given iteration."""
    import jax
    import jax.numpy as jnp

    e_state = {}
    for var in ("X", "mu_l", "mode_l", "sigma_l", "mu_f", "sigma_f", "mu_s", "sigma_s", "coef", "intercept"):
        if var not in results:
            continue
        values = (
            results[var].sel(iteration=iteration).values if "iteration" in results[var].dims else results[var].values
        )
        if np.all(np.isnan(values)):
            continue
        e_state[var] = jax.device_put(jnp.array(values), device)
    return e_state


def restore_M_step_state(results: xr.Dataset, iteration: int, n_neurons: int, n_bins: int, device) -> dict:
    """Restore M-step state dict from *results* at the given iteration."""
    import jax
    import jax.numpy as jnp

    m_state = {}
    for var in ("F", "F_odd_minutes", "F_even_minutes", "PX"):
        if var not in results:
            continue
        values = (
            results[var].sel(iteration=iteration).values if "iteration" in results[var].dims else results[var].values
        )
        if var.startswith("F"):
            values = np.asarray(values).reshape(n_neurons, -1)
        m_state[var] = jax.device_put(jnp.array(values), device)
    if "FX" in results:
        m_state["FX"] = jax.device_put(jnp.array(results["FX"].sel(iteration=iteration).values), device)
    elif "FX_last_iteration" in results:
        m_state["FX"] = jax.device_put(jnp.array(results["FX_last_iteration"].values), device)
    return m_state


# ──────────────────────────────────────────────────────────────────────────────
# Place-field analysis
# ──────────────────────────────────────────────────────────────────────────────


def get_field_peaks(F: jax.Array, coords: jax.Array) -> jax.Array:
    """Get argmax spatial position for each neuron's receptive field.

    Parameters
    ----------
    F : jnp.ndarray, shape (N_neurons, N_bins)
        Receptive fields.
    coords : jnp.ndarray, shape (N_bins, D)
        Spatial coordinates for each bin (e.g. ``environment.flattened_discretised_coords``).

    Returns
    -------
    jnp.ndarray, shape (N_neurons, D)
        Peak spatial position for each neuron.
    """
    argmax_bins = np.argmax(F, axis=1)
    return coords[argmax_bins]


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
            mu, mode, cov = fit_gaussian_legacy(xF, pf.flatten())
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

    $$I = \\sum_x r(x) \\log_2 \\frac{r(x)}{\\bar{r}} \\, P(x)$$

    where \\(r(x)\\) is the firing rate at position \\(x\\), \\(\\bar{r}\\) is the mean firing rate,
    and \\(P(x)\\) is the occupancy probability.

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


def calculate_mutual_information(
    F: jax.Array,
    PX: jax.Array,
    dt: float,
) -> jax.Array:
    """Calculate the exact mutual information between spike count and position per neuron.

    Computes I(X; Y) = Σ_x Σ_k P(x) · Poisson(k; λ(x)) · log₂[Poisson(k; λ(x)) / P(k)]

    where λ(x) = F[n, x] is the expected spike count (spikes/bin) at position x,
    and P(k) = Σ_x P(x) · Poisson(k; λ(x)) is the marginal spike count distribution.

    Unlike the Skaggs spatial information (which assumes infinitesimal time bins),
    this accounts for the full discrete spike count distribution.

    Parameters
    ----------
    F : jax.Array, shape (N_neurons, N_bins)
        Firing rate maps in spikes per bin (not Hz).
    PX : jax.Array, shape (N_bins,)
        Occupancy probability over spatial bins (sums to 1).
    dt : float
        Time bin size in seconds, used to convert from bits/bin to bits/s.

    Returns
    -------
    mi : jax.Array, shape (N_neurons,)
        Mutual information per neuron in bits/s.
    """
    # Determine K_max: largest expected count + margin
    max_rate = jnp.max(F)
    K_max = jnp.clip(max_rate + 10 * jnp.sqrt(max_rate + 1), a_min=10, a_max=50).astype(int)
    k = jnp.arange(K_max, dtype=float)  # (K,)

    # Poisson log-probabilities: log P(Y=k | x) for each neuron and bin
    # F: (N, B), k: (K,) -> log_p_yx: (N, B, K)
    lam = F[:, :, None]  # (N, B, 1)
    log_p_yx = k[None, None, :] * jnp.log(lam + 1e-30) - lam - jax.lax.lgamma(k[None, None, :] + 1)

    # Marginal: P(Y=k) = Σ_x P(x) · P(Y=k|x), per neuron
    # p_yx: (N, B, K), PX: (B,) -> p_y: (N, K)
    p_yx = jnp.exp(log_p_yx)
    p_y = jnp.sum(p_yx * PX[None, :, None], axis=1)  # (N, K)

    # MI = Σ_x Σ_k P(x) · P(Y=k|x) · log2[P(Y=k|x) / P(Y=k)]
    log_ratio = log_p_yx - jnp.log(p_y[:, None, :] + 1e-30)  # (N, B, K)
    mi = jnp.sum(PX[None, :, None] * p_yx * log_ratio, axis=(1, 2)) / jnp.log(2)

    return mi / dt
