import jax.numpy as jnp
from jax import random
import numpy as np 
import sklearn.cross_decomposition
import xarray as xr 

def gaussian_pdf(x : jnp.ndarray,
                 mu : jnp.ndarray, 
                 sigma : jnp.ndarray,) -> jnp.ndarray:
    """ Calculates the gaussian pdf of a multivariate normal distribution of mean mu and covariance sigma at x

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

def gaussian_norm_const(sigma : jnp.ndarray) -> jnp.ndarray:
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

def fit_gaussian(x, likelihood):
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

def fit_gaussian_vmap(x, likelihood):
    """
    Parameters
    ----------
    x : jnp.ndarray, shape (N_bins,D,)
        The position bins in which the likelihood is calculated
    likelihood : jnp.ndarray, shape (N_t,N_bins,)
        The combined likelihood (not log-likelihood) of the neurons firing at each position bin
    """  
    likelihood_sum = likelihood.sum(axis=-1)
    mu = (likelihood @ x) / likelihood_sum[:,None]
    mode = x[jnp.argmax(likelihood, axis=1)]
    sigma = ((x - mu[:,None]) * likelihood[:,:,None]).transpose(0,2,1) @ (x - mu[:,None]) / likelihood_sum[:,None,None]
    return mu, mode, sigma

def gaussian_sample(key, mu, sigma):
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
    D = mu.shape[0]
    sample = random.multivariate_normal(key, mu, sigma)
    return sample

def coefficient_of_determination(
        X : jnp.ndarray,
        Y : jnp.ndarray,
):
    """Calculates the coefficient of determination (R^2) between X and Y. This reflects the proportion of the variance in Y that is predictable from X.
    
    Parameters
    ----------
    X : jnp.ndarray, shape (N, D)
        The predicted latent positions
    Y : jnp.ndarray, shape (N, D)
        The true latent positions"""
    assert X.shape == Y.shape, "The predicted and true latent positions must have the same shape."
    D = X.shape[1]
    SST = jnp.sum((Y - jnp.mean(Y, axis=0))**2)
    SSR = jnp.sum((Y - X)**2)
    R2 = 1 - SSR / SST
    return R2

def cca(X : jnp.ndarray, Y : jnp.ndarray):
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
    cca.fit(X,Y)
    coef = cca.coef_ # / cca._x_std # this randomly changed at some point 
    intercept = cca.intercept_ - cca._x_mean @ coef.T
    return coef, intercept

def correlation_at_lag(X1, X2, lag : int):
    """Calculates to correlation between X1 and X2[lag:]. If X is D-dimensional, calculates the average correlation across dimensions

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
        X2 = X2[lag:,:]
        X1 = X1[:T-lag]
    else:
        lag = -lag
        X1 = X1[lag:,:]
        X2 = X2[:T-lag]
    return jnp.mean(jnp.diag(jnp.corrcoef(X1.T,X2.T)[D:,:D]))

def coarsen_dt(dataset : xr.Dataset, dt_multiplier : int):
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
    dataset = dataset.coarsen(dim={'time':dt_multiplier}).mean()
    dataset['X'] = dataset['X'] * dt_multiplier
    return dataset

def create_speckled_mask(size, sparsity=0.1, block_size=10):
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
    for row in range(size[1]):
        for block in range(num_blocks_per_row):
            # Randomly choose starting positions within the bounds
            start_idx = np.random.randint(0, size[0] - block_size)
            end_idx = min(start_idx + block_size, size[0])
            mask[start_idx:end_idx, row] = False
    return jnp.array(mask)

def load_datafile(name='gridcelldata.npz'):
    # Use pkg_resources.files to get a pathlib.Path object
    import importlib.resources as pkg_resources
    path = pkg_resources.files('rnem').joinpath('data/'+name)
    data_npz = np.load(path)
    return data_npz


def prepare_data(
        Y : np.ndarray,
        Xb : np.ndarray,
        time : np.ndarray,
        dims : np.ndarray = None,
        neurons : np.ndarray = None,
        Xt : np.ndarray = None,
        Ft : np.ndarray = None,
        Ft_coords_dict : dict = None,
    ) -> xr.Dataset:

    """
    Prepare data for rnem model fitting.

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
        Dictionary of coordinates for the tuning curves. For example if D=2, Ft_coords_dict = {'x': xbins, 'y': ybins} where xbins and ybins are the coordinates for the centres of the tuning curve bins.
    
    Returns
    -------
    xr.Dataset
        Data for rnem model fitting.
    """
    

    assert Y.shape[0] == Xb.shape[0]
    assert Y.shape[0] == len(time)
    if Xt is not None:
        assert Y.shape[0] == Xt.shape[0]

    if neurons is None:
        neurons = np.arange(Y.shape[1])
    if dims is None:
        dims = np.arange(Xb.shape[1])
    
    Y = xr.DataArray(Y, dims=['time', 'neuron'], coords={'time': time, 'neuron': neurons})
    Xb = xr.DataArray(Xb, dims=['time', 'dim'], coords={'time': time, 'dim': dims})
    if Xt is not None:
        Xt = xr.DataArray(Xt, dims=['time', 'dim'], coords={'time': time, 'dim': dims})
    if Ft is not None:
        Ft = xr.DataArray(Ft, dims=['neuron', *Ft_coords_dict.keys()], coords={'neuron': neurons, **Ft_coords_dict})

    data = xr.Dataset({'Y': Y, 'Xb': Xb})
    if Xt is not None:
        data['Xt'] = Xt
    if Ft is not None:
        data['Ft'] = Ft
    
    return data