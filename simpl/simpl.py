# Jax, for the majority of the calculations
import skimage.measure
from jax import jit, vmap
import jax.random as random
import jax.numpy as jnp
import numpy as np 

# Other libraries
import xarray as xr
import scipy
import skimage
import warnings
from typing import Callable
from tqdm import tqdm
import time 

# Internal libraries
import simpl
from simpl.utils import coefficient_of_determination, cca, create_speckled_mask, gaussian_sample, save_results_to_netcdf
from simpl.environment import Environment

# Kalmax package handles the Kalman filtering and KDE
from kalmax.kalman import KalmanFilter
from kalmax.kde import kde, kde_circular1d
from kalmax.kde import poisson_log_likelihood, poisson_log_likelihood_trajectory
from kalmax.utils import fit_gaussian
from kalmax.kernels import gaussian_kernel

class SIMPL:

    def __init__(self,
                # Data 
                data : xr.Dataset,
                # Environment 
                environment : Environment,
                # Model parameters
                kernel : Callable = gaussian_kernel,
                kernel_bandwidth : float = 0.02,
                observation_noise_std : float = 0.00, #TODO probably remove this unused parameter
                speed_prior = 0.1,
                behaviour_prior = None, 
                test_frac : float = 0.1,
                speckle_block_size_seconds : float = 1,
                # Analysis parameters
                manifold_align_against : str = 'behaviour',
                evaluate_each_epoch : bool = True,
                save_likelihood_maps : bool = False,
                resample_spike_mask : bool = False,
                is_circular : bool = False,
                ):
        
        """Initializes the SIMPL class.

        Overview: 
            SIMPL is a class which takes in a data set of spikes and initial latent estimates and iteratively "redecodes" the latent by (i) fitting receptive fields by KDE to the spikes (the "M-step") and (ii) running a Kalman filter on the MLE estimates from the spikes to redecode the latent positions (the "E-step"). This procedure is reminiscent (indeed -- theoretically equivalent to, see paper) of the EM-algorithm for latent variable optimisation. 
            
        Terminology:
            `Y` refers to spike counts, shape (T, N_neurons)
            `X` refers to the latent process (i.e. the position of the agent), shape (T, DIMS) 
            `F` refers to the receptive fields, shape (N_neurons, N_bins)
            `logPYXF` refers to the log-likelihood of the spikes given the position and receptive fields, shape (T, N_bins)
            `mu_s`, `mu_f`, `sigma_s` etc. are the Kalman filtered/smoothed means/covariances of the latent positions.
        
        Results format: 
            Data is stored and returned in one _ginormous_ xarray Dataset, `self.results`. This dataset contains _every_ variable at every epoch include the raw data (`Y`, `X`,`Xt` etc. ), outputs of the E-step and M-step such as receptive fields, likelihood maps, kalman filtered/smoothed means and covariances etc. (`mu_s`, `F` etc.) and evaluation baselines such as R2, X_err, and F_err and metrics (trajectory stability, field stability, field sparsity, spatial information), spike masks (used to isolate test and train splits), and linear rescaling coefficients. xarrays are like numpy arrays which allow you to save data arrays along with their associated coordinates. xarray-datasets are like dictionaries of xarrays, where each xarray is a variable and the dictionary keys are the variable names where many variables can share the same coordinates.
            
            So you can then access (for example) "the smoothed latent position y-coordinate of at time t on the e'th epochs" by calling `self.results['mu_s'].sel(epoch=e, dim='y', time=t)`. Two epochs (-2 and -1) are reserved for special cases ("exact" and "best" models which are calculated using the ground truth data (see `calculate_baselines`).
        
        Key methods include:
        - `train_epoch()` which runs an epoch of the EM algorithm 
        - `_E_step()` which runs the E-step of the EM algorithm
        - `_M_step()` which runs the M-step of the EM algorithm

        
        Parameters
        ----------
        data : xr.Dataset
            The data to be decoded. This should contain the following variables:
            - 'Y' : the spike counts of the neurons at each time step (T x N_neurons)
            - 'Xb' : the position of the agent at each time step (T x DIMS)
            ...and if available, ground truth data...
            - 'Xt' : the ground truth latent positions (T x DIMS)
            - 'Ft' : the ground truth place fields (N_neurons x N_bins)
            Along with the associated coordinates for each of these variables.
            - 'neurons' : the array of neuron indices [0, 1, 2, ...]
            - 'time' : the array of time stamps [0, 0.05, 0.1, ...]
            - 'dim' : the array of dimension names e.g. ['x', 'y'] in 2D space
            - 'x' : the array of x positions [0, 0.1, 0.2, ...]
            - 'y' : the array of y positions [0, 0.1, 0.2, ...] (if 2D space)
        environment : Object
            The environment in which the data can live. This should contain the following attributes (satisfied by the environment.Environment() class):
            - 'D' : the number of dimensions of the environment
            - 'flattened_dicretised_coords' : a list of coordinates for all bins covering in the environment (N_bins, D)
            - 'dim' : the array of dimension names e.g. ['x', 'y'] in 2D space
            - 'coords_dict' : a dictionary mapping dim to their coordinate values
            - `discrete_env_shape` : the shape of the discretised environment e.g. (N_xbins, N_ybins) in 2D space
        kernel : Callable, optional
            The kernel function to use for the KDE, by default gaussian_kernel
        kernel_bandwidth : float, optional
            The bandwidth of the kernel, by default 0.02
        observation_noise_std : float, optional
            A small fixed component added to the observation noise covariance of the Kalman filter. By default 0.00 m. Probably will be deprecated.
        speed_prior : float, optional
            The prior speed of the agent in units of meters per second, by default 0.1 m/s.
        behaviour_prior : Optional, optional
            Prior over how far the latent positions can deviate from the behaviour positions in units of meters, by default None (no prior). This should typically be off, or very large, unless you have good reason to believe the behaviour prior should be enforced strongly. 
        test_frac : float, optional
            The fraction of the data to use for testing, by default 0.1. Testing data is generated using a speckled mask.
        speckle_block_size_seconds : float, optional
            The size of contiguous blocks of False in the speckled mask, by default 1.0 second. 
        manifold_align_against : str, optional
            The variable to align the latent positions against, by default 'behaviour'. This can be 'behaviour' or 'ground_truth' or 'none' (no ma nifold alignment is performed).
        evaluate_each_epoch: bool, optional
            Whether to evaluate the model and save results each epoch (costing extra memory and compute) into the results dataset, by default True. 
            If False, the results can only be saved at the end of the training when self.evaluate_epoch() is manually called. Epoch 0 is also always evaluated.
        save_likelihood_maps : bool, optional
            Whether to save the likelihood maps of the spikes at each time step (these a size env x time so cost a LOT of memory, only save if needed), by default False
        is_circular : bool, optional
            Whether the latent space is circular (e.g. head direction data). If True, a kde_circular1d is used in the M-step, by default False. 
            Currently it only supports 1D circular data, so if True, the environment should have D=1.
            It expects the coordinates of the environment to be in radians and to range from -pi to pi.

        
        """
        # PREPARE THE DATA INTO JAX ARRAYS
        self.data = data.copy() 
        self.Y = jnp.array(data.Y.values) # (T, N_neurons)
        self.Xb = jnp.array(data.Xb.values) # (T, D)
        self.time = jnp.array(data.time.values) # (T,)
        self.neuron = jnp.array(data.neuron.values) # (N_neurons,)
        self.dt = self.time[1] - self.time[0] # time step size
        # INTEGER VARIABLES 
        self.D = data.Xb.shape[1] # number of dimensions of the latent space
        self.T = len(data.time) # number of time steps
        self.N_neurons = data.Y.shape[1]
        self.N_PFmax = 20 # to keep a fixed shape each tuning curve has max possible number of place fields


        # SET UP THE ENVIRONMENT
        self.environment = environment; assert self.D == environment.D, "The environment and data dimensions must match"
        self.xF = jnp.array(environment.flattened_discretised_coords) # (N_bins, D)
        self.xF_shape = environment.discrete_env_shape
        self.N_bins = len(self.xF)
    
        # INITIALSE SOME VARIABLES
        self.lastF, self.lastX = None, None
        self.epoch = -1
        self.evaluate_each_epoch = evaluate_each_epoch
        self.save_likelihood_maps = save_likelihood_maps

        # KERNEL STUFF 
        self.kernel = kernel
        self.kernel_bandwidth = kernel_bandwidth
        
        # CREATE SPIKE MASKS 
        self.resample_spike_mask = resample_spike_mask
        self.test_frac = test_frac
        self.block_size = int(speckle_block_size_seconds / self.dt)
        self.spike_mask = create_speckled_mask(size=(self.T, self.N_neurons), # train/test specle mask
                                               sparsity=test_frac, 
                                               block_size=self.block_size)
        self.odd_minute_mask = jnp.stack([jnp.array(self.time // 60 % 2 == 0)] * self.N_neurons, axis=1) # mask for odd minutes
        self.even_minute_mask = ~self.odd_minute_mask # mask for even minutes

        # INITIALISE THE KALMAN FILTER
        mu0 = self.Xb.mean(axis=0) # initial state estimate (estimate from behaviour)
        sigma0 = (1 / self.T) * (((self.Xb - mu0).T) @ (self.Xb - mu0)) # initial state covariance (estimate from behaviour)
        speed_sigma = speed_prior * self.dt
        behaviour_sigma = behaviour_prior if behaviour_prior is not None else 1e6 # if no behaviour prior, set to a large value (effectively no prior)
        lam = behaviour_sigma**2 / (speed_sigma**2 + behaviour_sigma**2)
        sigma_eff_square = speed_sigma**2 * behaviour_sigma**2 / (speed_sigma**2 + behaviour_sigma**2)

        F = lam * jnp.eye(self.D) # state transition matrix
        B = (1 - lam) * jnp.eye(self.D) # control input matrix
        Q = sigma_eff_square * jnp.eye(self.D) # process noise covariance
        H = jnp.eye(self.D) # observation matrix

        self.R_base = observation_noise_std**2 * jnp.eye(self.D) # base observation noise 
        self.kalman_filter = KalmanFilter(
            dim_Z = self.D, 
            dim_Y = self.D,
            dim_U = self.D,
            mu0 = mu0, 
            sigma0 = sigma0, 
            F = F, 
            B = B, 
            Q = Q,
            H = H, 
            R = None,
            )
        
        # SET UP THE DIMENSIONS AND VARIABLES DICTIONARY
        self.dim = self.environment.dim # as ordered in positon variables X = ['x', 'y', ...]
        self.variable_info_dict = self.init_variables_dict()
        self.N_PFmax = 20 # to keep a fixed shape each tuning curve has max possible number of place fields
        self.coordinates_dict = {
            'neuron' : self.neuron,
            'time' : self.time,
            'dim' : self.dim,
            'dim_' : self.dim, # for covariance matrices, two coords can't be the same 
            **self.environment.coords_dict,
            'place_field':jnp.arange(self.N_PFmax),
        }


        # INITIALISE THE RESULTS DATASET 
        self.results = xr.Dataset(coords={'epoch':jnp.array([],dtype=int)})
        self.results.attrs = { #env meta data in case you need it later
            'env_extent':self.environment.extent, 
            'env_pad':self.environment.pad, 
            'env_bin_size':self.environment.bin_size
            }
        self.results = xr.merge([self.results, self.dict_to_dataset({'Xb':self.Xb, 'Y':self.Y, 'spike_mask':self.spike_mask})]) # add spikes and behaviour to the results
        self.loglikelihoods = xr.Dataset(coords={'epoch':jnp.array([],dtype=int)}) # a smaller dict just to save likelihoods for online evaluation during training

        # ESTABLISH GROUND TRUTH (IF AVAILABLE) 
        self.ground_truth_available = ('Xt' in list(data.keys()))
        self.Ft, self.Xt = None, None
        if 'Xt' in list(self.data.keys()):
            self.Xt = jnp.array(self.data.Xt)
            self.results = xr.merge([self.results, self.dict_to_dataset({'Xt':self.Xt})])
        if 'Ft' in list(self.data.keys()): # interpolate the "true" receptive fields onto the coordinates of the environment
            Ft = self.data.Ft.interp(**self.environment.coords_dict, method='linear', kwargs={"fill_value": "extrapolate"}) * self.dt
            Ft = Ft.transpose('neuron', *self.environment.dim) # make coord order matches those of this class
            self.Ft = jnp.array(Ft.values).reshape(self.N_neurons, self.N_bins) # flatten to shape (N_neurons, N_bins)
            self.Ft = jnp.where(self.Ft < 0, 0, self.Ft) #threshold Ft at 0 just in case they weren't already
            self.results = xr.merge([self.results, self.dict_to_dataset({'Ft':self.Ft})])

        # MANIFOLD ALIGNMENT 
        if manifold_align_against == 'behaviour': self.Xalign = self.Xb
        elif manifold_align_against == 'ground_truth': self.Xalign = self.Xt
        elif manifold_align_against == 'none': self.Xalign = None

        self.is_circular = is_circular
        if is_circular:
            self.kde = kde_circular1d
        else:
            self.kde = kde
    
    def train_N_epochs(self, 
                       N : int=5, 
                       verbose : bool = True):
        """Trains the model for N epochs, allowing for KeyboardInterrupt to stop training early. This function is really just a wrapper on self.train_epoch() which does the hard work and could be looped over manually by the user. Loading bar 
        
        Parameters
        ----------
        N : int
            The number of epochs to train for.
        verbose : bool, optional
            Whether to print a loading bar and the training progress, by default True.
        """


        pbar = tqdm(range(self.epoch, self.epoch+N)) if verbose else range(self.epoch, self.epoch+N)
        self._set_pbar_desc(pbar)
        for epoch in pbar:
            try:
                self.train_epoch()
                self._set_pbar_desc(pbar)
            except KeyboardInterrupt:
                print(f"Training interrupted after {self.epoch} epochs.")
                break
        if self.evaluate_each_epoch == False: self.evaluate_epoch() # Always evaluate at the end of training if not done each epoch

        return
    
    def train_epoch(self,):
        """Runs an epoch of the EM algorithm.   
            1. INCREMENT: The epoch counter is incremented.    
            2. E-STEP: The E-step is performed by running the Kalman decder on the previous epoch's place fields.
            2.1. TRANSFORM: A linear transformation is applied to the latent positions so they maximally correlate with the behaviour.
            3. M_STEP: The M-step is performed by fitting the place fields to the new latent positions.
            4. EVALUATE: The R2, X_err, and F_err metrics are calculated between the true and estimated latent positions and place fields (if available).
            5. STORE: The results are converted to xarrays and concatenated to the results dataset.

        """
        # =========== INCREMENT EPOCH ===========
        self.epoch += 1
        if self.resample_spike_mask and self.epoch > 0:
            self.spike_mask = create_speckled_mask(size=(self.T, self.N_neurons), # train/test specle mask
                                                   sparsity=self.test_frac, 
                                                   block_size=self.block_size)

        # =========== E-STEP ===========
        if self.epoch == 0: self.E = {'X':self.Xb}
        else: self.E = self._E_step(Y=self.Y, F=self.lastF)
        
        # =========== M-STEP ===========
        X = self.E['X']
        self.M = self._M_step(Y=self.Y, X=X)
    
        # =========== EVALUATE AND SAVE RESULTS ===========
        if self.evaluate_each_epoch or self.epoch == 0: 
            self.evaluate_epoch()  # stores ALL metrics and the current trajectory, fields etc. in self.results 
        # Regardless of the above, always save the spike likelihoods
        loglikelihoods = self.dict_to_dataset(self.get_loglikelihoods(Y=self.Y, FX=self.M['FX'])).expand_dims({'epoch':[self.epoch]})
        self.loglikelihoods = xr.concat([self.loglikelihoods, loglikelihoods], dim='epoch', data_vars="minimal")

        # =========== STORE THE RESULTS FOR THE NEXT EPOCH ===========
        self.lastF = self.M['F'] # save the place fields for the next epoch
        self.lastX = self.E['X'] # save the latent positions for the next epoch
        
        return 

    def evaluate_epoch(self):
        """Evaluates the current model (i.e. calculates all the "metrics") and saves the results in the self.results dataset. By default this is done at the end of each epoch but can be turned off, see __init__() and done manually by calling this function after training
        
        Nothing needs to be passed as this function will use the current class attributes as it's data (self.E, self.M, self.epoch etc.)
        """

        evals = {}
        evals = self.get_metrics(
            X = self.E['X'], 
            F = self.M['F'], 
            Y = self.Y, 
            FX = self.M['FX'],
            F_odd_mins = self.M['F_odd_minutes'], 
            F_even_mins = self.M['F_even_minutes'], 
            X_prev=self.lastX,
            F_prev=self.lastF,
            Xt=self.Xt, 
            Ft=self.Ft,
            pos=self.M['pos_density']
            )
        results = self.dict_to_dataset({**self.M, **self.E, **evals}).expand_dims({'epoch':[self.epoch]})
        self.results = xr.concat([self.results, results], dim='epoch', data_vars="minimal")
        return 

    def _E_step(self, Y: jnp.ndarray, F:jnp.array) -> xr.Dataset:
        """E-STEP of the EM algorithm. 
           1. LIKELIHOOD: The log-likelihood maps of the spikes, as a function of position, is calculated for each time step.
           2. FIT GAUSSIANS: Gaussians (mu, mode, sigma), are fitted to the log-likelihoods maps.
           3. KALMAN FILTER: The Kalman filter is run on the modes (i.e. the MLE position) of the likelihoods to calculate the latent positions. The observation noise is the sigma of the Gaussians (wide likelihoods = high noise => weak effect on the latent positions).
           4. KALMAN SMOOTHER: The filtered datapoints are Kalman smoothed. 
           5. LINEAR SCALING: The latent positions are linearly scaled to maximally correlate with the behaviour.
           6. SAMPLE: The posterior of the latent positions is sampled.
           7. EVALUATE: The posterior likelihood of the data (mode observations) under the model is calculated.
           8. STORE: The results are stored in a dictionary.
        
        Parameters
        ----------
        Y : jnp.ndarray, shape (T, N_neurons)
            The spike counts of the neurons at each time step
        F : jnp.ndarray, shape (N_neurons, N_bins)
            The place fields of the neurons (expected no. spikes of each neuron at each position in one time bin)
            
        Returns 
        -------
        E : dict
            The results of the E-step"""

    
        # Batch this 
        logPYXF_maps = poisson_log_likelihood(Y, F, mask=self.spike_mask) # Calc. log-likelihood maps
        no_spikes = (jnp.sum(Y * self.spike_mask, axis=1) == 0)
        # Batch this 
        mu_l, mode_l, sigma_l = vmap(fit_gaussian, in_axes=(None, 0,))(self.xF, jnp.exp(logPYXF_maps)) # fit Gaussians
        
        # Kalman observation noise is base observation noise + the covariance of the likelihoods (artificially inflated when there are no spikes)
        observation_noise = self.R_base + sigma_l
        observation_noise = jnp.where(no_spikes[:,None,None], jnp.eye(self.D)*1e6, observation_noise)

        mu_f, sigma_f = self.kalman_filter.filter(
            Y = mode_l, 
            U = self.Xb,
            R = observation_noise)
        mu_s, sigma_s = self.kalman_filter.smooth(
            mus_f = mu_f, 
            sigmas_f = sigma_f)
        
        # use this E-step to calculate the data likelihood 
        logPYF = self.kalman_filter.loglikelihood(Y=mode_l,R=observation_noise,mu=mu_s,sigma=sigma_s).sum()
        
        # Test: here we ask "does the trajectory decoded from the training spikes have a high likelihood under the testing spikes?"
        logPYXF_maps_test = poisson_log_likelihood(Y, F, mask=~self.spike_mask) # Calc. log-likelihood maps
        no_spikes = (jnp.sum(Y * ~self.spike_mask, axis=1) == 0)
        mu_l_test, mode_l_test, sigma_l_test = vmap(fit_gaussian, in_axes=(None, 0,))(self.xF, jnp.exp(logPYXF_maps_test)) # fit Gaussians
        observation_noise_test = jnp.where(no_spikes[:,None,None], jnp.eye(self.D)*1e6, sigma_l_test) + self.R_base
        logPYF_test = self.kalman_filter.loglikelihood(Y=mode_l_test,R=observation_noise_test,mu=mu_s,sigma=sigma_s).sum()

        X = mu_s
        # By default X, the latent used for the next M-step is just mu_s. However we can optinally also align this latent (wlog) against the behaviour or ground truth using a linear transform.
        align_dict = {}
        if self.Xalign is not None:
            coef, intercept = cca(mu_s, self.Xalign) # linear manifold alignment
            X = mu_s @ coef.T + intercept
            align_dict = {'coef':coef, 'intercept':intercept}

        # make this all into a dictionary
        E = {
            'X': X,
            'mu_l': mu_l,
            'mode_l': mode_l,
            'sigma_l': sigma_l,
            'mu_f': mu_f,
            'sigma_f': sigma_f,
            'mu_s': mu_s,
            'sigma_s': sigma_s,
            'logPYF': logPYF,
            'logPYF_test': logPYF_test,
            **align_dict,
            **({'logPYXF_maps':logPYXF_maps} if self.save_likelihood_maps else {})
            }
    
        return E
    
    def _M_step(self, 
               Y : jnp.ndarray,
               X : jnp.ndarray) -> xr.Dataset:
        """Maximisation step of the EM algorithm. This step calculates the receptive fields of the neurons. F is the probability of the neurons firing at each position in one time step. We calcualte three versions of this, (i) using the full data (training spikes only), (ii) using the odd minutes of the data, and (iii) using the even minutes of the data.

        Parameters
        ----------
        Y : jnp.ndarray, shape (T, N_neurons)
            The spikes of the neurons at each time step
        X : jnp.ndarray, shape (T, D)
            The positions of the agent at each time step
        
        Returns
        -------
        dict : 
            The results of the M-step. This includes 

            - F : jnp.ndarray, shape (N_neurons, N_bins)
                    The place fields of the neurons (probability of the neurons firing at each position in one time step)
            - F_odd_minutes : jnp.ndarray, shape (N_neurons, N_bins)
                    The place fields of the neurons (probability of the neurons firing at each position in one time step) calculated from the odd minutes of the data
            - F_even_minutes : jnp.ndarray, shape (N_neurons, N_bins)
                    The place fields of the neurons (probability of the neurons firing at each position in one time step) calculated from the even minutes of the data
            

        """
        # Takes a mask and returns the receptive fields calcualted using that mask
        t0 = time.time()
        kde_func = lambda mask : self.kde(
            bins = self.xF,
            trajectory = X,
            spikes = Y,
            kernel = self.kernel,
            kernel_bandwidth = self.kernel_bandwidth,
            mask=mask,
            return_position_density=True) # fit place fields
        
        # vmap over the mask input (avoids a lot of redundant computation)
        # TODO It would be cleaner to the odd/even masks in get_metrics() but then I couldn't exploit the vmap. 
        stacked_masks = jnp.array([self.spike_mask, self.odd_minute_mask, self.even_minute_mask])
        all_F, all_pos = vmap(kde_func)(stacked_masks)
        F, F_odd_minutes, F_even_minutes = all_F[0], all_F[1], all_F[2]
        pos = all_pos[0] # (N_neurons, N_bins) Since each neuron has different mask, we need to store all of them.
        # Interpolates the rate maps just calculated onto the latent trajectory to establish a "smoothed" continuous estimate of the firing rates (note using KDE func directly would be too slow here)
        FX = self.interpolate_firing_rates(X, F)
        M = {'F':F,
             'F_odd_minutes':F_odd_minutes,
             'F_even_minutes':F_even_minutes,
             'FX':FX,
             'pos_density': pos
             }
      
        
        
        return M

    def get_loglikelihoods(self, 
                           Y:jnp.ndarray, 
                           FX:jnp.ndarray) -> dict:
        """Calculates the log-likelihoods of the spikes given the firing rates. This is the sum of the log-likelihood of the spikes given the firing rates at each time step. And is normalised per neuron per time step. This uses the Kalmax `poisson_log_likelihood_trajectory` function which takes the spike trains, Y, and an equally shaped array of predicted firing rates, FX, for every neuron at every timestep, and returns the log-likelihood of the spikes given the firing rates. This is then normalised by the number of neurons and time steps (accounting for the mask). 

        LL = sum_t sum_n log(P(Y_tn | X_t, F_n)) / T*N_neurons

        Params
        ------
        Y : jnp.ndarray, shape (T, N_neurons)
            The spike counts of the neurons at each time step
        FX : jnp.ndarray, shape (T, N_neurons)
            The estimated firing rates of the neurons at each time step
        
        Returns
        -------
        dict
            A dictionary containing the log-likelihood of the spikes given the firing rates.
        """
        LLs = {}
        logPYXF = poisson_log_likelihood_trajectory(Y, FX, mask=self.spike_mask).sum() / self.spike_mask.sum()
        logPYXF_test = poisson_log_likelihood_trajectory(Y, FX, mask=~self.spike_mask).sum() / (~self.spike_mask).sum()
        LLs['logPYXF'] = logPYXF
        LLs['logPYXF_test'] = logPYXF_test
        return LLs

    def get_metrics(self, X:jnp.ndarray=None, 
                          F:jnp.ndarray=None, 
                          Y:jnp.ndarray=None, 
                          FX:jnp.ndarray=None, 
                          F_odd_mins:jnp.ndarray=None, 
                          F_even_mins:jnp.ndarray=None, 
                          X_prev:jnp.ndarray=None, 
                          F_prev:jnp.ndarray=None, 
                          Xt:jnp.ndarray=None, 
                          Ft:jnp.ndarray=None,
                          pos:jnp.ndarray=None) -> dict:
        """Calculates important metrics and baselines on the current epochs results. Warning: this is a relaxed function; pass in whatever data you have and it will return whatever metrics it is able to calculate. These are: 
        
        - X_R2 : the R2 between the true and estimated latent positions
        - X_err : the mean position error between the true and estimated latent positions
        - F_err : the mean field error between the true and estimated place fields
        - information : the spatial information of the receptive fields, -sum(F * log(F))
        - sparsity : how sparse are the place fields, defined as the fraction of bins where the firing is greater than 0.1 * max firing rate
        - stability : correlation between receptive fields estimated seperately using spikes from odd and even minutes. 
        - field_count : the number of distinct, stable fields in the place fields
        - field_size : the average size of the fields in the place fields
        - field_change : how much the fields have shifted from the last epoch (if available)
        - trajectory_change : the change in the latent positions from the last epoch (if available)
        - pos_density : the density of the latent trajectory through each bin of the environment (i.e. how much data supports each bin of the place fields)

        Only variables which _can_ be calculated are calculated (i.e. if the true latent positions are not available, the X_R2 metric will not be calculated nor returned in the dictionary).
        
        Parameters
        ----------
        X : jnp.ndarray, shape (T, D)
            The estimated latent positions
        F : jnp.ndarray, shape (N_neurons, N_bins)
            The estimated place fields
        Y : jnp.ndarray, shape (T, N_neurons)
            The spike counts of the neurons at each time step
        FX : jnp.ndarray, shape (T, N_neurons)
            The estimated firing rates of the neurons at each time step
        F_odd_mins : jnp.ndarray, shape (N_neurons, N_bins)
            The estimated place fields from the odd minutes of the data
        F_even_mins : jnp.ndarray, shape (N_neurons, N_bins)
            The estimated place fields from the even minutes of the data
        X_prev : jnp.ndarray, shape (T, D)
            The estimated latent positions from the previous epoch
        F_prev : jnp.ndarray, shape (N_neurons, N_bins)
            The estimated place fields from the previous epoch
        Xt : jnp.ndarray, shape (T, D)
            The true latent positions, defaults to self.Xt
        Ft : jnp.ndarray, shape (N_neurons, N_bins)
            The true place fields, defaults to self.Ft
        
        Returns
        -------
        dict
            A dictionary containing any metrics that could be calculated from the input variables provided.
        """
        metrics = {} 

        # LOGPYXF The log-likelihood of the spikes given the trajectory
        if Y is not None and FX is not None:
            LLs = self.get_loglikelihoods(Y, FX)
            metrics.update(LLs)        

        # R2 between the true and estimated latent positions
        if X is not None and Xt is not None:
            metrics['X_R2'] = coefficient_of_determination(X, Xt)
        
        # MEAN POSITION ERROR between the true and estimated latent positions
        if X is not None and Xt is not None:
            metrics['X_err'] = jnp.mean(jnp.linalg.norm(X - Xt, axis=1))
        
        # MEAN FIELD ERROR between the true and estimated place fields
        if F is not None and Ft is not None:
            metrics['F_err'] = jnp.mean(jnp.linalg.norm(F - Ft, axis=1))

        # NEGATIVE ENTROPY
        if F is not None:
            F_pdf = (F+1e-6) / jnp.sum(F, axis=1)[:,None] # normalise the place fields
            I_F = jnp.sum(F_pdf * jnp.log(F_pdf), axis=1) # negative entropy of the place fields
            metrics['negative_entropy'] = I_F

        # SPARSITY
        if F is not None:
            rho_F = jnp.mean(F < 1.0*self.dt, axis=1) # fraction of bins where the firing is greater less than 1 Hz
            metrics['sparsity'] = rho_F

        # STABILITY
        if F_odd_mins is not None and F_even_mins is not None:
            corr = jnp.corrcoef(F_odd_mins, F_even_mins)
            cross_corr = corr[:self.N_neurons, self.N_neurons:]
            stability = jnp.diag(cross_corr) 
            metrics['stability'] = stability
        
        # PLACE FIELD ANALYSIS (number, size, position and shape)
        if F is not None and self.environment.D==2:
            metrics.update(self.analyse_place_fields(F))
                
        # FIELD CHANGE  
        if F_prev is not None and F is not None:
            delta_F = jnp.linalg.norm(F - F_prev, axis=1)
            metrics['field_change'] = delta_F
        
        # TRAJECTORY CHANGE
        delta_X = None
        if X_prev is not None and X is not None:
            delta_X = jnp.linalg.norm(X - X_prev, axis=1)
            metrics['trajectory_change'] = delta_X

        # SPATIAL INFORMATION
        # Following An Information-Theoretic Approach to Deciphering the Hippocampal Code,
        # the formula to compute the spatial information is as follows:
        # $I=\int_x \lambda(x) \log _2 \frac{\lambda(x)}{\lambda} p(x) d x$
        # where $\lambda(x)$ is the place field of the cell, $\lambda$ is the mean firing rate of the cell,
        # and $p(x)$ is the spatial occupancy.
        # https://proceedings.neurips.cc/paper/1992/file/5dd9db5e033da9c6fb5ba83c7a7ebea9-Paper.pdf
        if F is not None and pos is not None:
            lambda_x = F / self.dt  # (neuron, position_bin)
            p_x = pos  # (neuron, position_bin,)
            p_x = p_x / jnp.sum(p_x, axis=1)[:, None]
            # lambda_ = lambda_x @ p_x  # Mean firing rate (neuron, ) unit : hz
            lambda_ = jnp.sum(lambda_x * p_x, axis=1)  # Mean firing rate (neuron, ) unit : hz
            I_F = jnp.sum((lambda_x * jnp.log2(lambda_x / (lambda_[:, None] + 1e-6) + 1e-6)) * p_x, axis=1)
            I_F = I_F / lambda_  # bits/spike

            assert (
                np.allclose(p_x.sum(axis=1), 1.0)
            ), f"p_x does not sum to 1, sum is {p_x.sum(axis=1)}"  # p_x is a probability distribution
            if lambda_.mean() < 0.01 or lambda_.mean() > 100:
                print(
                    f"Warning: mean firing rate is {lambda_.mean():.4f} Hz, which is outside the expected range. Check if DT is correct."
                )

            metrics['spatial_information'] = I_F

        return metrics
    
    def calculate_baselines(self):
        """There are two particular special models that can/should be used as baselines:
        - Ft ("exact") model : the exact receptive fields are loaded from the data
        - Ft_hat ("best") : e.g. when Ft is 'unknown'. The true latent positions are used to fit the receptive fields using the KDE. 

        These should be similar except that Ft_hat is bottlenecked by amount of data that is available. If the data is too sparse, Ft_hat will be a poor estimate of Ft. Ft_hat therefore represents a more reasonable baseline for the Kalman model which is also bottlenecked by the amount of data available.

        This function fits/sets the place fields for both these models then runs and evaluates and E-step and saves the results in the results dataset under epoch labels -2 ("exact") and -1 ("best"). 
        """
        if self.ground_truth_available == False:
            warnings.warn("Ground truth data not available, so the baselines cannot be calculated.")
            return
        
        if self.Ft is not None:
            # EXACT MODEL: Ft (fit place fields using the exact receptive fields)
            M = {'F':self.Ft, 'FX':self.interpolate_firing_rates(self.Xt, self.Ft)}
            E = self._E_step(self.Y, self.Ft)
            evals = self.get_metrics(
                    X = E['X'],
                    F = self.Ft,
                    Y = self.Y,
                    FX = M['FX'],
                    F_odd_mins = None,
                    F_even_mins = None,
                    X_prev = None,
                    F_prev = None,
                    Xt = self.Xt,
                    Ft = self.Ft,
                    pos=None)
            results = self.dict_to_dataset({**M, **E, **evals}).expand_dims({'epoch':[-2]})
            self.results = xr.concat([self.results, results], dim='epoch', data_vars="minimal")
        else:
            warnings.warn("Exact place fields not provided so baselines against the exact model cannot be calculated.")  

        # BEST MODEL: Ft_hat (fit place fields using the true latent positions)
        M = self._M_step(self.Y, self.Xt)
        E = self._E_step(self.Y, M['F']) 
        evals = self.get_metrics(
                X = E['X'],
                F = M['F'],
                Y = self.Y,
                FX = M['FX'],
                F_odd_mins = M['F_odd_minutes'],
                F_even_mins = M['F_even_minutes'],
                X_prev = None,
                F_prev = None,
                Xt = self.Xt,
                Ft = self.Ft,
                pos = M['pos_density']
                )
        results = self.dict_to_dataset({**M, **E, **evals}).expand_dims({'epoch':[-1]})
        self.results = xr.concat([self.results, results], dim='epoch', data_vars="minimal")

        return 
    
    def interpolate_firing_rates(self, X, F): 
            """If you already have access to the discretised fields you can 'predict' the firing rate at new positions by interpolating the fields onto these positions (avoiding a potentially more expensive full KDE calculation ). This is much faster than the KDE calculation. We use the nearest available position bin for interpolation. TODO: try linear FX = data.F.interp(**coord_args, method='linear', kwargs={"fill_value": 0}).values.T
             
            Parameters
            ----------
            X : jnp.ndarray, shape (T, D)
                The latent positions to want to interpolate onto
            F : jnp.ndarray, shape (N_neurons, N_bins)
                The place fields of the neurons (expected number of spikes in one time steps)
                
                Returns
                -------
            FX : jnp.ndarray, shape (T, N_neurons)
                The firing rates (expected number of spikes in one time step) of the neurons at each position in X"""
            F = np.array(F); X = np.array(X) # only need this for jax>0.4.28 ?!?!
            data = self.dict_to_dataset({'F': F, 'X': X}) # reshape F into the correct shape 
            coord_args = {dim: data.X.sel(dim=dim) for dim in self.dim} # get the coordinates of the latent positions
            FX = data.F.sel(**coord_args, method='nearest').T # interpolate the fields onto the latent positions using xarray.sel()
            return FX.data

    def dict_to_dataset(self, data : dict, coords : dict = None):
        """Converts a dictionary to an xarray Dataset. Loops over any item in the dictionary and converts it to a DataArray then concatenates these in a xr.Dataset. If the data is a scalar, it is converted to a DataArray with no dimensions. 
        
        If the data name isn't recognized it is saved as an array with no dimension or coordinate data. 
        
        Parameters
        ----------
        data : dict
            The dictionary to convert to an xarray Dataset.
        coords : dict
            A dictionary containing the coordinates of the data. If not provided, the coordinates are taken from the self.coordinates_dict. These coords include: 
            - 'neuron' : the array of neuron indices
            - 'time' : the array of time indices
            - 'dim' : the array of dimension names, e.g. ['x', 'y'] in 2D space
            - 'x' : the array of x positions
            - 'y' : the array of y positions (if 2D space)

        Returns
        -------
        xr.Dataset
            The xarray Dataset containing the data in the dictionary"""
        dataset = xr.Dataset()
        if coords is None: coords = self.coordinates_dict
        for variable_name in data.keys():
            if variable_name in self.variable_info_dict:
                variable_info = self.variable_info_dict[variable_name]
                variable_coords = {k:coords[k] for k in variable_info['dims']}
                intended_variable_shape = tuple([len(variable_coords[c]) for c in variable_info['dims']]) # shape implied by the dimensions
                if 'reshape' in variable_info and variable_info['reshape']:
                    variable_data = data[variable_name].reshape(intended_variable_shape) # reshape the data to the intended shape
                else:
                    variable_data = data[variable_name]
                dataarray = xr.DataArray(variable_data, dims=variable_info['dims'], coords=variable_coords, attrs=variable_info)
            else: 
                warnings.warn(f"Variable {variable_name} not recognised, it will be saved without coordinate or dimension info unless you add it to the variable_info_dict in the init_variables_dict method.")
                dataarray = xr.DataArray(data[variable_name])
            dataset[variable_name] = dataarray
        return dataset
    
    def init_variables_dict(self):
        """Initialises a dictionary of variables and their metadata. This dictionary summarises _all_ the variable that are used and returned by the SIMPL class. Each entry is attached as the `attrs` to the DataArray when these variables are saved and is itself a dictionary taking the following form:
        key : dict
            name : str
                The name of the variable
            description : str
                A brief description of the variable   
            dims : list
                The dimensions of the variable (excluding any epoch dimension)
            axis_title : str
                The title of the axis when plotting the variable
            formula : str
                A formula for the variable (if applicable)
            reshape : bool
                Whether the variable should be forcefully reshaped to those of the dimensions given in `dims` when saved. This is useful for variables that are calculated in a different shape to the final intended shape (e.g. receptive fields).
        """
        variable_info_dict = {
            # Core data variables
            'Y': {
                'name':'Spikes',
                'description':'The spikes of the neurons at each time step. Each a binary vector of length N_neurons.',
                'dims':['time', 'neuron'],
                'axis_title':'Spike counts',
                'formula':r'$y(t)$',
            },
            'X': {
                'name':'Latent',
                'description':'The latent positions of the agent at each time step. This is (typically) the smoothed output of the Kalman filter, scaled to correlate maximally with behaviour (X <-- mu_s, X <-- X @ coef.T + intercept). For epoch 0, X == the behaviour.',
                'dims':['time', 'dim'],
                'axis_title':'Position',
                'formula':r'$x(t)$',
            },
            'F': {
                'name':'Model',
                'description':'The receptive fields of the neurons (probability of the neurons firing at each position in one time step (of length dt).',
                'dims':['neuron', *self.dim],
                'axis_title':'Receptive field',
                'reshape':True,
                'formula':r'$r(x)$',
            },
            'F_odd_minutes': {
                'name':'Model (odd minutes)',
                'description':'The receptive fields of the neurons (probability of the neurons firing at each position in one time step) calculated from the odd minutes of the data.',
                'dims':['neuron', *self.dim],
                'axis_title':'Receptive field (odd mins)',
                'Formula':r'$r_{\textrm{odd}}(x)$',
                'reshape':True,
            },
            'F_even_minutes': {
                'name':'Model (even minutes)',
                'description':'The receptive fields of the neurons (probability of the neurons firing at each position in one time step) calculated from the even minutes of the data.',
                'dims':['neuron', *self.dim],
                'axis_title':'Receptive field (even mins)',
                'Formula':r'$r_{\textrm{even}}(x)$',
                'reshape':True,
            },
            'FX': {
                'name':'Smoothed firing rates',
                'description':'An estimate of the firing rate of the neurons at each time step based on the latest position estimates nad their receptive fields.',
                'dims':['time', 'neuron'],
                'axis_title':r'Firing rate',
                'formula':r'$f(t)$',
            },
            # Ground truth data variables
            'Xb': {
                'name':'Position (behaviour)',
                'description':'The position of the agent (i.e. not necessarily the "true" latent position which generated the spikes) at each time step. This is the behaviour of the agent and acts as the starting conditions for the algorithm X[epoch=0] == Xb.',
                'dims':['time', 'dim'],
                'axis_title':'Position (behaviour)',
                'formula':r'$x_{\textrm{beh}}(t)$',
            },
            'Xt': {
                'name':'Latent (ground truth)',
                'description':'The ground truth latent positions of the agent at each time step. If using real neural data, this is typically not available.',
                'dims':['time', 'dim'],
                'axis_title':'Position (true)',
                'formula':r'$x_{\textrm{true}}(t)$',
            },
            'Ft': {
                'name':'Model (ground truth)',
                'description':'The ground truth receptive fields. These are the true receptive fields of the neurons used to generate the data. If using real neural data, this is typically not available.',
                'dims':['neuron', *self.dim],
                'reshape':True,
                'axis_title':'Receptive field (true)',
                'formula':r'$r_{\textrm{true}}(x)$',
            },

            # Likelihood maps and Gaussian posterior parameters
            'logPYXF_maps': {
                'name':'Log-likelihoods',
                'description':'The log-likelihood maps of the spikes, as a function of position, is calculated for each time step.',
                'dims':['time', *self.dim],
                'axis_title':'Log-likelihood map',
                'formula':r'$\log P(Y|x, \Theta)$',
                'reshape':True,
            },
            'mu_l': {
                'name':'Likelihood mean',
                'description':'The mean of the Gaussian fitted to the log-likelihood maps.',
                'dims':['time', 'dim'],
                'axis_title':'Likelihood mean',
                'formula':r'$\mu_l(t)$',
            },
            'mode_l': {
                'name':'Likelihood mode',
                'description':'The mode of the Gaussian fitted to the log-likelihood maps.',
                'dims':['time', 'dim'],
                'axis_title':'Likelihood mode',
                'formula':r'$\textrm{Mo}(t)$',

            },
            'sigma_l': {
                'name':'Likelihood covariance',
                'description':'The covariance matrix of the Gaussian fitted to the log-likelihood maps.',
                'dims':['time', 'dim', 'dim_'],
                'axis_title':'Likelihood covariance',
                'formula':r'$\Sigma_l(t)$',
            },
            'mu_f': {
                'name':'Kalman filtered mean',
                'description':'The mean of the Kalman filtered posterior of the latent positions.',
                'dims':['time', 'dim'],
                'axis_title':'Kalman filtered mean',
                'formula':r'$\mu_f(t)$',
            },
            'sigma_f': {
                'name':'Kalman filtered covariance',
                'description':'The covariance matrix of the Kalman filtered posterior of the latent positions.',
                'dims':['time', 'dim', 'dim_'],
                'axis_title':'Kalman filtered covariance',
                'formula':r'$\Sigma_f(t)$',
            },
            'mu_s': {
                'name':'Kalman smoothed mean',
                'description':'The mean of the Kalman smoothed posterior of the latent positions.',
                'dims':['time', 'dim'],
                'axis_title':'Kalman smoothed mean',
                'formula':r'$\mu_s(t)$',
            },
            'sigma_s': {
                'name':'Kalman smoothed covariance',
                'description':'The covariance matrix of the Kalman smoothed posterior of the latent positions.',
                'dims':['time', 'dim', 'dim_'],
                'axis_title':'Kalman smoothed covariance',
                'formula':r'$\Sigma_s(t)$',
            },
            # Linear scaling parameters
            'coef': {
                'name':'Linear scaling coefficients',
                'description':'The linear scaling matrix coefficients applied, wlog, to the latent positions to maximally correlate it with the behaviour.',
                'dims':['dim', 'dim_'],
                'axis title':'Linear scaling coefficients',
                'formula':r'$\mathbf{M}$',
            },
            'intercept': {
                'name':'Linear scaling intercept',
                'description':'The linear scaling intercept vector applied, wlog, to the latent positions to maximally correlate with the behaviour.',
                'dims':['dim'],
                'axis title':'Linear scaling intercept',
                'formula':r'$\mathbf{c}$',
            },
            # Place field stuff
            'place_field_count': {
                'name' : 'Number of place fields indentified',
                'description' : 'The number of place fields identified per neuron. Place fields are defined by thresholding hte tuning curves (2D only) at 1 Hz and all identifying continuous regions which are both (i) under one-third the full size of the environment and (ii) have a maximum firing rate over 2 Hz.',
                'dims' : ['neuron'],
                'formula':r'$N_{pf}$',
                'axis title':'Number of place fields',
            },
            'place_field_position': {
                'name':'Place field position',
                'description':'The mean of the Gaussian fitted to each place field.',
                'dims':['neuron','place_field','dim'],
                'axis title':'Place field position',
                'formula':r'$\mu_{pf}$',

            },
            'place_field_covariance': {
                'name':'Place field covariance',
                'description':'The covariance of the Gaussian fitted to each place field',
                'dims':['neuron','place_field','dim','dim_'],
                'axis title':'Place field covariance',
                'formula':r'$\Sigma_{pf}$',
            },
            'place_field_size' :{
                'name':'Place field size',
                'description':'The size of the place field (units of m^2)',
                'dims':['neuron','place_field'],
                'axis title':'Place field size',
                'formula':r'$S_{pf}$',
            }, 
            'place_field_outlines' :{
                'name':'Place field outlines',
                'description':'The outlines of the identified place fields (if any)',
                'dims':['neuron',*self.dim],
                'axis title':'Place field outlines',
            },
            'place_field_roundness' :{
                'name':'Place field roundness',
                'description':'The roundness of the identified place fields (if any). defined as r = 4 * pi * area / perimeter^2',
                'dims':['neuron','place_field'],
                'axis title':'Place field roundness',
            },
            'place_field_max_firing_rate' :{
                'name':'Maximum firing rate of the place field',
                'description':'Maximum firing rate -- in units of spikes per time bin rather than Hz -- of the place field',
                'dims':['neuron','place_field'],
                'axis title': 'Place field max. firing rate',
            },
            # Result metrics
            'logPYF': {
                'name':'Total data log-likelihood',
                'description':'Suppose we have two sets of observations, Y and Y_prime. The log-likelihood of the observations := P(Y | Y_prime) can be found by using Y_prime to estimate the latent posterior P(X | Y_prime) (i.e. run the Kalman model) and then marginalise out X from the likelihood of the new observations given the latent: P(Y | Y_prime) = int_X P(Y | X) P(X | Y_prime). This is called the posterior predictive and, for Kalman models, has analytic form  P(Y_i | Y_prime) = Normal(Y_i | Y_hat, S) where S = H @ sigma_i @ H.T + R (the posterior observation covariance combined with the observation noise covariance) and Y_hat = H @ mu_i (the predicted observation), see page 361 of the Advanced Murphy book. Strictly this metric (i) returns not the full distribution (a product of many Gaussians) but the probability density evaluated at the observation locations and (ii) uses the same observations for both sets Y = Y_prime = observations_from_training_spikes. It is a good gauge on the poerfromance of the model "were the observations (~spikes) P(Y|X) likely under the decoded trajectory P(X|Y_prime)?" but note that Y are the observations not the actual spikes so this is not strictly the likelihood of the spikes.',
                'dims':[],
                'axis title':'Data log-likelihood',
                'formula':r'$\log P(Y|Y_{\textrm{train}})$',
            },
            'logPYF_test': {
                'name':'Total data log-likelihood (test)',
                'description':'See logPYF but for observations derived from the testing spikes i.e. logPYF_test = P(Y_test|Y_train) = int_X P(Y_test|X)P(X|Y_train). Its analagous to asking "were the test observations (~test spikes) P(Y_test|X) likely under the decoded trajectory P(X|Y_train)?" or in other words "can the train spikes be used to predict the test spikes?"',
                'dims':[],
                'axis title':'Data log-likelihood (test)',
                'formula':r'$\log P(Y_{\textrm{test}}|Y_{\textrm{train}})$',
            },
            'logPYXF': {
                'name':'Mean spike log-likelihood given trajectory',
                'description':'This is the poisson log-likelihood of the spikes along the trajectory. It is different to the logPYF metric in that it doesnt account for the posterior uncertainty in X and is a direct reading of the spike likelihood rather than the likelihood of the observations of those spikes (likelihood map modes). It is normalised by the number of spike-time bins to make it comparable across test and train datasets.',
                'dims':[],
                'axis title':'Spike log-likelihood',
                'formula':r'$\log P(Y|X(t), \Theta)$',  
            },
            'logPYXF_test': {
                'name':'Mean spike log-likelihood given trajectory (test)',
                'description':'See logPYXF but for observations derived from the testing spikes. This is the poisson log-likelihood of the test spikes along the trajectory derived from the train spikes.',
                'dims':[],
                'axis title':'Spike log-likelihood (test)',
                'formula':r'$\log P(Y_{\textrm{test}}|X(t), \Theta)$',  
            },
            'negative_entropy': {
                'name':'Negative entropy',
                'description':'The negative entropy of each of the receptive fields, -sum(F * log(F)).',
                'dims':['neuron'],
                'axis title':'Negative entropy',
                'formula':r'$- \sum F \log F$',
            },
            'sparsity': {
                'name':'Sparsity',
                'description':'The fraction of bins where the firing is greater than 0.1 * max firing rate.',
                'dims':['neuron'],
                'axis title':'Spatial sparsity',
                'formula':r'$\rho(r(x))$',
            },
            'stability': {
                'name':'Stability',
                'description':'The correlation between receptive fields estimated seperately using spikes from odd and even minutes.',
                'dims':['neuron'],
                'axis title':'Field stability',
                'formula':r'$\textrm{Corr}(r_{\textrm{odd}}(x), r_{\textrm{even}}(x))$',
            },
            'field_count': {
                'name':'Number of fields',
                'description':'A heuristic on the number of distinct fields in the place fields calculated as the number of connected components in the receptive fields thresholded at 0.5 * their maximum firing rate.',
                'dims':['neuron'],
                'axis title':'Number of fields',
                'formula':r'$N_{\textrm{fields}}$',
            },
            'field_size': {
                'name':'Field size',
                'description':'The average size of the fields in the individual fields, as defined in field_count.',
                'dims':['neuron'],
                'axis title':'Field size',
                'formula':r'',
            },

            'field_change': {
                'name':'Field shift',
                'description':'The change in the place fields from the last epoch.',    
                'dims':['neuron'],
                'axis title':'Field change',
                'formula':r'$\|r_{e}(x) - r_{e-1}(x)\|$',
            },
            'trajectory_change': {
                'name':'Latent shift',
                'description':'The average change in the latent positions from the last epoch.',    
                'dims':['time'],
                'axis title':'Latent change',
                'formula':r'$\|x_{e}(t) - x_{e-1}(t)\|$',
            },
            # Baselines : calculated when ground truth data is available
            'X_R2': {
                'name':'Latent R2',
                'description':'The R2 score between the true and estimated latent positions.',
                'dims':[],
                'axis title':'Latent R2',
                'formula':r'$R^2$',
            },
            'X_err': {
                'name':'Latent error',
                'description':'The mean squared error between the true and estimated latent positions.',
                'dims':[],
                'axis title':'Latent error',
                'formula':r'$\|x_{\textrm{true}}(t) - x_{\textrm{est}}(t)\|$',
            },
            'F_err': {
                'name':'Model error',
                'description':'The mean squared error between the true and estimated place fields.',
                'dims':[],
                'axis title':'Receptive field error',
                'formula':r'$\|r_{\textrm{true}}(x) - r_{\textrm{est}}(x)\|$',
            },
            # Other variables
            'spike_mask': {
                'name':'Spike mask',
                'description':'The mask used to determine which spikes are used for training and testing. This is usually a speckled mask (see Williams et al. 2020, fig. 2) of mostly `True` interspersed with contiguous blocks of `False`. Where False, spikes are masked and NOT used for training.',
                'dims':['time', 'neuron'],
                'axis_title':'Spike mask',
                'formula':r'$\textrm{mask}(t, n)$',
            },
            'spatial_information': {
                "axis title": "Spatial Information (bits/spike)",
                "name": "Spatial information",
                "description": "The spatial information of each of the receptive fields, -sum(F * log(F)).",
                "dims": ["neuron"],
            },
            'pos_density': {
                "axis title": "Position density",
                "name": "Position density",
                "description": "The density of the positions in the environment, estimated from the behaviour using a kernel density estimator.",
                "dims": ['neuron', *self.dim],
                'reshape':True,
            },
        }
        return variable_info_dict
    
    def analyse_place_fields(self, F : jnp.array):
        """Analyses the tuning curves and returns a dictionary of information about the number, size, position and shape of place fields (pfs) for each neuron. 
        Terminology: field is the _whole_ tuning curve. place feild (pf) is the portion of the whole tuning curve identified as a particular place field. 
        
        Params
        ------
        F : jnp.ndarray, shape (N_neurons, N_bins)
            The estimated place fields
        
        Returns
        -------
        dict : The place field results dictionary
        """

        # Initialise arrays
        pf_count = np.zeros((self.N_neurons))
        pf_size = np.nan*np.ones((self.N_neurons,self.N_PFmax))
        pf_position = np.nan*np.ones((self.N_neurons,self.N_PFmax,self.D))
        pf_covariance = np.nan*np.ones((self.N_neurons,self.N_PFmax,self.D,self.D))
        pf_maxfr = np.nan*np.zeros((self.N_neurons, self.N_PFmax))
        pf_edges = np.zeros((self.N_neurons, *self.xF_shape))
        pf_roundness = np.nan*np.zeros((self.N_neurons, self.N_PFmax))

        # Reshape the fields 
        F_fields = F.reshape(self.N_neurons, *self.xF_shape) # reshape F into fields 
        
        #Threshold the fields
        F_1Hz = jnp.where(F_fields > 1.0*self.dt, 1, 0) #threshold at 1Hz
        
        # Total environment size 
        dx = self.environment.bin_size
        volume_element = dx**self.D
        env_size = self.environment.flattened_discretised_coords.shape[0]*volume_element
        
        # For each cell in turn, analyse the place fields 
        for n in range(self.N_neurons):
            # Finds contiguous field areas, O/False is considered background and labelled "0". 
            # Doesn't count diagonal pixel-connections as connections
            field = F_fields[n]
            field_thresh = F_1Hz[n]
            putative_pfs, putative_pfs_count = scipy.ndimage.label(field_thresh) 
            n_pfs = 0 # some of which won't meet out criteria so we use our own counter
            combined_pf_mask = np.zeros_like(field)
            for f in range(1,min(self.N_PFmax,putative_pfs_count+1)):
                pf_mask = jnp.where(putative_pfs == f, 1, 0)
                pf = jnp.where(putative_pfs == f, field, 0)
                # Check the field isn't too large 
                size = pf_mask.sum()*volume_element
                if size > (1/2)*env_size: 
                    continue
                # Check max firing rate is over 2Hz 
                maxfr = jnp.max(pf)
                if maxfr < 2.0*self.dt: 
                    continue 
                # Assuming it's passed these, it's a legit field. Now fit a Gaussian.
                perimeter = dx * skimage.measure.perimeter(pf_mask)
                perimeter_dilated = dx * skimage.measure.perimeter(scipy.ndimage.binary_dilation(pf_mask))
                perimeter = (perimeter + perimeter_dilated) / 2
                roundness = 4*np.pi*size / perimeter**2
                combined_pf_mask += pf_mask 
                mu, mode, cov = fit_gaussian(self.xF, pf.flatten())
                pf_size[n,n_pfs] = size
                pf_position[n,n_pfs] = mu
                pf_covariance[n,n_pfs] = cov
                pf_maxfr[n, n_pfs] = maxfr
                pf_roundness[n,n_pfs] = roundness
                n_pfs += 1
            # pad combined_pf_mask with zeros 
            is_pf = combined_pf_mask > 0
            pf_edges[n] = scipy.ndimage.binary_dilation(is_pf) ^ is_pf
            pf_count[n] = n_pfs
            
        place_field_results = {'place_field_count' : jnp.array(pf_count),
                               'place_field_size' : jnp.array(pf_size),
                               'place_field_position' : jnp.array(pf_position),
                               'place_field_covariance' : jnp.array(pf_covariance),
                               'place_field_max_firing_rate' : jnp.array(pf_maxfr),
                               'place_field_roundness' : jnp.array(pf_roundness),
                               'place_field_outlines' : jnp.array(pf_edges)}
        
        return place_field_results
    
    def _set_pbar_desc(self, pbar):
        """Tries to set the description of the progress bar to the current log-likelihoods. If this fails, it just sets the epoch number.
        
        Parameters
        ----------
        pbar : tqdm
            The progress bar to set the description of (can be any iterable, not just a tqdm bar, in which case the description is not set).
            """
        try: 
            likelihood = self.loglikelihoods.logPYXF.sel(epoch=self.epoch).values
            likelihood_test = self.loglikelihoods.logPYXF_test.sel(epoch=self.epoch).values
            likelihood0 = self.loglikelihoods.logPYXF.sel(epoch=0).values
            likelihood_test0 = self.loglikelihoods.logPYXF_test.sel(epoch=0).values
            pbar.set_description(f"Epoch {self.epoch}: Train LL: {likelihood:.3f} (Δ{likelihood-likelihood0:.3f}), Test LL: {likelihood_test:.3f} (Δ{likelihood_test-likelihood_test0:.3f})")
        except: 
            try: pbar.set_description(f"Epoch {max(0,self.epoch)}")
            except: pass

    def save_results(self, path: str):
        """Saves the results of the SIMPL model to a netCDF file at the given path. The results are saved as an xarray Dataset. 

        Parameters
        ----------
        path : str
            The path to save the results to.
        """
        save_results_to_netcdf(self.results, path)
if __name__ == "__main__":
    pass