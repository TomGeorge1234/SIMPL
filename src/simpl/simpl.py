# Jax, for the majority of the calculations
import warnings
import threading
import time as _time

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax import vmap
from tqdm import tqdm

# Internal libraries
from simpl._variable_registry import build_variable_info_dict
from simpl.environment import Environment
from simpl.kalman import KalmanFilter
from simpl.kde import gaussian_kernel, kde, kde_angular, poisson_log_likelihood, poisson_log_likelihood_trajectory
from simpl.utils import (
    print_data_summary,
    analyse_place_fields,
    calculate_spatial_information,
    cca,
    coefficient_of_determination,
    create_speckled_mask,
    fit_gaussian,
    save_results_to_netcdf,
)


class SIMPL:
    def __init__(
        self,
        # Data
        data: xr.Dataset,
        # Environment
        environment: Environment,
        # Model parameters
        kernel_bandwidth: float = 0.02,
        speed_prior: float = 0.1,
        behaviour_prior: float | None = None,
        is_circular: bool = False,
        # Mask and analysis parameters
        test_frac: float = 0.1,
        speckle_block_size_seconds: float = 1,
        resample_spike_mask: bool = False,
        random_seed: int = 0,
        manifold_align_against: str = "behaviour",
        evaluate_each_epoch: bool = True,
        save_likelihood_maps: bool = False,
        verbose: bool = True,
    ) -> None:
        """Initializes the SIMPL class.

        Overview:
            SIMPL takes in a data set of spikes and initial latent estimates and iteratively "redecodes" the
            latent by (i) fitting receptive fields by KDE to the spikes (the "M-step") and (ii) running a
            Kalman filter on the MLE estimates from the spikes to redecode the latent positions (the "E-step").
            This procedure is reminiscent (indeed -- theoretically equivalent to, see paper) of the EM-algorithm
            for latent variable optimisation.

        Terminology:
            `Y` refers to spike counts, shape (T, N_neurons)
            `X` refers to the latent process (i.e. the position of the agent), shape (T, DIMS)
            `F` refers to the receptive fields, shape (N_neurons, N_bins)
            `logPYXF` refers to the log-likelihood of the spikes given the position and receptive fields, (T, N_bins)
            `mu_s`, `mu_f`, `sigma_s` etc. are the Kalman filtered/smoothed means/covariances of the latent positions.

        Results format:
            Data is stored and returned in one _ginormous_ xarray Dataset, `self.results`. This dataset contains
            _every_ variable at every epoch including the raw data (`Y`, `X`, `Xt` etc.), outputs of the E-step
            and M-step such as receptive fields, likelihood maps, kalman filtered/smoothed means and covariances
            etc. (`mu_s`, `F` etc.) and evaluation baselines such as R2, X_err, and F_err and metrics (trajectory
            stability, field stability, field sparsity, spatial information), spike masks (used to isolate test
            and train splits), and linear rescaling coefficients. xarrays are like numpy arrays which allow you
            to save data arrays along with their associated coordinates. xarray-datasets are like dictionaries of
            xarrays, where each xarray is a variable and the dictionary keys are the variable names where many
            variables can share the same coordinates.

            So you can then access (for example) "the smoothed latent position y-coordinate at time t on the
            e'th epoch" by calling `self.results['mu_s'].sel(epoch=e, dim='y', time=t)`. Two epochs (-2 and -1)
            are reserved for special cases ("exact" and "best" models which are calculated using the ground truth
            data (see `calculate_baselines`).

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
            Along with the associated coordinates for each of these variables:
            - 'neurons' : the array of neuron indices [0, 1, 2, ...]
            - 'time' : the array of time stamps [0, 0.05, 0.1, ...]
            - 'dim' : the array of dimension names e.g. ['x', 'y'] in 2D space
            - 'x' : the array of x positions [0, 0.1, 0.2, ...]
            - 'y' : the array of y positions [0, 0.1, 0.2, ...] (if 2D space)
        environment : Object
            The environment in which the data can live. This should contain the following attributes
            (satisfied by the environment.Environment() class):
            - 'D' : the number of dimensions of the environment
            - 'flattened_discretised_coords' : coordinates for all bins in the environment (N_bins, D)
            - 'dim' : the array of dimension names e.g. ['x', 'y'] in 2D space
            - 'coords_dict' : a dictionary mapping dim to their coordinate values
            - 'discrete_env_shape' : the shape of the discretised environment e.g. (N_xbins, N_ybins) in 2D
        kernel_bandwidth : float, optional
            The bandwidth of the kernel in meters for doing kde on the spikes (default a gaussian kernel unless using a
            1Dcircular environment in which case a von Mises kernel is used and the effective units are radians)
            by default 0.02.
        speed_prior : float, optional
            The prior speed of the agent in units of meters per second, by default 0.1 m/s.
        behaviour_prior : Optional, optional
            Prior over how far the latent positions can deviate from the behaviour positions in units of
            meters, by default None (no prior). This should typically be off, or very large, unless you have
            good reason to believe the behaviour prior should be enforced strongly.
        test_frac : float, optional
            The fraction of the data to use for testing, by default 0.1. Testing data is generated using a
            speckled mask.
        is_circular : bool, optional
            Whether the latent space is circular (e.g. head direction data). If True, kde_angular is used in
            the M-step, by default False. Currently only supports 1D circular data, so if True, the environment
            should have D=1. Expects coordinates in radians ranging from -pi to pi.
        speckle_block_size_seconds : float, optional
            The size of contiguous blocks of False in the speckled mask, by default 1.0 second.
        resample_spike_mask : bool, optional
            Whether to resample the speckled mask each epoch, by default False. If True, generates a new random
            speckled mask each epoch, which can help ensure the model is not overfitting to a particular
            train/test split. If False, the same speckled mask is used each epoch.
        random_seed : int, optional
            The random seed for reproducibility of the speckled mask, by default 0. Only used if
            resample_spike_mask is True.
        manifold_align_against : str, optional
            The variable to align the latent positions against, by default 'behaviour'. Can be 'behaviour',
            'ground_truth', or 'none' (no manifold alignment is performed).
        evaluate_each_epoch: bool, optional
            Whether to evaluate the model and save results each epoch (costing extra memory and compute) into
            the results dataset, by default True. If False, results can only be saved at the end of training
            when self.evaluate_epoch() is manually called. Epoch 0 is also always evaluated.
        save_likelihood_maps : bool, optional
            Whether to save the likelihood maps of the spikes at each time step (these are size env x time so
            cost a LOT of memory, only save if needed), by default False.
        """
        # PREPARE THE DATA INTO JAX ARRAYS
        self.data = data.copy()
        self.Y = jnp.array(data.Y.values)  # (T, N_neurons)
        self.Xb = jnp.array(data.Xb.values)  # (T, D)
        self.time = jnp.array(data.time.values)  # (T,)
        self.neuron = jnp.array(data.neuron.values)  # (N_neurons,)
        self.dt = self.time[1] - self.time[0]  # time step size

        # INTEGER VARIABLES
        self.D = data.Xb.shape[1]  # number of dimensions of the latent space
        self.T = len(data.time)  # number of time steps
        self.N_neurons = data.Y.shape[1]
        self.N_PFmax = 20  # to keep a fixed shape each tuning curve has max possible number of place fields

        # SET TRIAL BOUNDARIES (IF GIVEN)
        self.trial_boundaries = np.array(
            [
                0,
            ]
        )
        self.trial_slices = [
            slice(0, self.T),
        ]
        if "trial_boundaries" in data.keys():
            self.trial_boundaries = data.trial_boundaries.values
            self.trial_slices = data.trial_slices.values

        # SET UP THE ENVIRONMENT
        self.environment = environment
        assert self.D == environment.D, "The environment and data dimensions must match"
        self.xF = jnp.array(environment.flattened_discretised_coords)  # (N_bins, D)
        self.xF_shape = environment.discrete_env_shape
        self.N_bins = len(self.xF)

        # INITIALSE SOME VARIABLES
        self.lastF, self.lastX = None, None
        self.epoch = -1
        self.evaluate_each_epoch = evaluate_each_epoch
        self.save_likelihood_maps = save_likelihood_maps

        # KERNEL STUFF
        self.kernel_bandwidth = kernel_bandwidth

        # CREATE SPIKE MASKS
        self.resample_spike_mask = resample_spike_mask
        self.test_frac = test_frac
        self.block_size = int(speckle_block_size_seconds / self.dt)
        self.random_seed = random_seed
        self._seed_seq = np.random.SeedSequence(self.random_seed)
        self.random_seed_epoch_ = self._next_seed() if self.resample_spike_mask else self.random_seed
        self.spike_mask = create_speckled_mask(
            size=(self.T, self.N_neurons),  # train/test specle mask
            sparsity=test_frac,
            block_size=self.block_size,
            random_seed=self.random_seed_epoch_,
        )
        # mask for odd minutes
        self.odd_minute_mask = jnp.stack([jnp.array(self.time // 60 % 2 == 0)] * self.N_neurons, axis=1)
        self.even_minute_mask = ~self.odd_minute_mask  # mask for even minutes

        # INITIALISE THE KALMAN FILTER
        speed_sigma = speed_prior * self.dt
        # if no behaviour prior, set to a large value (effectively no prior)
        behaviour_sigma = behaviour_prior if behaviour_prior is not None else 1e6
        lam = behaviour_sigma**2 / (speed_sigma**2 + behaviour_sigma**2)
        sigma_eff_square = speed_sigma**2 * behaviour_sigma**2 / (speed_sigma**2 + behaviour_sigma**2)

        F = lam * jnp.eye(self.D)  # state transition matrix
        B = (1 - lam) * jnp.eye(self.D)  # control input matrix
        Q = sigma_eff_square * jnp.eye(self.D)  # process noise covariance
        H = jnp.eye(self.D)  # observation matrix

        # Prepare Kalman Filter
        self.kalman_filter = KalmanFilter(
            dim_Z=self.D,
            dim_Y=self.D,
            dim_U=self.D,
            F=F,
            B=B,
            Q=Q,
            H=H,
            R=None,
        )

        # SET UP THE DIMENSIONS AND VARIABLES DICTIONARY
        self.dim = self.environment.dim  # as ordered in positon variables X = ['x', 'y', ...]
        self.variable_info_dict = build_variable_info_dict(self.dim)
        self.N_PFmax = 20  # to keep a fixed shape each tuning curve has max possible number of place fields
        self.coordinates_dict = {
            "neuron": self.neuron,
            "time": self.time,
            "dim": self.dim,
            "dim_": self.dim,  # for covariance matrices, two coords can't be the same
            **self.environment.coords_dict,
            "place_field": jnp.arange(self.N_PFmax),
        }

        # INITIALISE THE RESULTS DATASET
        self.results = xr.Dataset(coords={"epoch": jnp.array([], dtype=int)})
        self.results.attrs = {  # env meta data in case you need it later
            "env_extent": self.environment.extent,
            "env_pad": self.environment.pad,
            "env_bin_size": self.environment.bin_size,
            "trial_boundaries": self.trial_boundaries,
            "trial_slices": self.trial_slices,
        }
        # add spikes and behaviour to the results
        data_dict = {"Xb": self.Xb, "Y": self.Y, "spike_mask": self.spike_mask}
        self.results = xr.merge([self.results, self.dict_to_dataset(data_dict)])
        # a smaller dict just to save likelihoods for online evaluation
        self.loglikelihoods = xr.Dataset(coords={"epoch": jnp.array([], dtype=int)})

        # ESTABLISH GROUND TRUTH (IF AVAILABLE)
        self.ground_truth_available = "Xt" in list(data.keys())
        self.Ft, self.Xt = None, None
        if "Xt" in list(self.data.keys()):
            self.Xt = jnp.array(self.data.Xt)
            self.results = xr.merge([self.results, self.dict_to_dataset({"Xt": self.Xt})])
        # interpolate the "true" receptive fields onto the environment coords
        if "Ft" in list(self.data.keys()):
            Ft = (
                self.data.Ft.interp(
                    **self.environment.coords_dict,
                    method="linear",
                    kwargs={"fill_value": "extrapolate"},
                )
                * self.dt
            )
            Ft = Ft.transpose("neuron", *self.environment.dim)  # make coord order matches those of this class
            self.Ft = jnp.array(Ft.values).reshape(self.N_neurons, self.N_bins)  # flatten to shape (N_neurons, N_bins)
            self.Ft = jnp.where(self.Ft < 0, 0, self.Ft)  # threshold Ft at 0 just in case they weren't already
            self.results = xr.merge([self.results, self.dict_to_dataset({"Ft": self.Ft})])

        # MANIFOLD ALIGNMENT
        if manifold_align_against == "behaviour":
            self.Xalign = self.Xb
        elif manifold_align_against == "ground_truth":
            self.Xalign = self.Xt
        elif manifold_align_against == "none":
            self.Xalign = None

        self.is_circular = is_circular
        if is_circular:
            self.kde = kde_angular
        else:
            self.kde = kde

        # RUN EPOCH 0 (M-step on behavioural trajectory)
        if verbose:
            print_data_summary(data)

        _epoch0_done = threading.Event()

        def _delayed_message():
            if not _epoch0_done.wait(3):
                print("  ...[estimating spatial receptive fields from Xb (epoch 0)]")

        _timer = threading.Thread(target=_delayed_message, daemon=True)
        _timer.start()
        self.train_epoch()
        _epoch0_done.set()

        if verbose:
            si = np.array(self.results.spatial_information.sel(epoch=0))  # bits/s per neuron
            info_rate = float(si.sum())
            si_min = float(si.min())
            si_q25 = float(np.percentile(si, 25))
            si_med = float(np.median(si))
            si_q75 = float(np.percentile(si, 75))
            si_max = float(si.max())
            si_mean = float(si.mean())
            print(
                f"  Spatial info (bits/s per neuron): mean {si_mean:.2f}, "
                f"min {si_min:.2f}, Q1 {si_q25:.2f}, "
                f"median {si_med:.2f}, Q3 {si_q75:.2f}, max {si_max:.2f}"
            )
            print(f"  Spatial info rate (total): {info_rate:.1f} bits/s")
            print()

            # Warnings
            active_per_bin = (np.array(self.Y) > 0).sum(axis=1)
            frac_2plus = float(np.mean(active_per_bin >= 2))
            if frac_2plus < 0.05:
                print(
                    "  WARNING: fewer than 5% of time bins have 2+ active "
                    "neurons. The Poisson likelihood will be weak in most "
                    "bins. Consider coarsening dt or adding more neurons."
                )
            if info_rate < 1.0:
                print(
                    "  WARNING: spatial information rate is very low "
                    f"({info_rate:.1f} bits/s). The neurons may not carry "
                    "enough spatial information for reliable decoding."
                )

    def _next_seed(self) -> int:
        """Spawn a new seed from the seed sequence."""
        return self._seed_seq.spawn(1)[0].generate_state(1)[0]

    def train_N_epochs(self, N: int = 5, verbose: bool = True) -> None:
        """Trains the model for N epochs, allowing for KeyboardInterrupt to stop training early.

        This is really just a wrapper on self.train_epoch() which does the hard work and could be looped
        over manually by the user.

        Parameters
        ----------
        N : int
            The number of epochs to train for.
        verbose : bool, optional
            Whether to print a loading bar and the training progress, by default True.
        """

        pbar = tqdm(range(self.epoch, self.epoch + N)) if verbose else range(self.epoch, self.epoch + N)
        self._set_pbar_desc(pbar)
        for epoch in pbar:
            try:
                self.train_epoch()
                self._set_pbar_desc(pbar)
            except KeyboardInterrupt:
                print(f"Training interrupted after {self.epoch} epochs.")
                break
        if not self.evaluate_each_epoch:
            self.evaluate_epoch()  # Always evaluate at the end if not done each epoch

        return

    def train_epoch(
        self,
    ) -> None:
        """Runs an epoch of the EM algorithm.

        1. INCREMENT: The epoch counter is incremented.
        2. E-STEP: The Kalman decoder is run on the previous epoch's place fields.
        2.1. TRANSFORM: A linear transformation is applied so latent positions maximally correlate with behaviour.
        3. M-STEP: Place fields are fitted to the new latent positions.
        4. EVALUATE: R2, X_err, and F_err metrics are calculated between true and estimated values (if available).
        5. STORE: Results are converted to xarrays and concatenated to the results dataset.
        """
        # =========== INCREMENT EPOCH ===========
        self.epoch += 1
        if self.resample_spike_mask and self.epoch > 0:
            self.spike_mask = create_speckled_mask(
                size=(self.T, self.N_neurons),  # train/test specle mask
                sparsity=self.test_frac,
                block_size=self.block_size,
                random_seed=self._next_seed(),
            )

        # =========== E-STEP ===========
        if self.epoch == 0:
            self.E = {"X": self.Xb}
        else:
            self.E = self._E_step(Y=self.Y, F=self.lastF)

        # =========== M-STEP ===========
        X = self.E["X"]
        self.M = self._M_step(Y=self.Y, X=X)

        # =========== EVALUATE AND SAVE RESULTS ===========
        if self.evaluate_each_epoch or self.epoch == 0:
            self.evaluate_epoch()  # stores ALL metrics and the current trajectory, fields etc. in self.results
        # Regardless of the above, always save the spike likelihoods
        ll_data = self.get_loglikelihoods(Y=self.Y, FX=self.M["FX"])
        loglikelihoods = self.dict_to_dataset(ll_data).expand_dims({"epoch": [self.epoch]})
        self.loglikelihoods = xr.concat(
            [self.loglikelihoods, loglikelihoods],
            dim="epoch",
            data_vars="minimal",
        )

        # =========== STORE THE RESULTS FOR THE NEXT EPOCH ===========
        self.lastF = self.M["F"]  # save the place fields for the next epoch
        self.lastX = self.E["X"]  # save the latent positions for the next epoch

        return

    def evaluate_epoch(self) -> None:
        """Evaluates the current model (i.e. calculates all the "metrics") and saves the results in self.results.

        By default this is done at the end of each epoch but can be turned off (see __init__()) and done
        manually by calling this function after training. Nothing needs to be passed as this function will
        use the current class attributes (self.E, self.M, self.epoch etc.).
        """

        evals = {}
        evals = self.get_metrics(
            X=self.E["X"],
            F=self.M["F"],
            Y=self.Y,
            FX=self.M["FX"],
            F_odd_mins=self.M["F_odd_minutes"],
            F_even_mins=self.M["F_even_minutes"],
            X_prev=self.lastX,
            F_prev=self.lastF,
            Xt=self.Xt,
            Ft=self.Ft,
            PX=self.M["PX"],
        )
        results = self.dict_to_dataset({**self.M, **self.E, **evals}).expand_dims({"epoch": [self.epoch]})
        self.results = xr.concat([self.results, results], dim="epoch", data_vars="minimal")
        return

    def _E_step(self, Y: jax.Array, F: jax.Array) -> dict:
        """E-STEP of the EM algorithm.

        1. LIKELIHOOD: Log-likelihood maps of spikes as a function of position, calculated for each time step.
        2. FIT GAUSSIANS: Gaussians (mu, mode, sigma) are fitted to the log-likelihood maps.
        3. KALMAN FILTER: Run on the modes (MLE positions) of the likelihoods to calculate latent positions.
           The observation noise is the sigma of the Gaussians (wide likelihoods = high noise => weak effect).
        4. KALMAN SMOOTHER: The filtered datapoints are Kalman smoothed.
        5. LINEAR SCALING: Latent positions are linearly scaled to maximally correlate with the behaviour.
        6. SAMPLE: The posterior of the latent positions is sampled.
        7. EVALUATE: The posterior likelihood of the data (mode observations) under the model is calculated.
        8. STORE: The results are stored in a dictionary.

        Parameters
        ----------
        Y : jnp.ndarray, shape (T, N_neurons)
            The spike counts of the neurons at each time step.
        F : jnp.ndarray, shape (N_neurons, N_bins)
            The place fields of the neurons (expected no. spikes of each neuron at each position in one time bin).

        Returns
        -------
        E : dict
            The results of the E-step.
        """

        # Batch this
        # Calc. log-likelihood maps
        logPYXF_maps = poisson_log_likelihood(Y, F, mask=self.spike_mask)
        no_spikes = jnp.sum(Y * self.spike_mask, axis=1) == 0
        # Fit Gaussians
        mu_l, mode_l, sigma_l = vmap(
            fit_gaussian,
            in_axes=(
                None,
                0,
            ),
        )(self.xF, jnp.exp(logPYXF_maps))

        # Kalman observation noise is the covariance of the likelihoods
        # (artificially inflated when there are no spikes)
        observation_noise = jnp.where(
            no_spikes[:, None, None],
            jnp.eye(self.D) * 1e6,
            sigma_l,
        )

        # Process each trial (or full dataset if no trial boundaries are specified)
        mu_f_list, sigma_f_list, mu_s_list, sigma_s_list = [], [], [], []
        logPYF_list, logPYF_test_list = [], []
        for trial_slice in self.trial_slices:
            # Extract data from this slice
            _mode_l = mode_l[trial_slice]
            _Xb = self.Xb[trial_slice]
            _R = observation_noise[trial_slice]
            _Y = Y[trial_slice]
            spike_mask = self.spike_mask[trial_slice]

            # Calculate initial state estimates for this trial
            _mu0 = _Xb.mean(axis=0)
            _sigma0 = (1 / len(_Xb)) * (((_Xb - _mu0).T) @ (_Xb - _mu0))

            # Run filter and smoother
            mu_f, sigma_f = self.kalman_filter.filter(mu0=_mu0, sigma0=_sigma0, Y=_mode_l, U=_Xb, R=_R)
            mu_s, sigma_s = self.kalman_filter.smooth(mus_f=mu_f, sigmas_f=sigma_f)

            # Calculate likelihoods
            logPYF = self.kalman_filter.loglikelihood(Y=_mode_l, R=_R, mu=mu_s, sigma=sigma_s).sum()

            # Test likelihood
            logPYXF_maps_test = poisson_log_likelihood(_Y, F, mask=~spike_mask)
            no_spikes_test = jnp.sum(_Y * ~spike_mask, axis=1) == 0
            _, mode_l_test, sigma_l_test = vmap(
                fit_gaussian,
                in_axes=(
                    None,
                    0,
                ),
            )(self.xF, jnp.exp(logPYXF_maps_test))
            observation_noise_test = jnp.where(
                no_spikes_test[:, None, None],
                jnp.eye(self.D) * 1e6,
                sigma_l_test,
            )
            logPYF_test = self.kalman_filter.loglikelihood(
                Y=mode_l_test,
                R=observation_noise_test,
                mu=mu_s,
                sigma=sigma_s,
            ).sum()

            # Store results
            mu_f_list.append(mu_f)
            sigma_f_list.append(sigma_f)
            mu_s_list.append(mu_s)
            sigma_s_list.append(sigma_s)
            logPYF_list.append(logPYF)
            logPYF_test_list.append(logPYF_test)

        # Concatenate results
        mu_f = jnp.concatenate(mu_f_list, axis=0)
        sigma_f = jnp.concatenate(sigma_f_list, axis=0)
        mu_s = jnp.concatenate(mu_s_list, axis=0)
        sigma_s = jnp.concatenate(sigma_s_list, axis=0)
        logPYF = sum(logPYF_list)
        logPYF_test = sum(logPYF_test_list)

        # Get position from kalman filtering
        X = mu_s

        # By default X, the latent used for the next M-step is just
        # mu_s. However we can optionally also align this latent (wlog)
        # against the behaviour or ground truth using a linear transform.
        align_dict = {}
        if self.Xalign is not None:
            coef, intercept = cca(mu_s, self.Xalign)  # linear manifold alignment
            X = mu_s @ coef.T + intercept
            align_dict = {"coef": coef, "intercept": intercept}

        # make this all into a dictionary
        E = {
            "X": X,
            "mu_l": mu_l,
            "mode_l": mode_l,
            "sigma_l": sigma_l,
            "mu_f": mu_f,
            "sigma_f": sigma_f,
            "mu_s": mu_s,
            "sigma_s": sigma_s,
            "logPYF": logPYF,
            "logPYF_test": logPYF_test,
            **align_dict,
            **({"logPYXF_maps": logPYXF_maps} if self.save_likelihood_maps else {}),
        }

        return E

    def _M_step(self, Y: jax.Array, X: jax.Array) -> dict:
        """Maximisation step of the EM algorithm.

        Calculates the receptive fields of the neurons. F is the probability of the neurons firing at each
        position in one time step. We calculate three versions: (i) using the full data (training spikes only),
        (ii) using the odd minutes, and (iii) using the even minutes.

        Parameters
        ----------
        Y : jnp.ndarray, shape (T, N_neurons)
            The spikes of the neurons at each time step.
        X : jnp.ndarray, shape (T, D)
            The positions of the agent at each time step.

        Returns
        -------
        dict :
            The results of the M-step. This includes:
            - F : jnp.ndarray, shape (N_neurons, N_bins) — place fields (full training data).
            - F_odd_minutes : jnp.ndarray, shape (N_neurons, N_bins) — place fields from odd minutes.
            - F_even_minutes : jnp.ndarray, shape (N_neurons, N_bins) — place fields from even minutes.
        """

        # Takes a mask and returns the receptive fields calculated using that mask
        def kde_func(mask):
            return self.kde(
                bins=self.xF,
                trajectory=X,
                spikes=Y,
                kernel=gaussian_kernel,
                kernel_bandwidth=self.kernel_bandwidth,
                mask=mask,
                return_position_density=True,
            )

        # vmap over the mask input (avoids a lot of redundant computation)
        # TODO It would be cleaner to the odd/even masks in get_metrics() but then I couldn't exploit the vmap.
        stacked_masks = jnp.array([self.spike_mask, self.odd_minute_mask, self.even_minute_mask])
        all_F, all_PX = vmap(kde_func)(stacked_masks)
        F, F_odd_minutes, F_even_minutes = all_F[0], all_F[1], all_F[2]
        PX = all_PX[0]  # (N_bins,) Normalized position density.
        # Interpolates the rate maps just calculated onto the latent
        # trajectory to establish a "smoothed" continuous estimate of
        # the firing rates (note using KDE func directly would be too
        # slow here)
        FX = self.interpolate_firing_rates(X, F)
        M = {"F": F, "F_odd_minutes": F_odd_minutes, "F_even_minutes": F_even_minutes, "FX": FX, "PX": PX}

        return M

    def get_loglikelihoods(self, Y: jax.Array, FX: jax.Array) -> dict:
        """Calculates the log-likelihoods of the spikes given the firing rates.

        This is the sum of the log-likelihood of the spikes given the firing rates at each time step,
        normalised per neuron per time step. Uses the `poisson_log_likelihood_trajectory` function which
        takes the spike trains Y and an equally shaped array of predicted firing rates FX, for every neuron
        at every timestep. The result is normalised by the number of neurons and time steps (accounting for
        the mask).

        LL = sum_t sum_n log(P(Y_tn | X_t, F_n)) / T*N_neurons

        Parameters
        ----------
        Y : jnp.ndarray, shape (T, N_neurons)
            The spike counts of the neurons at each time step.
        FX : jnp.ndarray, shape (T, N_neurons)
            The estimated firing rates of the neurons at each time step.

        Returns
        -------
        dict
            A dictionary containing the log-likelihood of the spikes given the firing rates.
        """
        LLs = {}
        logPYXF = poisson_log_likelihood_trajectory(Y, FX, mask=self.spike_mask).sum() / self.spike_mask.sum()
        logPYXF_test = poisson_log_likelihood_trajectory(Y, FX, mask=~self.spike_mask).sum() / (~self.spike_mask).sum()
        LLs["logPYXF"] = logPYXF
        LLs["logPYXF_test"] = logPYXF_test
        return LLs

    def get_metrics(
        self,
        X: jax.Array | None = None,
        F: jax.Array | None = None,
        Y: jax.Array | None = None,
        FX: jax.Array | None = None,
        F_odd_mins: jax.Array | None = None,
        F_even_mins: jax.Array | None = None,
        X_prev: jax.Array | None = None,
        F_prev: jax.Array | None = None,
        Xt: jax.Array | None = None,
        Ft: jax.Array | None = None,
        PX: jax.Array | None = None,
    ) -> dict:
        """Calculates important metrics and baselines on the current epoch's results.

        Warning: this is a relaxed function; pass in whatever data you have and it will return whatever
        metrics it is able to calculate. These are:

        - X_R2 : R2 between true and estimated latent positions
        - X_err : mean position error between true and estimated latent positions
        - F_err : mean field error between true and estimated place fields
        - information : spatial information of receptive fields, -sum(F * log(F))
        - sparsity : fraction of bins where firing is greater than 0.1 * max firing rate
        - stability : correlation between receptive fields estimated from odd and even minutes
        - field_count : number of distinct, stable fields in the place fields
        - field_size : average size of the fields
        - field_change : how much the fields have shifted from the last epoch (if available)
        - trajectory_change : change in latent positions from the last epoch (if available)
        - PX : density of the latent trajectory through each bin (i.e. how much data supports each bin)

        Only variables which _can_ be calculated are calculated (i.e. if the true latent positions are not
        available, X_R2 will not be calculated nor returned).

        Parameters
        ----------
        X : jnp.ndarray, shape (T, D)
            The estimated latent positions.
        F : jnp.ndarray, shape (N_neurons, N_bins)
            The estimated place fields.
        Y : jnp.ndarray, shape (T, N_neurons)
            The spike counts of the neurons at each time step.
        FX : jnp.ndarray, shape (T, N_neurons)
            The estimated firing rates of the neurons at each time step.
        PX : jnp.ndarray, shape (N_bins,)
            The normalized position density of the latent trajectory through each bin.
        F_odd_mins : jnp.ndarray, shape (N_neurons, N_bins)
            The estimated place fields from the odd minutes of the data.
        F_even_mins : jnp.ndarray, shape (N_neurons, N_bins)
            The estimated place fields from the even minutes of the data.
        X_prev : jnp.ndarray, shape (T, D)
            The estimated latent positions from the previous epoch.
        F_prev : jnp.ndarray, shape (N_neurons, N_bins)
            The estimated place fields from the previous epoch.
        Xt : jnp.ndarray, shape (T, D)
            The true latent positions, defaults to self.Xt.
        Ft : jnp.ndarray, shape (N_neurons, N_bins)
            The true place fields, defaults to self.Ft.

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
            metrics["X_R2"] = coefficient_of_determination(X, Xt)

        # MEAN POSITION ERROR between the true and estimated latent positions
        if X is not None and Xt is not None:
            metrics["X_err"] = jnp.mean(jnp.linalg.norm(X - Xt, axis=1))

        # MEAN FIELD ERROR between the true and estimated place fields
        if F is not None and Ft is not None:
            metrics["F_err"] = jnp.mean(jnp.linalg.norm(F - Ft, axis=1))

        # NEGATIVE ENTROPY
        if F is not None:
            F_pdf = (F + 1e-6) / jnp.sum(F, axis=1)[:, None]  # normalise the place fields
            I_F = jnp.sum(F_pdf * jnp.log(F_pdf), axis=1)  # negative entropy of the place fields
            metrics["negative_entropy"] = I_F

        # SPARSITY
        if F is not None:
            rho_F = jnp.mean(F < 1.0 * self.dt, axis=1)  # fraction of bins where the firing is greater less than 1 Hz
            metrics["sparsity"] = rho_F

        # STABILITY
        if F_odd_mins is not None and F_even_mins is not None:
            corr = jnp.corrcoef(F_odd_mins, F_even_mins)
            cross_corr = corr[: self.N_neurons, self.N_neurons :]
            stability = jnp.diag(cross_corr)
            metrics["stability"] = stability

        # PLACE FIELD ANALYSIS (number, size, position and shape)
        if F is not None and self.environment.D == 2:
            metrics.update(
                analyse_place_fields(
                    F,
                    N_neurons=self.N_neurons,
                    N_PFmax=self.N_PFmax,
                    D=self.D,
                    xF_shape=self.xF_shape,
                    xF=self.xF,
                    dt=self.dt,
                    bin_size=self.environment.bin_size,
                    n_bins=self.N_bins,
                )
            )

        # FIELD CHANGE
        if F_prev is not None and F is not None:
            delta_F = jnp.linalg.norm(F - F_prev, axis=1)
            metrics["field_change"] = delta_F

        # TRAJECTORY CHANGE
        delta_X = None
        if X_prev is not None and X is not None:
            delta_X = jnp.linalg.norm(X - X_prev, axis=1)
            metrics["trajectory_change"] = delta_X

        # SPATIAL INFORMATION
        if F is not None and PX is not None:
            metrics["spatial_information"] = calculate_spatial_information(F / self.dt, PX)
            metrics["spatial_information_rate"] = float(jnp.sum(metrics["spatial_information"]))

        return metrics

    def calculate_baselines(self) -> None:
        """Calculates two special baseline models using ground truth data.

        - Ft ("exact") model: the exact receptive fields are loaded from the data.
        - Ft_hat ("best"): the true latent positions are used to fit receptive fields using KDE.

        These should be similar except that Ft_hat is bottlenecked by the amount of data available. If the
        data is too sparse, Ft_hat will be a poor estimate of Ft. Ft_hat therefore represents a more
        reasonable baseline for the Kalman model which is also data-bottlenecked.

        This function fits/sets the place fields for both models, then runs and evaluates an E-step and saves
        the results in the results dataset under epoch labels -2 ("exact") and -1 ("best").
        """
        if not self.ground_truth_available:
            warnings.warn("Ground truth data not available, so the baselines cannot be calculated.")
            return

        # BEST MODEL: Ft_hat (fit place fields using the true latent positions)
        M_best = self._M_step(self.Y, self.Xt)
        E_best = self._E_step(self.Y, M_best["F"])
        evals_best = self.get_metrics(
            X=E_best["X"],
            F=M_best["F"],
            Y=self.Y,
            FX=M_best["FX"],
            F_odd_mins=M_best["F_odd_minutes"],
            F_even_mins=M_best["F_even_minutes"],
            X_prev=None,
            F_prev=None,
            Xt=self.Xt,
            Ft=self.Ft,
            PX=M_best["PX"],
        )
        results = self.dict_to_dataset({**M_best, **E_best, **evals_best}).expand_dims({"epoch": [-1]})
        self.results = xr.concat([self.results, results], dim="epoch", data_vars="minimal")

        if self.Ft is not None:
            # EXACT MODEL: Ft (fit place fields using the exact receptive fields)
            M_exact = {
                "F": self.Ft,
                "FX": self.interpolate_firing_rates(self.Xt, self.Ft),
                "PX": M_best["PX"],  # the occupancy is the same as the best model.
            }
            E_exact = self._E_step(self.Y, self.Ft)
            evals_exact = self.get_metrics(
                X=E_exact["X"],
                F=self.Ft,
                Y=self.Y,
                FX=M_exact["FX"],
                F_odd_mins=None,
                F_even_mins=None,
                X_prev=None,
                F_prev=None,
                Xt=self.Xt,
                Ft=self.Ft,
                PX=M_exact["PX"],
            )
            results = self.dict_to_dataset({**M_exact, **E_exact, **evals_exact}).expand_dims({"epoch": [-2]})
            self.results = xr.concat([self.results, results], dim="epoch", data_vars="minimal")
            # sort results by epoch so exact comes before best model
            self.results = self.results.sortby("epoch")
        else:
            warnings.warn("Exact place fields not provided so baselines against the exact model cannot be calculated.")

        return

    def interpolate_firing_rates(self, X: jax.Array, F: jax.Array) -> jax.Array:
        """Predict firing rates at new positions by interpolating the discretised fields.

        Much faster than a full KDE calculation. Uses nearest-bin interpolation.

        Parameters
        ----------
        X : jnp.ndarray, shape (T, D)
            The latent positions to interpolate onto.
        F : jnp.ndarray, shape (N_neurons, N_bins)
            The place fields of the neurons (expected number of spikes in one time step).

        Returns
        -------
        FX : jnp.ndarray, shape (T, N_neurons)
            The firing rates (expected number of spikes in one time step) of the neurons at each position in X.
        """
        F = np.array(F)
        X = np.array(X)
        # reshape F into the correct shape
        data = self.dict_to_dataset({"F": F, "X": X})
        # get the coordinates of the latent positions
        coord_args = {dim: data.X.sel(dim=dim) for dim in self.dim}
        # interpolate fields onto latent positions
        FX = data.F.sel(**coord_args, method="nearest").T
        return FX.data

    def dict_to_dataset(self, data: dict, coords: dict | None = None) -> xr.Dataset:
        """Converts a dictionary to an xarray Dataset.

        Loops over any item in the dictionary and converts it to a DataArray then concatenates these in a
        xr.Dataset. If the data is a scalar, it is converted to a DataArray with no dimensions. If the data
        name isn't recognized it is saved as an array with no dimension or coordinate data.

        Parameters
        ----------
        data : dict
            The dictionary to convert to an xarray Dataset.
        coords : dict
            A dictionary containing the coordinates of the data. If not provided, the coordinates are taken
            from self.coordinates_dict. These coords include: 'neuron', 'time', 'dim', 'x', 'y' (if 2D), etc.

        Returns
        -------
        xr.Dataset
            The xarray Dataset containing the data in the dictionary.
        """
        dataset = xr.Dataset()
        if coords is None:
            coords = self.coordinates_dict
        for variable_name in data.keys():
            if variable_name in self.variable_info_dict:
                variable_info = self.variable_info_dict[variable_name]
                variable_coords = {k: coords[k] for k in variable_info["dims"]}
                # shape implied by the dimensions
                intended_variable_shape = tuple([len(variable_coords[c]) for c in variable_info["dims"]])
                if "reshape" in variable_info and variable_info["reshape"]:
                    # reshape the data to the intended shape
                    variable_data = data[variable_name].reshape(intended_variable_shape)
                else:
                    variable_data = data[variable_name]
                dataarray = xr.DataArray(
                    variable_data,
                    dims=variable_info["dims"],
                    coords=variable_coords,
                    attrs=variable_info,
                )
            else:
                warnings.warn(
                    f"Variable {variable_name} not recognised, "
                    f"it will be saved without coordinate or "
                    f"dimension info unless you add it to the "
                    f"variable_info_dict in "
                    f"_variable_registry.py."
                )
                dataarray = xr.DataArray(data[variable_name])
            dataset[variable_name] = dataarray
        return dataset

    def _set_pbar_desc(self, pbar: tqdm | range) -> None:
        """Tries to set the progress bar description to the current log-likelihoods. Falls back to epoch number.

        Parameters
        ----------
        pbar : tqdm
            The progress bar to set the description of (can be any iterable, not just a tqdm bar, in which
            case the description is not set).
        """
        try:
            likelihood = self.loglikelihoods.logPYXF.sel(epoch=self.epoch).values
            likelihood_test = self.loglikelihoods.logPYXF_test.sel(epoch=self.epoch).values
            likelihood0 = self.loglikelihoods.logPYXF.sel(epoch=0).values
            likelihood_test0 = self.loglikelihoods.logPYXF_test.sel(epoch=0).values
            pbar.set_description(
                f"Epoch {self.epoch}: "
                f"Train LL: {likelihood:.3f} "
                f"(\u0394{likelihood - likelihood0:.3f}), "
                f"Test LL: {likelihood_test:.3f} "
                f"(\u0394{likelihood_test - likelihood_test0:.3f})"
            )
        except Exception:
            try:
                pbar.set_description(f"Epoch {max(0, self.epoch)}")
            except Exception:
                pass

    def save_results(self, path: str) -> None:
        """Saves the results of the SIMPL model to a netCDF file at the given path.

        Parameters
        ----------
        path : str
            The path to save the results to.
        """
        save_results_to_netcdf(self.results, path)


if __name__ == "__main__":
    pass
