# Jax, for the majority of the calculations
import threading
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax import vmap

# Internal libraries
from simpl._variable_registry import build_variable_info_dict
from simpl.environment import Environment
from simpl.kalman import KalmanFilter
from simpl.kde import gaussian_kernel, kde, kde_angular, poisson_log_likelihood, poisson_log_likelihood_trajectory
from simpl.utils import (
    analyse_place_fields,
    calculate_spatial_information,
    cca,
    coefficient_of_determination,
    create_speckled_mask,
    fit_gaussian,
    print_data_summary,
    save_results_to_netcdf,
)


class SIMPL:
    def __init__(
        self,
        # Model hyperparameters
        kernel_bandwidth: float = 0.02,
        speed_prior: float = 0.1,
        use_kalman_smoothing: bool = True,
        behaviour_prior: float | None = None,
        is_circular: bool = False,
        # Environment parameters
        bin_size: float = 0.02,
        env_pad: float = 0.1,
        env_lims: tuple | None = None,
        environment: Environment | None = None,
        # Mask and analysis parameters
        test_frac: float = 0.1,
        speckle_block_size_seconds: float = 1,
        resample_spike_mask: bool = False,
        random_seed: int = 0,
        evaluate_each_epoch: bool = True,
        save_likelihood_maps: bool = False,
        verbose: bool = True,
        # Optional metadata
        dim_names: np.ndarray | None = None,
        neurons: np.ndarray | None = None,
    ) -> None:
        """Initialise the SIMPL model with hyperparameters only (no data, no computation).

        Call ``fit()`` to provide data and train the model. After fitting, optimised latents,
        receptive fields, metrics and other diagnostics can be found in ``self.results_``
        (an ``xarray.Dataset``). Quick access to the final decoded latent and receptive fields
        is available via ``self.X_`` and ``self.F_``. The ``predict()`` method can decode a
        new set of spikes using the fitted receptive fields, without needing behavioural input.

        Overview
        --------
        SIMPL takes in a dataset of spikes ``Y`` and initial latent estimates ``Xb`` and
        iteratively "redecodes" the latent by:

        1. **M-step** — fitting receptive fields (place fields) via kernel density estimation
           (KDE) from the current latent trajectory and spikes.
        2. **E-step** — running a Kalman filter/smoother on the maximum-likelihood estimates
           derived from spikes and the newly fitted receptive fields to produce an updated
           latent trajectory.

        This procedure is reminiscent of (indeed, theoretically equivalent to, see paper) the
        EM algorithm for latent variable optimisation. Epoch 0 runs the M-step only on the
        behavioural initialisation trajectory ``Xb``. Subsequent epochs alternate E-step and
        M-step.

        Terminology
        -----------
        - **Y** : Spike counts, shape (T, N_neurons). Can be binary (0/1) or integer counts (0/1/2/3...).
        - **Xb** : Behavioural initialisation, shape (T, D). The starting estimate of latent positions,
          typically derived from tracked behaviour (e.g. animal position).
        - **Xt** : Ground truth latent positions (if available), shape (T, D). Used for evaluation only.
        - **F** : Receptive fields (place fields), shape (N_neurons, N_bins) — the estimated firing rate of each neuron
          as a function of position, discretised on the environment grid.
        - **FX** : Firing rates along the trajectory, shape (T, N_neurons) — F evaluated at the latent positions X.

        Parameters
        ----------
        kernel_bandwidth : float, optional
            The bandwidth of the Gaussian kernel (in the same units as the latent space, e.g.
            meters) used for KDE when fitting receptive fields. Smaller values give sharper
            fields but are noisier; larger values smooth more. By default 0.02.
        speed_prior : float, optional
            Prior on agent speed in units of meters per second. This controls the strength of
            the Kalman smoother: a low speed prior constrains the decoded trajectory to be
            smooth, while a high value lets the trajectory follow the spike likelihood more
            closely. By default 0.1 m/s.
        use_kalman_smoothing : bool, optional
            Whether to use Kalman smoothing dynamics in the E-step. If False, the speed prior
            is set very high, effectively disabling temporal smoothing and letting the
            trajectory follow the per-bin maximum-likelihood estimate. By default True.
        behaviour_prior : float or None, optional
            Prior on how far the latent positions can deviate from the behavioural positions,
            in units of meters. This acts as a soft constraint pulling the decoded trajectory
            towards ``Xb``. None means no prior (the latent is free to move anywhere).
            By default None.
        is_circular : bool, optional
            Whether the latent space is circular (e.g. head direction data in radians). If
            True, angular KDE is used and the space is assumed to be in [-pi, pi].
            By default False.
        bin_size : float, optional
            Spatial bin size for discretising the environment, in the same units as the latent
            space. Controls the resolution of the receptive field grid. Smaller bins give
            higher resolution but increase computation and memory. By default 0.02.
        env_pad : float, optional
            Padding added outside the data bounds when constructing the environment grid. This
            ensures that receptive fields near the boundary of the explored space are not
            clipped. In the same units as the latent space. By default 0.1.
        env_lims : tuple or None, optional
            Force the environment limits to specific values instead of inferring them from the
            data. Format: ``((min_dim1, min_dim2, ...), (max_dim1, max_dim2, ...))``.
            By default None (auto-inferred from ``Xb``).
        environment : Environment or None, optional
            A pre-built ``Environment`` instance for power users. If provided, ``bin_size``,
            ``env_pad``, and ``env_lims`` are all ignored. By default None.
        test_frac : float, optional
            Fraction of spike observations held out for testing, implemented via a speckled
            (block-structured) mask. Used to compute held-out log-likelihood for monitoring
            overfitting. By default 0.1.
        speckle_block_size_seconds : float, optional
            Temporal size (in seconds) of contiguous blocks in the speckled test mask. Larger
            blocks give more temporally coherent held-out segments. By default 1.0.
        resample_spike_mask : bool, optional
            Whether to resample the speckled test mask at each epoch. If False, the same mask
            is used throughout training. By default False.
        random_seed : int, optional
            Random seed for reproducibility (controls the spike mask generation).
            By default 0.
        evaluate_each_epoch : bool, optional
            Whether to compute and store all metrics (spatial information, stability, place
            field analysis, etc.) at every epoch. If False, metrics are only computed after
            the final epoch. Set to False for faster training when intermediate metrics are
            not needed. By default True.
        save_likelihood_maps : bool, optional
            Whether to save the full log-likelihood maps at each E-step. These are large
            arrays (T x N_bins) and are usually not needed. By default False.
        verbose : bool, optional
            Whether to print progress information (data summary, epoch summaries, warnings)
            during ``fit()``. By default True.
        dim_names : np.ndarray or None, optional
            Names for the latent dimensions, e.g. ``['x', 'y']``. These are used as
            coordinate labels in the results Dataset. If None, auto-generated from the
            Environment (``'x'``, ``'y'``, ``'z'`` for ≤3D; ``'x1'``, ``'x2'``, ... for
            higher). By default None.
        neurons : np.ndarray or None, optional
            Array of neuron identifiers (e.g. cluster IDs). Used as coordinate labels in the
            results Dataset. If None, auto-generated as ``[0, 1, ..., N_neurons-1]``.
            By default None.

        Examples
        --------
        >>> model = SIMPL(speed_prior=0.4, kernel_bandwidth=0.02, bin_size=0.02, env_pad=0.0)
        >>> model.fit(Y, Xb, time, n_epochs=5)
        >>> print(model.X_.shape)  # decoded latent positions
        >>> print(model.F_.shape)  # fitted receptive fields
        """
        # Model hyperparameters
        self.kernel_bandwidth = kernel_bandwidth
        self.speed_prior = speed_prior
        self.use_kalman_smoothing = use_kalman_smoothing
        self.behaviour_prior = behaviour_prior
        self.is_circular = is_circular

        # Environment config
        self.bin_size = bin_size
        self.env_pad = env_pad
        self.env_lims = env_lims
        self._environment_override = environment  # power-user pre-built Environment

        # Mask and analysis parameters
        self.test_frac = test_frac
        self.speckle_block_size_seconds = speckle_block_size_seconds
        self.resample_spike_mask = resample_spike_mask
        self.random_seed = random_seed
        self.evaluate_each_epoch = evaluate_each_epoch
        self.save_likelihood_maps = save_likelihood_maps
        self.verbose = verbose

        # Optional metadata
        self.dim_names = dim_names
        self.neurons = neurons

        # Fitted flag
        self.is_fitted_ = False

    def fit(
        self,
        Y: np.ndarray,
        Xb: np.ndarray,
        time: np.ndarray,
        n_epochs: int = 5,
        trial_boundaries: np.ndarray | None = None,
        align_to_behaviour: bool = True,
        resume: bool = False,
        verbose: bool | None = None,
    ) -> "SIMPL":
        """Fit the SIMPL model to data.

        This is the main entry point for training. It performs the following steps:

        1. **Setup** — validates inputs, creates the ``Environment`` (spatial discretisation
           grid), sets up the Kalman filter, and builds the speckled test mask.
        2. **Epoch 0** — runs the M-step only on the behavioural trajectory ``Xb`` to produce
           the initial receptive fields. Prints a data summary and spatial information
           diagnostics (if ``verbose=True``).
        3. **Epochs 1..n_epochs** — alternates E-step (Kalman decoding using current receptive
           fields) and M-step (KDE re-fitting of receptive fields from updated trajectory).

        After fitting, results are available via:

        - ``self.X_`` — the final decoded latent positions, shape (T, D).
        - ``self.F_`` — the final receptive fields, shape (N_neurons, N_bins).
        - ``self.results_`` — full ``xarray.Dataset`` with all epochs, metrics, and
          intermediates (receptive fields, trajectories, log-likelihoods, spatial information,
          stability, etc.).
        - ``self.loglikelihoods_`` — per-epoch train/test log-likelihoods.

        Parameters
        ----------
        Y : np.ndarray, shape (T, N_neurons)
            Spike counts. Can be binary (0/1) or integer-valued. Each row is one time bin,
            each column is one neuron.
        Xb : np.ndarray, shape (T, D)
            Behavioural initialisation positions. This is the starting estimate of the latent
            trajectory, typically the tracked position of the animal. D is the number of
            latent dimensions (e.g. 2 for 2D position).
        time : np.ndarray, shape (T,)
            Time stamps (in seconds) for each time bin. Must be uniformly spaced. The time
            step ``dt`` is inferred as ``time[1] - time[0]``.
        n_epochs : int, optional
            Number of EM epochs to train after epoch 0. Set to 0 to run only the initial
            M-step (useful for manual epoch control via ``train_epoch()``). By default 5.
        trial_boundaries : np.ndarray or None, optional
            Array of indices where new trials start, e.g. ``[0, 1000, 2000]``. The first
            element must be 0. The Kalman filter runs independently within each trial to
            prevent smoothing across trial boundaries (e.g. between separate recording
            sessions). If None, all data is treated as a single trial. By default None.
        align_to_behaviour : bool, optional
            Whether to linearly align (via CCA) the decoded latent positions to the
            behavioural trajectory ``Xb`` after each E-step. This keeps the latent space in
            a coordinate system consistent with the behaviour, which is important for
            interpretable receptive fields. By default True.
        resume : bool, optional
            If True, continue training from the current state without re-initialising. The
            ``Y``, ``Xb``, and ``time`` arguments are ignored when resuming — training
            continues on the original data. Useful when the model has not yet converged.
            By default False.
        verbose : bool or None, optional
            Override the instance-level ``verbose`` setting for this call. If None, uses the
            value set at ``__init__``. By default None.

        Returns
        -------
        self : SIMPL
            The fitted model instance (for method chaining).

        Raises
        ------
        ValueError
            If Y, Xb, and time have inconsistent shapes, or if the data dimensionality does
            not match the environment.
        RuntimeError
            If ``resume=True`` but the model has not been fitted yet.

        Examples
        --------
        >>> model = SIMPL(speed_prior=0.4)
        >>> model.fit(Y, Xb, time, n_epochs=5)
        >>> print(model.results_)  # xarray Dataset with all epochs

        Resume training for more epochs:

        >>> model.fit(Y, Xb, time, n_epochs=3, resume=True)
        """
        verbose = self.verbose if verbose is None else verbose

        if resume:
            if not self.is_fitted_:
                raise RuntimeError("Cannot resume: model has not been fitted yet. Call fit() first.")
            self._train_N_epochs(n_epochs, verbose=verbose)
            self.X_ = self.E_["X"]
            self.F_ = self.M_["F"]
            return self

        # ── Validate inputs ──
        if Y.shape[0] != Xb.shape[0]:
            raise ValueError(f"Y and Xb must have the same number of time bins (got {Y.shape[0]} and {Xb.shape[0]})")
        if Y.shape[0] != len(time):
            raise ValueError(f"Y and time must have the same length (got {Y.shape[0]} and {len(time)})")

        # ── Create or use Environment ──
        if self._environment_override is not None:
            self.environment_ = self._environment_override
        else:
            self.environment_ = Environment(
                Xb, pad=self.env_pad, bin_size=self.bin_size, force_lims=self.env_lims, verbose=False
            )

        # ── Extract dimensions ──
        self.D_ = Xb.shape[1]
        self.T_ = Y.shape[0]
        self.N_neurons_ = Y.shape[1]
        self.N_PFmax_ = 20

        if self.D_ != self.environment_.D:
            raise ValueError(f"Data has {self.D_} dimensions but environment has {self.environment_.D}")

        # ── Set up coordinate metadata ──
        neurons = self.neurons if self.neurons is not None else np.arange(self.N_neurons_)

        # ── Convert to JAX arrays ──
        self.Y_ = jnp.array(Y)
        self.Xb_ = jnp.array(Xb)
        self.time_ = jnp.array(time)
        self.neuron_ = jnp.array(neurons)
        self.dt_ = float(self.time_[1] - self.time_[0])

        # ── Environment bins ──
        self.xF_ = jnp.array(self.environment_.flattened_discretised_coords)
        self.xF_shape_ = self.environment_.discrete_env_shape
        self.N_bins_ = len(self.xF_)

        # ── Trial boundaries ──
        self.trial_boundaries_, self.trial_slices_ = self._validate_trial_boundaries(trial_boundaries, self.T_)

        # ── Kalman filter ──
        self._init_kalman_filter()

        # ── Spike mask ──
        self._seed_seq = np.random.SeedSequence(self.random_seed)
        self.block_size_ = int(self.speckle_block_size_seconds / self.dt_)
        self.random_seed_epoch_ = self._next_seed() if self.resample_spike_mask else self.random_seed
        self.spike_mask_ = create_speckled_mask(
            size=(self.T_, self.N_neurons_),
            sparsity=self.test_frac,
            block_size=self.block_size_,
            random_seed=self.random_seed_epoch_,
        )

        # ── Stability masks (odd/even minutes) ──
        self.odd_minute_mask_ = jnp.stack([jnp.array(self.time_ // 60 % 2 == 0)] * self.N_neurons_, axis=1)
        self.even_minute_mask_ = ~self.odd_minute_mask_

        # ── Manifold alignment ──
        self.Xalign_ = self.Xb_ if align_to_behaviour else None

        # ── KDE function ──
        self._kde = kde_angular if self.is_circular else kde

        # ── Epoch tracking ──
        self.lastF_, self.lastX_ = None, None
        self.epoch_ = -1

        # ── Variable registry and coordinates ──
        self.dim_ = self.environment_.dim
        self.variable_info_dict_ = build_variable_info_dict(self.dim_)
        self.coordinates_dict_ = {
            "neuron": self.neuron_,
            "time": self.time_,
            "dim": self.dim_,
            "dim_": self.dim_,
            **self.environment_.coords_dict,
            "place_field": jnp.arange(self.N_PFmax_),
        }

        # ── Initialise results dataset ──
        self.results_ = xr.Dataset(coords={"epoch": jnp.array([], dtype=int)})
        self.results_.attrs = {
            "env_extent": self.environment_.extent,
            "env_pad": self.environment_.pad,
            "env_bin_size": self.environment_.bin_size,
            "trial_boundaries": self.trial_boundaries_,
            "trial_slices": self.trial_slices_,
        }
        data_dict = {"Xb": self.Xb_, "Y": self.Y_, "spike_mask": self.spike_mask_}
        self.results_ = xr.merge([self.results_, self._dict_to_dataset(data_dict)])
        self.loglikelihoods_ = xr.Dataset(coords={"epoch": jnp.array([], dtype=int)})

        # ── Ground truth (not stored — use add_baselines_to_results) ──
        self.Xt_ = None
        self.Ft_ = None
        self.ground_truth_available_ = False

        # ── Print data summary ──
        if verbose:
            self._print_data_summary()

        # ── Run epoch 0 (M-step on behavioural trajectory) ──
        self._run_epoch_zero(verbose)

        # ── Train for n_epochs ──
        self._train_N_epochs(n_epochs, verbose=verbose)

        # ── Set convenience attributes ──
        self.X_ = self.E_["X"]
        self.F_ = self.M_["F"]
        self.is_fitted_ = True

        return self

    def predict(
        self,
        Y: np.ndarray,
        trial_boundaries: np.ndarray | None = None,
        return_std: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Decode latent positions from new spikes using the fitted receptive fields.

        This method uses the receptive fields learned during ``fit()`` (stored in
        ``self.F_``) and a Kalman smoother to decode latent positions from a new set of
        spike observations. No behavioural input is required — the Kalman filter runs with
        zero control input (``U=0``), acting as a pure random-walk smoother constrained
        only by the spike likelihoods and the speed prior.

        The decoded trajectory is returned directly. For power users, the full decode
        results (filtered/smoothed means and covariances, log-likelihoods, etc.) are also
        stored in ``self.prediction_results_``.

        .. important::

            The spike data must be binned at the same ``dt`` as the training data (i.e. the
            ``time`` array passed to ``fit()``). The Kalman filter dynamics and receptive
            fields are calibrated to the training ``dt`` (available as ``self.dt_``). Using
            a different bin width will produce incorrect results. If your new data has a
            different temporal resolution, rebin it to match before calling ``predict()``.

        Parameters
        ----------
        Y : np.ndarray, shape (T_new, N_neurons)
            Spike counts for the new data. Must have the same number of neurons as the
            training data, and must be binned at the same ``dt`` as used during ``fit()``.
        trial_boundaries : np.ndarray or None, optional
            Trial start indices for the new data, e.g. ``[0, 500]``. The Kalman filter runs
            independently per trial. If None, all data is a single trial. By default None.
        return_std : bool, optional
            If True, also return the Kalman-smoothed covariance matrices, which can be used
            as a measure of decoding uncertainty. By default False.

        Returns
        -------
        X_decoded : np.ndarray, shape (T_new, D)
            Decoded latent positions.
        sigma_s : np.ndarray, shape (T_new, D, D)
            Smoothed covariance matrices (only returned if ``return_std=True``).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If the number of neurons in ``Y`` does not match the training data.

        Examples
        --------
        >>> model = SIMPL(speed_prior=0.4)
        >>> model.fit(Y_train, Xb_train, time_train, n_epochs=5)
        >>> X_decoded = model.predict(Y_test)
        >>> X_decoded, sigma = model.predict(Y_test, return_std=True)
        """
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        if Y.shape[1] != self.N_neurons_:
            raise ValueError(f"Y has {Y.shape[1]} neurons but model was fitted with {self.N_neurons_}")

        Y_jax = jnp.array(Y)
        T_new = Y_jax.shape[0]

        _, trial_slices = self._validate_trial_boundaries(trial_boundaries, T_new)

        # Decode using fitted receptive fields, no behaviour input (mask=None → all spikes)
        E = self._decode(
            Y=Y_jax,
            F=self.F_,
            trial_slices=trial_slices,
        )

        X_decoded = np.array(E["mu_s"])

        # Store full decode results for power users
        self.prediction_results_ = E

        if return_std:
            return X_decoded, np.array(E["sigma_s"])
        return X_decoded

    def train_epoch(self) -> None:
        """Run a single epoch of the EM algorithm (E-step then M-step).

        This is the low-level method for manual epoch control. It increments the epoch
        counter, runs the E-step (Kalman decoding — skipped at epoch 0), then the M-step
        (KDE receptive field fitting), and stores the results. The convenience attributes
        ``self.X_`` and ``self.F_`` are updated after each call.

        Must be called after ``fit()`` has been called at least once (at minimum with
        ``n_epochs=0`` to set up all internal state).

        Raises
        ------
        RuntimeError
            If the model has not been initialised via ``fit()`` yet.
        """
        if not hasattr(self, "Y_"):
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

        # ── Increment epoch ──
        self.epoch_ += 1
        if self.resample_spike_mask and self.epoch_ > 0:
            self.spike_mask_ = create_speckled_mask(
                size=(self.T_, self.N_neurons_),
                sparsity=self.test_frac,
                block_size=self.block_size_,
                random_seed=self._next_seed(),
            )

        # ── E-step ──
        if self.epoch_ == 0:
            self.E_ = {"X": self.Xb_}
        else:
            assert self.lastF_ is not None
            self.E_ = self._E_step(Y=self.Y_, F=self.lastF_)

        # ── M-step ──
        X = self.E_["X"]
        self.M_ = self._M_step(Y=self.Y_, X=X)

        # ── Evaluate and save results ──
        if self.evaluate_each_epoch or self.epoch_ == 0:
            self.evaluate_epoch()
        ll_data = self.get_loglikelihoods(Y=self.Y_, FX=self.M_["FX"])
        loglikelihoods = self._dict_to_dataset(ll_data).expand_dims({"epoch": [self.epoch_]})
        self.loglikelihoods_ = xr.concat(
            [self.loglikelihoods_, loglikelihoods],
            dim="epoch",
            data_vars="minimal",
        )

        # ── Store for next epoch ──
        self.lastF_ = self.M_["F"]
        self.lastX_ = self.E_["X"]

        # ── Update convenience attributes ──
        self.X_ = self.E_["X"]
        self.F_ = self.M_["F"]

    def evaluate_epoch(self) -> None:
        """Evaluate the current epoch's metrics and append to the results Dataset.

        Computes log-likelihoods, spatial information, stability, place field analysis,
        and (if ground truth is available) R², trajectory error, and field error. The
        results are stored under the current epoch in ``self.results_``.
        """
        evals = self.get_metrics(
            X=self.E_["X"],
            F=self.M_["F"],
            Y=self.Y_,
            FX=self.M_["FX"],
            F_odd_mins=self.M_["F_odd_minutes"],
            F_even_mins=self.M_["F_even_minutes"],
            X_prev=self.lastX_,
            F_prev=self.lastF_,
            Xt=self.Xt_,
            Ft=self.Ft_,
            PX=self.M_["PX"],
        )
        results = self._dict_to_dataset({**self.M_, **self.E_, **evals}).expand_dims({"epoch": [self.epoch_]})
        self.results_ = xr.concat([self.results_, results], dim="epoch", data_vars="minimal")

    def add_baselines_to_results(
        self,
        Xt: np.ndarray,
        Ft: np.ndarray | None = None,
        Ft_coords_dict: dict | None = None,
    ) -> None:
        """Compute baseline models from ground truth and append to the results Dataset.

        This method computes two special baseline epochs that serve as upper bounds on
        model performance. These are useful for evaluating how close the learned model is
        to the best achievable:

        - **Epoch -1 ("best")** — Receptive fields are fit via KDE to the *true* latent
          positions ``Xt`` (rather than the decoded trajectory). This represents the best
          possible KDE-based model given perfect position knowledge. The E-step is then
          run using these fields to decode positions.
        - **Epoch -2 ("exact")** — The exact ground truth receptive fields ``Ft`` are used
          directly (if provided). This represents the best possible model given both perfect
          fields and perfect decoding. Only computed if ``Ft`` is provided.

        After calling this method, the baseline epochs are appended to ``self.results_``
        and metrics like ``X_R2``, ``X_err``, and ``F_err`` become available across all
        epochs (including baselines) for comparison.

        Parameters
        ----------
        Xt : np.ndarray, shape (T, D)
            Ground truth latent positions. Must have the same number of time bins as the
            training data.
        Ft : np.ndarray or None, optional
            Ground truth receptive fields, shape ``(N_neurons, *spatial_dims)``. For example,
            for 2D data with 100x100 bins, shape would be ``(N_neurons, 100, 100)``. These
            are interpolated onto the model's environment grid. By default None.
        Ft_coords_dict : dict or None, optional
            Coordinate arrays for ``Ft``, mapping dimension names to bin centres. For example:
            ``{"x": np.linspace(0, 1, 100), "y": np.linspace(0, 1, 100)}``. Required if
            ``Ft`` is provided. By default None.

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If ``Xt`` has a different number of time bins than the training data.

        Examples
        --------
        >>> model.fit(Y, Xb, time, n_epochs=5)
        >>> model.add_baselines_to_results(
        ...     Xt=Xt,
        ...     Ft=Ft,
        ...     Ft_coords_dict={"x": xbins, "y": ybins},
        ... )
        >>> print(model.results_.X_R2)  # R² vs ground truth, per epoch
        """
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        if Xt.shape[0] != self.T_:
            raise ValueError(f"Xt has {Xt.shape[0]} time bins but model was fitted with {self.T_}")

        Xt_jax = jnp.array(Xt)
        self.Xt_ = Xt_jax
        self.ground_truth_available_ = True

        # Store Xt in results
        self.results_ = xr.merge([self.results_, self._dict_to_dataset({"Xt": Xt_jax})])

        # Interpolate Ft onto environment grid if provided
        if Ft is not None and Ft_coords_dict is not None:
            Ft_da = xr.DataArray(
                Ft,
                dims=["neuron", *Ft_coords_dict.keys()],
                coords={"neuron": self.neuron_, **Ft_coords_dict},
            )
            Ft_interp = (
                Ft_da.interp(
                    **self.environment_.coords_dict,
                    method="linear",
                    kwargs={"fill_value": "extrapolate"},
                )
                * self.dt_
            )
            Ft_interp = Ft_interp.transpose("neuron", *self.environment_.dim)
            self.Ft_ = jnp.array(Ft_interp.values).reshape(self.N_neurons_, self.N_bins_)
            self.Ft_ = jnp.where(self.Ft_ < 0, 0, self.Ft_)
            self.results_ = xr.merge([self.results_, self._dict_to_dataset({"Ft": self.Ft_})])

        # BEST MODEL: Ft_hat (fit place fields using the true latent positions)
        M_best = self._M_step(self.Y_, self.Xt_)
        E_best = self._E_step(self.Y_, M_best["F"])
        evals_best = self.get_metrics(
            X=E_best["X"],
            F=M_best["F"],
            Y=self.Y_,
            FX=M_best["FX"],
            F_odd_mins=M_best["F_odd_minutes"],
            F_even_mins=M_best["F_even_minutes"],
            X_prev=None,
            F_prev=None,
            Xt=self.Xt_,
            Ft=self.Ft_,
            PX=M_best["PX"],
        )
        results = self._dict_to_dataset({**M_best, **E_best, **evals_best}).expand_dims({"epoch": [-1]})
        self.results_ = xr.concat([self.results_, results], dim="epoch", data_vars="minimal")

        if self.Ft_ is not None:
            # EXACT MODEL: Ft (use the exact receptive fields)
            M_exact = {
                "F": self.Ft_,
                "FX": self.interpolate_firing_rates(self.Xt_, self.Ft_),
                "PX": M_best["PX"],
            }
            E_exact = self._E_step(self.Y_, self.Ft_)
            evals_exact = self.get_metrics(
                X=E_exact["X"],
                F=self.Ft_,
                Y=self.Y_,
                FX=M_exact["FX"],
                F_odd_mins=None,
                F_even_mins=None,
                X_prev=None,
                F_prev=None,
                Xt=self.Xt_,
                Ft=self.Ft_,
                PX=M_exact["PX"],
            )
            results = self._dict_to_dataset({**M_exact, **E_exact, **evals_exact}).expand_dims({"epoch": [-2]})
            self.results_ = xr.concat([self.results_, results], dim="epoch", data_vars="minimal")
            self.results_ = self.results_.sortby("epoch")
        else:
            warnings.warn("Exact place fields not provided so baselines against the exact model cannot be calculated.")

    def interpolate_firing_rates(self, X: jax.Array, F: jax.Array) -> jax.Array:
        """Predict firing rates at arbitrary positions by interpolating the discretised fields.

        Given a set of latent positions ``X`` and receptive fields ``F`` (discretised on
        the environment grid), this method computes the expected firing rate of each neuron
        at each position using nearest-bin lookup. This is much faster than a full KDE
        recalculation and is used internally during the M-step to compute ``FX``.

        Parameters
        ----------
        X : jnp.ndarray, shape (T, D)
            Latent positions to interpolate onto.
        F : jnp.ndarray, shape (N_neurons, N_bins)
            Receptive fields (place fields) of each neuron, flattened over spatial bins.

        Returns
        -------
        FX : jnp.ndarray, shape (T, N_neurons)
            Estimated firing rate (expected spike count per time bin) of each neuron at
            each position in ``X``.
        """
        F = np.array(F)
        X = np.array(X)
        data = self._dict_to_dataset({"F": F, "X": X})
        coord_args = {dim: data.X.sel(dim=dim) for dim in self.dim_}
        FX = data.F.sel(**coord_args, method="nearest").T
        return FX.data

    def get_loglikelihoods(self, Y: jax.Array, FX: jax.Array) -> dict:
        """Calculate log-likelihoods of spikes given firing rates.

        Parameters
        ----------
        Y : jnp.ndarray, shape (T, N_neurons)
            Spike counts.
        FX : jnp.ndarray, shape (T, N_neurons)
            Estimated firing rates.

        Returns
        -------
        dict
            Dictionary with 'logPYXF' and 'logPYXF_test' keys.
        """
        LLs = {}
        logPYXF = poisson_log_likelihood_trajectory(Y, FX, mask=self.spike_mask_).sum() / self.spike_mask_.sum()
        logPYXF_test = (
            poisson_log_likelihood_trajectory(Y, FX, mask=~self.spike_mask_).sum() / (~self.spike_mask_).sum()
        )
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
        """Calculate metrics on the current epoch's results.

        This is a relaxed function: pass in whatever data you have and it will return
        whatever metrics it can calculate.

        Parameters
        ----------
        X : jnp.ndarray, shape (T, D), optional
            Estimated latent positions.
        F : jnp.ndarray, shape (N_neurons, N_bins), optional
            Estimated place fields.
        Y : jnp.ndarray, shape (T, N_neurons), optional
            Spike counts.
        FX : jnp.ndarray, shape (T, N_neurons), optional
            Estimated firing rates.
        F_odd_mins : jnp.ndarray, optional
            Place fields from odd minutes.
        F_even_mins : jnp.ndarray, optional
            Place fields from even minutes.
        X_prev : jnp.ndarray, optional
            Latent positions from previous epoch.
        F_prev : jnp.ndarray, optional
            Place fields from previous epoch.
        Xt : jnp.ndarray, optional
            True latent positions.
        Ft : jnp.ndarray, optional
            True place fields.
        PX : jnp.ndarray, optional
            Position occupancy.

        Returns
        -------
        dict
            Dictionary of calculated metrics.
        """
        metrics = {}

        if Y is not None and FX is not None:
            LLs = self.get_loglikelihoods(Y, FX)
            metrics.update(LLs)

        if X is not None and Xt is not None:
            metrics["X_R2"] = coefficient_of_determination(X, Xt)

        if X is not None and Xt is not None:
            metrics["X_err"] = jnp.mean(jnp.linalg.norm(X - Xt, axis=1))

        if F is not None and Ft is not None:
            metrics["F_err"] = jnp.mean(jnp.linalg.norm(F - Ft, axis=1))

        if F is not None:
            F_pdf = (F + 1e-6) / jnp.sum(F, axis=1)[:, None]
            I_F = jnp.sum(F_pdf * jnp.log(F_pdf), axis=1)
            metrics["negative_entropy"] = I_F

        if F is not None:
            rho_F = jnp.mean(F < 1.0 * self.dt_, axis=1)
            metrics["sparsity"] = rho_F

        if F_odd_mins is not None and F_even_mins is not None:
            corr = jnp.corrcoef(F_odd_mins, F_even_mins)
            cross_corr = corr[: self.N_neurons_, self.N_neurons_ :]
            stability = jnp.diag(cross_corr)
            metrics["stability"] = stability

        if F is not None and self.environment_.D == 2:
            metrics.update(
                analyse_place_fields(
                    F,
                    N_neurons=self.N_neurons_,
                    N_PFmax=self.N_PFmax_,
                    D=self.D_,
                    xF_shape=self.xF_shape_,
                    xF=self.xF_,
                    dt=self.dt_,
                    bin_size=self.environment_.bin_size,
                    n_bins=self.N_bins_,
                )
            )

        if F_prev is not None and F is not None:
            delta_F = jnp.linalg.norm(F - F_prev, axis=1)
            metrics["field_change"] = delta_F

        if X_prev is not None and X is not None:
            delta_X = jnp.linalg.norm(X - X_prev, axis=1)
            metrics["trajectory_change"] = delta_X

        if F is not None and PX is not None:
            metrics["spatial_information"] = calculate_spatial_information(F / self.dt_, PX)
            metrics["spatial_information_rate"] = float(jnp.sum(metrics["spatial_information"]))

        return metrics

    def save_results(self, path: str) -> None:
        """Save the results Dataset to a netCDF file.

        The saved file can be loaded back with ``simpl.load_results(path)``, which returns
        an ``xarray.Dataset`` with all epochs, metrics, and fitted variables.

        Parameters
        ----------
        path : str
            File path to save to (typically ending in ``.nc``).
        """
        save_results_to_netcdf(self.results_, path)

    # ──────────────────────────────────────────────────────────────────────────
    # Private methods
    # ──────────────────────────────────────────────────────────────────────────

    def _init_kalman_filter(self) -> None:
        """Set up the Kalman filter from prior parameters."""
        speed_sigma = self.speed_prior * self.dt_

        # Kalman configuration
        self.speed_prior_requested_ = self.speed_prior
        self.kalman_off_speed_prior_ = 1e10
        speed_prior_effective = (
            self.speed_prior if self.use_kalman_smoothing else max(self.speed_prior, self.kalman_off_speed_prior_)
        )
        self.speed_prior_effective_ = speed_prior_effective
        speed_sigma = speed_prior_effective * self.dt_

        behaviour_sigma = self.behaviour_prior if self.behaviour_prior is not None else 1e6
        lam = behaviour_sigma**2 / (speed_sigma**2 + behaviour_sigma**2)
        sigma_eff_square = speed_sigma**2 * behaviour_sigma**2 / (speed_sigma**2 + behaviour_sigma**2)

        F = lam * jnp.eye(self.D_)
        B = (1 - lam) * jnp.eye(self.D_)
        Q = sigma_eff_square * jnp.eye(self.D_)
        H = jnp.eye(self.D_)

        self.kalman_filter_ = KalmanFilter(dim_Z=self.D_, dim_Y=self.D_, dim_U=self.D_, F=F, B=B, Q=Q, H=H, R=None)

    def _next_seed(self) -> int:
        """Spawn a new seed from the seed sequence."""
        return self._seed_seq.spawn(1)[0].generate_state(1)[0]

    @staticmethod
    def _validate_trial_boundaries(trial_boundaries, T):
        """Validate trial boundaries and create trial slices.

        Parameters
        ----------
        trial_boundaries : np.ndarray or None
            Trial start indices. If None, treats all data as one trial.
        T : int
            Total number of time bins.

        Returns
        -------
        trial_boundaries : np.ndarray
            Validated boundaries array.
        trial_slices : list[slice]
            List of slice objects for each trial.
        """
        if trial_boundaries is None:
            trial_boundaries = np.array([0])
        else:
            trial_boundaries = np.array(trial_boundaries)

        if trial_boundaries[0] != 0:
            raise ValueError("First trial boundary must be 0")
        if trial_boundaries[-1] >= T:
            raise ValueError(f"Last trial boundary must be < T (got {trial_boundaries[-1]} with T={T})")
        if len(trial_boundaries) > 1 and not np.all(np.diff(trial_boundaries) > 0):
            raise ValueError("Trial boundaries must be strictly increasing")

        trial_slices = [slice(trial_boundaries[i], trial_boundaries[i + 1]) for i in range(len(trial_boundaries) - 1)]
        trial_slices.append(slice(trial_boundaries[-1], T))
        return trial_boundaries, trial_slices

    def _decode(
        self,
        Y: jax.Array,
        F: jax.Array,
        trial_slices: list,
        mask: jax.Array | None = None,
        Xb: jax.Array | None = None,
    ) -> dict:
        """Core decoding: likelihood maps -> Gaussian fit -> Kalman filter/smooth.

        Shared by ``_E_step`` (fit-time) and ``predict`` (inference-time).

        Parameters
        ----------
        Y : jnp.ndarray, shape (T, N_neurons)
            Spike counts.
        F : jnp.ndarray, shape (N_neurons, N_bins)
            Place fields.
        trial_slices : list[slice]
            Trial boundaries as slices.
        mask : jnp.ndarray or None, shape (T, N_neurons)
            Boolean mask (True = use for likelihood). If None, all spikes are used.
        Xb : jnp.ndarray or None, shape (T, D)
            Behavioural positions. If None, Kalman runs with U=0 (pure smoother).

        Returns
        -------
        dict
            Decode results including mu_l, mode_l, sigma_l, mu_f, sigma_f, mu_s, sigma_s,
            logPYF, logPYF_test, and optionally logPYXF_maps.
        """
        if mask is None:
            mask = jnp.ones(Y.shape, dtype=bool)

        # Log-likelihood maps
        logPYXF_maps = poisson_log_likelihood(Y, F, mask=mask)
        no_spikes = jnp.sum(Y * mask, axis=1) == 0

        # Fit Gaussians
        mu_l, mode_l, sigma_l = vmap(fit_gaussian, in_axes=(None, 0))(self.xF_, jnp.exp(logPYXF_maps))

        # Observation noise (inflated when no spikes)
        observation_noise = jnp.where(
            no_spikes[:, None, None],
            jnp.eye(self.D_) * 1e6,
            sigma_l,
        )

        # Process each trial
        mu_f_list, sigma_f_list, mu_s_list, sigma_s_list = [], [], [], []
        logPYF_list, logPYF_test_list = [], []

        for trial_slice in trial_slices:
            _mode_l = mode_l[trial_slice]
            _R = observation_noise[trial_slice]
            _Y = Y[trial_slice]
            _mask = mask[trial_slice]
            T_trial = _mode_l.shape[0]

            # Control input: behaviour if available, zeros otherwise
            if Xb is not None:
                _U = Xb[trial_slice]
                _mu0 = _U.mean(axis=0)
                _sigma0 = (1 / len(_U)) * (((_U - _mu0).T) @ (_U - _mu0))
            else:
                _U = jnp.zeros((T_trial, self.D_))
                # Initialise from first few likelihood modes
                _mu0 = _mode_l[: min(10, T_trial)].mean(axis=0)
                _sigma0 = jnp.eye(self.D_) * 1.0

            # Filter and smooth
            mu_f, sigma_f = self.kalman_filter_.filter(mu0=_mu0, sigma0=_sigma0, Y=_mode_l, U=_U, R=_R)
            mu_s, sigma_s = self.kalman_filter_.smooth(mus_f=mu_f, sigmas_f=sigma_f)

            # Likelihoods
            logPYF = self.kalman_filter_.loglikelihood(Y=_mode_l, R=_R, mu=mu_s, sigma=sigma_s).sum()

            # Test likelihood
            logPYXF_maps_test = poisson_log_likelihood(_Y, F, mask=~_mask)
            no_spikes_test = jnp.sum(_Y * ~_mask, axis=1) == 0
            _, mode_l_test, sigma_l_test = vmap(fit_gaussian, in_axes=(None, 0))(self.xF_, jnp.exp(logPYXF_maps_test))
            observation_noise_test = jnp.where(
                no_spikes_test[:, None, None],
                jnp.eye(self.D_) * 1e6,
                sigma_l_test,
            )
            logPYF_test = self.kalman_filter_.loglikelihood(
                Y=mode_l_test, R=observation_noise_test, mu=mu_s, sigma=sigma_s
            ).sum()

            mu_f_list.append(mu_f)
            sigma_f_list.append(sigma_f)
            mu_s_list.append(mu_s)
            sigma_s_list.append(sigma_s)
            logPYF_list.append(logPYF)
            logPYF_test_list.append(logPYF_test)

        # Concatenate
        mu_f = jnp.concatenate(mu_f_list, axis=0)
        sigma_f = jnp.concatenate(sigma_f_list, axis=0)
        mu_s = jnp.concatenate(mu_s_list, axis=0)
        sigma_s = jnp.concatenate(sigma_s_list, axis=0)
        logPYF = sum(logPYF_list)
        logPYF_test = sum(logPYF_test_list)

        E = {
            "mu_l": mu_l,
            "mode_l": mode_l,
            "sigma_l": sigma_l,
            "mu_f": mu_f,
            "sigma_f": sigma_f,
            "mu_s": mu_s,
            "sigma_s": sigma_s,
            "logPYF": logPYF,
            "logPYF_test": logPYF_test,
        }
        if self.save_likelihood_maps:
            E["logPYXF_maps"] = logPYXF_maps

        return E

    def _E_step(self, Y: jax.Array, F: jax.Array) -> dict:
        """E-step: decode latent positions and optionally align to behaviour.

        Parameters
        ----------
        Y : jnp.ndarray, shape (T, N_neurons)
            Spike counts.
        F : jnp.ndarray, shape (N_neurons, N_bins)
            Place fields from previous M-step.

        Returns
        -------
        dict
            E-step results including X (aligned latent positions).
        """
        E = self._decode(
            Y=Y,
            F=F,
            mask=self.spike_mask_,
            trial_slices=self.trial_slices_,
            Xb=self.Xb_,
        )

        # Manifold alignment (fit-time only)
        align_dict = {}
        if self.Xalign_ is not None:
            coef, intercept = cca(E["mu_s"], self.Xalign_)
            E["X"] = E["mu_s"] @ coef.T + intercept
            align_dict = {"coef": coef, "intercept": intercept}
        else:
            E["X"] = E["mu_s"]

        E.update(align_dict)
        return E

    def _M_step(self, Y: jax.Array, X: jax.Array) -> dict:
        """M-step: fit receptive fields via KDE.

        Parameters
        ----------
        Y : jnp.ndarray, shape (T, N_neurons)
            Spike counts.
        X : jnp.ndarray, shape (T, D)
            Latent positions.

        Returns
        -------
        dict
            M-step results: F, F_odd_minutes, F_even_minutes, FX, PX.
        """

        def kde_func(mask):
            return self._kde(
                bins=self.xF_,
                trajectory=X,
                spikes=Y,
                kernel=gaussian_kernel,
                kernel_bandwidth=self.kernel_bandwidth,
                mask=mask,
                return_position_density=True,
            )

        stacked_masks = jnp.array([self.spike_mask_, self.odd_minute_mask_, self.even_minute_mask_])
        all_F, all_PX = vmap(kde_func)(stacked_masks)
        F, F_odd_minutes, F_even_minutes = all_F[0], all_F[1], all_F[2]
        PX = all_PX[0]
        FX = self.interpolate_firing_rates(X, F)
        M = {"F": F, "F_odd_minutes": F_odd_minutes, "F_even_minutes": F_even_minutes, "FX": FX, "PX": PX}
        return M

    def _train_N_epochs(self, N: int, verbose: bool = True) -> None:
        """Train for N epochs with KeyboardInterrupt support."""
        if N <= 0:
            return
        self._print_epoch_summary()
        for _ in range(N):
            try:
                self.train_epoch()
                if verbose:
                    self._print_epoch_summary()
            except KeyboardInterrupt:
                print(f"Training interrupted after {self.epoch_} epochs.")
                break
        if not self.evaluate_each_epoch:
            self.evaluate_epoch()

    def _run_epoch_zero(self, verbose: bool) -> None:
        """Run epoch 0 (M-step on behavioural trajectory) and print diagnostics."""
        _epoch0_done = threading.Event()

        def _delayed_message():
            if not _epoch0_done.wait(3):
                print("  ...[estimating spatial receptive fields from Xb (epoch 0)]")

        _timer = threading.Thread(target=_delayed_message, daemon=True)
        _timer.start()
        self.train_epoch()
        _epoch0_done.set()

        if verbose:
            si = np.array(self.results_.spatial_information.sel(epoch=0))
            info_rate = float(si.sum())
            si_min = float(si.min())
            si_q25 = float(np.percentile(si, 25))
            si_med = float(np.median(si))
            si_q75 = float(np.percentile(si, 75))
            si_max = float(si.max())
            si_mean = float(si.mean())
            print(
                f"  Spatial information (bits/s per neuron): mean {si_mean:.2f}, "
                f"min {si_min:.2f}, Q1 {si_q25:.2f}, "
                f"median {si_med:.2f}, Q3 {si_q75:.2f}, max {si_max:.2f}"
            )
            print(f"  Total spatial information rate: {info_rate:.1f} bits/s")
            print()

            active_per_bin = (np.array(self.Y_) > 0).sum(axis=1)
            frac_2plus = float(np.mean(active_per_bin >= 2))
            if frac_2plus < 0.05:
                print(
                    "  WARNING: fewer than 5% of time bins have 2+ active "
                    "neurons. The Poisson likelihood will be weak in most "
                    "bins. Try coarsen_dt() or accumulate_spikes() to "
                    "increase spike density, or add more neurons."
                )
            if info_rate < 1.0:
                print(
                    "  WARNING: spatial information rate is very low "
                    f"({info_rate:.1f} bits/s). The neurons may not carry "
                    "enough spatial information for reliable decoding. "
                    "Try coarsen_dt() or accumulate_spikes() to increase "
                    "spike density, or add more neurons."
                )

    def _print_data_summary(self) -> None:
        """Print a summary of the input data."""
        # Build a minimal xr.Dataset for print_data_summary
        data = xr.Dataset(
            {
                "Y": xr.DataArray(
                    self.Y_, dims=["time", "neuron"], coords={"time": self.time_, "neuron": self.neuron_}
                ),
                "Xb": xr.DataArray(self.Xb_, dims=["time", "dim"], coords={"time": self.time_}),
            }
        )
        data["trial_slices"] = self.trial_slices_
        print_data_summary(data)

    def _print_epoch_summary(self) -> None:
        """Print a one-line summary of the current epoch's metrics."""
        e = self.epoch_
        try:
            train_ll = float(self.loglikelihoods_.logPYXF.sel(epoch=e).values)
            test_ll = float(self.loglikelihoods_.logPYXF_test.sel(epoch=e).values)
            si = float(self.results_.spatial_information.sel(epoch=e).mean().values)
            parts = [
                f"Epoch {e:<3d}:    log-likelihood(train={train_ll:.3f}, test={test_ll:.3f})",
                f"spatial_info={si:.3f} bits/s/neuron",
            ]
            print("     ".join(parts))
            if e > 0:
                epoch0_test_ll = float(self.loglikelihoods_.logPYXF_test.sel(epoch=0).values)
                if test_ll < epoch0_test_ll:
                    print("    WARNING: test LL below epoch 0. Model may be overfitting.")
        except Exception:
            print(f"Epoch {e}")

    def _dict_to_dataset(self, data: dict, coords: dict | None = None) -> xr.Dataset:
        """Convert a dictionary to an xarray Dataset.

        Parameters
        ----------
        data : dict
            Dictionary of variable_name -> array.
        coords : dict or None
            Coordinate arrays. If None, uses self.coordinates_dict_.

        Returns
        -------
        xr.Dataset
        """
        dataset = xr.Dataset()
        if coords is None:
            coords = self.coordinates_dict_
        for variable_name in data.keys():
            if variable_name in self.variable_info_dict_:
                variable_info = self.variable_info_dict_[variable_name]
                variable_coords = {k: coords[k] for k in variable_info["dims"]}
                intended_variable_shape = tuple([len(variable_coords[c]) for c in variable_info["dims"]])
                if "reshape" in variable_info and variable_info["reshape"]:
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


if __name__ == "__main__":
    pass
