"""Core SIMPL class implementing the EM-style optimisation loop.

This module provides the ``SIMPL`` class, the primary user-facing API.  It
follows scikit-learn conventions: ``__init__`` stores hyperparameters only,
``fit()`` accepts data and runs the EM algorithm, and ``predict()`` decodes new
spike data using the fitted receptive fields.

Each EM epoch proceeds as:

1. **E-step** — Kalman filter/smoother decodes latent positions from spike
   observations.
2. **M-step** — Kernel density estimation re-estimates receptive fields from
   the decoded positions.
3. **Evaluate** — Metrics (log-likelihood, spatial information, stability,
   etc.) are computed and stored in an ``xarray.Dataset``.
"""

# Jax, for the majority of the calculations
import threading
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import xarray as xr
from jax import vmap

# Internal libraries
from simpl._variable_registry import _build_variable_info_dict, _dict_to_dataset
from simpl.environment import Environment
from simpl.kalman import KalmanFilter
from simpl.kde import gaussian_kernel, kde, kde_angular, poisson_log_likelihood, poisson_log_likelihood_trajectory
from simpl.utils import (
    _wrap_minuspi_pi,
    analyse_place_fields,
    calculate_spatial_information,
    cca,
    cca_angular,
    coefficient_of_determination,
    create_speckled_mask,
    fit_gaussian,
    get_field_peaks,
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
        behavior_prior: float | None = None,
        # Environment parameters
        is_1D_angular: bool = False,
        bin_size: float = 0.02,
        env_pad: float = 0.1,
        env_lims: tuple | None = None,
        environment: Environment | None = None,
        # Mask and analysis parameters
        val_frac: float = 0.1,
        speckle_block_size_seconds: float = 1,
        random_seed: int = 0,
        # Device
        use_gpu: bool | str = "if_available",
    ) -> None:
        """Initialise the SIMPL model with hyperparameters only (no data, no computation).

        Call ``fit()`` to provide data and train the model. After fitting, optimised latents,
        receptive fields, metrics and other diagnostics can be found in ``self.results_``
        (an ``xarray.Dataset``). Quick access to the final decoded latent and receptive fields
        is available via ``self.X_`` and ``self.F_``. The ``predict()`` method can decode a
        new set of spikes using the fitted receptive fields, without needing behavioral input.

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
        behavioral initialisation trajectory ``Xb``. Subsequent epochs alternate E-step and
        M-step.

        Terminology
        -----------
        - **Y** : Spike counts, shape (T, N_neurons). Can be binary (0/1) or integer counts (0/1/2/3...).
        - **Xb** : Behavioral initialisation, shape (T, D). The starting estimate of latent positions,
          typically derived from tracked behavior (e.g. animal position).
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
        behavior_prior : float or None, optional
            Prior on how far the latent positions can deviate from the behavioral positions,
            in units of meters. This acts as a soft constraint pulling the decoded trajectory
            towards ``Xb``. None means no prior (the latent is free to move anywhere).
            By default None.
        is_1D_angular : bool, optional
            Whether the latent space is 1D angular/circular (e.g. head direction data in
            radians). If True, angular KDE is used, the Kalman filter wraps its state to
            [-pi, pi) after every predict/update/smooth step, and the environment is fixed
            to [-pi, pi). The wrapped Kalman approximation assumes a tight posterior
            (sigma << 2*pi); results may degrade when posterior uncertainty is large relative
            to the circular domain. By default False.
        bin_size : float, optional
            Spatial bin size for discretising the environment, in the same units as the latent
            space. Controls the resolution of the receptive field grid. Smaller bins give
            higher resolution but increase computation and memory. By default 0.02.
        env_pad : float, optional
            Padding added outside the data bounds when constructing the environment grid. This
            ensures that receptive fields near the boundary of the explored space are not
            clipped. In the same units as the latent space. Ignored when
            ``is_1D_angular=True`` because the circular domain is fixed to ``[-pi, pi)``.
            By default 0.1.
        env_lims : tuple or None, optional
            Force the environment limits to specific values instead of inferring them from the
            data. Format: ``((min_dim1, min_dim2, ...), (max_dim1, max_dim2, ...))``.
            By default None (auto-inferred from ``Xb``). When ``is_1D_angular=True``, the
            domain is fixed to ``[-pi, pi)`` and incompatible limits raise an error.
        environment : Environment or None, optional
            A pre-built ``Environment`` instance for power users. If provided, ``bin_size``,
            ``env_pad``, and ``env_lims`` are all ignored. In circular mode the provided
            environment must also represent the full ``[-pi, pi)`` domain. By default None.
        val_frac : float, optional
            Fraction of spike observations held out for validation, implemented via a speckled
            (block-structured) mask. Used to compute held-out log-likelihood for monitoring
            overfitting. By default 0.1.
        speckle_block_size_seconds : float, optional
            Temporal size (in seconds) of contiguous blocks in the speckled validation mask. Larger
            blocks give more temporally coherent held-out segments. By default 1.0.
        random_seed : int, optional
            Random seed for reproducibility (controls the spike mask generation).
            By default 0.
        use_gpu : bool or str, optional
            Controls GPU usage. ``True`` forces GPU and raises an error if none is
            available. ``False`` forces CPU even when a GPU is present.
            ``"if_available"`` (default) uses GPU when one is detected, otherwise
            falls back to CPU.

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
        self.behavior_prior = behavior_prior
        self.is_1D_angular = is_1D_angular

        # Environment config
        self.bin_size = bin_size
        self.env_pad = env_pad
        self.env_lims = env_lims
        self._environment_override = environment  # power-user pre-built Environment

        # Mask and analysis parameters
        self.val_frac = val_frac
        self.speckle_block_size_seconds = speckle_block_size_seconds
        self.random_seed = random_seed

        # Fitted flag
        self.is_fitted_ = False

        # Device setup
        gpu_available = jax.default_backend() == "gpu"
        if use_gpu is True:
            if not gpu_available:
                raise RuntimeError(
                    "use_gpu=True but no GPU is available. "
                    "Install a GPU-enabled JAX build or use use_gpu='if_available'."
                )
            self.use_gpu_ = True
        elif use_gpu is False:
            self.use_gpu_ = False
        elif use_gpu == "if_available":
            self.use_gpu_ = gpu_available
        else:
            raise ValueError(f"use_gpu must be True, False, or 'if_available', got {use_gpu!r}")

        if self.use_gpu_:
            device = jax.devices("gpu")[0]
            print(f"SIMPL: Using GPU ({device.device_kind})")
        else:
            print("SIMPL: Using CPU")

    def _jax_device(self):
        """Return the JAX device to place arrays on."""
        if self.use_gpu_:
            return jax.devices("gpu")[0]
        return jax.devices("cpu")[0]

    def fit(
        self,
        Y: np.ndarray,
        Xb: np.ndarray,
        time: np.ndarray,
        n_epochs: int = 5,
        trial_boundaries: np.ndarray | None = None,
        align_to_behavior: bool | str = "trajectory",
        resume: bool = False,
        save_full_history: bool = False,
        early_stopping: bool = True,
        verbose: bool = True,
    ) -> "SIMPL":
        """Fit the SIMPL model to data.

        This is the main entry point for training. It performs the following steps:

        1. **Setup** — validates inputs, creates the ``Environment`` (spatial discretisation
           grid), sets up the Kalman filter, and builds the speckled validation mask.
        2. **Epoch 0** — runs the M-step only on the behavioral trajectory ``Xb`` to produce
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
        - ``self.loglikelihoods_`` — per-epoch train/validation log-likelihoods.

        Parameters
        ----------
        Y : np.ndarray, shape (T, N_neurons)
            Spike counts. Can be binary (0/1) or integer-valued. Each row is one time bin,
            each column is one neuron.
        Xb : np.ndarray, shape (T, D)
            Behavioral initialisation positions. This is the starting estimate of the latent
            trajectory, typically the tracked position of the animal. D is the number of
            latent dimensions (e.g. 2 for 2D position).
        time : np.ndarray, shape (T,)
            Time stamps (in seconds) for each time bin. Values should be uniformly
            increasing (Kalman filter is poorly defined otherwise. ``dt`` is automatically
            inferred as ``median(diff(time))``.
        n_epochs : int, optional
            Number of EM epochs to train after epoch 0. Set to 0 to run only the initial
            M-step (useful for manual epoch control via ``_fit_epoch()``). By default 5.
        trial_boundaries : np.ndarray or None, optional
            Array of indices where new trials start, e.g. ``[0, 1000, 2000]``. The first
            element must be 0. The Kalman filter runs independently within each trial to
            prevent smoothing across trial boundaries (e.g. between separate recording
            sessions). If None, all data is treated as a single trial. By default None.
        align_to_behavior : bool or str, optional
            How to linearly align (via CCA) the decoded latent positions to the behavioral
            coordinate system after each E-step. Options:

            - ``"trajectory"`` (default) — align the decoded trajectory ``mu_s`` directly
              to ``Xb``.
            - ``"fields"`` — align based on peak positions of receptive fields. CAn be useful
            in 1D where the latent position distribution can be bimodal. Unstable / not
            recommended if fields are likely to have multiple peaks
            - ``True`` — alias for ``"trajectory"``.
            - ``False`` — no alignment.
        resume : bool, optional
            If True, continue training from the current state without re-initialising. The
            ``Y``, ``Xb``, and ``time`` arguments are ignored when resuming — training
            continues on the original data. Useful when the model has not yet converged.
            By default False.
        save_full_history : bool, optional
            Controls storage of large per-timestep arrays. When False
            (the default), the per-epoch firing rate trajectories ``FX`` are not stored and ``logPYXF_maps`` is
            not stored at all. When True, ``FX`` is stored for every epoch and
            ``logPYXF_maps`` is stored for the last epoch only (due to its huge size).
            Metrics, receptive fields, decoded trajectories, and Kalman outputs are
            unaffected and stored for every epoch regardless. By default False.
        early_stopping : bool, optional
            If True, training stops early when the validation log-likelihood has not improved
            for 3 consecutive epochs. By default True.
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
        self.save_full_history_ = save_full_history

        if resume:
            if not self.is_fitted_:
                raise RuntimeError("Cannot resume: model has not been fitted yet. Call fit() first.")
            self._fit_N_epochs(n_epochs, early_stopping=early_stopping, verbose=verbose)
            self.results_["FX_lastepoch"] = _dict_to_dataset(
                {"FX_lastepoch": self.M_["FX"]}, self.variable_info_dict_, self.coordinates_dict_
            )["FX_lastepoch"]
            self.X_ = self.E_["X"]
            self.F_ = self.M_["F"]
            return self

        self._init_from_data(Y, Xb, time, trial_boundaries, align_to_behavior, verbose)

        # ── Run epoch 0 (M-step on behavioral trajectory) ──
        self._run_epoch_zero(verbose)

        # ── Train for n_epochs ──
        self._fit_N_epochs(n_epochs, early_stopping=early_stopping, verbose=verbose)

        # ── Attach FX_firstepoch and FX_lastepoch (always, without epoch dim) ──
        self.results_["FX_firstepoch"] = _dict_to_dataset(
            {"FX_firstepoch": self.FX_firstepoch_}, self.variable_info_dict_, self.coordinates_dict_
        )["FX_firstepoch"]
        self.results_["FX_lastepoch"] = _dict_to_dataset(
            {"FX_lastepoch": self.M_["FX"]}, self.variable_info_dict_, self.coordinates_dict_
        )["FX_lastepoch"]

        # ── Attach logPYXF_maps for the final epoch only (too large to store per epoch) ──
        if self.save_full_history_:
            self.results_["logPYXF_maps"] = _dict_to_dataset(
                {"logPYXF_maps": self.E_["logPYXF_maps"]}, self.variable_info_dict_, self.coordinates_dict_
            )["logPYXF_maps"]

        # ── Set convenience attributes ──
        self.X_ = self.E_["X"]
        self.F_ = self.M_["F"]
        self.is_fitted_ = True

        # ── Compute baseline epochs if ground truth was registered before fit ──
        if self.ground_truth_available_:
            self._apply_baselines_to_results()

        return self

    def predict(
        self,
        Y: np.ndarray,
        trial_boundaries: np.ndarray | None = None,
    ) -> np.ndarray:
        """Decode latent positions from new spikes using the fitted receptive fields.

        This method uses the receptive fields learned during ``fit()`` (stored in
        ``self.F_``) and a Kalman smoother to decode latent positions from a new set of
        spike observations. No behavioral input is required — the Kalman filter runs with
        zero control input (``U=0``), acting as a pure random-walk smoother constrained
        only by the spike likelihoods and the speed prior.

        The decoded trajectory is returned directly. The full decode results
        (filtered/smoothed means and covariances, log-likelihoods, etc.) are stored as an
        ``xr.Dataset`` in ``self.prediction_results_``.

        !!! important

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

        Returns
        -------
        X_decoded : np.ndarray, shape (T_new, D)
            Decoded latent positions.

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
        >>> sigma = model.prediction_results_["sigma_s"]  # smoothed covariances
        """
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        if Y.shape[1] != self.N_neurons_:
            raise ValueError(f"Y has {Y.shape[1]} neurons but model was fitted with {self.N_neurons_}")

        Y_jax = jax.device_put(jnp.array(Y), self._jax_device())
        T_new = Y_jax.shape[0]

        trial_boundaries_validated, trial_slices, _, _ = self._validate_trial_boundaries(trial_boundaries, T_new)

        # Decode using fitted receptive fields, no behavior input (mask=None → all spikes)
        E = self._decode(
            Y=Y_jax,
            F=self.F_,
            trial_boundaries=trial_boundaries_validated,
        )

        X_decoded = np.array(E["mu_s"])
        if self.is_1D_angular:
            X_decoded = np.array(_wrap_minuspi_pi(jnp.array(X_decoded)))

        # Store full decode results as an xr.Dataset
        pred_time = np.arange(T_new) * self.dt_
        pred_coords = {**self.coordinates_dict_, "time": pred_time}
        self.prediction_results_ = _dict_to_dataset(E, self.variable_info_dict_, pred_coords)
        self.prediction_results_.attrs = self._build_dataset_attrs(
            trial_boundaries=trial_boundaries if trial_boundaries is not None else [0],
        )

        return X_decoded

    def analyse_place_fields(self, epochs: list[int] | None = None) -> "SIMPL":
        """Run place field morphology analysis and add results to ``self.results_``.

        Identifies individual place fields in each neuron's receptive field
        using connected-component labelling and computes per-field statistics
        (size, position, roundness, peak firing rate, etc.). Only applicable to
        2D environments.

        This analysis uses scipy and scikit-image and can be slow for large
        neuron counts, which is why it is not run automatically during
        ``fit()``. Results are added to ``self.results_`` in-place.

        Parameters
        ----------
        epochs : list of int, optional
            Which epochs to analyse. If None (default), analyses only the
            final epoch.

        Returns
        -------
        self : SIMPL
            The model instance (for method chaining).

        Raises
        ------
        RuntimeError
            If the model has not been fitted yet.
        ValueError
            If the environment is not 2D.

        Examples
        --------
        >>> model = SIMPL(speed_prior=0.4)
        >>> model.fit(Y, Xb, time, n_epochs=5)
        >>> model.analyse_place_fields()
        >>> print(model.results_["place_field_count"])
        """
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        if self.environment_.D != 2:
            raise ValueError("Place field analysis is only supported for 2D environments.")

        if epochs is None:
            epochs = [int(self.results_.epoch.values[-1])]

        for epoch in epochs:
            F = self.results_["F"].sel(epoch=epoch).values
            F_jax = jnp.array(F)
            pf_data = analyse_place_fields(
                F_jax,
                N_neurons=self.N_neurons_,
                N_PFmax=self.N_PFmax_,
                D=self.D_,
                xF_shape=self.xF_shape_,
                xF=self.xF_,
                dt=self.dt_,
                bin_size=self.environment_.bin_size,
                n_bins=self.N_bins_,
            )
            pf_ds = _dict_to_dataset(pf_data, self.variable_info_dict_, self.coordinates_dict_).expand_dims(
                {"epoch": [epoch]}
            )
            for var in pf_ds.data_vars:
                if var in self.results_:
                    self.results_[var] = pf_ds[var]
                else:
                    self.results_[var] = pf_ds[var]

        return self

    def add_baselines(
        self,
        Xt: np.ndarray,
        Ft: np.ndarray | None = None,
        Ft_coords_dict: dict | None = None,
    ) -> "SIMPL":
        """Register ground truth data for baseline comparison.

        Can be called **before** or **after** ``fit()``:

        - **Before fit** — ground truth positions ``Xt`` are stored so that per-epoch
          metrics like ``X_R2`` and ``X_err`` are computed during training. Baseline
          epochs (-1, -2) are computed automatically at the end of ``fit()``.
        - **After fit** — baseline epochs are computed immediately and appended to
          ``self.results_``.

        Two special baseline epochs are created:

        - **Epoch -1 ("best")** — Receptive fields fit via KDE to the *true* positions
          ``Xt``, representing the best KDE model given perfect position knowledge.
        - **Epoch -2 ("exact")** — The exact ground truth fields ``Ft`` are used directly
          (only if provided).

        Parameters
        ----------
        Xt : np.ndarray, shape (T, D)
            Ground truth latent positions. Must have the same number of time bins as the
            training data (if already fitted).
        Ft : np.ndarray or None, optional
            Ground truth receptive fields, shape ``(N_neurons, *spatial_dims)``. These
            are interpolated onto the model's environment grid. By default None.
        Ft_coords_dict : dict or None, optional
            Coordinate arrays for ``Ft``, mapping dimension names to bin centres. Required
            if ``Ft`` is provided. By default None.

        Returns
        -------
        self : SIMPL
            The model instance (for chaining).

        Examples
        --------
        Call before fit to get ground truth metrics during training:

        >>> model = SIMPL()
        >>> model.add_baselines(Xt=Xt, Ft=Ft, Ft_coords_dict={"x": xbins, "y": ybins})
        >>> model.fit(Y, Xb, time, n_epochs=5)
        >>> print(model.results_.X_R2)  # R² vs ground truth, per epoch

        Or call after fit (baseline epochs computed immediately):

        >>> model.fit(Y, Xb, time, n_epochs=5)
        >>> model.add_baselines(Xt=Xt)
        """
        # Store raw arrays (processed later in _apply_baselines_to_results)
        self._Xt_raw = np.asarray(Xt)
        self._Ft_raw = Ft
        self._Ft_coords_dict_raw = Ft_coords_dict
        self.ground_truth_available_ = True

        if self.is_fitted_:
            # Model already fitted — apply immediately
            self._apply_baselines_to_results()

        return self

    def add_baselines_to_results(
        self,
        Xt: np.ndarray,
        Ft: np.ndarray | None = None,
        Ft_coords_dict: dict | None = None,
    ) -> None:
        """Deprecated: use ``add_baselines`` instead."""
        warnings.warn("add_baselines_to_results is deprecated, use add_baselines instead.", DeprecationWarning)
        self.add_baselines(Xt=Xt, Ft=Ft, Ft_coords_dict=Ft_coords_dict)

    def _apply_baselines_to_results(self) -> None:
        """Process stored ground truth and compute baseline epochs."""
        Xt = self._Xt_raw
        if Xt.shape[0] != self.T_:
            raise ValueError(f"Xt has {Xt.shape[0]} time bins but model was fitted with {self.T_}")

        Xt_jax = jnp.array(Xt)
        self.Xt_ = Xt_jax

        # Store Xt in results
        self.results_ = xr.merge(
            [self.results_, _dict_to_dataset({"Xt": Xt_jax}, self.variable_info_dict_, self.coordinates_dict_)],
            compat="override",
        )

        # Interpolate Ft onto environment grid if provided
        Ft = self._Ft_raw
        Ft_coords_dict = self._Ft_coords_dict_raw
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
            self.results_ = xr.merge(
                [self.results_, _dict_to_dataset({"Ft": self.Ft_}, self.variable_info_dict_, self.coordinates_dict_)],
                compat="override",
            )

        # BEST MODEL: Ft_hat (fit place fields using the true latent positions)
        M_best = self._M_step(self.Y_, self.Xt_)
        E_best = self._E_step(self.Y_, M_best["F"])
        self._append_baseline_epoch(M_best, E_best, epoch_id=-1)

        if self.Ft_ is not None:
            # EXACT MODEL: Ft (use the exact receptive fields)
            M_exact = {
                "F": self.Ft_,
                "FX": self._interpolate_firing_rates(self.Xt_, self.Ft_),
                "PX": M_best["PX"],
            }
            E_exact = self._E_step(self.Y_, self.Ft_)
            self._append_baseline_epoch(M_exact, E_exact, epoch_id=-2)
            self.results_ = self.results_.sortby("epoch")
        else:
            warnings.warn("Exact place fields not provided so baselines against the exact model cannot be calculated.")

        # Backfill F_err for training epochs now that Ft_ is available
        if self.Ft_ is not None and "F_err" in self.results_:
            f_err_vals = np.array(self.results_["F_err"].values, dtype=np.float32)
            for i, ep in enumerate(self.results_.epoch.values):
                if ep >= 0:
                    F_ep = jnp.array(self.results_.F.sel(epoch=ep).values.reshape(self.N_neurons_, self.N_bins_))
                    f_err_vals[i] = float(jnp.mean(jnp.linalg.norm(F_ep - self.Ft_, axis=1)))
            self.results_["F_err"] = xr.DataArray(
                f_err_vals, dims=("epoch",), coords={"epoch": self.results_.epoch}, attrs=self.results_["F_err"].attrs
            )

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
    # Plotting
    # ──────────────────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")

    def plot_fitting_summary(
        self,
        show_neurons: bool = True,
        **plot_kwargs,
    ) -> np.ndarray:
        """Two-panel summary: log-likelihood (left) and spatial information (right).

        Parameters
        ----------
        show_neurons : bool
            Show individual neuron dots for per-neuron metrics.
        **plot_kwargs
            Forwarded to the main scatter calls.

        Returns
        -------
        axes : np.ndarray of Axes, shape (2,)
        """
        from simpl.plotting import plot_fitting_summary

        self._check_fitted()
        return plot_fitting_summary(self.results_, show_neurons=show_neurons, **plot_kwargs)

    def plot_latent_trajectory(
        self,
        time_range: tuple[float, float] | None = None,
        epochs: int | tuple[int, ...] | None = None,
        include_ground_truth: bool = True,
        **plot_kwargs,
    ) -> np.ndarray:
        """Plot decoded latent trajectory (one subplot per spatial dimension).

        Parameters
        ----------
        time_range : tuple, optional
            ``(t_start, t_end)`` in seconds.  Default: first 120 s.
        epochs : int or tuple of ints, optional
            Which epoch(s) to show.  Negative values index from the end
            (``-1`` = last epoch).  Default: all epochs.
        include_ground_truth : bool
            Show ``Xt`` as ``"k--"`` if present.
        **plot_kwargs
            Forwarded to ``ax.plot``.

        Returns
        -------
        axes : np.ndarray of Axes, shape (D,)
        """
        from simpl.plotting import plot_latent_trajectory

        self._check_fitted()
        return plot_latent_trajectory(
            self.results_,
            time_range=time_range,
            epochs=epochs,
            include_ground_truth=include_ground_truth,
            **plot_kwargs,
        )

    def plot_receptive_fields(
        self,
        epochs: int | tuple[int, ...] | None = None,
        neurons: list[int] | np.ndarray | None = None,
        include_baselines: bool = False,
        sort_by_spatial_information: bool = False,
        ncols: int = 4,
        **plot_kwargs,
    ) -> np.ndarray:
        """Plot receptive fields for selected neurons.

        Parameters
        ----------
        epochs : int or tuple of int, optional
            Which epoch(s) to show.  Negative values index from the end
            (``-1`` = last epoch).  Default: ``(0, -1)``.
        neurons : array-like, optional
            Subset of neuron indices.  Default: all neurons.
        include_baselines : bool
            Show ground-truth fields (``Ft``) if present, else ``F`` at epoch -1.
        sort_by_spatial_information : bool
            If ``True``, reorder neurons so that the most spatially informative
            appear first (uses the last training epoch).
        ncols : int
            Maximum number of neuron-columns in the grid.
        **plot_kwargs
            Forwarded to ``imshow`` (2-D) or ``plot`` (1-D).

        Returns
        -------
        axes : np.ndarray of Axes
        """
        from simpl.plotting import plot_receptive_fields

        self._check_fitted()
        extent = getattr(self, "environment_", None)
        if extent is not None:
            extent = self.environment_.extent
        return plot_receptive_fields(
            self.results_,
            extent=extent,
            epochs=epochs,
            neurons=neurons,
            include_baselines=include_baselines,
            sort_by_spatial_information=sort_by_spatial_information,
            ncols=ncols,
            **plot_kwargs,
        )

    def plot_all_metrics(
        self,
        show_neurons: bool = True,
        ncols: int = 3,
        **plot_kwargs,
    ) -> np.ndarray:
        """Auto-discover and plot all per-epoch metrics.

        Parameters
        ----------
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
        from simpl.plotting import plot_all_metrics

        self._check_fitted()
        return plot_all_metrics(self.results_, show_neurons=show_neurons, ncols=ncols, **plot_kwargs)

    def plot_spikes(
        self,
        time_range: tuple[float, float] | None = None,
        neurons: list[int] | np.ndarray | None = None,
        sort_by_spatial_information: bool = False,
        cmap: str = "Greys",
        **plot_kwargs,
    ) -> np.ndarray:
        """Visualise spike counts as a heatmap (time × neurons).

        Parameters
        ----------
        time_range : tuple, optional
            ``(t_start, t_end)`` in seconds.  Default: first 120 s.
        neurons : array-like, optional
            Subset of neuron indices to display.  Default: all neurons.
        sort_by_spatial_information : bool
            If ``True``, reorder neurons so that the most spatially informative
            appear at the top of the heatmap (uses the last training epoch).
        cmap : str
            Colormap for ``imshow``.  Default: ``"Greys"``.
        **plot_kwargs
            Forwarded to ``ax.imshow``.

        Returns
        -------
        ax : matplotlib Axes
        """
        from simpl.plotting import plot_spikes

        self._check_fitted()
        return plot_spikes(
            self.results_,
            time_range=time_range,
            neurons=neurons,
            sort_by_spatial_information=sort_by_spatial_information,
            cmap=cmap,
            **plot_kwargs,
        )

    def plot_prediction(
        self,
        Xb: np.ndarray | None = None,
        Xt: np.ndarray | None = None,
        time_range: tuple[float, float] | None = None,
        **plot_kwargs,
    ) -> np.ndarray:
        """Plot predicted trajectory from the most recent ``predict()`` call.

        Parameters
        ----------
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
        from simpl.plotting import plot_prediction

        if not hasattr(self, "prediction_results_"):
            raise RuntimeError("No prediction results. Call predict() first.")
        return plot_prediction(
            self.prediction_results_,
            Xb=Xb,
            Xt=Xt,
            time_range=time_range,
            **plot_kwargs,
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Core algorithm
    # ──────────────────────────────────────────────────────────────────────────

    def _E_step(self, Y: jax.Array, F: jax.Array) -> dict:
        """E-step: decode latent positions and optionally align to behavior.

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
            trial_boundaries=self.trial_boundaries_,
            U=self.Xb_,
            mask=self.spike_mask_,
        )

        # Manifold alignment (fit-time only)
        align_dict = {}
        if self.align_mode_ == "fields":
            current_peaks = get_field_peaks(F, self.xF_)
            source, target = current_peaks, self.Falign_peaks_
        elif self.align_mode_ == "trajectory":
            source, target = E["mu_s"], self.Xalign_
        else:
            source, target = None, None

        if source is not None:
            if self.is_1D_angular:
                angle, _ = cca_angular(source, target)
                E["X"] = _wrap_minuspi_pi(E["mu_s"] + angle)
                align_dict = {"intercept": jnp.atleast_1d(angle)}
            else:
                coef, intercept = cca(source, target)
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
        FX = self._interpolate_firing_rates(X, F)
        M = {"F": F, "F_odd_minutes": F_odd_minutes, "F_even_minutes": F_even_minutes, "FX": FX, "PX": PX}
        return M

    def _decode(
        self,
        Y: jax.Array,
        F: jax.Array,
        trial_boundaries: np.ndarray,
        U: jax.Array | None = None,
        mask: jax.Array | None = None,
    ) -> dict:
        """Core decoding: likelihood maps -> Gaussian fit -> Kalman filter/smooth.

        Shared by ``_E_step`` (fit-time) and ``predict`` (inference-time).
        Trial boundary masks and per-trial initial states are computed
        internally from ``trial_boundaries`` and the likelihood modes.

        Parameters
        ----------
        Y : jnp.ndarray, shape (T, N_neurons)
            Spike counts.
        F : jnp.ndarray, shape (N_neurons, N_bins)
            Place fields.
        trial_boundaries : np.ndarray
            Array of indices where new trials start, e.g. ``[0, 1000, 2000]``.
        U : jnp.ndarray or None, shape (T, D)
            Control inputs (Xb at fit-time, None/zeros at predict-time).
        mask : jnp.ndarray or None, shape (T, N_neurons)
            Boolean mask (True = use for likelihood). If None, all spikes are used.

        Returns
        -------
        dict
            Decode results including mu_l, mode_l, sigma_l, mu_f, sigma_f, mu_s, sigma_s,
            logPYF, logPYF_val, and optionally logPYXF_maps.
        """
        T = Y.shape[0]
        if mask is None:
            mask = jnp.ones(Y.shape, dtype=bool)
        if U is None:
            U = jnp.zeros((T, self.D_))

        _, trial_slices, is_boundary, is_trial_end = self._validate_trial_boundaries(trial_boundaries, T)

        # Log-likelihood maps
        logPYXF_maps = poisson_log_likelihood(Y, F, mask=mask)
        no_spikes = jnp.sum(Y * mask, axis=1) == 0

        # Fit Gaussians
        mu_l, mode_l, sigma_l = fit_gaussian(self.xF_, jnp.exp(logPYXF_maps))

        # Observation noise (inflated when no spikes)
        observation_noise = jnp.where(
            no_spikes[:, None, None],
            jnp.eye(self.D_) * 1e6,
            sigma_l,
        )

        # Per-trial initial states (mean and covariance of likelihood modes over each trial)
        mu0_all, sigma0_all = self._per_trial_initial_states(mode_l, trial_slices)

        # Single-pass filter and smooth
        mu_f, sigma_f = self.kalman_filter_.filter(
            mu0=mu0_all[0],
            sigma0=sigma0_all[0],
            Y=mode_l,
            U=U,
            R=observation_noise,
            is_boundary=is_boundary,
            mu0_all=mu0_all,
            sigma0_all=sigma0_all,
        )
        mu_s, sigma_s = self.kalman_filter_.smooth(
            mus_f=mu_f,
            sigmas_f=sigma_f,
            is_trial_end=is_trial_end,
        )

        # Likelihoods (full arrays, single sum)
        logPYF = self.kalman_filter_.loglikelihood(Y=mode_l, R=observation_noise, mu=mu_s, sigma=sigma_s).sum()

        # Validation likelihood
        logPYXF_maps_val = poisson_log_likelihood(Y, F, mask=~mask)
        no_spikes_val = jnp.sum(Y * ~mask, axis=1) == 0
        _, mode_l_val, sigma_l_val = fit_gaussian(self.xF_, jnp.exp(logPYXF_maps_val))
        observation_noise_val = jnp.where(
            no_spikes_val[:, None, None],
            jnp.eye(self.D_) * 1e6,
            sigma_l_val,
        )
        logPYF_val = self.kalman_filter_.loglikelihood(
            Y=mode_l_val, R=observation_noise_val, mu=mu_s, sigma=sigma_s
        ).sum()

        E = {
            "mu_l": mu_l,
            "mode_l": mode_l,
            "sigma_l": sigma_l,
            "mu_f": mu_f,
            "sigma_f": sigma_f,
            "mu_s": mu_s,
            "sigma_s": sigma_s,
            "logPYF": logPYF,
            "logPYF_val": logPYF_val,
        }
        E["logPYXF_maps"] = logPYXF_maps

        return E

    # ──────────────────────────────────────────────────────────────────────────
    # Training loop
    # ──────────────────────────────────────────────────────────────────────────

    def _fit_epoch(self) -> None:
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
        self._evaluate_epoch()
        ll_data = self._get_loglikelihoods(Y=self.Y_, FX=self.M_["FX"])
        loglikelihoods = _dict_to_dataset(ll_data, self.variable_info_dict_, self.coordinates_dict_).expand_dims(
            {"epoch": [self.epoch_]}
        )
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

    def _evaluate_epoch(self) -> None:
        """Evaluate the current epoch's metrics and append to the results Dataset.

        Computes log-likelihoods, spatial information, stability, place field analysis,
        and (if ground truth is available) R², trajectory error, and field error. The
        results are stored under the current epoch in ``self.results_``.
        """
        evals = self._get_metrics(
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
        epoch_data = {**self.M_, **self.E_, **evals}
        if not self.save_full_history_:
            epoch_data.pop("FX", None)
        epoch_data.pop("logPYXF_maps", None)
        results = _dict_to_dataset(epoch_data, self.variable_info_dict_, self.coordinates_dict_).expand_dims(
            {"epoch": [self.epoch_]}
        )
        self.results_ = xr.concat([self.results_, results], dim="epoch", data_vars="minimal")

    def _fit_N_epochs(self, N: int, early_stopping: bool = True, verbose: bool = True) -> None:
        """Train for N epochs with KeyboardInterrupt and early stopping support."""
        if N <= 0:
            return
        patience = 3
        best_val_ll = -np.inf
        epochs_without_improvement = 0
        self._print_epoch_summary()
        for _ in range(N):
            try:
                self._fit_epoch()
                if verbose:
                    self._print_epoch_summary()
                if early_stopping:
                    val_ll = float(self.loglikelihoods_.logPYXF_val.sel(epoch=self.epoch_).values)
                    if val_ll > best_val_ll:
                        best_val_ll = val_ll
                        epochs_without_improvement = 0
                    else:
                        epochs_without_improvement += 1
                        if epochs_without_improvement >= patience:
                            if verbose:
                                print(
                                    f"  Early stopping: validation log-likelihood has not "
                                    f"improved for {patience} epochs."
                                )
                            break
            except KeyboardInterrupt:
                print(f"Training interrupted after {self.epoch_} epochs.")
                break

    def _run_epoch_zero(self, verbose: bool) -> None:
        """Run epoch 0 (M-step on behavioral trajectory) and print diagnostics."""
        _epoch0_done = threading.Event()

        def _delayed_message():
            if not _epoch0_done.wait(3):
                print("  ...[estimating spatial receptive fields from Xb (epoch 0)]")

        _timer = threading.Thread(target=_delayed_message, daemon=True)
        _timer.start()
        self._fit_epoch()
        self.FX_firstepoch_ = self.M_["FX"]
        if self.align_mode_ == "fields":
            self.Falign_peaks_ = get_field_peaks(self.M_["F"], self.xF_)
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
            print("FITTING:")

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

    def _append_baseline_epoch(self, M: dict, E: dict, epoch_id: int) -> None:
        """Evaluate a baseline model and append it as a single epoch to results."""
        evals = self._get_metrics(
            X=E["X"],
            F=M["F"],
            Y=self.Y_,
            FX=M["FX"],
            F_odd_mins=M.get("F_odd_minutes"),
            F_even_mins=M.get("F_even_minutes"),
            X_prev=None,
            F_prev=None,
            Xt=self.Xt_,
            Ft=self.Ft_,
            PX=M["PX"],
        )
        data = {**M, **E, **evals}
        if not self.save_full_history_:
            data.pop("FX", None)
        data.pop("logPYXF_maps", None)
        results = _dict_to_dataset(data, self.variable_info_dict_, self.coordinates_dict_).expand_dims(
            {"epoch": [epoch_id]}
        )
        self.results_ = xr.concat([self.results_, results], dim="epoch", data_vars="minimal")

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation
    # ──────────────────────────────────────────────────────────────────────────

    def _init_from_data(self, Y, Xb, time, trial_boundaries, align_to_behavior, verbose) -> None:
        """Validate inputs and initialise all internal state from raw data."""
        # ── Validate inputs ──
        time = np.asarray(time, dtype=float)

        if len(Xb.shape) == 1:  # Handle 1D case where Xb is (T,) instead of (T, 1)
            Xb = Xb[:, np.newaxis]
        if Y.shape[0] != Xb.shape[0]:
            raise ValueError(f"Y and Xb must have the same number of time bins (got {Y.shape[0]} and {Xb.shape[0]})")
        if time.ndim != 1:
            raise ValueError(f"time must be a 1D array (got shape {time.shape})")
        if Y.shape[0] != len(time):
            raise ValueError(f"Y and time must have the same length (got {Y.shape[0]} and {len(time)})")
        if len(time) < 2:
            raise ValueError("time must contain at least 2 samples so dt can be inferred")
        if not np.all(np.isfinite(time)):
            raise ValueError("time must contain only finite values")

        dt = np.diff(time)
        if not np.all(dt > 0):
            raise ValueError("time must be strictly increasing")
        dt_median = np.median(dt)
        if (
            self.use_kalman_smoothing
            and dt_median > 0
            and np.max(np.abs(dt - dt_median)) / dt_median > 0.01
            and trial_boundaries is None
        ):
            warnings.warn(
                f"time is not uniformly spaced (dt varies by more than 1% around the median dt={dt_median:.6g}s). "
                "The Kalman filter assumes a constant dt. Consider using `trial_boundaries` to separate "
                "non-contiguous segments"
            )
        if not 0 < self.val_frac < 1:
            raise ValueError(f"val_frac must be between 0 and 1 (exclusive), got {self.val_frac}")
        if self.speckle_block_size_seconds <= 0:
            raise ValueError(
                "speckle_block_size_seconds must be positive so the held-out mask spans at least one time bin"
            )

        # ── Build environment and extract dimensions ──
        self.D_ = Xb.shape[1]
        self.T_ = Y.shape[0]
        self.N_neurons_ = Y.shape[1]
        self.N_PFmax_ = 20

        if self.is_1D_angular:
            if self.D_ != 1:
                raise ValueError("Circular mode currently supports only 1D latent variables")

            circular_lims = ((-np.pi,), (np.pi,))
            if self._environment_override is not None:
                self.environment_ = self._environment_override
            else:
                if self.env_lims is not None:
                    env_lims_array = np.asarray(self.env_lims, dtype=float)
                    if not np.allclose(env_lims_array, circular_lims, atol=1e-6):
                        raise ValueError("Circular mode requires env_lims to span the full [-pi, pi) domain")
                if self.env_pad != 0:
                    warnings.warn("env_pad is ignored when is_1D_angular=True; using the full [-pi, pi) domain.")
                self.environment_ = Environment(
                    Xb, pad=0.0, bin_size=self.bin_size, force_lims=circular_lims, verbose=False
                )

            environment_lims = np.asarray(self.environment_.lims, dtype=float)
            if self.environment_.D != 1 or not np.allclose(environment_lims, np.asarray(circular_lims), atol=1e-6):
                raise ValueError("Circular mode requires an Environment spanning the full [-pi, pi) domain")
        else:
            if self._environment_override is not None:
                self.environment_ = self._environment_override
            else:
                self.environment_ = Environment(
                    Xb, pad=self.env_pad, bin_size=self.bin_size, force_lims=self.env_lims, verbose=False
                )

        if self.D_ != self.environment_.D:
            raise ValueError(f"Data has {self.D_} dimensions but environment has {self.environment_.D}")

        env_lo = np.asarray(self.environment_.lims[0])
        env_hi = np.asarray(self.environment_.lims[1])
        data_lo = Xb.min(axis=0)
        data_hi = Xb.max(axis=0)
        if np.any(data_lo < env_lo) or np.any(data_hi > env_hi):
            warnings.warn(
                f"Behavioural data spans [{data_lo}, {data_hi}] but the environment only covers "
                f"[{env_lo}, {env_hi}]. Positions outside the environment have no matching spatial bins, "
                "which may degrade receptive field estimates and decoding accuracy. "
                "Consider increasing `env_pad` or adjusting `env_lims`."
            )

        # ── Convert data to JAX arrays (on the chosen device) ──
        neurons = np.arange(self.N_neurons_)
        device = self._jax_device()
        self.Y_ = jax.device_put(jnp.array(Y), device)
        self.Xb_ = jax.device_put(jnp.array(Xb), device)
        self.time_ = jax.device_put(jnp.array(time), device)
        self.neuron_ = jax.device_put(jnp.array(neurons), device)
        self.dt_ = float(np.median(dt))

        self.xF_ = jnp.array(self.environment_.flattened_discretised_coords)
        self.xF_shape_ = self.environment_.discrete_env_shape
        self.N_bins_ = len(self.xF_)

        # ── Check speed prior against data ──
        displacements = np.sqrt(np.sum(np.diff(Xb, axis=0) ** 2, axis=1))
        median_speed = float(np.median(displacements / self.dt_))
        if median_speed > 0 and self.speed_prior < 0.2 * median_speed:
            warnings.warn(
                f"speed_prior ({self.speed_prior:.4g}) is much slower than the median behavioural speed "
                f"({median_speed:.4g}). This may impede the decoded trajectory. "
                f"Consider increasing speed_prior (e.g. to {median_speed:.2g} or higher)."
            )

        # ── Set up Kalman filter, masks, and alignment ──
        self.trial_boundaries_, self.trial_slices_, _, _ = self._validate_trial_boundaries(trial_boundaries, self.T_)
        self._init_kalman_filter()

        self.block_size_ = max(1, int(np.ceil(self.speckle_block_size_seconds / self.dt_)))
        if self.block_size_ >= self.T_:
            raise ValueError(
                "speckle_block_size_seconds must be shorter than the recording duration so both train and validation "
                f"observations remain available (got block_size={self.block_size_} bins for T={self.T_})"
            )
        self.spike_mask_ = create_speckled_mask(
            size=(self.T_, self.N_neurons_),
            sparsity=self.val_frac,
            block_size=self.block_size_,
            random_seed=self.random_seed,
        )
        n_train = int(np.asarray(self.spike_mask_).sum())
        if n_train == 0:
            raise ValueError(
                "The held-out mask produced an empty train split (no spikes are used for fitting). "
                "Adjust val_frac or speckle_block_size_seconds."
            )

        self.odd_minute_mask_ = jnp.stack([jnp.array(self.time_ // 60 % 2 == 0)] * self.N_neurons_, axis=1)
        self.even_minute_mask_ = ~self.odd_minute_mask_

        if align_to_behavior is True:
            align_to_behavior = "trajectory"
        if align_to_behavior and align_to_behavior not in ("trajectory", "fields"):
            raise ValueError(
                f"align_to_behavior must be True, False, 'trajectory', or 'fields', got {align_to_behavior!r}"
            )
        self.align_mode_ = align_to_behavior if align_to_behavior else None
        self.Xalign_ = self.Xb_ if self.align_mode_ else None
        self._kde = kde_angular if self.is_1D_angular else kde

        self.lastF_, self.lastX_ = None, None
        self.epoch_ = -1

        # ── Build coordinate system and variable registry ──
        self.dim_ = self.environment_.dim
        self.variable_info_dict_ = _build_variable_info_dict(self.dim_)
        self.coordinates_dict_ = {
            "neuron": self.neuron_,
            "time": self.time_,
            "dim": self.dim_,
            "dim_": self.dim_,
            **self.environment_.coords_dict,
            "place_field": jnp.arange(self.N_PFmax_),
        }

        # ── Initialise empty results datasets ──
        self.results_ = xr.Dataset(coords={"epoch": jnp.array([], dtype=int)})
        self.results_.attrs = self._build_dataset_attrs(trial_boundaries=self.trial_boundaries_)
        data_dict = {"Xb": self.Xb_, "Y": self.Y_, "spike_mask": self.spike_mask_}
        self.results_ = xr.merge(
            [self.results_, _dict_to_dataset(data_dict, self.variable_info_dict_, self.coordinates_dict_)],
            compat="override",
        )
        self.loglikelihoods_ = xr.Dataset(coords={"epoch": jnp.array([], dtype=int)})

        # Preserve ground truth if add_baselines was called before fit
        if not getattr(self, "ground_truth_available_", False):
            self.Xt_ = None
            self.Ft_ = None
            self.ground_truth_available_ = False
        else:
            # Xt_ will be set from raw data in _apply_baselines_to_results after fit
            self.Xt_ = jnp.array(self._Xt_raw)
            self.Ft_ = None  # Ft needs environment grid, processed in _apply_baselines_to_results

        if verbose:
            self._print_data_summary()

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

        behavior_sigma = self.behavior_prior if self.behavior_prior is not None else 1e6
        lam = behavior_sigma**2 / (speed_sigma**2 + behavior_sigma**2)
        sigma_eff_square = speed_sigma**2 * behavior_sigma**2 / (speed_sigma**2 + behavior_sigma**2)

        F = lam * jnp.eye(self.D_)
        B = (1 - lam) * jnp.eye(self.D_)
        Q = sigma_eff_square * jnp.eye(self.D_)
        H = jnp.eye(self.D_)

        self.kalman_filter_ = KalmanFilter(
            dim_Z=self.D_,
            dim_Y=self.D_,
            dim_U=self.D_,
            batch_size=self.T_,  # i.e. one long batch.
            F=F,
            B=B,
            Q=Q,
            H=H,
            R=None,
            is_1D_angular=self.is_1D_angular,
            force_cpu=True,  # Kalman almost always faster on CPU due to low per-step compute and GPU kernel overhead
        )

    # ──────────────────────────────────────────────────────────────────────────
    # Display
    # ──────────────────────────────────────────────────────────────────────────

    def _print_data_summary(self) -> None:
        """Print a summary of the input data and environment."""
        # Build a minimal xr.Dataset for print_data_summary
        data = xr.Dataset(
            {
                "Y": xr.DataArray(
                    self.Y_, dims=["time", "neuron"], coords={"time": self.time_, "neuron": self.neuron_}
                ),
                "Xb": xr.DataArray(self.Xb_, dims=["time", "dim"], coords={"time": self.time_}),
            }
        )
        Xb_np = np.asarray(self.Xb_)
        env = self.environment_
        Xb_extent_str = " x ".join(f"[{Xb_np[:, i].min():.2f}, {Xb_np[:, i].max():.2f}]" for i in range(self.D_))
        extent_str = " x ".join(f"[{env.extent[2 * i]:.2f}, {env.extent[2 * i + 1]:.2f}]" for i in range(env.D))
        print("ENVIRONMENT:")
        print(f"  Extent:     {extent_str} (Xb: {Xb_extent_str})")
        print(f"  Bin size:   {env.bin_size}")
        print(f"  Grid shape: {env.discrete_env_shape} ({self.N_bins_} bins)")
        print(f"  Padding:    {env.pad}")
        print()
        print_data_summary(data)

    def _print_epoch_summary(self) -> None:
        """Print a one-line summary of the current epoch's metrics."""
        e = self.epoch_
        try:
            train_ll = float(self.loglikelihoods_.logPYXF.sel(epoch=e).values)
            val_ll = float(self.loglikelihoods_.logPYXF_val.sel(epoch=e).values)
            bps_train = float(self.loglikelihoods_.bits_per_spike.sel(epoch=e).values)
            bps_val = float(self.loglikelihoods_.bits_per_spike_val.sel(epoch=e).values)
            si = float(self.results_.spatial_information.sel(epoch=e).mean().values)
            parts = [
                f"Epoch {e:<3d}:    log-likelihood(train={train_ll:.4f}, val={val_ll:.4f})",
                f"bits/spike(train={bps_train:.4f}, val={bps_val:.4f})",
                f"spatial_info={si:.4f} bits/s/neuron",
            ]
            print("     ".join(parts))
            if e > 0:
                epoch0_val_ll = float(self.loglikelihoods_.logPYXF_val.sel(epoch=0).values)
                if val_ll < epoch0_val_ll:
                    print("    WARNING: validation LL below epoch 0. Model may be overfitting.")
        except Exception:
            print(f"Epoch {e}")

    # ──────────────────────────────────────────────────────────────────────────
    # Utilities and static methods
    # ──────────────────────────────────────────────────────────────────────────

    def _interpolate_firing_rates(self, X: jax.Array, F: jax.Array) -> jax.Array:
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
        data = _dict_to_dataset({"F": F, "X": X}, self.variable_info_dict_, self.coordinates_dict_)
        coord_args = {dim: data.X.sel(dim=dim) for dim in self.dim_}
        FX = data.F.sel(**coord_args, method="nearest").T
        return FX.data

    def _get_loglikelihoods(self, Y: jax.Array, FX: jax.Array) -> dict:
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
            Dictionary with 'logPYXF' and 'logPYXF_val' keys.
        """
        LLs = {}
        train_mask = self.spike_mask_
        val_mask = ~self.spike_mask_

        ll_model_train = poisson_log_likelihood_trajectory(Y, FX, mask=train_mask).sum()
        ll_model_val = poisson_log_likelihood_trajectory(Y, FX, mask=val_mask).sum()
        logPYXF = ll_model_train / train_mask.sum()
        logPYXF_val = ll_model_val / val_mask.sum()
        LLs["logPYXF"] = logPYXF
        LLs["logPYXF_val"] = logPYXF_val

        # Bits per spike: (ll_model - ll_mean_rate) / (n_spikes * log2)
        mean_FX = (Y * train_mask).sum(axis=0, keepdims=True) / train_mask.sum(axis=0, keepdims=True)
        mean_FX = jnp.broadcast_to(mean_FX, FX.shape)

        ll_model_by_suffix = {"": ll_model_train, "_val": ll_model_val}

        for mask, suffix in [(train_mask, ""), (val_mask, "_val")]:
            ll_model = ll_model_by_suffix[suffix]
            ll_baseline = poisson_log_likelihood_trajectory(Y, mean_FX, mask=mask).sum()
            n_spikes = (Y * mask).sum()
            bps = jnp.where(n_spikes > 0, (ll_model - ll_baseline) / (n_spikes * jnp.log(2.0)), 0.0)
            LLs[f"bits_per_spike{suffix}"] = bps

        return LLs

    def _get_metrics(
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
            LLs = self._get_loglikelihoods(Y, FX)
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
            # Per-neuron Pearson correlation (avoids building a full (2N x 2N) matrix)
            odd_c = F_odd_mins - jnp.mean(F_odd_mins, axis=1, keepdims=True)
            even_c = F_even_mins - jnp.mean(F_even_mins, axis=1, keepdims=True)
            num = jnp.sum(odd_c * even_c, axis=1)
            denom = jnp.sqrt(jnp.sum(odd_c**2, axis=1) * jnp.sum(even_c**2, axis=1))
            stability = num / (denom + 1e-12)
            metrics["stability"] = stability

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

    @staticmethod
    def _validate_trial_boundaries(trial_boundaries, T):
        """Validate trial boundaries and build boolean masks for the Kalman filter.

        Parameters
        ----------
        trial_boundaries : np.ndarray or None
            Trial start indices, e.g. ``[0, 1000, 2000]``.  If None, treats
            all data as one trial.
        T : int
            Total number of time bins.

        Returns
        -------
        trial_boundaries : np.ndarray
            Validated boundaries array.
        trial_slices : list[slice]
            List of slice objects for each trial.
        is_boundary : jax.Array, shape (T,)
            True at the first timestep of each trial.
        is_trial_end : jax.Array, shape (T,)
            True at the last timestep of each trial.
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

        is_boundary = jnp.zeros(T, dtype=bool)
        is_trial_end = jnp.zeros(T, dtype=bool)
        for trial_slice in trial_slices:
            is_boundary = is_boundary.at[trial_slice.start].set(True)
            is_trial_end = is_trial_end.at[trial_slice.stop - 1].set(True)
        return trial_boundaries, trial_slices, is_boundary, is_trial_end

    @staticmethod
    def _per_trial_initial_states(mode_l, trial_slices):
        """Compute per-trial initial states from likelihood modes.

        For each trial, the initial mean is the average of the likelihood modes
        over the trial, and the initial covariance is the sample covariance.

        Parameters
        ----------
        mode_l : jax.Array, shape (T, D)
            Likelihood modes at each timestep.
        trial_slices : list[slice]
            Trial boundaries as slices.

        Returns
        -------
        mu0_all : jax.Array, shape (T, D)
            Per-timestep initial means (meaningful only at trial starts).
        sigma0_all : jax.Array, shape (T, D, D)
            Per-timestep initial covariances (meaningful only at trial starts).
        """
        T, D = mode_l.shape
        mu0_all = jnp.zeros((T, D))
        sigma0_all = jnp.zeros((T, D, D))
        for trial_slice in trial_slices:
            modes = mode_l[trial_slice]
            mu = modes.mean(axis=0)
            sigma = (1 / len(modes)) * ((modes - mu).T @ (modes - mu))
            mu0_all = mu0_all.at[trial_slice.start].set(mu)
            sigma0_all = sigma0_all.at[trial_slice.start].set(sigma)
        return mu0_all, sigma0_all

    def _build_dataset_attrs(self, trial_boundaries) -> dict:
        """Build the standard attrs dict for results datasets."""
        return {
            "env_extent": self.environment_.extent,
            "env_pad": self.environment_.pad,
            "bin_size": self.environment_.bin_size,
            "trial_boundaries": trial_boundaries,
            "kernel_bandwidth": self.kernel_bandwidth,
            "speed_prior": self.speed_prior,
            "use_kalman_smoothing": int(self.use_kalman_smoothing),
            "behavior_prior": np.nan if self.behavior_prior is None else self.behavior_prior,
            "is_1D_angular": int(self.is_1D_angular),
            "environment_provided": int(self._environment_override is not None),
            "val_frac": self.val_frac,
            "speckle_block_size_seconds": self.speckle_block_size_seconds,
            "random_seed": self.random_seed,
        }


if __name__ == "__main__":
    pass
