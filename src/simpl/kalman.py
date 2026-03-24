"""Kalman filter and smoother implementation in JAX.

Provides the ``KalmanFilter`` class — the primary interface used by the SIMPL
E-step — as well as lower-level JIT-compiled helper functions for prediction,
update, filtering, smoothing, and parameter fitting.

The Kalman dynamics are:

$$
z_t = F \\, z_{t-1} + B \\, u_t + q_t, \\qquad y_t = H \\, z_t + r_t
$$

where \\(q_t \\sim \\mathcal{N}(0, Q)\\) and
\\(r_t \\sim \\mathcal{N}(0, R)\\).

* **Filtering** estimates the causal posterior
  \\(P(z_t \\mid y_{1:t})\\).
* **Smoothing** refines this to the full posterior
  \\(P(z_t \\mid y_{1:T})\\) using all observations.

For 1-D angular state spaces (``is_1D_angular=True``), the filter and smoother
wrap \\(\\mu\\) to \\([-\\pi, \\pi)\\) after every predict, update, and
smooth step.

The lower-level functions (prefixed with ``_``) mirror ``KalmanFilter``
methods and are not intended for direct use.
"""

import math

import jax
import jax.numpy as jnp
from jax import jit, vmap

from simpl.utils import _wrap_minuspi_pi, gaussian_pdf, log_gaussian_pdf

__all__ = [
    "KalmanFilter",
    "_kalman_filter",
    "_kalman_smoother",
    "_kalman_predict",
    "_kalman_update",
    "_kalman_likelihoods",
    "_calculate_S_matrix",
    "_calculate_K_matrix",
    "_fit_parameters",
    "_fit_mu0",
    "_fit_sigma0",
    "_fit_F",
    "_fit_Q",
    "_fit_H",
    "_fit_R",
]


class KalmanFilter:
    """A Kalman filter class. This class is used to filter the data and fit the model.

    Written in jax, the lower level functions are jit compiled for speed.
    The filtering and smoothing loops are processed in batches using
    jax.lax.scan(): higher batch sizes will run faster but at the cost of
    a one-off compilation time.

    The Kalman dynamics equations are as follows:

    $$
    z_t = F \\, z_{t-1} + B \\, u_t + q_t, \\qquad y_t = H \\, z_t + r_t
    $$

    where \\(z_t\\) is the hidden state, \\(y_t\\) is the observation, \\(u_t\\) is the
    control input, \\(F\\) is the state transition matrix, \\(B\\) is the control
    matrix, \\(H\\) is the observation matrix, \\(q_t \\sim \\mathcal{N}(0, Q)\\) is the state
    transition noise, and \\(r_t \\sim \\mathcal{N}(0, R)\\) is the observation noise.

    Kalman _filtering_ takes observations and estimates the _causal_
    posterior distribution of the hidden state given the observations.
    Kalman _smoothing_ takes the filtered estimates and estimates the
    _posterior_ distribution of the hidden state given all the
    observations.

    \\(\\mu_{\\textrm{filter},t} = \\mathbb{E}[z_t \\mid y_{1:t}, u_{1:t}]\\),
    \\(\\Sigma_{\\textrm{filter},t} = \\textrm{Cov}[z_t \\mid y_{1:t}, u_{1:t}]\\)

    \\(\\mu_{\\textrm{smooth},t} = \\mathbb{E}[z_t \\mid y_{1:T}, u_{1:T}]\\),
    \\(\\Sigma_{\\textrm{smooth},t} = \\textrm{Cov}[z_t \\mid y_{1:T}, u_{1:T}]\\)

    Multi-trial support
    -------------------
    Both ``filter()`` and ``smooth()`` support processing multiple
    concatenated trials in a single pass. Trial boundaries are specified via
    boolean arrays (``is_boundary`` / ``is_trial_end``) that mark where one
    trial ends and the next begins. At these points the filter state is reset
    to per-trial initial conditions (``mu0_all``, ``sigma0_all``), and the
    smoother treats each trial independently by resetting its backward carry
    to the filtered terminal state. This avoids a Python-level loop over
    trials and keeps the full computation inside a single ``jax.lax.scan``.
    """

    def __init__(
        self,
        dim_Z: int,
        dim_Y: int,
        dim_U: int = 0,
        batch_size: int = 100,
        is_1D_angular: bool = False,
        # optional parameters
        mu0: jax.Array | None = None,
        sigma0: jax.Array | None = None,
        F: jax.Array | None = None,
        B: jax.Array | None = None,
        Q: jax.Array | None = None,
        H: jax.Array | None = None,
        R: jax.Array | None = None,
    ) -> None:
        """Initializes the Kalman class.

        The state has size dim_Z, the observations have size dim_Y, and
        the control input has size dim_U.

        If dim_U = 0, no control input is used.

        Parameters F, B, Q, H and R can either be:
        * Passed in at initialisation --> assumed constant over time
        * Passed in at runtime --> assumed to time-vary (additional
          time-dim in along the 0 axis matching the length of the
          observation data)

        Parameters
        ----------
        dim_Z : int
            The size of the state space
        dim_Y : int
            The size of the observation space
        dim_U : int, optional
            The size of the control space (default is 0, for no control)
        batch_size : int
            The batch size for the Kalman filter
        is_1D_angular : bool, optional
            Whether the state is a 1D angle in [-pi, pi). If True, the filter
            and smoother wrap mu to [-pi, pi) after every predict, update, and
            smooth step. Only valid when dim_Z == 1. The wrapped Kalman
            approximation assumes a tight posterior (sigma << 2*pi).
            By default False.

        Optional parameters
        -------------------
        mu0 : jax.Array, shape (dim_Z,)
            The initial state mean
        sigma0 : jax.Array, shape (dim_Z, dim_Z)
            The initial state covariance
        F : jax.Array, shape (dim_Z, dim_Z)
            The state transition matrix
        B : jax.Array, shape (dim_Z, dim_U)
            The control matrix
        Q : jax.Array, shape (dim_Z, dim_Z)
            The state transition noise covariance
        H : jax.Array, shape (dim_X, dim_Z)
            The observation matrix
        R : jax.Array, shape (dim_X, dim_X)
            The observation noise covariance
        """
        if is_1D_angular and dim_Z != 1:
            raise ValueError("is_1D_angular=True requires dim_Z == 1")

        self.is_1D_angular = jnp.array(is_1D_angular)
        self.dim_Z = dim_Z
        self.dim_Y = dim_Y
        self.dim_U = dim_U
        self.batch_size = batch_size

        # Optionally set parameters and initial conditions now
        if mu0 is not None:
            assert mu0.shape == (self.dim_Z,)
        if sigma0 is not None:
            assert sigma0.shape == (self.dim_Z, self.dim_Z)
        if F is not None:
            assert F.shape == (self.dim_Z, self.dim_Z)
        if B is not None:
            assert B.shape == (self.dim_Z, self.dim_U)
        if Q is not None:
            assert Q.shape == (self.dim_Z, self.dim_Z)
        if H is not None:
            assert H.shape == (self.dim_Y, self.dim_Z)
        if R is not None:
            assert R.shape == (self.dim_Y, self.dim_Y)

        self.mu0 = mu0
        self.sigma0 = sigma0
        self.F = F
        self.B = B if B is not None else jnp.zeros((self.dim_Z, self.dim_U))
        self.Q = Q
        self.H = H
        self.R = R

    def filter(
        self,
        Y: jax.Array,
        U: jax.Array | None = None,
        mu0: jax.Array | None = None,
        sigma0: jax.Array | None = None,
        F: jax.Array | None = None,
        B: jax.Array | None = None,
        Q: jax.Array | None = None,
        H: jax.Array | None = None,
        R: jax.Array | None = None,
        is_boundary: jax.Array | None = None,
        mu0_all: jax.Array | None = None,
        sigma0_all: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Run the Kalman filter, optionally over multiple concatenated trials.

        If parameters are not passed in, the class defaults are used.
        If they are passed in, they must have shape
        ``(T, *param_shape)`` where *T* is the number of time steps — this
        allows for time-varying parameters.

        For multi-trial data, pass ``is_boundary``, ``mu0_all``, and
        ``sigma0_all``.  At every timestep where ``is_boundary[t]`` is True
        the filter carry is reset to ``(mu0_all[t], sigma0_all[t])`` before
        the predict/update step, so each trial is filtered independently
        within a single ``jax.lax.scan`` pass.

        Parameters
        ----------
        Y : jax.Array, shape (T, dim_Y)
            The observation means.
        U : jax.Array, shape (T, dim_U), optional
            The control inputs (defaults to zeros if not provided).
        mu0 : jax.Array, shape (dim_Z,), optional
            The initial state mean (default is provided at initialisation).
        sigma0 : jax.Array, shape (dim_Z, dim_Z), optional
            The initial state covariance (default is provided at initialisation).
        F : jax.Array, shape (T, dim_Z, dim_Z), optional
            The state transition matrix (default is provided at initialisation).
        B : jax.Array, shape (T, dim_Z, dim_U), optional
            The control matrix (default is provided at initialisation).
        Q : jax.Array, shape (T, dim_Z, dim_Z), optional
            The state transition noise covariance (default is provided at initialisation).
        H : jax.Array, shape (T, dim_Y, dim_Z), optional
            The observation matrix (default is provided at initialisation).
        R : jax.Array, shape (T, dim_Y, dim_Y), optional
            The observation noise covariances (default is provided at initialisation).
        is_boundary : jax.Array, shape (T,), optional
            Boolean array. True at the first timestep of each trial. The filter
            state is reset to ``mu0_all[t]``, ``sigma0_all[t]`` at these points.
            If None, no resets occur (single-trial behaviour).
        mu0_all : jax.Array, shape (T, dim_Z), optional
            Per-timestep initial means (only read where ``is_boundary`` is True).
        sigma0_all : jax.Array, shape (T, dim_Z, dim_Z), optional
            Per-timestep initial covariances (only read where ``is_boundary`` is True).

        Returns
        -------
        mus_f : jax.Array, shape (T, dim_Z)
            The filtered means.
        sigmas_f : jax.Array, shape (T, dim_Z, dim_Z)
            The filtered covariances.
        """
        assert Y.ndim == 2
        assert Y.shape[1] == self.dim_Y
        T = Y.shape[0]  # number of time steps

        if mu0 is None:
            assert self.mu0 is not None, "You must either pass in the initial conditions or set them at initialisation"
            mu0 = self.mu0
        else:
            assert mu0.ndim == 1
            assert mu0.shape[0] == self.dim_Z
        if sigma0 is None:
            assert self.sigma0 is not None, (
                "You must either pass in the initial conditions or set them at initialisation"
            )
            sigma0 = self.sigma0
        else:
            assert sigma0.ndim == 2
            assert sigma0.shape[0] == self.dim_Z
            assert sigma0.shape[1] == self.dim_Z

        F = self._verify_and_tile(F, self.F, T, (self.dim_Z, self.dim_Z))
        B = self._verify_and_tile(B, self.B, T, (self.dim_Z, self.dim_U))
        Q = self._verify_and_tile(Q, self.Q, T, (self.dim_Z, self.dim_Z))
        H = self._verify_and_tile(H, self.H, T, (self.dim_Y, self.dim_Z))
        R = self._verify_and_tile(R, self.R, T, (self.dim_Y, self.dim_Y))

        if U is None:
            U = jnp.zeros((T, self.dim_U))
        else:
            assert U.ndim == 2
            assert U.shape[0] == T
            assert U.shape[1] == self.dim_U

        if is_boundary is None:
            is_boundary = jnp.zeros(T, dtype=bool)
            mu0_all = jnp.zeros((T, self.dim_Z))
            sigma0_all = jnp.zeros((T, self.dim_Z, self.dim_Z))

        mus_f, sigmas_f = [], []  # filtered means and covariances

        N_batch = math.ceil(T / self.batch_size)
        for i in range(N_batch):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, T)
            mu, sigma = _kalman_filter(
                Y=Y[start:end],
                U=U[start:end],
                mu0=mu0,
                sigma0=sigma0,
                F=F[start:end],
                B=B[start:end],
                Q=Q[start:end],
                H=H[start:end],
                R=R[start:end],
                is_1D_angular=self.is_1D_angular,
                is_boundary=is_boundary[start:end],
                mu0_all=mu0_all[start:end],
                sigma0_all=sigma0_all[start:end],
            )
            mus_f.append(mu)
            sigmas_f.append(sigma)
            mu0, sigma0 = mu[-1], sigma[-1]  # update initial conditions for next batch
        mus_f = jnp.concatenate(mus_f)
        sigmas_f = jnp.concatenate(sigmas_f)

        return mus_f, sigmas_f

    def smooth(
        self,
        mus_f: jax.Array,
        sigmas_f: jax.Array,
        U: jax.Array | None = None,
        F: jax.Array | None = None,
        B: jax.Array | None = None,
        Q: jax.Array | None = None,
        is_trial_end: jax.Array | None = None,
    ) -> tuple[jax.Array, jax.Array]:
        """Run the Rauch-Tung-Striebel smoother, optionally over multiple concatenated trials.

        For multi-trial data, pass ``is_trial_end``.  At every timestep where
        ``is_trial_end[t]`` is True the smoothed output is set to the filtered
        value (terminal condition) and the backward carry is reset, so each
        trial is smoothed independently within a single backward
        ``jax.lax.scan`` pass.

        Parameters
        ----------
        mus_f : jax.Array, shape (T, dim_Z)
            The filtered means.
        sigmas_f : jax.Array, shape (T, dim_Z, dim_Z)
            The filtered covariances.
        U : jax.Array, shape (T, dim_U), optional
            The control inputs (defaults to zeros if not provided).
        F : jax.Array, shape (T, dim_Z, dim_Z), optional
            The state transition matrix.
        B : jax.Array, shape (T, dim_Z, dim_U), optional
            The control matrix.
        Q : jax.Array, shape (T, dim_Z, dim_Z), optional
            The state transition noise covariance.
        is_trial_end : jax.Array, shape (T,), optional
            Boolean array. True at the last timestep of each trial. At these
            points the smoothed state is set to the filtered state (terminal
            condition) and the carry is reset for the next trial's backward
            pass.  If None, only the final timestep is treated as a trial end
            (single-trial behaviour).

        Returns
        -------
        mus_s : jax.Array, shape (T, dim_Z)
            The smoothed means.
        sigmas_s : jax.Array, shape (T, dim_Z, dim_Z)
            The smoothed covariances.
        """

        T = len(mus_f)
        muT = mus_f[-1]
        sigmaT = sigmas_f[-1]

        F = self._verify_and_tile(F, self.F, T, (self.dim_Z, self.dim_Z))
        B = self._verify_and_tile(B, self.B, T, (self.dim_Z, self.dim_U))
        Q = self._verify_and_tile(Q, self.Q, T, (self.dim_Z, self.dim_Z))

        if U is None:
            U = jnp.zeros((T, self.dim_U))
        else:
            assert U.ndim == 2
            assert U.shape[0] == T
            assert U.shape[1] == self.dim_U

        if is_trial_end is None:
            is_trial_end = jnp.zeros(T, dtype=bool).at[-1].set(True)

        mus_s, sigmas_s = [], []

        for i in range(math.ceil(T / self.batch_size)):
            start = max(0, T - (i + 1) * self.batch_size)
            end = T - i * self.batch_size
            mu, sigma = _kalman_smoother(
                mu=mus_f[start:end],
                sigma=sigmas_f[start:end],
                U=U[start:end],
                muT=muT,
                sigmaT=sigmaT,
                F=F[start:end],
                B=B[start:end],
                Q=Q[start:end],
                is_1D_angular=self.is_1D_angular,
                is_trial_end=is_trial_end[start:end],
            )
            mus_s.insert(0, mu)
            sigmas_s.insert(0, sigma)
            muT, sigmaT = mu[0], sigma[0]
        mus_s = jnp.concatenate(mus_s)
        sigmas_s = jnp.concatenate(sigmas_s)

        return mus_s, sigmas_s

    def loglikelihood(
        self,
        Y: jax.Array,
        mu: jax.Array,
        sigma: jax.Array,
        H: jax.Array | None = None,
        R: jax.Array | None = None,
    ) -> jax.Array:
        """Calculates the log-likelihood of the observations, Y.

        Marginalises over the hidden state [mu, sigma] (filtered or
        smoothed). This can be done analytically (see page 361 of the
        Advanced Murphy book).

        \\(P(Y) = \\mathcal{N}(Y \\mid \\hat{Y}, S)\\) where
        \\(S = H \\Sigma H^\\top + R\\) (the posterior observation covariance
        combined with the observation noise covariance) and
        \\(\\hat{Y} = H \\mu\\) (the predicted observation).

        Parameters
        ----------
        Y : jax.Array, shape (T, dim_Y)
            The observation means
        mu : jax.Array, shape (T, dim_Z)
            The posterior state means
        sigma : jax.Array, shape (T, dim_Z, dim_Z)
            The posterior state covariances
        H: jax.Array, shape (T, dim_Y, dim_Z)
            The observation matrix, optional
        R : jax.Array, shape (T, dim_Y, dim_Y)
            The observation noise covariances, optional

        Returns
        -------
        logP : jax.Array, shape (T,)
            The log-likelihood of the data given the model
        """

        T = len(mu)
        H = self._verify_and_tile(H, self.H, T, (self.dim_Y, self.dim_Z))
        R = self._verify_and_tile(R, self.R, T, (self.dim_Y, self.dim_Y))

        S = vmap(_calculate_S_matrix, (0, 0, 0))(sigma, H, R)
        Y_hat = jnp.einsum("ijk,ik->ij", H, mu)  # the "observation" mean
        logP = vmap(log_gaussian_pdf, (0, 0, 0))(Y, Y_hat, S)

        return logP

    def _verify_and_tile(
        self, param: jax.Array | None, default_param: jax.Array | None, T: int, intended_shape: tuple
    ) -> jax.Array:
        """Verifies the shape of the parameter.

        If the parameter is not passed in, the default parameter
        (presumably set at initialisation) is tiled T-times along a new
        0th axis and used.

        Parameters
        ----------
        param : jax.Array or None
            The parameter to be verified
        default_param : jax.Array
            The default parameter
        T : int
            The number of times to tile the parameter
        intended_shape : tuple
            The intended shape of the parameter

        Returns
        -------
        param : jax.Array
            The parameter, shape (T, *intended_shape)
        """
        if param is None:
            assert default_param is not None, "You must either pass in the parameter or set it at initialisation"
            param = jnp.tile(default_param, (T,) + (1,) * default_param.ndim)
        else:
            assert param.shape == (T,) + intended_shape, (
                f"Parameter shape is {param.shape} but should be {(T,) + intended_shape}"
            )
        return param


@jit
def _kalman_filter(
    Y: jax.Array,
    U: jax.Array,
    mu0: jax.Array,
    sigma0: jax.Array,
    F: jax.Array,
    B: jax.Array,
    Q: jax.Array,
    H: jax.Array,
    R: jax.Array,
    is_1D_angular: jax.Array = jnp.array(False),
    is_boundary: jax.Array | None = None,
    mu0_all: jax.Array | None = None,
    sigma0_all: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Kalman filters a batch of observation data, Y.

    Parameters
    ----------
    Y : jax.Array, shape (T, dim_Y)
        The observation means
    U : jax.Array, shape (T, dim_U)
        The control inputs
    mu0 : jax.Array, shape (dim_Z,)
        The initial state mean
    sigma0 : jax.Array, shape (dim_Z, dim_Z)
        The initial state covariance
    F : jax.Array, shape (T, dim_Z, dim_Z)
        The state transition matrix
    B : jax.Array, shape (T, dim_Z, dim_U)
        The control matrix
    Q : jax.Array, shape (T, dim_Z, dim_Z)
        The state transition noise covariance
    H : jax.Array, shape (T, dim_Y, dim_Z)
        The observation matrix
    R : jax.Array, shape (T, dim_Y, dim_Y)
        The observation noise covariances
    is_1D_angular : jax.Array, optional
        Scalar bool. If True, wrap mu to [-pi, pi) after predict and update steps
        and wrap the innovation for angular data. By default False.
    is_boundary : jax.Array, shape (T,), optional
        Boolean array. True at the first timestep of each trial. When True, the
        filter state is reset to ``mu0_all[t]``, ``sigma0_all[t]``.
    mu0_all : jax.Array, shape (T, dim_Z), optional
        Per-timestep initial means (only used where ``is_boundary`` is True).
    sigma0_all : jax.Array, shape (T, dim_Z, dim_Z), optional
        Per-timestep initial covariances (only used where ``is_boundary`` is True).

    Returns
    -------
    mu : jax.Array, shape (T, dim_Z)
        The filtered posterior state means
    sigma : jax.Array, shape (T, dim_Z, dim_Z)
        The filtered posterior state covariances
    """
    T = Y.shape[0]
    dim_Z = mu0.shape[0]
    if is_boundary is None:
        is_boundary = jnp.zeros(T, dtype=bool)
        mu0_all = jnp.zeros((T, dim_Z))
        sigma0_all = jnp.zeros((T, dim_Z, dim_Z))

    def loop(carry, inputs):
        mu, sigma = carry
        Y, u, F, B, Q, H, R, is_bound, mu0_t, sigma0_t = inputs
        # Reset state at trial boundaries
        mu = jnp.where(is_bound, mu0_t, mu)
        sigma = jnp.where(is_bound, sigma0_t, sigma)
        mu_p, sigma_p = _kalman_predict(mu, sigma, F, Q, B, u)
        mu_p = jnp.where(is_1D_angular, _wrap_minuspi_pi(mu_p), mu_p)
        mu_u, sigma_u = _kalman_update(mu_p, sigma_p, H, R, Y, is_1D_angular=is_1D_angular)
        mu_u = jnp.where(is_1D_angular, _wrap_minuspi_pi(mu_u), mu_u)
        return (mu_u, sigma_u), (mu_u, sigma_u)  # carry, output

    _, (mu_all, sigma_all) = jax.lax.scan(
        loop, (mu0, sigma0), (Y, U, F, B, Q, H, R, is_boundary, mu0_all, sigma0_all)
    )
    return jnp.stack(mu_all), jnp.stack(sigma_all)


@jit
def _kalman_smoother(
    mu: jax.Array,
    sigma: jax.Array,
    U: jax.Array,
    muT: jax.Array,
    sigmaT: jax.Array,
    F: jax.Array,
    B: jax.Array,
    Q: jax.Array,
    is_1D_angular: jax.Array = jnp.array(False),
    is_trial_end: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array]:
    """Runs the Kalman smoother on the data.

    mu and sigma are in forward order, ie.
    mu = [mu[0], mu[1], ... mu[T]] and they are looped over in reverse
    order, so you can still batch the data.


    Parameters
    ----------
    mu : jax.Array, shape (T, dim_Z)
        The filtered posterior state means
    sigma : jax.Array, shape (T, dim_Z, dim_Z)
        The filtered posterior state covariances
    U : jax.Array, shape (T, dim_U)
        The control inputs
    muT : jax.Array, shape (dim_Z,)
        The final state mean - by definition this should have already been smoothed
    sigmaT : jax.Array, shape (dim_Z, dim_Z)
        The final state covariance - by definition this should have already been smoothed
    F : jax.Array, shape (T, dim_Z, dim_Z)
        The state transition matrix
    B : jax.Array, shape (T, dim_Z, dim_U)
        The control matrix
    Q : jax.Array, shape (T, dim_Z, dim_Z)
        The state transition noise covariance
    is_1D_angular : jax.Array, optional
        Scalar bool. If True, wrap mu and angular differences to [-pi, pi)
        during smoothing. By default False.
    is_trial_end : jax.Array, shape (T,), optional
        Boolean array. True at the last timestep of each trial. At these points
        the smoothed state is set to the filtered state (terminal condition) and
        the carry is reset for the next trial's backward pass.

    Returns
    -------
    mu_smooth : jax.Array, shape (T, dim_Z)
        The smoothed state means
    sigma_smooth : jax.Array, shape (T, dim_Z, dim_Z)
        The smoothed state covariances
    """
    T = mu.shape[0]
    if is_trial_end is None:
        is_trial_end = jnp.zeros(T, dtype=bool)

    def loop(carry, inputs):
        mu_s_next, sigma_s_next = carry
        mu_f_t, sigma_f_t, u, F, B, Q, is_end = inputs
        # Standard RTS smoother step
        mu_predict, sigma_predict = _kalman_predict(mu_f_t, sigma_f_t, F, Q, B, u)
        mu_predict = jnp.where(is_1D_angular, _wrap_minuspi_pi(mu_predict), mu_predict)
        J = sigma_f_t @ F.T @ jnp.linalg.inv(sigma_predict)
        diff = mu_s_next - mu_predict
        diff = jnp.where(is_1D_angular, _wrap_minuspi_pi(diff), diff)
        mu_smoothed = mu_f_t + J @ diff
        mu_smoothed = jnp.where(is_1D_angular, _wrap_minuspi_pi(mu_smoothed), mu_smoothed)
        sigma_smoothed = sigma_f_t + J @ (sigma_s_next - sigma_predict) @ J.T
        # At trial ends: override with filtered value (terminal condition)
        mu_out = jnp.where(is_end, mu_f_t, mu_smoothed)
        sigma_out = jnp.where(is_end, sigma_f_t, sigma_smoothed)
        return (mu_out, sigma_out), (mu_out, sigma_out)

    _, (mus_all, sigmas_all) = jax.lax.scan(
        loop,
        (muT, sigmaT),
        (mu[::-1], sigma[::-1], U[::-1], F[::-1], B[::-1], Q[::-1], is_trial_end[::-1]),
    )
    mus_all = mus_all[::-1]
    sigmas_all = sigmas_all[::-1]  # reverse the order back to forward

    return mus_all, sigmas_all


@jit
def _kalman_likelihoods(
    Z: jax.Array,
    Y: jax.Array,
    mu: jax.Array,
    sigma: jax.Array,
    F: jax.Array,
    Q: jax.Array,
    H: jax.Array,
    R: jax.Array,
    B: jax.Array | None = None,
    U: jax.Array | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Calculates the prior P(Z), likelihood P(Y | Z), and posterior P(Z | Y).

    Evaluates any state trajectory (Z) and observations (Y, R) under
    the fitted kalman model. Note although Z and Y can, in principle,
    be _any_ trajectory and observations, typically Z == mu and Y ==
    the observations which were used to fit the model in the first
    place.

    Parameters
    ----------
    Z : jax.Array, shape (T, dim_Z)
        The trajectory of the agent (typical this might just be the same as mu)
    Y : jax.Array, shape (T, dim_Y)
        The observations to be evalauted
    mu : jax.Array, shape (T, dim_Z)
        The posterior state means
    sigma : jax.Array, shape (T, dim_Z, dim_Z)
        The posterior state covariances
    F : jax.Array, shape (T, dim_Z, dim_Z)
        The state transition matrix
    Q : jax.Array, shape (T, dim_Z, dim_Z)
        The state transition noise covariance
    H : jax.Array, shape (T, dim_Y, dim_Z)
        The observation matrix
    R : jax.Array, shape (T, dim_Y, dim_Y)
        The observation noise covariances
    B : jax.Array, shape (T, dim_Z, dim_U), optional
        The control matrix
    U : jax.Array, shape (T, dim_U), optional
        The control inputs

    Returns
    -------
    PZ : jax.Array, shape (T,)
        The likelihood of the state given the previous state
    PZXF : jax.Array, shape (T,)
        The likelihood of the state given the observation
    PXZF : jax.Array, shape (T,)
        The likelihood of the observation given the state
    """

    T = Z.shape[0]
    dim_Z = Z.shape[1]
    if B is None:
        dim_U = 0
        B = jnp.zeros((T, dim_Z, dim_U))
        U = jnp.zeros((T, dim_U))
    elif U is None:
        dim_U = B.shape[-1]
        U = jnp.zeros((T, dim_U))

    Z_ = jnp.concatenate((Z[0][None], Z))  # prepend Z0 to Z so its [Z0, Z0, Z1, Z2, ... ZT]
    U_ = jnp.concatenate((U[0][None], U))  # prepend U0 to U so its [U0, U0, U1, U2, ... UT]
    Q_ = jnp.concatenate((Q[0][None], Q))  # prepend Q0 to Q so its [Q0, Q0, Q1, Q2, ... QT]
    F_ = jnp.concatenate((F[0][None], F))  # prepend F0 to F so its [F0, F0, F1, F2, ... FT]
    B_ = jnp.concatenate((B[0][None], B))  # prepend B0 to B so its [B0, B0, B1, B2, ... BT]

    mu_p = jnp.einsum("ijk,ik->ij", F_, Z_) + jnp.einsum("ijk,ik->ij", B_, U_)

    Y_hat = jnp.einsum("ijk,ik->ij", H, mu)  # the "observation" mean
    PZ = vmap(gaussian_pdf, (0, 0, 0))(Z_[1:], mu_p[:-1], Q_[1:])  # zt ~ N(F*zt-1 + B*ut, Qt)
    PZXF = vmap(gaussian_pdf, (0, 0, 0))(Z, mu, sigma)  # zt ~ N(mu, sigma)
    PXZF = vmap(gaussian_pdf, (0, 0, 0))(Y, Y_hat, R)
    return PZ, PZXF, PXZF


def _kalman_predict(
    mu: jax.Array,
    sigma: jax.Array,
    F: jax.Array,
    Q: jax.Array,
    B: jax.Array,
    u: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Predicts the next state of the system given the current state and the state transition matrix.

    $$
    \\mu_{\\textrm{predict}} = F \\mu + B u, \\quad \\Sigma_{\\textrm{predict}} = F \\Sigma F^\\top + Q
    $$

    Parameters
    ----------
    mu : jax.Array, shape (dim_Z,)
        The current state mean
    sigma : jax.Array, shape (dim_Z, dim_Z)
        The current state covariance
    F : jax.Array, shape (dim_Z, dim_Z)
        The state transition matrix
    Q : jax.Array, shape (dim_Z, dim_Z)
        The state transition noise covariance
    B : jax.Array, shape (dim_Z, dim_U)
        The control matrix
    u : jax.Array, shape (dim_U,)
        The control input

    Returns
    -------
    mu_next : jax.Array, shape (dim_Z,)
        The predicted next state mean
    sigma_next : jax.Array, shape (dim_Z, dim_Z)
        The predicted next state covariance
    """
    mu_next = F @ mu + B @ u
    sigma_next = F @ sigma @ F.T + Q
    return mu_next, sigma_next


def _kalman_update(
    mu: jax.Array,
    sigma: jax.Array,
    H: jax.Array,
    R: jax.Array,
    y: jax.Array,
    is_1D_angular: jax.Array = jnp.array(False),
) -> tuple[jax.Array, jax.Array]:
    """Updates the state estimate given an observation.

    Innovation: \\(v = y - H\\mu\\),
    Kalman gain: \\(K = \\Sigma H^\\top S^{-1}\\),
    Posterior: \\(\\mu_{\\textrm{post}} = \\mu + Kv\\),
    \\(\\Sigma_{\\textrm{post}} = (I - KH)\\Sigma\\).

    Parameters
    ----------
    mu : jax.Array, shape (dim_Z,)
        The current state mean
    sigma : jax.Array, shape (dim_Z, dim_Z)
        The current state covariance
    H : jax.Array, shape (dim_Y, dim_Z)
        The observation matrix
    R : jax.Array, shape (dim_Y, dim_Y)
        The observation noise covariance
    y : jax.Array, shape (dim_Y,)
        The state observation
    is_1D_angular : jax.Array, optional
        Scalar bool. If True, wrap the innovation (y - y_hat) to [-pi, pi)
        for angular data. By default False.

    Returns
    -------
    mu_post : jax.Array, shape (dim_Z,)
        The posterior state mean
    sigma_post : jax.Array, shape (dim_Z, dim_Z)
        The posterior state covariance
    """
    S = _calculate_S_matrix(sigma, H, R)
    y_hat = H @ mu
    K = _calculate_K_matrix(sigma, H, S)
    innovation = y - y_hat
    innovation = jnp.where(is_1D_angular, _wrap_minuspi_pi(innovation), innovation)
    mu_post = mu + K @ innovation
    sigma_post = (jnp.eye(len(mu)) - K @ H) @ sigma

    return mu_post, sigma_post


def _calculate_S_matrix(sigma: jax.Array, H: jax.Array, R: jax.Array) -> jax.Array:
    """Calculates the S matrix, \\(S = H \\Sigma H^\\top + R\\), for the Kalman filter.

    This doesn't really need to be it's own function but it's useful
    for readability and I vmap it later.

    Parameters
    ----------
    sigma : jax.Array, shape (dim_Z, dim_Z)
        The state covariance
    H : jax.Array, shape (dim_Y, dim_Z)
        The observation matrix
    R : jax.Array, shape (dim_Y, dim_Y)
        The observation noise covariance

    Returns
    -------
    S : jax.Array, shape (dim_Y, dim_Y)
        The S matrix
    """
    return H @ sigma @ H.T + R


def _calculate_K_matrix(sigma: jax.Array, H: jax.Array, S: jax.Array) -> jax.Array:
    """Calculates the Kalman gain matrix, \\(K = \\Sigma H^\\top S^{-1}\\), for the Kalman filter.

    This doesn't really need to be it's own function but it's useful
    for readability and I vmap it later.

    Parameters
    ----------
    sigma : jax.Array, shape (dim_Z, dim_Z)
        The state covariance
    H : jax.Array, shape (dim_Y, dim_Z)
        The observation matrix
    S : jax.Array, shape (dim_Y, dim_Y)
        The S matrix

    Returns
    -------
    K : jax.Array, shape (dim_Z, dim_Y)
        The K matrix
    """
    return sigma @ H.T @ jnp.linalg.inv(S)


def _fit_parameters(
    Z: jax.Array,
    Y: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Fits the optimal stationary parameters of the Kalman filter.

    Assuming a training set exists where hidden states Z and
    observations Y are known, this function returns those parameters
    that maximise the likelihood of the data and the state:
    \\(\\mathcal{L}(\\Theta) = \\log p(\\{z\\}, \\{y\\} \\mid \\Theta)\\).
    These solutions are (relatively) easy to derive, I took them from
    Byron Yu's lecture notes (they look a lot like linear regression solutions):

    **NOTE: This function assumes NO control input (B=0).** Fitting B
    would require U as an input and a different regression setup
    (e.g., regressing \\(z_{t+1}\\) on \\([z_t, u_t]\\)).

    $$
    \\begin{aligned}
    \\mu_0 &= \\frac{1}{T} \\sum_t z_t \\\\
    \\Sigma_0 &= \\frac{1}{T} \\sum_t (z_t - \\mu_0)(z_t - \\mu_0)^\\top \\\\
    F &= \\left(\\sum_t z_{t+1} z_t^\\top\\right) \\left(\\sum_t z_t z_t^\\top\\right)^{-1} \\\\
    Q &= \\frac{1}{T-1} \\sum_t (z_t - F z_{t-1})(z_t - F z_{t-1})^\\top \\\\
    H &= \\left(\\sum_t y_t z_t^\\top\\right) \\left(\\sum_t z_t z_t^\\top\\right)^{-1} \\\\
    R &= \\frac{1}{T} \\sum_t (y_t - H z_t)(y_t - H z_t)^\\top
    \\end{aligned}
    $$

    Parameters
    ----------
    Z : jax.Array, shape (T, dim_Z)
        The hidden states (training data)
    Y : jax.Array, shape (T, dim_Y)
        The observations (training data)

    Returns
    -------
    mu0 : jax.Array, shape (dim_Z,)
        The initial state mean
    sigma0 : jax.Array, shape (dim_Z, dim_Z)
        The initial state covariance
    F : jax.Array, shape (dim_Z, dim_Z)
        The state transition matrix
    Q : jax.Array, shape (dim_Z, dim_Z)
        The state transition noise covariance
    H : jax.Array, shape (dim_Y, dim_Z)
        The observation matrix
    R : jax.Array, shape (dim_Y, dim_Y)
        The observation noise covariance

    """

    assert Z.ndim == 2
    assert Y.ndim == 2
    T = Z.shape[0]
    assert Y.shape[0] == T

    mu0 = Z.mean(axis=0)
    sigma0 = (1 / T) * ((Z - mu0).T @ (Z - mu0))
    F = (Z[1:].T @ Z[:-1]) @ jnp.linalg.inv(Z.T @ Z)
    Q = (1 / (T - 1)) * (Z[1:] - Z[:-1] @ F.T).T @ (Z[1:] - Z[:-1] @ F.T)
    H = (Y.T @ Z) @ jnp.linalg.inv(Z.T @ Z)
    R = (1 / T) * (Y - Z @ H.T).T @ (Y - Z @ H.T)

    return mu0, sigma0, F, Q, H, R


def _fit_mu0(Z: jax.Array) -> jax.Array:
    """Fits the initial state mean of the Kalman filter.

    Assumes stationary dynamics, see `_fit_parameters` for more details.

    Parameters
    ----------
    Z : jax.Array, shape (T, dim_Z)
        The hidden states (training data)

    Returns
    -------
    mu0 : jax.Array, shape (dim_Z,)
        The initial state mean
    """
    return Z.mean(axis=0)


def _fit_sigma0(Z: jax.Array) -> jax.Array:
    """Fits the initial state covariance of the Kalman filter.

    Assumes stationary dynamics, see `_fit_parameters` for more details.

    Parameters
    ----------
    Z : jax.Array, shape (T, dim_Z)
        The hidden states (training data)

    Returns
    -------
    sigma0 : jax.Array, shape (dim_Z, dim_Z)
        The initial state covariance
    """
    T = Z.shape[0]
    mu0 = Z.mean(axis=0)
    return (1 / T) * ((Z - mu0).T @ (Z - mu0))


def _fit_F(Z: jax.Array) -> jax.Array:
    """Fits the state transition matrix of the Kalman filter.

    Assumes stationary dynamics **and no control input**, see
    `_fit_parameters` for more details.

    Parameters
    ----------
    Z : jax.Array, shape (T, dim_Z)
        The hidden states (training data)

    Returns
    -------
    F : jax.Array, shape (dim_Z, dim_Z)
        The state transition matrix
    """
    return (Z[1:].T @ Z[:-1]) @ jnp.linalg.inv(Z.T @ Z)


def _fit_Q(Z: jax.Array) -> jax.Array:
    """Fits the state transition noise covariance of the Kalman filter.

    Assumes stationary dynamics **and no control input**, see
    `_fit_parameters` for more details.

    Parameters
    ----------
    Z : jax.Array, shape (T, dim_Z)
        The hidden states (training data)

    Returns
    -------
    Q : jax.Array, shape (dim_Z, dim_Z)
        The state transition noise covariance
    """
    T = Z.shape[0]
    F = (Z[1:].T @ Z[:-1]) @ jnp.linalg.inv(Z.T @ Z)
    return (1 / (T - 1)) * (Z[1:] - Z[:-1] @ F.T).T @ (Z[1:] - Z[:-1] @ F.T)


def _fit_H(Z: jax.Array, Y: jax.Array) -> jax.Array:
    """Fits the observation matrix of the Kalman filter.

    Assumes stationary dynamics, see `_fit_parameters` for more details.

    Parameters
    ----------
    Z : jax.Array, shape (T, dim_Z)
        The hidden states (training data)
    Y : jax.Array, shape (T, dim_Y)
        The observations (training data)

    Returns
    -------
    H : jax.Array, shape (dim_Y, dim_Z)
        The observation matrix
    """
    return (Y.T @ Z) @ jnp.linalg.inv(Z.T @ Z)


def _fit_R(Z: jax.Array, Y: jax.Array) -> jax.Array:
    """Fits the observation noise covariance of the Kalman filter.

    Assumes stationary dynamics, see `_fit_parameters` for more details.

    Parameters
    ----------
    Z : jax.Array, shape (T, dim_Z)
        The hidden states (training data)
    Y : jax.Array, shape (T, dim_Y)
        The observations (training data)

    Returns
    -------
    R : jax.Array, shape (dim_Y, dim_Y)
        The observation noise covariance
    """
    T = Z.shape[0]
    H = _fit_H(Z, Y)
    return (1 / T) * (Y - Z @ H.T).T @ (Y - Z @ H.T)
