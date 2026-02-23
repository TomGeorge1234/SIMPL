"""Kalman filter and smoother implementation in JAX.

Provides a high-level ``KalmanFilter`` class as well as lower-level JIT-compiled
functions for prediction, update, filtering, smoothing, and parameter fitting.
"""

import math

import jax
import jax.numpy as jnp
from jax import jit, vmap

from simpl.utils import gaussian_pdf, log_gaussian_pdf

__all__ = [
    "KalmanFilter",
    "kalman_filter",
    "kalman_smoother",
    "kalman_predict",
    "kalman_update",
    "kalman_likelihoods",
    "calculate_S_matrix",
    "calculate_K_matrix",
    "fit_parameters",
    "fit_mu0",
    "fit_sigma0",
    "fit_F",
    "fit_Q",
    "fit_H",
    "fit_R",
]


class KalmanFilter:
    """A Kalman filter class. This class is used to filter the data and fit the model.

    Written in jax, the lower level functions are jit compiled for speed.
    The filtering and smoothing loops are processed in batches using
    jax.lax.scan(): higher batch sizes will run faster but at the cost of
    a one-off compilation time.

    The Kalman dynamics equations are as follows:
    z_t = F @ z_t-1 + B @ u_t + q_t
    y_t = H @ z_t + r_t
    where z_t is the hidden state, y_t is the observation, u_t is the
    control input, F is the state transition matrix, B is the control
    matrix, H is the observation matrix, q_t ~ N(0, Q) is the state
    transition noise, and r_t ~ N(0, R) is the observation noise.

    Kalman _filtering_ takes observations and estimates the _causal_
    posterior distribution of the hidden state given the observations.
    Kalman _smoothing_ takes the filtered estimates and estimates the
    _posterior_ distribution of the hidden state given all the
    observations.
    mu_filter_t = E[z_t | y_1:t, u_1:t]
    sigma_filter_t = Cov[z_t | y_1:t, u_1:t]
    mu_smooth_t = E[z_t | y_1:T, u_1:T]
    sigma_smooth_t = Cov[z_t | y_1:T, u_1:T]
    """

    def __init__(
        self,
        dim_Z: int,
        dim_Y: int,
        dim_U: int = 0,
        batch_size: int = 100,
        # optional parameters
        mu0: jnp.ndarray = None,
        sigma0: jnp.ndarray = None,
        F: jnp.ndarray = None,
        B: jnp.ndarray = None,
        Q: jnp.ndarray = None,
        H: jnp.ndarray = None,
        R: jnp.ndarray = None,
    ):
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

        Optional parameters
        -------------------
        mu0 : jnp.ndarray, shape (dim_Z,)
            The initial state mean
        sigma0 : jnp.ndarray, shape (dim_Z, dim_Z)
            The initial state covariance
        F : jnp.ndarray, shape (dim_Z, dim_Z)
            The state transition matrix
        B : jnp.ndarray, shape (dim_Z, dim_U)
            The control matrix
        Q : jnp.ndarray, shape (dim_Z, dim_Z)
            The state transition noise covariance
        H : jnp.ndarray, shape (dim_X, dim_Z)
            The observation matrix
        R : jnp.ndarray, shape (dim_X, dim_X)
            The observation noise covariance
        """

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
        Y,
        U=None,
        mu0=None,
        sigma0=None,
        F=None,
        B=None,
        Q=None,
        H=None,
        R=None,
    ):
        """Takes sequences of observations, control inputs, and noise covariances and runs the Kalman filter.

        If parameters are not passed in, the class defaults are used.
        If they are passed in, they must have shape
        (T, *param_shape,) where T is the number of time steps - this
        allows for time-varying parameters.

        Parameters
        ----------
        Y : jnp.ndarray, shape (T, dim_Y)
            The observation means
        U : jnp.ndarray, shape (T, dim_U), optional
            The control inputs (defaults to zeros if not provided)
        mu0 : jnp.ndarray, shape (dim_Z,)
            The initial state mean, optional (default is provided at initialisation)
        sigma0 : jnp.ndarray, shape (dim_Z, dim_Z)
            The initial state covariance, optional (default is provided at initialisation)
        F : jnp.ndarray, shape (T, dim_Z, dim_Z)
            The state transition matrix, optional (default is provided at initialisation)
        B : jnp.ndarray, shape (T, dim_Z, dim_U)
            The control matrix, optional (default is provided at initialisation)
        Q : jnp.ndarray, shape (T, dim_Z, dim_Z)
            The state transition noise covariance, optional (default is provided at initialisation)
        H : jnp.ndarray, shape (T, dim_Z, dim_Z)
            The observation matrix, optional (default is provided at initialisation)
        R : jnp.ndarray, shape (T, dim_Z, dim_Z)
            The observation noise covariances, optional (default is provided at initialisation)


        Returns
        -------
        mus_f : jnp.ndarray, shape (T, dim_Z)
            The filtered means
        sigmas_f : jnp.ndarray, shape (T, dim_Z, dim_Z)
            The filtered covariances
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

        mus_f, sigmas_f = [], []  # filtered means and covariances

        N_batch = math.ceil(T / self.batch_size)
        for i in range(N_batch):
            start = i * self.batch_size
            end = min((i + 1) * self.batch_size, T)
            mu, sigma = kalman_filter(
                Y=Y[start:end],
                U=U[start:end],
                mu0=mu0,
                sigma0=sigma0,
                F=F[start:end],
                B=B[start:end],
                Q=Q[start:end],
                H=H[start:end],
                R=R[start:end],
            )
            mus_f.append(mu)
            sigmas_f.append(sigma)
            mu0, sigma0 = mu[-1], sigma[-1]  # update initial conditions for next batch
        mus_f = jnp.concatenate(mus_f)
        sigmas_f = jnp.concatenate(sigmas_f)

        return mus_f, sigmas_f

    def smooth(
        self,
        mus_f,
        sigmas_f,
        U=None,
        F=None,
        B=None,
        Q=None,
    ):
        """Takes the filtered means, covariances, and control inputs and runs the Kalman smoother on the data.

        Parameters
        ----------
        mus_f : jnp.ndarray, shape (T, dim_Z)
            The filtered means
        sigmas_f : jnp.ndarray, shape (T, dim_Z, dim_Z)
            The filtered covariances
        U : jnp.ndarray, shape (T, dim_U), optional
            The control inputs (defaults to zeros if not provided)
        F : jnp.ndarray, shape (T, dim_Z, dim_Z)
            The state transition matrix, optional
        B : jnp.ndarray, shape (T, dim_Z, dim_U)
            The control matrix, optional
        Q : jnp.ndarray, shape (T, dim_Z, dim_Z)
            The state transition noise covariance, optional

        Returns
        -------
        mus_s : jnp.ndarray, shape (T, dim_Z)
            The smoothed means
        sigmas_s : jnp.ndarray, shape (T, dim_Z, dim_Z)
            The smoothed covariances
        """

        T = len(mus_f)
        muT = mus_f[-1]
        sigmaT = sigmas_f[-1]
        mus_s, sigmas_s = [jnp.array([muT])], [jnp.array([sigmaT])]

        F = self._verify_and_tile(F, self.F, T, (self.dim_Z, self.dim_Z))
        B = self._verify_and_tile(B, self.B, T, (self.dim_Z, self.dim_U))
        Q = self._verify_and_tile(Q, self.Q, T, (self.dim_Z, self.dim_Z))

        if U is None:
            U = jnp.zeros((T, self.dim_U))
        else:
            assert U.ndim == 2
            assert U.shape[0] == T
            assert U.shape[1] == self.dim_U

        for i in range(math.ceil((T - 1) / (self.batch_size))):
            start = max(0, T - 1 - (i + 1) * self.batch_size)
            end = T - 1 - i * self.batch_size
            mu, sigma = kalman_smoother(
                mu=mus_f[start:end],
                sigma=sigmas_f[start:end],
                U=U[start:end],
                muT=muT,
                sigmaT=sigmaT,
                F=F[start:end],
                B=B[start:end],
                Q=Q[start:end],
            )
            mus_s.insert(0, mu)
            sigmas_s.insert(0, sigma)
            muT, sigmaT = mu[0], sigma[0]
        mus_s = jnp.concatenate(mus_s)
        sigmas_s = jnp.concatenate(sigmas_s)

        return mus_s, sigmas_s

    def loglikelihood(
        self,
        Y,
        mu,
        sigma,
        H=None,
        R=None,
    ):
        """Calculates the log-likelihood of the observations, Y.

        Marginalises over the hidden state [mu, sigma] (filtered or
        smoothed). This can be done analytically (see page 361 of the
        Advanced Murphy book).

        P(Y) = Normal(Y | Y_hat, S) where
        S = H @ sigma @ H.T + R (the posterior observation covariance
        combined with the observation noise covariance)
        Y_hat = H @ mu (the predicted observation).

        Parameters
        ----------
        Y : jnp.ndarray, shape (T, dim_Y)
            The observation means
        mu : jnp.ndarray, shape (T, dim_Z)
            The posterior state means
        sigma : jnp.ndarray, shape (T, dim_Z, dim_Z)
            The posterior state covariances
        H: jnp.ndarray, shape (T, dim_Y, dim_Z)
            The observation matrix, optional
        R : jnp.ndarray, shape (T, dim_Y, dim_Y)
            The observation noise covariances, optional

        Returns
        -------
        logP : jnp.ndarray, shape (T,)
            The log-likelihood of the data given the model
        """

        T = len(mu)
        H = self._verify_and_tile(H, self.H, T, (self.dim_Y, self.dim_Z))
        R = self._verify_and_tile(R, self.R, T, (self.dim_Y, self.dim_Y))

        S = vmap(calculate_S_matrix, (0, 0, 0))(sigma, H, R)
        Y_hat = jnp.einsum("ijk,ik->ij", H, mu)  # the "observation" mean
        logP = vmap(log_gaussian_pdf, (0, 0, 0))(Y, Y_hat, S)

        return logP

    def _verify_and_tile(self, param, default_param, T, intended_shape):
        """Verifies the shape of the parameter.

        If the parameter is not passed in, the default parameter
        (presumably set at initialisation) is tiled T-times along a new
        0th axis and used.

        Parameters
        ----------
        param : jnp.ndarray or None
            The parameter to be verified
        default_param : jnp.ndarray
            The default parameter
        T : int
            The number of times to tile the parameter
        intended_shape : tuple
            The intended shape of the parameter

        Returns
        -------
        param : jnp.ndarray
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
def kalman_filter(Y, U, mu0, sigma0, F, B, Q, H, R):
    """Kalman filters a batch of observation data, Y.

    Parameters
    ----------
    Y : jnp.ndarray, shape (T, dim_Y)
        The observation means
    U : jnp.ndarray, shape (T, dim_U)
        The control inputs
    mu0 : jnp.ndarray, shape (dim_Z,)
        The initial state mean
    sigma0 : jnp.ndarray, shape (dim_Z, dim_Z)
        The initial state covariance
    F : jnp.ndarray, shape (T, dim_Z, dim_Z)
        The state transition matrix
    B : jnp.ndarray, shape (T, dim_Z, dim_U)
        The control matrix
    Q : jnp.ndarray, shape (T, dim_Z, dim_Z)
        The state transition noise covariance
    H : jnp.ndarray, shape (T, dim_Y, dim_Z)
        The observation matrix
    R : jnp.ndarray, shape (T, dim_Y, dim_Y)
        The observation noise covariances


    Returns
    -------
    mu : jnp.ndarray, shape (T, dim_Z)
        The filtered posterior state means
    sigma : jnp.ndarray, shape (T, dim_Z, dim_Z)
        The filtered posterior state covariances
    """

    def loop(carry, inputs):
        mu, sigma = carry
        (
            Y,
            u,
            F,
            B,
            Q,
            H,
            R,
        ) = inputs
        mu_p, sigma_p = kalman_predict(mu, sigma, F, Q, B, u)
        mu_u, sigma_u = kalman_update(mu_p, sigma_p, H, R, Y)
        return (mu_u, sigma_u), (mu_u, sigma_u)  # carry, output

    _, (mu_all, sigma_all) = jax.lax.scan(loop, (mu0, sigma0), (Y, U, F, B, Q, H, R))
    return jnp.stack(mu_all), jnp.stack(sigma_all)


@jit
def kalman_smoother(mu, sigma, U, muT, sigmaT, F, B, Q):
    """Runs the Kalman smoother on the data.

    mu and sigma are in forward order, ie.
    mu = [mu[0], mu[1], ... mu[T]] and they are looped over in reverse
    order, so you can still batch the data.


    Parameters
    ----------
    mu : jnp.ndarray, shape (T, dim_Z)
        The filtered posterior state means
    sigma : jnp.ndarray, shape (T, dim_Z, dim_Z)
        The filtered posterior state covariances
    U : jnp.ndarray, shape (T, dim_U)
        The control inputs
    muT : jnp.ndarray, shape (dim_Z,)
        The final state mean - by definition this should have already been smoothed
    sigmaT : jnp.ndarray, shape (dim_Z, dim_Z)
        The final state covariance - by definition this should have already been smoothed
    F : jnp.ndarray, shape (T, dim_Z, dim_Z)
        The state transition matrix
    B : jnp.ndarray, shape (T, dim_Z, dim_U)
        The control matrix
    Q : jnp.ndarray, shape (T, dim_Z, dim_Z)
        The state transition noise covariance

    Returns
    -------
    mu_smooth : jnp.ndarray, shape (T, dim_Z)
        The smoothed state means
    sigma_smooth : jnp.ndarray, shape (T, dim_Z, dim_Z)
        The smoothed state covariances
    """

    def loop(carry, inputs):
        mu, sigma = carry
        mu_, sigma_, u, F, B, Q = inputs
        mu_predict, sigma_predict = kalman_predict(mu_, sigma_, F, Q, B, u)
        J = sigma_ @ F.T @ jnp.linalg.inv(sigma_predict)
        mu_smoothed = mu_ + J @ (mu - mu_predict)
        sigma_smoothed = sigma_ + J @ (sigma - sigma_predict) @ J.T
        return (mu_smoothed, sigma_smoothed), (mu_smoothed, sigma_smoothed)

    _, (mus_all, sigmas_all) = jax.lax.scan(
        loop, (muT, sigmaT), (mu[::-1], sigma[::-1], U[::-1], F[::-1], B[::-1], Q[::-1])
    )
    mus_all = mus_all[::-1]
    sigmas_all = sigmas_all[::-1]  # reverse the order back to forward

    return mus_all, sigmas_all


@jit
def kalman_likelihoods(Z, Y, mu, sigma, F, Q, H, R, B=None, U=None):
    """Calculates the prior P(Z), likelihood P(Y | Z), and posterior P(Z | Y).

    Evaluates any state trajectory (Z) and observations (Y, R) under
    the fitted kalman model. Note although Z and Y can, in principle,
    be _any_ trajectory and observations, typically Z == mu and Y ==
    the observations which were used to fit the model in the first
    place.

    Parameters
    ----------
    Z : jnp.ndarray, shape (T, dim_Z)
        The trajectory of the agent (typical this might just be the same as mu)
    Y : jnp.ndarray, shape (T, dim_Y)
        The observations to be evalauted
    mu : jnp.ndarray, shape (T, dim_Z)
        The posterior state means
    sigma : jnp.ndarray, shape (T, dim_Z, dim_Z)
        The posterior state covariances
    F : jnp.ndarray, shape (T, dim_Z, dim_Z)
        The state transition matrix
    Q : jnp.ndarray, shape (T, dim_Z, dim_Z)
        The state transition noise covariance
    H : jnp.ndarray, shape (T, dim_Y, dim_Z)
        The observation matrix
    R : jnp.ndarray, shape (T, dim_Y, dim_Y)
        The observation noise covariances
    B : jnp.ndarray, shape (T, dim_Z, dim_U), optional
        The control matrix
    U : jnp.ndarray, shape (T, dim_U), optional
        The control inputs

    Returns
    -------
    PZ : jnp.ndarray, shape (T,)
        The likelihood of the state given the previous state
    PZXF : jnp.ndarray, shape (T,)
        The likelihood of the state given the observation
    PXZF : jnp.ndarray, shape (T,)
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


def kalman_predict(mu, sigma, F, Q, B, u):
    """Predicts the next state of the system given the current state and the state transition matrix.

    Parameters
    ----------
    mu : jnp.ndarray, shape (dim_Z,)
        The current state mean
    sigma : jnp.ndarray, shape (dim_Z, dim_Z)
        The current state covariance
    F : jnp.ndarray, shape (dim_Z, dim_Z)
        The state transition matrix
    Q : jnp.ndarray, shape (dim_Z, dim_Z)
        The state transition noise covariance
    B : jnp.ndarray, shape (dim_Z, dim_U)
        The control matrix
    u : jnp.ndarray, shape (dim_U,)
        The control input

    Returns
    -------
    mu_next : jnp.ndarray, shape (dim_Z,)
        The predicted next state mean
    sigma_next : jnp.ndarray, shape (dim_Z, dim_Z)
        The predicted next state covariance
    """
    mu_next = F @ mu + B @ u
    sigma_next = F @ sigma @ F.T + Q
    return mu_next, sigma_next


def kalman_update(mu, sigma, H, R, y):
    """Updates the state estimate given an observation.

    Parameters
    ----------
    mu : jnp.ndarray, shape (dim_Z,)
        The current state mean
    sigma : jnp.ndarray, shape (dim_Z, dim_Z)
        The current state covariance
    H : jnp.ndarray, shape (dim_Y, dim_Z)
        The observation matrix
    R : jnp.ndarray, shape (dim_Y, dim_Y)
        The observation noise covariance
    y : jnp.ndarray, shape (dim_Y,)
        The state observation

    Returns
    -------
    mu_post : jnp.ndarray, shape (dim_Z,)
        The posterior state mean
    sigma_post : jnp.ndarray, shape (dim_Z, dim_Z)
        The posterior state covariance
    """
    S = calculate_S_matrix(sigma, H, R)
    y_hat = H @ mu
    K = calculate_K_matrix(sigma, H, S)
    mu_post = mu + K @ (y - y_hat)
    sigma_post = (jnp.eye(len(mu)) - K @ H) @ sigma

    return mu_post, sigma_post


def calculate_S_matrix(sigma, H, R):
    """Calculates the S matrix for the Kalman filter.

    This doesn't really need to be it's own function but it's useful
    for readability and I vmap it later.

    Parameters
    ----------
    sigma : jnp.ndarray, shape (dim_Z, dim_Z)
        The state covariance
    H : jnp.ndarray, shape (dim_Y, dim_Z)
        The observation matrix
    R : jnp.ndarray, shape (dim_Y, dim_Y)
        The observation noise covariance

    Returns
    -------
    S : jnp.ndarray, shape (dim_Y, dim_Y)
        The S matrix
    """
    return H @ sigma @ H.T + R


def calculate_K_matrix(sigma, H, S):
    """Calculates the K matrix for the Kalman filter.

    This doesn't really need to be it's own function but it's useful
    for readability and I vmap it later.

    Parameters
    ----------
    sigma : jnp.ndarray, shape (dim_Z, dim_Z)
        The state covariance
    H : jnp.ndarray, shape (dim_Y, dim_Z)
        The observation matrix
    S : jnp.ndarray, shape (dim_Y, dim_Y)
        The S matrix

    Returns
    -------
    K : jnp.ndarray, shape (dim_Z, dim_Y)
        The K matrix
    """
    return sigma @ H.T @ jnp.linalg.inv(S)


def fit_parameters(Z, Y):
    """Fits the optimal stationary parameters of the Kalman filter.

    Assuming a training set exists where hidden states Z and
    observations Y are known, this function returns those parameters
    that maximise the likelihood of the data and the state:
    L(Theta) = log({z},{y} | Theta). These solutions are (relatively)
    easy to derive, I took them from Byron Yu's lecture notes (they
    look a lot like linear regression solutions):

    **NOTE: This function assumes NO control input (B=0).** Fitting B
    would require U as an input and a different regression setup
    (e.g., regressing z_t+1 on [z_t, u_t]).

    mu0 = (1/T) sum{zt}
    sigma0 = (1/T) sum{zt - mu0}{zt - mu0}.T
    F = sum{zt+1 @ zt.T} sum{zt zt.T}^-1
    Q = (1/T-1) sum{zt - F @ zt-1}{zt - F @ zt-1}.T
    H = sum{yt @ zt.T} sum{zt zt.T}^-1
    R = (1/T) sum{yt - H @ zt}{yt - H @ zt}.T

    Parameters
    ----------
    Z : jnp.ndarray, shape (T, dim_Z)
        The hidden states (training data)
    Y : jnp.ndarray, shape (T, dim_Y)
        The observations (training data)

    Returns
    -------
    mu0 : jnp.ndarray, shape (dim_Z,)
        The initial state mean
    sigma0 : jnp.ndarray, shape (dim_Z, dim_Z)
        The initial state covariance
    F : jnp.ndarray, shape (dim_Z, dim_Z)
        The state transition matrix
    Q : jnp.ndarray, shape (dim_Z, dim_Z)
        The state transition noise covariance
    H : jnp.ndarray, shape (dim_Y, dim_Z)
        The observation matrix
    R : jnp.ndarray, shape (dim_Y, dim_Y)
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


def fit_mu0(Z):
    """Fits the initial state mean of the Kalman filter.

    Assumes stationary dynamics, see `fit_parameters` for more details.

    Parameters
    ----------
    Z : jnp.ndarray, shape (T, dim_Z)
        The hidden states (training data)

    Returns
    -------
    mu0 : jnp.ndarray, shape (dim_Z,)
        The initial state mean
    """
    return Z.mean(axis=0)


def fit_sigma0(Z):
    """Fits the initial state covariance of the Kalman filter.

    Assumes stationary dynamics, see `fit_parameters` for more details.

    Parameters
    ----------
    Z : jnp.ndarray, shape (T, dim_Z)
        The hidden states (training data)

    Returns
    -------
    sigma0 : jnp.ndarray, shape (dim_Z, dim_Z)
        The initial state covariance
    """
    T = Z.shape[0]
    mu0 = Z.mean(axis=0)
    return (1 / T) * ((Z - mu0).T @ (Z - mu0))


def fit_F(Z):
    """Fits the state transition matrix of the Kalman filter.

    Assumes stationary dynamics **and no control input**, see
    `fit_parameters` for more details.

    Parameters
    ----------
    Z : jnp.ndarray, shape (T, dim_Z)
        The hidden states (training data)

    Returns
    -------
    F : jnp.ndarray, shape (dim_Z, dim_Z)
        The state transition matrix
    """
    return (Z[1:].T @ Z[:-1]) @ jnp.linalg.inv(Z.T @ Z)


def fit_Q(Z):
    """Fits the state transition noise covariance of the Kalman filter.

    Assumes stationary dynamics **and no control input**, see
    `fit_parameters` for more details.

    Parameters
    ----------
    Z : jnp.ndarray, shape (T, dim_Z)
        The hidden states (training data)

    Returns
    -------
    Q : jnp.ndarray, shape (dim_Z, dim_Z)
        The state transition noise covariance
    """
    T = Z.shape[0]
    F = (Z[1:].T @ Z[:-1]) @ jnp.linalg.inv(Z.T @ Z)
    return (1 / (T - 1)) * (Z[1:] - Z[:-1] @ F.T).T @ (Z[1:] - Z[:-1] @ F.T)


def fit_H(Z, Y):
    """Fits the observation matrix of the Kalman filter.

    Assumes stationary dynamics, see `fit_parameters` for more details.

    Parameters
    ----------
    Z : jnp.ndarray, shape (T, dim_Z)
        The hidden states (training data)
    Y : jnp.ndarray, shape (T, dim_Y)
        The observations (training data)

    Returns
    -------
    H : jnp.ndarray, shape (dim_Y, dim_Z)
        The observation matrix
    """
    return (Y.T @ Z) @ jnp.linalg.inv(Z.T @ Z)


def fit_R(Z, Y):
    """Fits the observation noise covariance of the Kalman filter.

    Assumes stationary dynamics, see `fit_parameters` for more details.

    Parameters
    ----------
    Z : jnp.ndarray, shape (T, dim_Z)
        The hidden states (training data)
    Y : jnp.ndarray, shape (T, dim_Y)
        The observations (training data)

    Returns
    -------
    R : jnp.ndarray, shape (dim_Y, dim_Y)
        The observation noise covariance
    """
    T = Z.shape[0]
    H = fit_H(Z, Y)
    return (1 / T) * (Y - Z @ H.T).T @ (Y - Z @ H.T)
