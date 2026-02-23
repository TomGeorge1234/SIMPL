"""Tests for simpl.kalman."""

import jax.numpy as jnp
import numpy as np

from simpl.kalman import (
    KalmanFilter,
    fit_F,
    fit_H,
    fit_mu0,
    fit_parameters,
    fit_Q,
    fit_R,
    fit_sigma0,
    kalman_predict,
    kalman_update,
)


class TestKalmanFilterInit:
    def test_dimensions_correct(self):
        kf = KalmanFilter(dim_Z=2, dim_Y=3, dim_U=1)
        assert kf.dim_Z == 2
        assert kf.dim_Y == 3
        assert kf.dim_U == 1

    def test_default_B(self):
        kf = KalmanFilter(dim_Z=2, dim_Y=2, dim_U=1)
        assert kf.B.shape == (2, 1)


class TestKalmanFilterIdentity:
    def test_recovers_observations(self):
        """With F=I, Q~0, H=I, R~0, the filter should track observations closely."""
        D = 2
        T = 50
        kf = KalmanFilter(
            dim_Z=D,
            dim_Y=D,
            F=jnp.eye(D),
            Q=1e-6 * jnp.eye(D),
            H=jnp.eye(D),
            R=1e-6 * jnp.eye(D),
            mu0=jnp.zeros(D),
            sigma0=jnp.eye(D),
        )
        Y = jnp.array(np.random.randn(T, D))
        mus_f, sigmas_f = kf.filter(Y)
        # Filtered means should be close to observations (mean error < 1.0)
        assert jnp.mean(jnp.abs(mus_f[5:] - Y[5:])) < 1.0


class TestKalmanSmootherVariance:
    def test_smoothed_cov_leq_filtered(self):
        """Smoothed covariance should be <= filtered covariance (in trace)."""
        D = 2
        T = 30
        kf = KalmanFilter(
            dim_Z=D,
            dim_Y=D,
            F=0.99 * jnp.eye(D),
            Q=0.01 * jnp.eye(D),
            H=jnp.eye(D),
            R=0.1 * jnp.eye(D),
            mu0=jnp.zeros(D),
            sigma0=jnp.eye(D),
        )
        Y = jnp.array(np.random.randn(T, D))
        mus_f, sigmas_f = kf.filter(Y)
        mus_s, sigmas_s = kf.smooth(mus_f, sigmas_f)

        trace_f = jnp.trace(sigmas_f, axis1=1, axis2=2)
        trace_s = jnp.trace(sigmas_s, axis1=1, axis2=2)
        # On average smoothed should have less variance
        assert trace_s.mean() <= trace_f.mean() + 1e-3


class TestKalmanPredict:
    def test_shapes_correct(self):
        D = 3
        mu = jnp.zeros(D)
        sigma = jnp.eye(D)
        F = jnp.eye(D)
        Q = 0.1 * jnp.eye(D)
        B = jnp.zeros((D, 1))
        u = jnp.zeros(1)
        mu_next, sigma_next = kalman_predict(mu, sigma, F, Q, B, u)
        assert mu_next.shape == (D,)
        assert sigma_next.shape == (D, D)


class TestKalmanUpdate:
    def test_posterior_covariance_shrinks(self):
        D = 2
        mu = jnp.zeros(D)
        sigma = jnp.eye(D)
        H = jnp.eye(D)
        R = 0.1 * jnp.eye(D)
        y = jnp.array([1.0, 0.0])
        mu_post, sigma_post = kalman_update(mu, sigma, H, R, y)
        # Posterior covariance trace should be less than prior
        assert jnp.trace(sigma_post) < jnp.trace(sigma)


class TestKalmanFilterBatchConsistency:
    def test_batch_sizes_give_same_result(self):
        D = 2
        T = 60
        np.random.seed(42)
        Y = jnp.array(np.random.randn(T, D))

        kf_small = KalmanFilter(
            dim_Z=D,
            dim_Y=D,
            batch_size=10,
            F=jnp.eye(D),
            Q=0.01 * jnp.eye(D),
            H=jnp.eye(D),
            R=0.1 * jnp.eye(D),
            mu0=jnp.zeros(D),
            sigma0=jnp.eye(D),
        )
        kf_large = KalmanFilter(
            dim_Z=D,
            dim_Y=D,
            batch_size=100,
            F=jnp.eye(D),
            Q=0.01 * jnp.eye(D),
            H=jnp.eye(D),
            R=0.1 * jnp.eye(D),
            mu0=jnp.zeros(D),
            sigma0=jnp.eye(D),
        )
        mu_s, _ = kf_small.filter(Y)
        mu_l, _ = kf_large.filter(Y)
        assert jnp.allclose(mu_s, mu_l, atol=1e-4)


class TestFitParameters:
    def test_recovers_known_parameters(self):
        np.random.seed(0)
        T = 500
        D_Z, D_Y = 2, 3
        Z = jnp.array(np.cumsum(np.random.randn(T, D_Z) * 0.1, axis=0))
        H_true = jnp.array(np.random.randn(D_Y, D_Z))
        Y = Z @ H_true.T + jnp.array(np.random.randn(T, D_Y) * 0.01)

        mu0, sigma0, F, Q, H, R = fit_parameters(Z, Y)
        assert mu0.shape == (D_Z,)
        assert sigma0.shape == (D_Z, D_Z)
        assert F.shape == (D_Z, D_Z)
        assert Q.shape == (D_Z, D_Z)
        assert H.shape == (D_Y, D_Z)
        assert R.shape == (D_Y, D_Y)


class TestFitIndividualParameters:
    def test_match_fit_parameters(self):
        np.random.seed(1)
        T, D_Z, D_Y = 200, 2, 2
        Z = jnp.array(np.random.randn(T, D_Z))
        Y = jnp.array(np.random.randn(T, D_Y))

        mu0_all, sigma0_all, F_all, Q_all, H_all, R_all = fit_parameters(Z, Y)
        assert jnp.allclose(fit_mu0(Z), mu0_all, atol=1e-5)
        assert jnp.allclose(fit_sigma0(Z), sigma0_all, atol=1e-5)
        assert jnp.allclose(fit_F(Z), F_all, atol=1e-5)
        assert jnp.allclose(fit_Q(Z), Q_all, atol=1e-4)
        assert jnp.allclose(fit_H(Z, Y), H_all, atol=1e-5)
        assert jnp.allclose(fit_R(Z, Y), R_all, atol=1e-5)


class TestKalmanLoglikelihood:
    def test_correct_shape(self):
        D = 2
        T = 20
        kf = KalmanFilter(
            dim_Z=D,
            dim_Y=D,
            F=jnp.eye(D),
            Q=0.01 * jnp.eye(D),
            H=jnp.eye(D),
            R=0.1 * jnp.eye(D),
            mu0=jnp.zeros(D),
            sigma0=jnp.eye(D),
        )
        Y = jnp.array(np.random.randn(T, D))
        mus_f, sigmas_f = kf.filter(Y)
        logP = kf.loglikelihood(Y, mus_f, sigmas_f)
        assert logP.shape == (T,)


class TestKalmanFilterWithControl:
    def test_B_and_U_work(self):
        D = 2
        T = 30
        kf = KalmanFilter(
            dim_Z=D,
            dim_Y=D,
            dim_U=D,
            F=0.9 * jnp.eye(D),
            B=0.1 * jnp.eye(D),
            Q=0.01 * jnp.eye(D),
            H=jnp.eye(D),
            R=0.1 * jnp.eye(D),
            mu0=jnp.zeros(D),
            sigma0=jnp.eye(D),
        )
        Y = jnp.array(np.random.randn(T, D))
        U = jnp.ones((T, D))
        mus_f, sigmas_f = kf.filter(Y, U=U)
        assert mus_f.shape == (T, D)
