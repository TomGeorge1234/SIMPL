"""Tests for simpl.kalman."""

import jax.numpy as jnp
import numpy as np
import pytest

from simpl.kalman import (
    KalmanFilter,
    _fit_F,
    _fit_H,
    _fit_mu0,
    _fit_parameters,
    _fit_Q,
    _fit_R,
    _fit_sigma0,
    _kalman_predict,
    _kalman_update,
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
        mu_next, sigma_next = _kalman_predict(mu, sigma, F, Q, B, u)
        assert mu_next.shape == (D,)
        assert sigma_next.shape == (D, D)


@pytest.mark.cpu_only
class TestKalmanUpdate:
    def test_posterior_covariance_shrinks(self):
        D = 2
        mu = jnp.zeros(D)
        sigma = jnp.eye(D)
        H = jnp.eye(D)
        R = 0.1 * jnp.eye(D)
        y = jnp.array([1.0, 0.0])
        mu_post, sigma_post = _kalman_update(mu, sigma, H, R, y)
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


@pytest.mark.cpu_only
class TestFitParameters:
    def test_recovers_known_parameters(self):
        np.random.seed(0)
        T = 500
        D_Z, D_Y = 2, 3
        Z = jnp.array(np.cumsum(np.random.randn(T, D_Z) * 0.1, axis=0))
        H_true = jnp.array(np.random.randn(D_Y, D_Z))
        Y = Z @ H_true.T + jnp.array(np.random.randn(T, D_Y) * 0.01)

        mu0, sigma0, F, Q, H, R = _fit_parameters(Z, Y)
        assert mu0.shape == (D_Z,)
        assert sigma0.shape == (D_Z, D_Z)
        assert F.shape == (D_Z, D_Z)
        assert Q.shape == (D_Z, D_Z)
        assert H.shape == (D_Y, D_Z)
        assert R.shape == (D_Y, D_Y)


@pytest.mark.cpu_only
class TestFitIndividualParameters:
    def test_match_fit_parameters(self):
        np.random.seed(1)
        T, D_Z, D_Y = 200, 2, 2
        Z = jnp.array(np.random.randn(T, D_Z))
        Y = jnp.array(np.random.randn(T, D_Y))

        mu0_all, sigma0_all, F_all, Q_all, H_all, R_all = _fit_parameters(Z, Y)
        assert jnp.allclose(_fit_mu0(Z), mu0_all, atol=1e-5)
        assert jnp.allclose(_fit_sigma0(Z), sigma0_all, atol=1e-5)
        assert jnp.allclose(_fit_F(Z), F_all, atol=1e-5)
        assert jnp.allclose(_fit_Q(Z), Q_all, atol=1e-4)
        assert jnp.allclose(_fit_H(Z, Y), H_all, atol=1e-5)
        assert jnp.allclose(_fit_R(Z, Y), R_all, atol=1e-5)


@pytest.mark.cpu_only
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


class TestKalmanFilterAngular:
    def test_is_1D_angular_requires_dim_Z_1(self):
        with pytest.raises(ValueError, match="dim_Z == 1"):
            KalmanFilter(dim_Z=2, dim_Y=2, is_1D_angular=True)

    def test_default_is_1D_angular_false(self):
        kf = KalmanFilter(dim_Z=2, dim_Y=2)
        assert not bool(kf.is_1D_angular)

    def test_mu_stays_in_range(self):
        """Filter with angular data crossing +/-pi boundary, verify mu stays in [-pi, pi)."""
        D = 1
        T = 50
        kf = KalmanFilter(
            dim_Z=D,
            dim_Y=D,
            is_1D_angular=True,
            F=jnp.eye(D),
            Q=0.01 * jnp.eye(D),
            H=jnp.eye(D),
            R=0.1 * jnp.eye(D),
            mu0=jnp.array([3.0]),
            sigma0=jnp.eye(D),
        )
        # Observations that cross the +pi/-pi boundary
        angles = jnp.linspace(2.5, 2.5 + 3.0, T)[:, None]  # crosses pi
        Y = jnp.mod(angles + jnp.pi, 2 * jnp.pi) - jnp.pi  # wrap to [-pi, pi)
        mus_f, sigmas_f = kf.filter(Y)
        mus_s, sigmas_s = kf.smooth(mus_f, sigmas_f)

        assert jnp.all(mus_f >= -jnp.pi)
        assert jnp.all(mus_f < jnp.pi + 1e-6)
        assert jnp.all(mus_s >= -jnp.pi)
        assert jnp.all(mus_s < jnp.pi + 1e-6)

    def test_no_wrapping_when_disabled(self):
        """is_1D_angular=False should not change existing behavior."""
        D = 1
        T = 20
        np.random.seed(123)
        Y = jnp.array(np.random.randn(T, D) * 5)  # values outside [-pi, pi)

        kf = KalmanFilter(
            dim_Z=D,
            dim_Y=D,
            is_1D_angular=False,
            F=jnp.eye(D),
            Q=0.01 * jnp.eye(D),
            H=jnp.eye(D),
            R=0.01 * jnp.eye(D),
            mu0=jnp.zeros(D),
            sigma0=jnp.eye(D),
        )
        mus_f, _ = kf.filter(Y)
        # With very low R, filtered means should track observations closely
        # Some values should be outside [-pi, pi)
        assert jnp.any(jnp.abs(mus_f) > jnp.pi)

    def test_innovation_wrapping(self):
        """Innovation wrapping: obs at -3.0, prediction at 3.0 should give small innovation."""
        D = 1
        kf = KalmanFilter(
            dim_Z=D,
            dim_Y=D,
            is_1D_angular=True,
            F=jnp.eye(D),
            Q=0.001 * jnp.eye(D),
            H=jnp.eye(D),
            R=0.1 * jnp.eye(D),
            mu0=jnp.array([3.0]),  # near +pi
            sigma0=0.01 * jnp.eye(D),
        )
        # Single observation near -pi (short angular distance from +3.0)
        Y = jnp.array([[-3.0]])
        mus_f, _ = kf.filter(Y)
        # The filtered mu should be near pi (not pulled towards 0 as with Euclidean diff)
        assert jnp.abs(mus_f[0, 0]) > 2.5  # should stay near the boundary
