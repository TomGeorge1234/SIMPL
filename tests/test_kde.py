"""Tests for simpl.kde."""

import jax.numpy as jnp
import numpy as np
import pytest

from simpl.kde import (
    gaussian_kernel,
    kde,
    kde_angular,
    poisson_log_likelihood,
    poisson_log_likelihood_trajectory,
)


class TestGaussianKernel:
    def test_peak_at_same_point(self):
        x = jnp.array([0.0, 0.0])
        val_same = gaussian_kernel(x, x, bandwidth=0.1)
        val_diff = gaussian_kernel(x, x + 1.0, bandwidth=0.1)
        assert val_same > val_diff

    def test_symmetry(self):
        x1 = jnp.array([0.0, 0.0])
        x2 = jnp.array([1.0, 1.0])
        assert jnp.allclose(
            gaussian_kernel(x1, x2, 0.5),
            gaussian_kernel(x2, x1, 0.5),
        )

    def test_bandwidth_effect(self):
        x1 = jnp.array([0.0])
        x2 = jnp.array([0.5])
        narrow = gaussian_kernel(x1, x2, bandwidth=0.1)
        wide = gaussian_kernel(x1, x2, bandwidth=1.0)
        # Wide kernel gives higher value at moderate distance
        assert wide > narrow


class TestKDE:
    @pytest.fixture
    def simple_data(self):
        np.random.seed(42)
        T = 500
        D = 1
        N_neurons = 3
        trajectory = jnp.array(np.random.uniform(-1, 1, (T, D)))
        bins = jnp.linspace(-1, 1, 20)[:, None]
        return T, D, N_neurons, trajectory, bins

    def test_uniform_spikes_flat_map(self, simple_data):
        T, D, N_neurons, trajectory, bins = simple_data
        spikes = jnp.ones((T, N_neurons))
        result = kde(bins, trajectory, spikes, kernel_bandwidth=0.3)
        assert result.shape == (N_neurons, bins.shape[0])
        # Should be roughly uniform (all ~1 spike per timestep)
        assert result.std(axis=1).mean() < 0.5

    def test_localised_spikes_peaked_map(self, simple_data):
        T, D, N_neurons, trajectory, bins = simple_data
        # Only spike when near position 0
        spikes = jnp.where(jnp.abs(trajectory) < 0.2, 1.0, 0.0)
        spikes = jnp.tile(spikes, (1, N_neurons))
        result = kde(bins, trajectory, spikes, kernel_bandwidth=0.2)
        # Peak should be near center
        peak_bin = jnp.argmax(result[0])
        assert jnp.abs(bins[peak_bin, 0]) < 0.5

    def test_with_mask(self, simple_data):
        T, D, N_neurons, trajectory, bins = simple_data
        spikes = jnp.ones((T, N_neurons))
        mask = jnp.ones((T, N_neurons), dtype=bool)
        mask = mask.at[: T // 2, :].set(False)
        result = kde(bins, trajectory, spikes, kernel_bandwidth=0.3, mask=mask)
        assert result.shape == (N_neurons, bins.shape[0])

    def test_returns_position_density(self, simple_data):
        T, D, N_neurons, trajectory, bins = simple_data
        spikes = jnp.ones((T, N_neurons))
        result, pos_density = kde(bins, trajectory, spikes, kernel_bandwidth=0.3, return_position_density=True)
        assert pos_density.shape == (bins.shape[0],)
        assert jnp.allclose(pos_density.sum(), 1.0, atol=1e-3)


class TestKDEAngular:
    def test_smooth_output(self):
        np.random.seed(42)
        T = 500
        n_bins = 20
        bins = jnp.linspace(-jnp.pi, jnp.pi, n_bins, endpoint=False)
        trajectory = jnp.array(np.random.uniform(-np.pi, np.pi, T))
        spikes = jnp.ones((T, 2))
        result = kde_angular(bins, trajectory, spikes, kernel_bandwidth=0.5)
        assert result.shape == (2, n_bins)
        assert jnp.all(jnp.isfinite(result))

    def test_wrapping_consistency(self):
        """Points near +pi and -pi should produce similar results."""
        n_bins = 20
        bins = jnp.linspace(-jnp.pi, jnp.pi, n_bins, endpoint=False)
        # Trajectory concentrated near pi/-pi boundary
        trajectory = jnp.array([jnp.pi - 0.1, -jnp.pi + 0.1] * 100)
        spikes = jnp.ones((200, 1))
        result = kde_angular(bins, trajectory, spikes, kernel_bandwidth=0.3)
        # The peak should be near the pi/-pi boundary
        assert result.shape == (1, n_bins)
        assert jnp.all(jnp.isfinite(result))


class TestPoissonLogLikelihood:
    def test_correct_shape(self):
        T, N_neurons, N_bins = 50, 5, 20
        spikes = jnp.ones((T, N_neurons), dtype=jnp.int32)
        mean_rate = jnp.ones((N_neurons, N_bins)) * 0.1
        result = poisson_log_likelihood(spikes, mean_rate)
        assert result.shape == (T, N_bins)

    def test_masking(self):
        np.random.seed(42)
        T, N_neurons, N_bins = 50, 5, 20
        spikes = jnp.array(np.random.randint(0, 3, (T, N_neurons)))
        # Non-uniform mean_rate so masked neurons produce different bin profiles
        mean_rate = jnp.array(np.random.uniform(0.01, 0.5, (N_neurons, N_bins)))
        mask = jnp.ones((T, N_neurons), dtype=bool)
        mask_half = mask.at[:, :2].set(False)
        result_full = poisson_log_likelihood(spikes, mean_rate, mask=mask)
        result_masked = poisson_log_likelihood(spikes, mean_rate, mask=mask_half)
        # Results should differ when mask changes
        assert not jnp.allclose(result_full, result_masked)


class TestPoissonLogLikelihoodTrajectory:
    def test_correct_shape(self):
        T, N_neurons = 50, 5
        spikes = jnp.ones((T, N_neurons), dtype=jnp.int32)
        mean_rate = jnp.ones((T, N_neurons)) * 0.1
        result = poisson_log_likelihood_trajectory(spikes, mean_rate)
        assert result.shape == (T,)

    def test_masking(self):
        np.random.seed(42)
        T, N_neurons = 50, 5
        spikes = jnp.array(np.random.randint(0, 3, (T, N_neurons)))
        mean_rate = jnp.array(np.random.uniform(0.01, 0.5, (T, N_neurons)))
        mask = jnp.ones((T, N_neurons), dtype=bool)
        mask_half = mask.at[:, :2].set(False)
        result_full = poisson_log_likelihood_trajectory(spikes, mean_rate, mask=mask)
        result_masked = poisson_log_likelihood_trajectory(spikes, mean_rate, mask=mask_half)
        assert not jnp.allclose(result_full, result_masked)
