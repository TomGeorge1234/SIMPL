"""Tests for simpl.utils."""

import jax.numpy as jnp
import jax.random as random
import numpy as np
import xarray as xr

from simpl.utils import (
    _bin_indices_minuspi_pi,
    _circular_conv_fft_1d,
    _wrap_minuspi_pi,
    accumulate_spikes,
    calculate_spatial_information,
    cca,
    coarsen_dt,
    coefficient_of_determination,
    correlation_at_lag,
    create_speckled_mask,
    fit_gaussian,
    fit_gaussian_vmap,
    gaussian_norm_const,
    gaussian_pdf,
    gaussian_sample,
    load_datafile,
    load_results,
    log_gaussian_pdf,
    print_data_summary,
)


class TestGaussianPdf:
    def test_correct_shape(self):
        x = jnp.array([0.0, 0.0])
        mu = jnp.array([0.0, 0.0])
        sigma = jnp.eye(2)
        result = gaussian_pdf(x, mu, sigma)
        assert result.shape == ()

    def test_known_value_1d(self):
        x = jnp.array([0.0])
        mu = jnp.array([0.0])
        sigma = jnp.array([[1.0]])
        result = gaussian_pdf(x, mu, sigma)
        expected = 1.0 / jnp.sqrt(2 * jnp.pi)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_peak_at_mean(self):
        mu = jnp.array([1.0, 2.0])
        sigma = 0.5 * jnp.eye(2)
        at_mean = gaussian_pdf(mu, mu, sigma)
        off_mean = gaussian_pdf(mu + 1.0, mu, sigma)
        assert at_mean > off_mean


class TestLogGaussianPdf:
    def test_consistent_with_pdf(self):
        x = jnp.array([0.5, -0.3])
        mu = jnp.array([0.0, 0.0])
        sigma = jnp.eye(2) * 2.0
        log_val = log_gaussian_pdf(x, mu, sigma)
        val = gaussian_pdf(x, mu, sigma)
        assert jnp.allclose(log_val, jnp.log(val), atol=1e-5)


class TestGaussianNormConst:
    def test_1d(self):
        sigma = jnp.array([[1.0]])
        result = gaussian_norm_const(sigma)
        expected = 1.0 / jnp.sqrt(2 * jnp.pi)
        assert jnp.allclose(result, expected, atol=1e-5)

    def test_2d(self):
        sigma = jnp.eye(2)
        result = gaussian_norm_const(sigma)
        expected = 1.0 / (2 * jnp.pi)
        assert jnp.allclose(result, expected, atol=1e-5)


class TestFitGaussian:
    def test_fit_to_known_distribution(self):
        # Create a known Gaussian-like likelihood on a grid
        x = jnp.linspace(-3, 3, 100)[:, None]
        true_mu = jnp.array([1.0])
        likelihood = jnp.exp(-0.5 * ((x - true_mu) ** 2).sum(axis=1))
        mu, mode, cov = fit_gaussian(x, likelihood)
        assert jnp.allclose(mu, true_mu, atol=0.1)
        assert jnp.allclose(mode, true_mu, atol=0.1)

    def test_returns_correct_shapes(self):
        x = jnp.ones((50, 2))
        x = x.at[:, 0].set(jnp.linspace(-1, 1, 50))
        x = x.at[:, 1].set(jnp.linspace(-1, 1, 50))
        likelihood = jnp.ones(50)
        mu, mode, cov = fit_gaussian(x, likelihood)
        assert mu.shape == (2,)
        assert mode.shape == (2,)
        assert cov.shape == (2, 2)


class TestFitGaussianVmap:
    def test_matches_unbatched(self):
        x = jnp.linspace(-3, 3, 50)[:, None]
        # Stack two likelihood vectors: peaked at 0 and 1
        L = jnp.stack(
            [
                jnp.exp(-0.5 * (x[:, 0] - 0.0) ** 2),
                jnp.exp(-0.5 * (x[:, 0] - 1.0) ** 2),
            ]
        )
        mus, modes, covs = fit_gaussian_vmap(x, L)
        mu0, mode0, cov0 = fit_gaussian(x, L[0])
        assert jnp.allclose(mus[0], mu0, atol=1e-5)
        assert jnp.allclose(modes[0], mode0, atol=1e-5)


class TestGaussianSample:
    def test_correct_shape(self):
        key = random.PRNGKey(0)
        mu = jnp.array([0.0, 0.0])
        sigma = jnp.eye(2)
        sample = gaussian_sample(key, mu, sigma)
        assert sample.shape == (2,)

    def test_reproducibility(self):
        mu = jnp.array([1.0])
        sigma = jnp.array([[0.5]])
        s1 = gaussian_sample(random.PRNGKey(42), mu, sigma)
        s2 = gaussian_sample(random.PRNGKey(42), mu, sigma)
        assert jnp.allclose(s1, s2)


class TestCoefficientOfDetermination:
    def test_perfect_prediction(self):
        X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        r2 = coefficient_of_determination(X, X)
        assert jnp.allclose(r2, 1.0, atol=1e-5)

    def test_mean_prediction(self):
        Y = jnp.array([[1.0], [2.0], [3.0]])
        X = jnp.full_like(Y, Y.mean())
        r2 = coefficient_of_determination(X, Y)
        assert jnp.allclose(r2, 0.0, atol=1e-5)


class TestCCA:
    def test_identity_mapping(self):
        np.random.seed(42)
        X = jnp.array(np.random.randn(100, 2))
        coef, intercept = cca(X, X)
        X_pred = X @ coef.T + intercept
        assert jnp.allclose(X, X_pred, atol=0.1)


class TestCorrelationAtLag:
    def test_autocorrelation_lag_zero(self):
        X = jnp.array(np.random.randn(100, 2))
        corr = correlation_at_lag(X, X, 0)
        assert jnp.allclose(corr, 1.0, atol=1e-5)


class TestCoarsenDt:
    def test_shape_halved(self):
        T, N, D = 100, 5, 2
        Y = np.random.randint(0, 3, (T, N))
        Xb = np.random.randn(T, D)
        time = np.arange(T, dtype=float) * 0.05
        Y_c, Xb_c, time_c = coarsen_dt(Y, Xb, time, dt_multiplier=2)
        assert Y_c.shape[0] == T // 2
        assert Xb_c.shape[0] == T // 2
        assert time_c.shape[0] == T // 2

    def test_spikes_are_summed(self):
        T, N = 10, 2
        Y = np.ones((T, N), dtype=int)
        Xb = np.zeros((T, 1))
        time = np.arange(T, dtype=float)
        Y_c, _, _ = coarsen_dt(Y, Xb, time, dt_multiplier=5)
        # Each coarsened bin should sum 5 spikes per neuron
        assert np.all(Y_c == 5)

    def test_with_Xt(self):
        T, N, D = 100, 5, 2
        Y = np.random.randint(0, 3, (T, N))
        Xb = np.random.randn(T, D)
        Xt = np.random.randn(T, D)
        time = np.arange(T, dtype=float) * 0.05
        Y_c, Xb_c, time_c, Xt_c = coarsen_dt(Y, Xb, time, dt_multiplier=2, Xt=Xt)
        assert Xt_c.shape[0] == T // 2


class TestCreateSpeckledMask:
    def test_shape(self):
        mask = create_speckled_mask((100, 10), sparsity=0.1, block_size=5)
        assert mask.shape == (100, 10)

    def test_dtype(self):
        mask = create_speckled_mask((100, 10))
        assert mask.dtype == jnp.bool_

    def test_sparsity(self):
        mask = create_speckled_mask((1000, 20), sparsity=0.2, block_size=10)
        false_frac = 1.0 - float(mask.sum()) / mask.size
        assert 0.05 < false_frac < 0.5  # rough check


class TestLoadDatafile:
    def test_loads_successfully(self):
        data = load_datafile()
        assert "Y" in data
        assert "Xb" in data
        assert "time" in data


class TestSaveAndLoadResults:
    def test_round_trip(self, tmp_path, demo_data):
        from simpl.simpl import SIMPL

        N = 500
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL(verbose=False)
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=1,
        )

        path = str(tmp_path / "results.nc")
        model.save_results(path)
        loaded = load_results(path)
        assert "Y" in loaded
        assert "F" in loaded


class TestCalculateSpatialInformation:
    def test_uniform_rate_map_zero_info(self):
        """A uniform firing rate map should have zero spatial information."""
        N_neurons, N_bins = 3, 50
        r = jnp.ones((N_neurons, N_bins)) * 10.0  # uniform 10 Hz everywhere
        PX = jnp.ones(N_bins) / N_bins
        si = calculate_spatial_information(r, PX)
        assert si.shape == (N_neurons,)
        assert jnp.allclose(si, 0.0, atol=1e-5)

    def test_peaked_rate_map_positive_info(self):
        """A peaked firing rate map should have positive spatial information."""
        N_bins = 100
        r = jnp.zeros((1, N_bins))
        r = r.at[0, 50].set(100.0)  # single peak
        PX = jnp.ones(N_bins) / N_bins
        si = calculate_spatial_information(r, PX)
        assert float(si[0]) > 0.0

    def test_returns_bits_per_second(self):
        """Output should scale with firing rate (bits/s, not bits/spike)."""
        N_bins = 50
        PX = jnp.ones(N_bins) / N_bins
        # Peaked rate map at two different overall scales
        r_low = jnp.zeros((1, N_bins)).at[0, 25].set(10.0)
        r_high = jnp.zeros((1, N_bins)).at[0, 25].set(100.0)
        si_low = calculate_spatial_information(r_low, PX)
        si_high = calculate_spatial_information(r_high, PX)
        # Higher rate → more bits per second
        assert float(si_high[0]) > float(si_low[0])

    def test_multiple_neurons(self):
        """Each neuron gets its own independent info value."""
        N_bins = 50
        PX = jnp.ones(N_bins) / N_bins
        r = jnp.ones((2, N_bins))
        # neuron 0: uniform → 0 info; neuron 1: peaked → positive info
        r = r.at[1, 25].set(100.0)
        si = calculate_spatial_information(r, PX)
        assert float(si[0]) < float(si[1])


class TestAccumulateSpikes:
    def _make_Y(self, T=20, N_neurons=3):
        """Create a minimal spike array for testing."""
        Y = np.zeros((T, N_neurons), dtype=int)
        Y[5, 0] = 1  # single spike at t=5 for neuron 0
        Y[10, 1] = 2  # two spikes at t=10 for neuron 1
        return Y

    def test_does_not_modify_original(self):
        """accumulate_spikes should not modify the original array."""
        Y = self._make_Y()
        original_Y = Y.copy()
        _ = accumulate_spikes(Y, window=3)
        np.testing.assert_array_equal(Y, original_Y)

    def test_window_1_is_identity(self):
        """Window of 1 should return the same Y."""
        Y = self._make_Y()
        result = accumulate_spikes(Y, window=1)
        np.testing.assert_array_equal(result, Y)

    def test_rolling_sum(self):
        """With window=3, a spike at t=5 should appear in bins 5, 6, 7."""
        Y = self._make_Y()
        Y_out = accumulate_spikes(Y, window=3)
        # Neuron 0 had a spike only at t=5
        assert Y_out[5, 0] == 1
        assert Y_out[6, 0] == 1
        assert Y_out[7, 0] == 1
        assert Y_out[8, 0] == 0  # spike falls out of window

    def test_causal(self):
        """Spikes should not appear before the original spike time."""
        Y = self._make_Y()
        Y_out = accumulate_spikes(Y, window=5)
        # Neuron 0 spike at t=5 — bins before t=5 should be unaffected
        assert Y_out[4, 0] == 0

    def test_preserves_dtype(self):
        """Output Y should have the same dtype as input Y."""
        Y = self._make_Y()
        result = accumulate_spikes(Y, window=3)
        assert result.dtype == Y.dtype


class TestPrintDataSummary:
    def test_prints_output(self, demo_data, capsys):
        """print_data_summary should produce output with key headings."""
        Y = demo_data["Y"][:500]
        Xb = demo_data["Xb"][:500]
        time = demo_data["time"][:500]
        data = xr.Dataset(
            {
                "Y": xr.DataArray(Y, dims=["time", "neuron"], coords={"time": time}),
                "Xb": xr.DataArray(Xb, dims=["time", "dim"], coords={"time": time}),
            }
        )
        data["trial_slices"] = [slice(0, 500)]
        print_data_summary(data)
        captured = capsys.readouterr().out
        assert "DATA SUMMARY" in captured
        assert "Neurons" in captured
        assert "Dimensions" in captured
        assert "dt" in captured
        assert "Neuron firing rate" in captured
        assert "Simultaneously active neurons per bin" in captured


class TestCircularHelpers:
    def test_wrap_minuspi_pi(self):
        theta = jnp.array([0.0, jnp.pi, -jnp.pi, 3 * jnp.pi, -3 * jnp.pi])
        wrapped = _wrap_minuspi_pi(theta)
        assert jnp.all(wrapped >= -jnp.pi)
        assert jnp.all(wrapped < jnp.pi)

    def test_bin_indices(self):
        theta = jnp.array([-jnp.pi, 0.0, jnp.pi - 0.01])
        idx = _bin_indices_minuspi_pi(theta, 10)
        assert idx[0] == 0
        assert idx[-1] == 9

    def test_circular_conv(self):
        x = jnp.zeros(16)
        x = x.at[8].set(1.0)  # delta
        k = jnp.zeros(16)
        k = k.at[0].set(1.0)  # identity kernel
        result = _circular_conv_fft_1d(x, k)
        assert jnp.allclose(result, x, atol=1e-5)
