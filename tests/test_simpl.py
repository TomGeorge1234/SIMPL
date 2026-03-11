"""Integration tests for simpl.simpl.SIMPL."""

import warnings

import numpy as np
import pytest
import xarray as xr

from simpl.environment import Environment
from simpl.simpl import SIMPL
from simpl.utils import load_results


class TestSIMPLInit:
    def test_config_only(self):
        """__init__ should only store config, no data or computation."""
        model = SIMPL(speed_prior=0.2, kernel_bandwidth=0.03)
        assert model.speed_prior == 0.2
        assert model.kernel_bandwidth == 0.03
        assert model.is_fitted_ is False
        assert not hasattr(model, "Y_")
        assert not hasattr(model, "results_")

    def test_use_kalman_smoothing_argument(self, demo_data):
        N = 500
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL(use_kalman_smoothing=False, speed_prior=0.1)
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=0,
        )
        assert model.use_kalman_smoothing is False
        assert model.speed_prior_effective_ >= model.kalman_off_speed_prior_


class TestSIMPLFit:
    def test_fit_returns_self(self, demo_data):
        N = 500
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL()
        result = model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=0,
        )
        assert result is model

    def test_fit_sets_attributes(self, small_simpl_model):
        model = small_simpl_model
        assert model.is_fitted_ is True
        assert hasattr(model, "Y_")
        assert hasattr(model, "Xb_")
        assert hasattr(model, "kalman_filter_")
        assert hasattr(model, "environment_")
        assert hasattr(model, "results_")
        assert hasattr(model, "X_")
        assert hasattr(model, "F_")
        assert model.D_ == 2
        assert model.epoch_ == 0

    def test_fit_creates_environment_internally(self, demo_data):
        N = 500
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL(bin_size=0.03, env_pad=0.05)
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=0,
        )
        assert model.environment_.bin_size == 0.03
        assert model.env_pad == 0.05

    def test_fit_with_custom_environment(self, demo_data):
        N = 500
        N_neurons = min(5, demo_data["Y"].shape[1])
        env = Environment(demo_data["Xb"][:N], bin_size=0.04)
        model = SIMPL(environment=env)
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=0,
        )
        assert model.environment_ is env

    def test_fit_validates_shapes(self):
        model = SIMPL()
        with pytest.raises(ValueError, match="same number of time bins"):
            model.fit(
                Y=np.zeros((100, 5)),
                Xb=np.zeros((200, 2)),
                time=np.arange(100) * 0.05,
                n_epochs=0,
            )

    def test_fit_validates_time_length(self):
        model = SIMPL()
        with pytest.raises(ValueError, match="same length"):
            model.fit(
                Y=np.zeros((100, 5)),
                Xb=np.zeros((100, 2)),
                time=np.arange(50) * 0.05,
                n_epochs=0,
            )

    def test_fit_requires_at_least_two_timepoints(self):
        model = SIMPL()
        with pytest.raises(ValueError, match="at least 2 samples"):
            model.fit(
                Y=np.zeros((1, 5)),
                Xb=np.zeros((1, 2)),
                time=np.array([0.0]),
                n_epochs=0,
            )

    def test_fit_validates_monotonic_time(self):
        model = SIMPL()
        with pytest.raises(ValueError, match="strictly increasing"):
            model.fit(
                Y=np.zeros((4, 5)),
                Xb=np.zeros((4, 2)),
                time=np.array([0.0, 0.05, 0.05, 0.10]),
                n_epochs=0,
            )

    def test_fit_uses_median_dt_when_time_has_gaps(self):
        time = np.array([0.0, 0.05, 0.11, 0.17, 0.23, 0.29, 0.35, 0.41, 0.47, 0.86])
        model = SIMPL(test_frac=0.2, speckle_block_size_seconds=0.11)
        model.fit(
            Y=np.zeros((len(time), 5)),
            Xb=np.zeros((len(time), 2)),
            time=time,
            n_epochs=0,
        )
        assert model.dt_ == pytest.approx(0.06)
        assert model.block_size_ == 2

    @pytest.mark.parametrize("test_frac", [0.0, 1.0, -0.1, 1.1])
    def test_fit_validates_test_frac(self, test_frac):
        model = SIMPL(test_frac=test_frac)
        with pytest.raises(ValueError, match="test_frac"):
            model.fit(
                Y=np.zeros((10, 5)),
                Xb=np.zeros((10, 2)),
                time=np.arange(10) * 0.05,
                n_epochs=0,
            )

    def test_fit_validates_speckle_block_size_duration(self):
        model = SIMPL(speckle_block_size_seconds=1.0)
        with pytest.raises(ValueError, match="shorter than the recording duration"):
            model.fit(
                Y=np.zeros((10, 5)),
                Xb=np.zeros((10, 2)),
                time=np.arange(10) * 0.05,
                n_epochs=0,
            )

    def test_fit_validates_nonempty_mask_split(self):
        model = SIMPL(test_frac=0.01, speckle_block_size_seconds=0.01)
        with pytest.raises(ValueError, match="empty train/test split"):
            model.fit(
                Y=np.zeros((10, 5)),
                Xb=np.zeros((10, 2)),
                time=np.arange(10) * 0.05,
                n_epochs=0,
            )


class TestSIMPLFitResume:
    def test_resume_continues_training(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL()
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=2,
        )
        epoch_after_first = model.epoch_

        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=3,
            resume=True,
        )
        assert model.epoch_ == epoch_after_first + 3

    def test_resume_before_fit_raises(self):
        model = SIMPL()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.fit(
                Y=np.zeros((100, 5)),
                Xb=np.zeros((100, 2)),
                time=np.arange(100) * 0.05,
                n_epochs=1,
                resume=True,
            )


class TestSIMPLTrainOneEpoch:
    def test_runs_and_populates_results(self, small_simpl_model):
        model = small_simpl_model
        assert model.epoch_ == 0
        assert "X" in model.results_
        assert "F" in model.results_


class TestSIMPLResultsStructure:
    def test_correct_xarray_dims(self, small_simpl_model):
        model = small_simpl_model
        assert "epoch" in model.results_.dims
        assert "time" in model.results_.dims
        assert "neuron" in model.results_.dims


class TestSIMPLTrainImprovesLikelihood:
    def test_likelihood_increases(self, demo_data):
        N = 2000
        N_neurons = min(10, demo_data["Y"].shape[1])
        model = SIMPL()
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=3,
        )

        ll_0 = float(model.loglikelihoods_.logPYXF.sel(epoch=0).values)
        ll_last = float(model.loglikelihoods_.logPYXF.sel(epoch=model.epoch_).values)
        assert ll_last >= ll_0 - 0.1


class TestSIMPLEvaluateEpoch:
    def test_metrics_dict_keys(self, small_simpl_model):
        model = small_simpl_model
        assert "logPYXF" in model.results_
        assert "logPYXF_test" in model.results_


class TestSIMPLWithGroundTruth:
    def test_baselines_computed(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL()
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=1,
        )
        model.add_baselines(
            Xt=demo_data["Xt"][:N],
            Ft=demo_data["Ft"][:N_neurons],
            Ft_coords_dict={"x": demo_data["x"], "y": demo_data["y"]},
        )
        assert "X_R2" in model.results_
        assert "X_err" in model.results_
        assert "F_err" in model.results_


class TestSIMPLWithoutGroundTruth:
    def test_works_without_Xt_Ft(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL()
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=1,
        )
        assert "F" in model.results_
        assert "X_R2" not in model.results_


class TestSIMPLTrialBoundaries:
    def test_per_trial_filtering(self, demo_data):
        N = 2000
        N_neurons = min(5, demo_data["Y"].shape[1])
        boundaries = np.array([0, N // 2])
        model = SIMPL()
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=0,
            trial_boundaries=boundaries,
        )
        assert len(model.trial_slices_) == 2
        assert model.epoch_ == 0


class TestSIMPLSaveLoadResults:
    def test_round_trip(self, tmp_path, demo_data):
        N = 500
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL(
            kernel_bandwidth=0.03,
            speed_prior=0.2,
            use_kalman_smoothing=False,
            behavior_prior=0.4,
            is_circular=False,
            bin_size=0.05,
            env_pad=0.0,
            env_lims=((0.0, 0.0), (1.0, 1.0)),
            test_frac=0.2,
            speckle_block_size_seconds=2.0,
            random_seed=7,
        )
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=1,
        )

        assert model.results_.attrs["kernel_bandwidth"] == 0.03
        assert model.results_.attrs["speed_prior"] == 0.2
        assert model.results_.attrs["use_kalman_smoothing"] == 0
        assert model.results_.attrs["behavior_prior"] == 0.4
        assert model.results_.attrs["is_circular"] == 0
        assert model.results_.attrs["bin_size"] == 0.05
        assert model.results_.attrs["env_pad"] == 0.0
        np.testing.assert_allclose(model.results_.attrs["env_extent"], np.array([0.0, 1.0, 0.0, 1.0]))
        assert model.results_.attrs["environment_provided"] == 0
        assert model.results_.attrs["test_frac"] == 0.2
        assert model.results_.attrs["speckle_block_size_seconds"] == 2.0
        assert model.results_.attrs["random_seed"] == 7

        path = str(tmp_path / "test_results.nc")
        model.save_results(path)
        loaded = load_results(path)
        assert "Y" in loaded
        assert "F" in loaded
        assert loaded.F.shape == model.results_.F.shape
        assert loaded.attrs["kernel_bandwidth"] == 0.03
        assert loaded.attrs["use_kalman_smoothing"] == 0
        np.testing.assert_allclose(loaded.attrs["env_extent"], np.array([0.0, 1.0, 0.0, 1.0]))


class TestSIMPLInterpolateFiringRates:
    def test_correct_shape(self, small_simpl_model):
        model = small_simpl_model
        F = model.M_["F"]
        X = model.E_["X"]
        FX = model._interpolate_firing_rates(X, F)
        assert FX.shape == (model.T_, model.N_neurons_)


class TestSIMPLGetLoglikelihoods:
    def test_expected_keys(self, small_simpl_model):
        model = small_simpl_model
        lls = model._get_loglikelihoods(model.Y_, model.M_["FX"])
        assert "logPYXF" in lls
        assert "logPYXF_test" in lls


class TestSIMPLSeeding:
    """Tests for the random seeding mechanism."""

    def _make_model(self, demo_data, random_seed=0):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL(random_seed=random_seed)
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=0,
        )
        return model

    def test_same_seed_same_mask(self, demo_data):
        m1 = self._make_model(demo_data, random_seed=42)
        m2 = self._make_model(demo_data, random_seed=42)
        assert np.array_equal(m1.spike_mask_, m2.spike_mask_)

    def test_different_seed_different_mask(self, demo_data):
        m1 = self._make_model(demo_data, random_seed=42)
        m2 = self._make_model(demo_data, random_seed=99)
        assert not np.array_equal(m1.spike_mask_, m2.spike_mask_)

    def test_mask_unchanged_across_epochs(self, demo_data):
        model = self._make_model(demo_data, random_seed=0)
        mask_init = np.array(model.spike_mask_)
        model._fit_epoch()
        model._fit_epoch()
        assert np.array_equal(mask_init, np.array(model.spike_mask_))


class TestSIMPLSpatialInformation:
    def test_spatial_information_in_results(self, small_simpl_model):
        model = small_simpl_model
        assert "spatial_information" in model.results_
        assert "spatial_information_rate" in model.results_

    def test_spatial_information_shape(self, small_simpl_model):
        model = small_simpl_model
        si = model.results_.spatial_information.sel(epoch=0)
        assert si.shape == (model.N_neurons_,)

    def test_spatial_information_nonnegative(self, small_simpl_model):
        model = small_simpl_model
        si = model.results_.spatial_information.sel(epoch=0).values
        assert np.all(si >= -1e-6)

    def test_spatial_information_rate_is_sum(self, small_simpl_model):
        model = small_simpl_model
        si = model.results_.spatial_information.sel(epoch=0).values
        sir = float(model.results_.spatial_information_rate.sel(epoch=0))
        assert np.isclose(sir, si.sum(), atol=1e-3)


class TestSIMPLEpochZeroInFit:
    def test_epoch_starts_at_zero(self, demo_data):
        N = 500
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL()
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=0,
        )
        assert model.epoch_ == 0
        assert "F" in model.results_


class TestSIMPLAddBaselines:
    def _make_model(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL()
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=0,
        )
        return model, N, N_neurons

    def test_populates_best_and_exact_epochs(self, demo_data):
        model, N, N_neurons = self._make_model(demo_data)
        model.add_baselines(
            Xt=demo_data["Xt"][:N],
            Ft=demo_data["Ft"][:N_neurons],
            Ft_coords_dict={"x": demo_data["x"], "y": demo_data["y"]},
        )
        assert -1 in model.results_.epoch.values
        assert -2 in model.results_.epoch.values

    def test_baseline_results_have_metrics(self, demo_data):
        model, N, N_neurons = self._make_model(demo_data)
        model.add_baselines(
            Xt=demo_data["Xt"][:N],
            Ft=demo_data["Ft"][:N_neurons],
            Ft_coords_dict={"x": demo_data["x"], "y": demo_data["y"]},
        )
        for epoch in [-2, -1]:
            assert "F" in model.results_.sel(epoch=epoch)
            assert "X" in model.results_.sel(epoch=epoch)
            assert "X_R2" in model.results_.sel(epoch=epoch)

    def test_best_only_without_Ft(self, demo_data):
        model, N, _ = self._make_model(demo_data)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.add_baselines(Xt=demo_data["Xt"][:N])
            assert any("Exact place fields" in str(warning.message) for warning in w)
        assert -1 in model.results_.epoch.values

    def test_stores_before_fit(self):
        model = SIMPL()
        model.add_baselines(Xt=np.zeros((100, 2)))
        assert model.ground_truth_available_
        assert model._Xt_raw is not None


class TestSIMPLManifoldAlignment:
    def _make_model(self, demo_data, align_to_behavior=True):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL()
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=1,
            align_to_behavior=align_to_behavior,
        )
        return model

    def test_cca_runs(self, small_simpl_model):
        model = small_simpl_model
        if model.epoch_ < 1:
            model._fit_epoch()
            model._fit_epoch()
        assert model.epoch_ >= 1
        assert "X" in model.E_

    def test_align_to_behavior(self, demo_data):
        model = self._make_model(demo_data, align_to_behavior=True)
        assert model.Xalign_ is not None
        assert "X" in model.E_

    def test_no_alignment(self, demo_data):
        model = self._make_model(demo_data, align_to_behavior=False)
        assert model.Xalign_ is None
        assert "X" in model.E_

    def test_align_trajectory_mode(self, demo_data):
        model = self._make_model(demo_data, align_to_behavior="trajectory")
        assert model.align_mode_ == "trajectory"
        assert model.Xalign_ is not None
        assert "coef" in model.E_
        assert "intercept" in model.E_

    def test_align_fields_mode(self, demo_data):
        model = self._make_model(demo_data, align_to_behavior="fields")
        assert model.align_mode_ == "fields"
        assert hasattr(model, "Falign_peaks_")
        assert model.Falign_peaks_.shape == (model.N_neurons_, model.D_)
        assert "coef" in model.E_
        assert "intercept" in model.E_

    def test_align_invalid_mode_raises(self, demo_data):
        with pytest.raises(ValueError, match="align_to_behavior"):
            self._make_model(demo_data, align_to_behavior="invalid")

    def test_align_angular(self):
        """Field-based angular alignment uses rotation, not CCA."""
        rng = np.random.default_rng(42)
        T, N_neurons = 2000, 15
        time = np.arange(T) * 0.02
        Xb = np.linspace(-np.pi, np.pi, T, endpoint=False)[:, None]
        # Simulate spikes with angular tuning
        preferred = np.linspace(-np.pi, np.pi, N_neurons, endpoint=False)
        rates = np.exp(3 * np.cos(Xb - preferred[None, :]))
        Y = rng.poisson(rates * 0.02)

        model = SIMPL(is_circular=True, bin_size=np.pi / 32, env_pad=0.0, speed_prior=0.1, kernel_bandwidth=0.3)
        model.fit(Y, Xb, time, n_epochs=1, align_to_behavior="fields")
        assert model.align_mode_ == "fields"
        assert "intercept" in model.E_
        # X should be wrapped to [-pi, pi)
        assert np.all(model.X_ >= -np.pi)
        assert np.all(model.X_ < np.pi)

    def test_align_angular_trajectory_mode(self):
        """Trajectory-based angular alignment also uses rotation."""
        rng = np.random.default_rng(42)
        T, N_neurons = 2000, 15
        time = np.arange(T) * 0.02
        Xb = np.linspace(-np.pi, np.pi, T, endpoint=False)[:, None]
        preferred = np.linspace(-np.pi, np.pi, N_neurons, endpoint=False)
        rates = np.exp(3 * np.cos(Xb - preferred[None, :]))
        Y = rng.poisson(rates * 0.02)

        model = SIMPL(is_circular=True, bin_size=np.pi / 32, env_pad=0.0, speed_prior=0.1, kernel_bandwidth=0.3)
        model.fit(Y, Xb, time, n_epochs=1, align_to_behavior="trajectory")
        assert model.align_mode_ == "trajectory"
        assert "intercept" in model.E_
        assert "coef" not in model.E_


class TestSIMPLPredict:
    def test_predict_returns_correct_shape(self, demo_data):
        N = 2000
        N_neurons = min(10, demo_data["Y"].shape[1])
        model = SIMPL()
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=2,
        )

        # Predict on a different slice
        N_pred = 500
        X_decoded = model.predict(
            Y=demo_data["Y"][N : N + N_pred, :N_neurons],
        )
        assert X_decoded.shape == (N_pred, model.D_)

    def test_predict_stores_prediction_results(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL()
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=1,
        )

        N_pred = 200
        model.predict(
            Y=demo_data["Y"][N : N + N_pred, :N_neurons],
        )
        assert hasattr(model, "prediction_results_")
        assert isinstance(model.prediction_results_, xr.Dataset)
        assert "mu_s" in model.prediction_results_
        assert "sigma_s" in model.prediction_results_
        assert model.prediction_results_["mu_s"].dims == ("time", "dim")
        assert model.prediction_results_["mu_s"].shape == (N_pred, model.D_)

    def test_predict_with_trial_boundaries(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL()
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=1,
        )

        N_pred = 400
        X_decoded = model.predict(
            Y=demo_data["Y"][N : N + N_pred, :N_neurons],
            trial_boundaries=np.array([0, N_pred // 2]),
        )
        assert X_decoded.shape == (N_pred, model.D_)

    def test_predict_requires_fitted(self):
        model = SIMPL()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(Y=np.zeros((100, 5)))

    def test_predict_validates_neuron_count(self, demo_data):
        N = 500
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL()
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=0,
        )
        with pytest.raises(ValueError, match="neurons"):
            model.predict(Y=np.zeros((100, N_neurons + 3)))


class TestSIMPLSaveFullHistory:
    def test_FX_firstepoch_and_lastepoch_always_stored(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL()
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=2,
        )
        # FX_firstepoch and FX_lastepoch should always be present without epoch dim
        assert "FX_firstepoch" in model.results_
        assert "FX_lastepoch" in model.results_
        assert model.results_.FX_firstepoch.dims == ("time", "neuron")
        assert model.results_.FX_lastepoch.dims == ("time", "neuron")
        # Per-epoch FX should NOT be stored by default
        assert "FX" not in model.results_

    def test_save_full_history_stores_FX_all_epochs(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL()
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=2,
            save_full_history=True,
        )
        # FX should be present for every epoch (0, 1, 2)
        FX_epochs = model.results_.FX.dropna("epoch", how="all").epoch.values
        assert len(FX_epochs) == model.epoch_ + 1

    def test_logPYXF_maps_excluded_by_default(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL()
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=1,
        )
        assert "logPYXF_maps" not in model.results_

    def test_save_full_history_stores_logPYXF_maps(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL()
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=1,
            save_full_history=True,
        )
        assert "logPYXF_maps" in model.results_
        assert "epoch" not in model.results_.logPYXF_maps.dims


class TestSIMPLConvenienceAttrs:
    def test_X_and_F_match_last_epoch(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL()
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=2,
        )
        last_epoch = model.epoch_
        X_from_results = model.results_.X.sel(epoch=last_epoch).values
        F_from_results = model.results_.F.sel(epoch=last_epoch).values

        assert np.allclose(model.X_, X_from_results)
        assert np.allclose(model.F_, F_from_results.reshape(model.N_neurons_, -1))
