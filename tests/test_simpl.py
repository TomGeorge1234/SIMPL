"""Integration tests for simpl.simpl.SIMPL."""

import warnings

import numpy as np
import pytest

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
        model = SIMPL(use_kalman_smoothing=False, speed_prior=0.1, verbose=False)
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
        model = SIMPL(verbose=False)
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
        model = SIMPL(bin_size=0.03, env_pad=0.05, verbose=False)
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
        env = Environment(demo_data["Xb"][:N], bin_size=0.04, verbose=False)
        model = SIMPL(environment=env, verbose=False)
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=0,
        )
        assert model.environment_ is env

    def test_fit_validates_shapes(self):
        model = SIMPL(verbose=False)
        with pytest.raises(ValueError, match="same number of time bins"):
            model.fit(
                Y=np.zeros((100, 5)),
                Xb=np.zeros((200, 2)),
                time=np.arange(100) * 0.05,
                n_epochs=0,
            )

    def test_fit_validates_time_length(self):
        model = SIMPL(verbose=False)
        with pytest.raises(ValueError, match="same length"):
            model.fit(
                Y=np.zeros((100, 5)),
                Xb=np.zeros((100, 2)),
                time=np.arange(50) * 0.05,
                n_epochs=0,
            )


class TestSIMPLFitResume:
    def test_resume_continues_training(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL(verbose=False)
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
        model = SIMPL(verbose=False)
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
        model = SIMPL(verbose=False)
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
        model = SIMPL(verbose=False)
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=1,
        )
        model.add_baselines_to_results(
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
        model = SIMPL(verbose=False)
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
        model = SIMPL(verbose=False)
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
        model = SIMPL(verbose=False)
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=1,
        )

        path = str(tmp_path / "test_results.nc")
        model.save_results(path)
        loaded = load_results(path)
        assert "Y" in loaded
        assert "F" in loaded
        assert loaded.F.shape == model.results_.F.shape


class TestSIMPLInterpolateFiringRates:
    def test_correct_shape(self, small_simpl_model):
        model = small_simpl_model
        F = model.M_["F"]
        X = model.E_["X"]
        FX = model.interpolate_firing_rates(X, F)
        assert FX.shape == (model.T_, model.N_neurons_)


class TestSIMPLGetLoglikelihoods:
    def test_expected_keys(self, small_simpl_model):
        model = small_simpl_model
        lls = model.get_loglikelihoods(model.Y_, model.M_["FX"])
        assert "logPYXF" in lls
        assert "logPYXF_test" in lls


class TestSIMPLSeeding:
    """Tests for the random seeding mechanism."""

    def _make_model(self, demo_data, random_seed=0, resample_spike_mask=False):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL(
            random_seed=random_seed,
            resample_spike_mask=resample_spike_mask,
            verbose=False,
        )
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=0,
        )
        return model

    def test_same_seed_same_initial_mask(self, demo_data):
        m1 = self._make_model(demo_data, random_seed=42, resample_spike_mask=True)
        m2 = self._make_model(demo_data, random_seed=42, resample_spike_mask=True)
        assert np.array_equal(m1.spike_mask_, m2.spike_mask_)

    def test_different_seed_different_mask(self, demo_data):
        m1 = self._make_model(demo_data, random_seed=42, resample_spike_mask=True)
        m2 = self._make_model(demo_data, random_seed=99, resample_spike_mask=True)
        assert not np.array_equal(m1.spike_mask_, m2.spike_mask_)

    def test_resample_changes_mask_each_epoch(self, demo_data):
        model = self._make_model(demo_data, random_seed=0, resample_spike_mask=True)
        mask_init = np.array(model.spike_mask_)
        model.train_epoch()
        model.train_epoch()
        mask_epoch1 = np.array(model.spike_mask_)
        assert not np.array_equal(mask_init, mask_epoch1)

    def test_no_resample_keeps_mask(self, demo_data):
        model = self._make_model(demo_data, random_seed=0, resample_spike_mask=False)
        mask_init = np.array(model.spike_mask_)
        model.train_epoch()
        model.train_epoch()
        assert np.array_equal(mask_init, np.array(model.spike_mask_))

    def test_next_seed_never_repeats(self, demo_data):
        model = self._make_model(demo_data, random_seed=0, resample_spike_mask=True)
        seeds = [model._next_seed() for _ in range(100)]
        assert len(set(seeds)) == len(seeds)

    def test_reproducible_across_runs(self, demo_data):
        m1 = self._make_model(demo_data, random_seed=7, resample_spike_mask=True)
        m2 = self._make_model(demo_data, random_seed=7, resample_spike_mask=True)
        m1.train_epoch()
        m1.train_epoch()
        m2.train_epoch()
        m2.train_epoch()
        assert np.array_equal(m1.spike_mask_, m2.spike_mask_)


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
        model = SIMPL(verbose=False)
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=0,
        )
        assert model.epoch_ == 0
        assert "F" in model.results_

    def test_verbose_false_suppresses_output(self, demo_data, capsys):
        N = 500
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL(verbose=False)
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=0,
        )
        captured = capsys.readouterr().out
        assert "DATA SUMMARY" not in captured
        assert "Spatial info" not in captured


class TestSIMPLAddBaselines:
    def _make_model(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL(verbose=False)
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=0,
        )
        return model, N, N_neurons

    def test_populates_best_and_exact_epochs(self, demo_data):
        model, N, N_neurons = self._make_model(demo_data)
        model.add_baselines_to_results(
            Xt=demo_data["Xt"][:N],
            Ft=demo_data["Ft"][:N_neurons],
            Ft_coords_dict={"x": demo_data["x"], "y": demo_data["y"]},
        )
        assert -1 in model.results_.epoch.values
        assert -2 in model.results_.epoch.values

    def test_baseline_results_have_metrics(self, demo_data):
        model, N, N_neurons = self._make_model(demo_data)
        model.add_baselines_to_results(
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
            model.add_baselines_to_results(Xt=demo_data["Xt"][:N])
            assert any("Exact place fields" in str(warning.message) for warning in w)
        assert -1 in model.results_.epoch.values

    def test_requires_fitted(self):
        model = SIMPL(verbose=False)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.add_baselines_to_results(Xt=np.zeros((100, 2)))


class TestSIMPLManifoldAlignment:
    def _make_model(self, demo_data, align_to_behaviour=True):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL(verbose=False)
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=1,
            align_to_behaviour=align_to_behaviour,
        )
        return model

    def test_cca_runs(self, small_simpl_model):
        model = small_simpl_model
        if model.epoch_ < 1:
            model.train_epoch()
            model.train_epoch()
        assert model.epoch_ >= 1
        assert "X" in model.E_

    def test_align_to_behaviour(self, demo_data):
        model = self._make_model(demo_data, align_to_behaviour=True)
        assert model.Xalign_ is not None
        assert "X" in model.E_

    def test_no_alignment(self, demo_data):
        model = self._make_model(demo_data, align_to_behaviour=False)
        assert model.Xalign_ is None
        assert "X" in model.E_


class TestSIMPLPredict:
    def test_predict_returns_correct_shape(self, demo_data):
        N = 2000
        N_neurons = min(10, demo_data["Y"].shape[1])
        model = SIMPL(verbose=False)
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

    def test_predict_with_return_std(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL(verbose=False)
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=1,
        )

        N_pred = 200
        X_decoded, sigma = model.predict(
            Y=demo_data["Y"][N : N + N_pred, :N_neurons],
            return_std=True,
        )
        assert X_decoded.shape == (N_pred, model.D_)
        assert sigma.shape == (N_pred, model.D_, model.D_)

    def test_predict_stores_prediction_results(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL(verbose=False)
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=1,
        )

        model.predict(
            Y=demo_data["Y"][N : N + 200, :N_neurons],
        )
        assert hasattr(model, "prediction_results_")
        assert "mu_s" in model.prediction_results_
        assert "sigma_s" in model.prediction_results_

    def test_predict_with_trial_boundaries(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL(verbose=False)
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
        model = SIMPL(verbose=False)
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.predict(Y=np.zeros((100, 5)))

    def test_predict_validates_neuron_count(self, demo_data):
        N = 500
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL(verbose=False)
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=0,
        )
        with pytest.raises(ValueError, match="neurons"):
            model.predict(Y=np.zeros((100, N_neurons + 3)))


class TestSIMPLConvenienceAttrs:
    def test_X_and_F_match_last_epoch(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        model = SIMPL(verbose=False)
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
