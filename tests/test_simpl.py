"""Integration tests for simpl.simpl.SIMPL."""

import warnings

import numpy as np

from simpl.environment import Environment
from simpl.simpl import SIMPL
from simpl.utils import load_datafile, load_results, prepare_data


class TestSIMPLInit:
    def test_correct_attributes(self, small_simpl_model):
        model = small_simpl_model
        assert hasattr(model, "Y")
        assert hasattr(model, "Xb")
        assert hasattr(model, "kalman_filter")
        assert hasattr(model, "environment")
        assert hasattr(model, "results")
        assert model.D == 2
        assert model.epoch == 0  # epoch 0 runs in __init__

    def test_kalman_argument_exposed(self, demo_data):
        N = 500
        N_neurons = min(5, demo_data["Y"].shape[1])
        data = prepare_data(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            dims=demo_data["dim"],
            neurons=np.arange(N_neurons),
        )
        env = Environment(demo_data["Xb"][:N], verbose=False)
        model = SIMPL(data=data, environment=env, kalman=False, speed_prior=0.1)
        assert model.kalman is False
        # kalman=False should enforce a high effective speed prior.
        assert model.speed_prior_effective >= model.kalman_off_speed_prior


class TestSIMPLTrainOneEpoch:
    def test_runs_and_populates_results(self, small_simpl_model):
        model = small_simpl_model
        # epoch 0 already ran in __init__, so results should be populated
        assert model.epoch == 0
        assert "X" in model.results
        assert "F" in model.results


class TestSIMPLResultsStructure:
    def test_correct_xarray_dims(self, small_simpl_model):
        model = small_simpl_model
        if model.epoch < 0:
            model.train_epoch()
        assert "epoch" in model.results.dims
        assert "time" in model.results.dims
        assert "neuron" in model.results.dims


class TestSIMPLTrainImprovesLikelihood:
    def test_likelihood_increases(self, demo_data, environment):
        N = 2000
        N_neurons = min(10, demo_data["Y"].shape[1])
        data = prepare_data(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            dims=demo_data["dim"],
            neurons=np.arange(N_neurons),
        )
        env = Environment(demo_data["Xb"][:N], verbose=False)
        model = SIMPL(data=data, environment=env, evaluate_each_epoch=True)
        model.train_N_epochs(3, verbose=False)

        ll_0 = float(model.loglikelihoods.logPYXF.sel(epoch=0).values)
        ll_last = float(model.loglikelihoods.logPYXF.sel(epoch=model.epoch).values)
        # Likelihood should generally improve (or at least not decrease much)
        assert ll_last >= ll_0 - 0.1


class TestSIMPLEvaluateEpoch:
    def test_metrics_dict_keys(self, small_simpl_model):
        model = small_simpl_model
        if model.epoch < 0:
            model.train_epoch()
        assert "logPYXF" in model.results
        assert "logPYXF_test" in model.results


class TestSIMPLWithGroundTruth:
    def test_r2_and_error_computed(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        data = prepare_data(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            dims=demo_data["dim"],
            neurons=np.arange(N_neurons),
            Xt=demo_data["Xt"][:N],
            Ft=demo_data["Ft"][:N_neurons],
            Ft_coords_dict={"x": demo_data["x"], "y": demo_data["y"]},
        )
        env = Environment(demo_data["Xb"][:N], verbose=False)
        model = SIMPL(data=data, environment=env)
        model.train_epoch()
        assert "X_R2" in model.results
        assert "X_err" in model.results
        assert "F_err" in model.results


class TestSIMPLWithoutGroundTruth:
    def test_works_without_Xt_Ft(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        data = prepare_data(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
        )
        env = Environment(demo_data["Xb"][:N], verbose=False)
        model = SIMPL(data=data, environment=env)
        model.train_epoch()
        assert "F" in model.results
        assert "X_R2" not in model.results


class TestSIMPLTrialBoundaries:
    def test_per_trial_filtering(self, demo_data):
        N = 2000
        N_neurons = min(5, demo_data["Y"].shape[1])
        boundaries = np.array([0, N // 2])
        data = prepare_data(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            trial_boundaries=boundaries,
        )
        env = Environment(demo_data["Xb"][:N], verbose=False)
        model = SIMPL(data=data, environment=env)
        assert len(model.trial_slices) == 2
        assert model.epoch == 0  # epoch 0 runs in __init__


class TestSIMPLSaveLoadResults:
    def test_round_trip(self, tmp_path, demo_data):
        N = 500
        N_neurons = min(5, demo_data["Y"].shape[1])
        data = prepare_data(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
        )
        env = Environment(demo_data["Xb"][:N], verbose=False)
        model = SIMPL(data=data, environment=env)
        model.train_epoch()

        path = str(tmp_path / "test_results.nc")
        model.save_results(path)
        loaded = load_results(path)
        assert "Y" in loaded
        assert "F" in loaded
        assert loaded.F.shape == model.results.F.shape


class TestSIMPLInterpolateFiringRates:
    def test_correct_shape(self, small_simpl_model):
        model = small_simpl_model
        if model.epoch < 0:
            model.train_epoch()
        F = model.M["F"]
        X = model.E["X"]
        FX = model.interpolate_firing_rates(X, F)
        assert FX.shape == (model.T, model.N_neurons)


class TestSIMPLGetLoglikelihoods:
    def test_expected_keys(self, small_simpl_model):
        model = small_simpl_model
        if model.epoch < 0:
            model.train_epoch()
        lls = model.get_loglikelihoods(model.Y, model.M["FX"])
        assert "logPYXF" in lls
        assert "logPYXF_test" in lls


class TestSIMPLSeeding:
    """Tests for the random seeding mechanism."""

    def _make_model(self, demo_data, random_seed=0, resample_spike_mask=False):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        data = prepare_data(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
        )
        env = Environment(demo_data["Xb"][:N], verbose=False)
        return SIMPL(
            data=data,
            environment=env,
            random_seed=random_seed,
            resample_spike_mask=resample_spike_mask,
        )

    def test_same_seed_same_initial_mask(self, demo_data):
        """Same random_seed should produce the same initial spike mask."""
        m1 = self._make_model(demo_data, random_seed=42, resample_spike_mask=True)
        m2 = self._make_model(demo_data, random_seed=42, resample_spike_mask=True)
        assert np.array_equal(m1.spike_mask, m2.spike_mask)

    def test_different_seed_different_mask(self, demo_data):
        """Different random_seed should produce different initial spike masks."""
        m1 = self._make_model(demo_data, random_seed=42, resample_spike_mask=True)
        m2 = self._make_model(demo_data, random_seed=99, resample_spike_mask=True)
        assert not np.array_equal(m1.spike_mask, m2.spike_mask)

    def test_resample_changes_mask_each_epoch(self, demo_data):
        """With resample_spike_mask=True, the mask should change each epoch."""
        model = self._make_model(demo_data, random_seed=0, resample_spike_mask=True)
        mask_init = np.array(model.spike_mask)
        model.train_epoch()
        model.train_epoch()
        mask_epoch1 = np.array(model.spike_mask)
        assert not np.array_equal(mask_init, mask_epoch1)

    def test_no_resample_keeps_mask(self, demo_data):
        """With resample_spike_mask=False, the mask should stay the same."""
        model = self._make_model(demo_data, random_seed=0, resample_spike_mask=False)
        mask_init = np.array(model.spike_mask)
        model.train_epoch()
        model.train_epoch()
        assert np.array_equal(mask_init, np.array(model.spike_mask))

    def test_next_seed_never_repeats(self, demo_data):
        """_next_seed should produce unique seeds across calls."""
        model = self._make_model(demo_data, random_seed=0, resample_spike_mask=True)
        seeds = [model._next_seed() for _ in range(100)]
        assert len(set(seeds)) == len(seeds)

    def test_reproducible_across_runs(self, demo_data):
        """Two models with the same seed should produce identical masks after training."""
        m1 = self._make_model(demo_data, random_seed=7, resample_spike_mask=True)
        m2 = self._make_model(demo_data, random_seed=7, resample_spike_mask=True)
        m1.train_epoch()
        m1.train_epoch()
        m2.train_epoch()
        m2.train_epoch()
        assert np.array_equal(m1.spike_mask, m2.spike_mask)


class TestSIMPLSpatialInformation:
    def test_spatial_information_in_results(self, small_simpl_model):
        """After epoch 0, spatial_information should be in results."""
        model = small_simpl_model
        assert "spatial_information" in model.results
        assert "spatial_information_rate" in model.results

    def test_spatial_information_shape(self, small_simpl_model):
        """spatial_information should have one value per neuron."""
        model = small_simpl_model
        si = model.results.spatial_information.sel(epoch=0)
        assert si.shape == (model.N_neurons,)

    def test_spatial_information_nonnegative(self, small_simpl_model):
        """Spatial information should be non-negative."""
        model = small_simpl_model
        si = model.results.spatial_information.sel(epoch=0).values
        assert np.all(si >= -1e-6)

    def test_spatial_information_rate_is_sum(self, small_simpl_model):
        """spatial_information_rate should be the sum of per-neuron values."""
        model = small_simpl_model
        si = model.results.spatial_information.sel(epoch=0).values
        sir = float(model.results.spatial_information_rate.sel(epoch=0))
        assert np.isclose(sir, si.sum(), atol=1e-3)


class TestSIMPLEpochZeroInInit:
    def test_epoch_starts_at_zero(self, demo_data):
        """Epoch 0 should run automatically during __init__."""
        N = 500
        N_neurons = min(5, demo_data["Y"].shape[1])
        data = prepare_data(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
        )
        env = Environment(demo_data["Xb"][:N], verbose=False)
        model = SIMPL(data=data, environment=env, verbose=False)
        assert model.epoch == 0
        assert "F" in model.results

    def test_verbose_false_suppresses_output(self, demo_data, capsys):
        """verbose=False should suppress data summary and spatial info prints."""
        N = 500
        N_neurons = min(5, demo_data["Y"].shape[1])
        data = prepare_data(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
        )
        env = Environment(demo_data["Xb"][:N], verbose=False)
        _ = SIMPL(data=data, environment=env, verbose=False)
        captured = capsys.readouterr().out
        assert "DATA SUMMARY" not in captured
        assert "Spatial info" not in captured


class TestSIMPLCalculateBaselines:
    def _make_model_with_gt(self, demo_data):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        data = prepare_data(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            Xt=demo_data["Xt"][:N],
            Ft=demo_data["Ft"][:N_neurons],
            Ft_coords_dict={"x": demo_data["x"], "y": demo_data["y"]},
        )
        env = Environment(demo_data["Xb"][:N], verbose=False)
        return SIMPL(data=data, environment=env, verbose=False)

    def test_populates_best_and_exact_epochs(self, demo_data):
        """calculate_baselines should add epochs -2 (exact) and -1 (best)."""
        model = self._make_model_with_gt(demo_data)
        model.calculate_baselines()
        assert -1 in model.results.epoch.values
        assert -2 in model.results.epoch.values

    def test_baseline_results_have_metrics(self, demo_data):
        """Baseline epochs should contain F, X, and error metrics."""
        model = self._make_model_with_gt(demo_data)
        model.calculate_baselines()
        for epoch in [-2, -1]:
            assert "F" in model.results.sel(epoch=epoch)
            assert "X" in model.results.sel(epoch=epoch)
            assert "X_R2" in model.results.sel(epoch=epoch)

    def test_warns_without_ground_truth(self, demo_data):
        """Without ground truth, calculate_baselines should warn and return."""
        N = 500
        N_neurons = min(5, demo_data["Y"].shape[1])
        data = prepare_data(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
        )
        env = Environment(demo_data["Xb"][:N], verbose=False)
        model = SIMPL(data=data, environment=env, verbose=False)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            model.calculate_baselines()
            assert any("Ground truth" in str(warning.message) for warning in w)


class TestSIMPLManifoldAlignment:
    def _make_model(self, demo_data, align_against, with_gt=False):
        N = 1000
        N_neurons = min(5, demo_data["Y"].shape[1])
        kwargs = dict(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
        )
        if with_gt:
            kwargs["Xt"] = demo_data["Xt"][:N]
        data = prepare_data(**kwargs)
        env = Environment(demo_data["Xb"][:N], verbose=False)
        return SIMPL(data=data, environment=env, manifold_align_against=align_against, verbose=False)

    def test_cca_runs(self, small_simpl_model):
        model = small_simpl_model
        if model.epoch < 1:
            model.train_epoch()  # epoch 0 uses Xb directly
            model.train_epoch()  # epoch 1 runs E-step with CCA
        # After epoch 1, alignment should have been applied
        assert model.epoch >= 1
        assert "X" in model.E

    def test_align_against_behaviour(self, demo_data):
        """manifold_align_against='behaviour' should align to Xb."""
        model = self._make_model(demo_data, "behaviour")
        assert model.Xalign is not None
        model.train_epoch()
        assert "X" in model.E

    def test_align_against_ground_truth(self, demo_data):
        """manifold_align_against='ground_truth' should align to Xt."""
        model = self._make_model(demo_data, "ground_truth", with_gt=True)
        assert model.Xalign is not None
        model.train_epoch()
        assert "X" in model.E

    def test_align_against_none(self, demo_data):
        """manifold_align_against='none' should skip alignment."""
        model = self._make_model(demo_data, "none")
        assert model.Xalign is None
        model.train_epoch()
        assert "X" in model.E

    def test_invalid_align_raises(self, demo_data):
        """Invalid manifold_align_against should raise ValueError."""
        try:
            self._make_model(demo_data, "invalid")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass
