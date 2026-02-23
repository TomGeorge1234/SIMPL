"""Integration tests for simpl.simpl.SIMPL."""

import numpy as np

from simpl.environment import Environment
from simpl.simpl import SIMPL
from simpl.utils import load_results, prepare_data


class TestSIMPLInit:
    def test_correct_attributes(self, small_simpl_model):
        model = small_simpl_model
        assert hasattr(model, "Y")
        assert hasattr(model, "Xb")
        assert hasattr(model, "kalman_filter")
        assert hasattr(model, "environment")
        assert hasattr(model, "results")
        assert model.D == 2
        assert model.epoch == -1


class TestSIMPLTrainOneEpoch:
    def test_runs_and_populates_results(self, small_simpl_model):
        model = small_simpl_model
        model.train_epoch()
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
        model.train_epoch()
        assert model.epoch == 0


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


class TestSIMPLManifoldAlignment:
    def test_cca_runs(self, small_simpl_model):
        model = small_simpl_model
        if model.epoch < 1:
            model.train_epoch()  # epoch 0 uses Xb directly
            model.train_epoch()  # epoch 1 runs E-step with CCA
        # After epoch 1, alignment should have been applied
        assert model.epoch >= 1
        assert "X" in model.E
