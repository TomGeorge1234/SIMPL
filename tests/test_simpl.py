"""Integration tests for simpl.simpl.SIMPL."""

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
        assert model.epoch == -1

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


class TestSIMPLManifoldAlignment:
    def test_cca_runs(self, small_simpl_model):
        model = small_simpl_model
        if model.epoch < 1:
            model.train_epoch()  # epoch 0 uses Xb directly
            model.train_epoch()  # epoch 1 runs E-step with CCA
        # After epoch 1, alignment should have been applied
        assert model.epoch >= 1
        assert "X" in model.E


class TestSIMPLKalmanDiagnostics:
    def test_mode_mu_s_rmse_small_when_kalman_off(self):
        demo_data = load_datafile("gridcelldata.npz")
        N = 1000
        N_neurons = min(8, demo_data["Y"].shape[1])
        data = prepare_data(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            dims=demo_data["dim"],
            neurons=np.arange(N_neurons),
        )
        env = Environment(demo_data["Xb"][:N], verbose=False)
        model = SIMPL(data=data, environment=env, kalman=False, evaluate_each_epoch=True)
        model.train_N_epochs(2, verbose=False)  # epoch 1 is first E-step

        mode_l = np.asarray(model.results.mode_l.sel(epoch=model.epoch).values)
        mu_s = np.asarray(model.results.mu_s.sel(epoch=model.epoch).values)
        diff = mode_l - mu_s
        rmse = float(np.sqrt(np.mean(diff**2)))
        mae = float(np.mean(np.abs(diff)))
        max_abs = float(np.max(np.abs(diff)))

        assert np.isfinite(rmse)
        assert np.isfinite(mae)
        assert np.isfinite(max_abs)
        assert rmse >= 0.0
        assert mae >= 0.0
        assert max_abs >= 0.0

    def test_mode_mu_s_allclose_when_kalman_off(self):
        demo_data = load_datafile("gridcelldata.npz")
        N = 1000
        N_neurons = min(8, demo_data["Y"].shape[1])
        data = prepare_data(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            dims=demo_data["dim"],
            neurons=np.arange(N_neurons),
        )
        env = Environment(demo_data["Xb"][:N], verbose=False)
        model = SIMPL(data=data, environment=env, kalman=False, evaluate_each_epoch=True)
        model.train_N_epochs(2, verbose=False)  # epoch 1 is first E-step

        mode_l = np.asarray(model.results.mode_l.sel(epoch=model.epoch).values)
        mu_s = np.asarray(model.results.mu_s.sel(epoch=model.epoch).values)
        assert np.allclose(mode_l, mu_s, atol=1e-2, rtol=0.0)
