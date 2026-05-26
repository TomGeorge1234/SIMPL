"""Tests for simpl.plotting functions."""

import matplotlib
import numpy as np
import pytest
import xarray as xr

matplotlib.use("Agg")  # non-interactive backend for CI

from simpl.plotting import (
    plot_all_metrics,
    plot_fitting_summary,
    plot_latent_trajectory,
    plot_metric,
    plot_prediction,
    plot_receptive_fields,
    plot_spikes,
)
from simpl.simpl import SIMPL

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def fitted_model(demo_data):
    """A small fitted model for plotting tests."""
    N = 2000
    N_neurons = 10
    model = SIMPL(bin_size=0.02, env_pad=0.0)
    model.fit(
        Y=demo_data["Y"][:N, :N_neurons],
        Xb=demo_data["Xb"][:N],
        time=demo_data["time"][:N],
        n_iterations=1,
    )
    return model


@pytest.fixture(scope="module")
def results(fitted_model):
    return fitted_model.results_


@pytest.fixture
def metric_results():
    """Small results Dataset with scalar, per-neuron, and place-field metrics."""
    iterations = np.array([-1, 0, 1])
    return xr.Dataset(
        data_vars={
            "scalar_metric": (
                ("iteration",),
                np.array([0.5, 1.0, 2.0]),
                {"axis_title": "Scalar metric"},
            ),
            "scalar_metric_val": (
                ("iteration",),
                np.array([np.nan, 0.8, 1.7]),
                {"axis_title": "Scalar metric validation"},
            ),
            "neuron_metric": (
                ("iteration", "neuron"),
                np.array([[np.nan, np.nan], [1.0, 3.0], [2.0, 4.0]]),
                {"axis title": "Neuron metric"},
            ),
            "place_field_metric": (
                ("iteration", "neuron", "place_field"),
                np.array(
                    [
                        [[np.nan, np.nan], [np.nan, np.nan]],
                        [[1.0, 3.0], [5.0, 7.0]],
                        [[2.0, 4.0], [6.0, 8.0]],
                    ]
                ),
                {"axis_title": "Place field metric"},
            ),
            "not_a_metric": (("iteration", "time"), np.ones((3, 2))),
        },
        coords={
            "iteration": iterations,
            "neuron": [0, 1],
            "place_field": [0, 1],
            "time": [0, 1],
        },
    )


# ── Standalone function tests ─────────────────────────────────────────────────


class TestPlotFittingSummary:
    def test_returns_two_axes(self, results):
        axes = plot_fitting_summary(results)
        assert len(axes) == 2
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_show_neurons_false(self, results):
        axes = plot_fitting_summary(results, show_neurons=False)
        assert len(axes) == 2
        import matplotlib.pyplot as plt

        plt.close("all")


class TestPlotLatentTrajectory:
    def test_returns_D_axes(self, results):
        axes = plot_latent_trajectory(results)
        D = len(results.dim.values)
        assert len(axes) == D
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_custom_time_range(self, results):
        t0 = float(results.time.values[0])
        axes = plot_latent_trajectory(results, time_range=(t0, t0 + 30))
        assert len(axes) == len(results.dim.values)
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_single_iteration(self, results):
        axes = plot_latent_trajectory(results, iterations=1)
        assert len(axes) == len(results.dim.values)
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_iteration_tuple(self, results):
        axes = plot_latent_trajectory(results, iterations=(0, 1))
        assert len(axes) == len(results.dim.values)
        import matplotlib.pyplot as plt

        plt.close("all")


class TestPlotReceptiveFields:
    def test_returns_axes_grid(self, results):
        axes = plot_receptive_fields(results, neurons=[0, 1])
        assert axes.shape[0] == 1  # 2 neurons, ncols=4 -> 1 row
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_include_baselines_without_Ft(self, results):
        # should work even if Ft not present — just no GT column
        axes = plot_receptive_fields(results, neurons=[0], include_baselines=True)
        assert axes.ndim == 2
        import matplotlib.pyplot as plt

        plt.close("all")

    @pytest.mark.filterwarnings("ignore:Exact place fields not provided")
    def test_include_baselines_fallback_iteration_minus1(self, demo_data):
        """When Ft is absent but iteration -1 exists, should show 'Best' column from F.sel(iteration=-1)."""
        import matplotlib.pyplot as plt

        N, N_neurons = 2000, 10
        model = SIMPL(bin_size=0.02, env_pad=0.0)
        # add_baselines with Xt only (no Ft) so iteration -1 is computed but Ft is not stored
        model.add_baselines(Xt=demo_data["Xb"][:N])
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_iterations=1,
        )
        res = model.results_
        assert "Ft" not in res, "Ft should not be in results when only Xt is provided"
        assert -1 in res.iteration.values, "Iteration -1 should exist from add_baselines(Xt=...)"

        # include_baselines=True should use the F.sel(iteration=-1) fallback
        axes = plot_receptive_fields(res, neurons=[0], include_baselines=True)
        # 4 columns: Ep 0 + Ep 1 + Best + colorbar
        assert axes.shape == (1, 4)
        plt.close("all")

    def test_single_iteration_int(self, results):
        """Single int iteration should show one iteration column per neuron."""
        axes = plot_receptive_fields(results, neurons=[0], iterations=1)
        # columns: Beh + Ep 1 = 2 (since iteration=1, include_behavior adds Beh)
        import matplotlib.pyplot as plt

        assert axes.ndim == 2
        plt.close("all")

    def test_iteration_tuple(self, results):
        """Tuple of iterations should show one column per iteration per neuron."""
        axes = plot_receptive_fields(results, neurons=[0], iterations=(0, 1))
        # columns: Ep 0 (behavior) + Ep 1 = 2 (0 is in iterations, so no extra Beh col)
        import matplotlib.pyplot as plt

        assert axes.ndim == 2
        plt.close("all")

    def test_iteration_none_shows_first_and_last(self, results):
        """Default iteration=None should show iteration 0 and last."""
        axes = plot_receptive_fields(results, neurons=[0])
        # iteration=None -> (0, 1), no extra Beh col since 0 in iterations -> 2 data cols
        import matplotlib.pyplot as plt

        assert axes.ndim == 2
        plt.close("all")

    def test_spacer_and_unused_axes_off(self, results):
        """With multiple neuron groups, spacer and trailing axes should be turned off."""
        axes = plot_receptive_fields(results, neurons=[0, 1, 2], ncols=2, iterations=1)
        # iterations=1 → 1 col per neuron + 1 cbar (2D), ncols=2 → 2 neuron groups per row
        # total cols = 2*(1+1) + 1 spacer = 5, total rows = 2
        # Layout: [data, cbar, spacer, data, cbar]
        import matplotlib.pyplot as plt

        # spacer column should be off
        assert not axes[0, 2].axison  # spacer
        # trailing unused axes on row 2 should be off (spacer + group 1 data + cbar)
        assert not axes[1, 2].axison
        assert not axes[1, 3].axison
        assert not axes[1, 4].axison
        plt.close("all")


class TestPlotAllMetrics:
    def test_returns_axes(self, results):
        axes = plot_all_metrics(results)
        assert axes.size > 0
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_wrapper_delegates_to_plot_metric(self, metric_results, monkeypatch):
        calls = []

        def fake_plot_metric(results, var_name, show_neurons=True, ax=None, **plot_kwargs):
            calls.append((results, var_name, show_neurons, ax, plot_kwargs))
            ax.set(xlabel="Iteration", ylabel=var_name)
            return ax

        monkeypatch.setattr("simpl.plotting.plot_metric", fake_plot_metric)

        axes = plot_all_metrics(metric_results, show_neurons=False, ncols=2, alpha=0.3)

        assert axes.shape == (2, 2)
        assert [call[1] for call in calls] == ["scalar_metric", "neuron_metric", "place_field_metric"]
        assert all(call[0] is metric_results for call in calls)
        assert all(call[2] is False for call in calls)
        assert [call[3] for call in calls] == list(axes.flat[:3])
        assert all(call[4] == {"alpha": 0.3} for call in calls)
        assert not axes.flat[3].get_visible()
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_show_neurons_false(self, results):
        axes = plot_all_metrics(results, show_neurons=False)
        assert axes.size > 0
        import matplotlib.pyplot as plt

        plt.close("all")


class TestPlotMetric:
    def test_scalar_metric_returns_axis_and_plots_validation(self, metric_results):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        returned = plot_metric(metric_results, "scalar_metric", ax=ax)

        assert returned is ax
        assert ax.get_xlabel() == "Iteration"
        assert ax.get_ylabel() == "Scalar metric"
        assert len(ax.collections) == 4  # train + validation scatter for two iterations
        assert any(line.get_linestyle() == "--" for line in ax.lines)
        plt.close("all")

    def test_per_neuron_metric_can_hide_individual_neurons(self, metric_results):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        returned = plot_metric(metric_results, "neuron_metric", show_neurons=False, ax=ax)

        assert returned is ax
        assert ax.get_ylabel() == "Neuron metric"
        assert len(ax.collections) == 2  # one mean marker per plotted iteration
        np.testing.assert_allclose(ax.collections[0].get_offsets()[0], [0.0, 2.0])
        np.testing.assert_allclose(ax.collections[1].get_offsets()[0], [1.0, 3.0])
        plt.close("all")

    def test_place_field_metric_averages_place_fields_before_neurons(self, metric_results):
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
        returned = plot_metric(metric_results, "place_field_metric", show_neurons=False, ax=ax)

        assert returned is ax
        assert ax.get_ylabel() == "Place field metric"
        np.testing.assert_allclose(ax.collections[0].get_offsets()[0], [0.0, 4.0])
        np.testing.assert_allclose(ax.collections[1].get_offsets()[0], [1.0, 5.0])
        plt.close("all")


class TestPlotSpikes:
    def test_returns_axes(self, results):
        ax = plot_spikes(results)
        import matplotlib.pyplot as plt

        assert ax is not None
        plt.close("all")

    def test_custom_time_range(self, results):
        t0 = float(results.time.values[0])
        ax = plot_spikes(results, time_range=(t0, t0 + 10))
        import matplotlib.pyplot as plt

        assert ax is not None
        plt.close("all")

    def test_neuron_subset(self, results):
        ax = plot_spikes(results, neurons=[0, 1])
        import matplotlib.pyplot as plt

        assert ax is not None
        plt.close("all")

    def test_custom_ncols(self, results):
        axes = plot_all_metrics(results, ncols=5)
        assert axes.shape[1] == 5
        import matplotlib.pyplot as plt

        plt.close("all")


class TestPlotPrediction:
    def test_returns_D_axes(self, fitted_model, demo_data):
        import matplotlib.pyplot as plt

        N_test = 200
        Y_test = demo_data["Y"][-N_test:, :10]
        fitted_model.predict(Y_test)
        axes = plot_prediction(fitted_model.prediction_results_)
        assert len(axes) == len(fitted_model.results_.dim.values)
        plt.close("all")

    def test_with_xb_and_xt(self, fitted_model, demo_data):
        import matplotlib.pyplot as plt

        N_test = 200
        Y_test = demo_data["Y"][-N_test:, :10]
        Xb_test = demo_data["Xb"][-N_test:]
        Xt_test = demo_data["Xb"][-N_test:]  # use Xb as fake Xt
        fitted_model.predict(Y_test)
        axes = plot_prediction(fitted_model.prediction_results_, Xb=Xb_test, Xt=Xt_test)
        assert len(axes) == len(fitted_model.results_.dim.values)
        plt.close("all")

    def test_shape_mismatch_raises(self, fitted_model, demo_data):
        import matplotlib.pyplot as plt

        N_test = 200
        Y_test = demo_data["Y"][-N_test:, :10]
        fitted_model.predict(Y_test)
        # Xb with wrong length should raise
        with pytest.raises(AssertionError, match="Xb length"):
            plot_prediction(fitted_model.prediction_results_, Xb=np.zeros((N_test + 10, 2)))
        plt.close("all")

    def test_with_time_range(self, fitted_model, demo_data):
        import matplotlib.pyplot as plt

        N_test = 200
        Y_test = demo_data["Y"][-N_test:, :10]
        fitted_model.predict(Y_test)
        pred = fitted_model.prediction_results_
        t0 = float(pred.time.values[0])
        axes = plot_prediction(pred, time_range=(t0, t0 + 5))
        assert len(axes) == len(fitted_model.results_.dim.values)
        plt.close("all")

    def test_wrapper_unfitted_raises(self):
        model = SIMPL()
        with pytest.raises(RuntimeError, match="No prediction results"):
            model.plot_prediction()

    def test_wrapper_returns_axes(self, fitted_model, demo_data):
        import matplotlib.pyplot as plt

        N_test = 200
        Y_test = demo_data["Y"][-N_test:, :10]
        fitted_model.predict(Y_test)
        axes = fitted_model.plot_prediction()
        assert len(axes) == len(fitted_model.results_.dim.values)
        plt.close("all")


# ── SIMPL wrapper tests ──────────────────────────────────────────────────────


class TestSIMPLPlotWrappers:
    def test_unfitted_raises(self):
        model = SIMPL()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.plot_fitting_summary()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.plot_latent_trajectory()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.plot_receptive_fields()
        with pytest.raises(RuntimeError, match="not been fitted"):
            model.plot_all_metrics()

    def test_wrappers_return_axes(self, fitted_model):
        import matplotlib.pyplot as plt

        axes = fitted_model.plot_fitting_summary()
        assert len(axes) == 2
        plt.close("all")

        axes = fitted_model.plot_latent_trajectory()
        assert len(axes) == len(fitted_model.results_.dim.values)
        plt.close("all")

        axes = fitted_model.plot_receptive_fields(neurons=[0, 1])
        assert axes.ndim == 2
        plt.close("all")

        axes = fitted_model.plot_all_metrics()
        assert axes.size > 0
        plt.close("all")


# ── 3D+ ValueError ───────────────────────────────────────────────────────────


class TestHighDimensionalError:
    def test_3d_receptive_fields_raises(self):
        import xarray as xr

        # Build a minimal fake 3D dataset
        fake = xr.Dataset(
            coords={"dim": ["x", "y", "z"], "iteration": [0], "neuron": [0]},
        )
        with pytest.raises(ValueError, match="only supports 1-D and 2-D"):
            plot_receptive_fields(fake, neurons=[0])
        import matplotlib.pyplot as plt

        plt.close("all")
