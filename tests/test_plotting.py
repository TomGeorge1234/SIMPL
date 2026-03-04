"""Tests for simpl.plotting functions."""

import matplotlib
import pytest

matplotlib.use("Agg")  # non-interactive backend for CI

from simpl.plotting import (
    plot_all_metrics,
    plot_fitting_summary,
    plot_latent_trajectory,
    plot_receptive_fields,
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
        n_epochs=1,
    )
    return model


@pytest.fixture(scope="module")
def results(fitted_model):
    return fitted_model.results_


# ── Standalone function tests ─────────────────────────────────────────────────


class TestPlotFittingSummary:
    def test_returns_two_axes(self, results):
        axes = plot_fitting_summary(results)
        assert len(axes) == 2
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_custom_cmap(self, results):
        axes = plot_fitting_summary(results, cmap="viridis")
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

    def test_no_behaviour(self, results):
        axes = plot_latent_trajectory(results, include_behaviour=False)
        assert len(axes) == len(results.dim.values)
        import matplotlib.pyplot as plt

        plt.close("all")


class TestPlotReceptiveFields:
    def test_returns_axes_grid(self, results):
        axes = plot_receptive_fields(results, neurons=[0, 1])
        assert axes.shape[0] == 1  # 2 neurons, ncols=8 -> 1 row
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_include_baselines_without_Ft(self, results):
        # should work even if Ft not present — just no GT column
        axes = plot_receptive_fields(results, neurons=[0], include_baselines=True)
        assert axes.ndim == 2
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_include_baselines_fallback_epoch_minus1(self, demo_data):
        """When Ft is absent but epoch -1 exists, should show 'Best' column from F.sel(epoch=-1)."""
        import matplotlib.pyplot as plt

        N, N_neurons = 2000, 10
        model = SIMPL(bin_size=0.02, env_pad=0.0)
        # add_baselines with Xt only (no Ft) so epoch -1 is computed but Ft is not stored
        model.add_baselines(Xt=demo_data["Xb"][:N])
        model.fit(
            Y=demo_data["Y"][:N, :N_neurons],
            Xb=demo_data["Xb"][:N],
            time=demo_data["time"][:N],
            n_epochs=1,
        )
        res = model.results_
        assert "Ft" not in res, "Ft should not be in results when only Xt is provided"
        assert -1 in res.epoch.values, "Epoch -1 should exist from add_baselines(Xt=...)"

        # include_baselines=True should use the F.sel(epoch=-1) fallback
        axes = plot_receptive_fields(res, neurons=[0], include_baselines=True)
        # 3 columns: Behaviour + Epoch 1 + Best
        assert axes.shape == (1, 3)
        plt.close("all")

    def test_single_neuron_no_behaviour(self, results):
        axes = plot_receptive_fields(results, neurons=[0], epoch=1, include_behaviour=False)
        assert axes.shape == (1, 1)
        import matplotlib.pyplot as plt

        plt.close("all")


class TestPlotAllMetrics:
    def test_returns_axes(self, results):
        axes = plot_all_metrics(results)
        assert axes.size > 0
        import matplotlib.pyplot as plt

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
            coords={"dim": ["x", "y", "z"], "epoch": [0], "neuron": [0]},
        )
        with pytest.raises(ValueError, match="only supports 1-D and 2-D"):
            plot_receptive_fields(fake, neurons=[0])
        import matplotlib.pyplot as plt

        plt.close("all")
