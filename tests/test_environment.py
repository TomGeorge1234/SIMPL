"""Tests for simpl.environment."""

import numpy as np

from simpl.environment import Environment


class TestEnvironment1D:
    def test_correct_dimensions(self):
        X = np.random.randn(100, 1)
        env = Environment(X, verbose=False)
        assert env.D == 1
        assert env.dim == ["x"]

    def test_correct_shape(self):
        X = np.random.randn(100, 1)
        env = Environment(X, bin_size=0.1, verbose=False)
        assert len(env.discrete_env_shape) == 1
        assert env.flattened_discretised_coords.shape[1] == 1


class TestEnvironment2D:
    def test_correct_dimensions(self):
        X = np.random.randn(100, 2)
        env = Environment(X, verbose=False)
        assert env.D == 2
        assert env.dim == ["x", "y"]

    def test_coords(self):
        X = np.random.randn(100, 2)
        env = Environment(X, verbose=False)
        assert "x" in env.coords_dict
        assert "y" in env.coords_dict


class TestEnvironment3D:
    def test_correct_dimensions(self):
        X = np.random.randn(100, 3)
        env = Environment(X, verbose=False)
        assert env.D == 3
        assert env.dim == ["x", "y", "z"]


class TestEnvironmentPadding:
    def test_limits_extend_beyond_data(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0]])
        env = Environment(X, pad=0.1, verbose=False)
        assert env.lims[0][0] < 0.0
        assert env.lims[1][0] > 1.0


class TestEnvironmentBinSize:
    def test_bin_count_matches_expected(self):
        X = np.array([[0.0], [1.0]])
        env = Environment(X, pad=0.0, bin_size=0.1, verbose=False)
        n_bins = env.discrete_env_shape[0]
        expected = int(np.ceil((1.0 - 0.0) / 0.1))
        assert abs(n_bins - expected) <= 1


class TestEnvironmentForceLims:
    def test_overrides_data_limits(self):
        X = np.random.randn(100, 2)
        lims = ((-5.0, -5.0), (5.0, 5.0))
        env = Environment(X, force_lims=lims, verbose=False)
        assert env.lims == lims


class TestEnvironmentDiscretisedCoords:
    def test_mesh_grid_correct(self):
        X = np.random.randn(50, 2)
        env = Environment(X, verbose=False)
        assert env.discretised_coords.shape[0] == 2
        assert env.discretised_coords.shape[1:] == env.discrete_env_shape


class TestEnvironmentFlattenedCoords:
    def test_shape(self):
        X = np.random.randn(50, 2)
        env = Environment(X, verbose=False)
        n_bins = np.prod(env.discrete_env_shape)
        assert env.flattened_discretised_coords.shape == (n_bins, 2)


class TestEnvironmentPlot:
    def test_returns_axes(self):
        import matplotlib

        matplotlib.use("Agg")
        X = np.random.randn(50, 2)
        env = Environment(X, verbose=False)
        ax = env.plot_environment()
        assert ax is not None
