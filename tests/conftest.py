"""Shared fixtures for SIMPL test suite."""

import numpy as np
import pytest

from simpl.environment import Environment
from simpl.utils import load_datafile, prepare_data


@pytest.fixture(scope="session")
def demo_data():
    """Load the bundled gridcelldata.npz."""
    return load_datafile("gridcelldata.npz")


@pytest.fixture(scope="session")
def prepared_data(demo_data):
    """Prepare demo data via prepare_data() — single trial."""
    return prepare_data(
        Y=demo_data["Y"],
        Xb=demo_data["Xb"],
        time=demo_data["time"],
        dims=demo_data["dim"],
        neurons=demo_data["neuron"],
        Xt=demo_data["Xt"],
        Ft=demo_data["Ft"],
        Ft_coords_dict={"x": demo_data["x"], "y": demo_data["y"]},
    )


@pytest.fixture(scope="session")
def prepared_data_with_trials(demo_data):
    """Prepare demo data with trial boundaries."""
    T = demo_data["Y"].shape[0]
    boundaries = np.array([0, T // 3, 2 * T // 3])
    return prepare_data(
        Y=demo_data["Y"],
        Xb=demo_data["Xb"],
        time=demo_data["time"],
        dims=demo_data["dim"],
        neurons=demo_data["neuron"],
        Xt=demo_data["Xt"],
        Ft=demo_data["Ft"],
        Ft_coords_dict={"x": demo_data["x"], "y": demo_data["y"]},
        trial_boundaries=boundaries,
    )


@pytest.fixture(scope="session")
def environment(demo_data):
    """Create an Environment from demo data."""
    return Environment(demo_data["Xb"], verbose=False)


@pytest.fixture(scope="session")
def small_simpl_model(demo_data, environment):
    """A SIMPL instance using a small subset of the data for speed."""
    from simpl.simpl import SIMPL

    N = 2000  # small subset
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
    return model
