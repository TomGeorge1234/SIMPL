"""Shared fixtures for SIMPL test suite."""

import pytest

from simpl.utils import load_datafile


@pytest.fixture(scope="session")
def demo_data():
    """Load the bundled gridcelldata.npz."""
    return load_datafile("gridcelldata.npz")


@pytest.fixture(scope="session")
def small_simpl_model(demo_data):
    """A SIMPL instance using a small subset of the data for speed."""
    from simpl.simpl import SIMPL

    N = 2000  # small subset
    N_neurons = min(10, demo_data["Y"].shape[1])
    model = SIMPL(evaluate_each_epoch=True, verbose=False)
    model.fit(
        Y=demo_data["Y"][:N, :N_neurons],
        Xb=demo_data["Xb"][:N],
        time=demo_data["time"][:N],
        n_epochs=0,
    )
    return model
