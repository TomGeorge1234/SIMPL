"""Shared fixtures for SIMPL test suite."""

import jax
import pytest

from simpl.utils import load_demo_data

_HAS_METAL = any(d.platform == "METAL" for d in jax.devices())


@pytest.fixture(autouse=True)
def _force_cpu_on_metal(request):
    """When running on Apple Metal, force CPU for tests marked ``cpu_only``.

    Metal lacks support for linalg.inv, cholesky, and fft.  Tests that call
    functions using these ops directly (outside the Kalman ``force_cpu`` path)
    need to run on CPU.
    """
    marker = request.node.get_closest_marker("cpu_only")
    if marker is not None and _HAS_METAL:
        cpu = jax.devices("cpu")[0]
        with jax.default_device(cpu):
            yield
    else:
        yield


@pytest.fixture(scope="session")
def demo_data():
    """Load the bundled gridcells_synthetic.npz."""
    return load_demo_data("gridcells_synthetic.npz")


@pytest.fixture(scope="session")
def small_simpl_model(demo_data):
    """A SIMPL instance using a small subset of the data for speed."""
    from simpl.simpl import SIMPL

    N = 2000  # small subset
    N_neurons = min(10, demo_data["Y"].shape[1])
    model = SIMPL()
    model.fit(
        Y=demo_data["Y"][:N, :N_neurons],
        Xb=demo_data["Xb"][:N],
        time=demo_data["time"][:N],
        n_iterations=0,
    )
    return model
