import simpl


def test_version_is_exposed():
    assert isinstance(simpl.__version__, str)
    assert simpl.__version__
    assert "__version__" in simpl.__all__
