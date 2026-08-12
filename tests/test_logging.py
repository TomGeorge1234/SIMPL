"""Tests for package-level logging configuration."""

import logging

import simpl  # noqa: F401 - importing installs the logging filter


def test_optional_backend_probe_messages_are_filtered():
    logger = logging.getLogger("jax._src.xla_bridge")
    filters = logger.filters

    def is_visible(message):
        record = logging.LogRecord(logger.name, logging.INFO, __file__, 1, message, (), None)
        return all(log_filter.filter(record) for log_filter in filters)

    assert not is_visible("Unable to initialize backend 'rocm': missing runtime")
    assert not is_visible("Unable to initialize backend 'tpu': missing libtpu.so")
    assert is_visible("Unable to initialize backend 'METAL': actual Metal problem")
