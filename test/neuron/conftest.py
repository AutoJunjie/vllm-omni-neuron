# SPDX-License-Identifier: Apache-2.0
"""Fixtures for the on-device tests under test/neuron/.

Every test here needs a trn2 node with the plugin installed and the
HunyuanImage-3.0-Instruct checkpoint on disk; ``require_neuron_device`` skips it
anywhere else. The heavy parts run in subprocesses (torchrun workers, ``run.py``), so
the pytest process itself never holds a NeuronCore.
"""

import glob
import os

import pytest

MODEL_PATH = os.environ.get("HY3_MODEL_PATH", "/workspace/models/HunyuanImage-3.0-Instruct")


def pytest_configure(config):
    config.addinivalue_line("markers", "neuron: needs Neuron hardware (/dev/neuron* present)")


@pytest.fixture(scope="session")
def require_neuron_device():
    if not glob.glob("/dev/neuron*"):
        pytest.skip("needs Neuron hardware (/dev/neuron*)")
    if not os.path.isfile(os.path.join(MODEL_PATH, "model.safetensors.index.json")):
        pytest.skip(f"needs the HunyuanImage-3.0-Instruct checkpoint at {MODEL_PATH} (HY3_MODEL_PATH)")
    return MODEL_PATH
