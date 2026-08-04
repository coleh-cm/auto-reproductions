"""conftest.py — pin determinism for the test suite."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pytest


@pytest.fixture(autouse=True)
def _deterministic():
    import torch, numpy, random
    torch.set_num_threads(1)
    torch.manual_seed(0)
    numpy.random.seed(0)
    random.seed(0)
    yield
