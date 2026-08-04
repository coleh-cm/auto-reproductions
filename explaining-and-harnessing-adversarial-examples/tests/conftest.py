"""Shared pytest fixtures for the EAE reproduction test suite.

This conftest exists to make the suite **deterministic regardless of test
ordering**. Several invariant/instrument/constructed-truth tests compare exact
integer counts or exact sign-equality on small random batches. Their result
previously depended on two non-deterministic inputs:

  1. **Inherited global RNG state.** A handful of tests call ``torch.rand`` /
     ``torch.randint`` without first setting ``torch.manual_seed``, so they
     inherited whatever state the preceding test left in the global generator.
     A particular collection order could therefore produce a near-zero gradient
     element and flip a result.

  2. **Thread-parallel float reductions.** ``nn.Linear`` / autograd matmuls run
     on multiple OMP threads; the summation order is non-associative, so a
     borderline gradient sign can flip across thread configurations, which is
     enough on a 128-sample batch to make two attackers' adversarial-error
     counts tie (``117 == 117``) where the test expects them to differ.

Both are float/statistical flakiness, not logic errors. The autouse fixture below
removes both sources: it pins the global torch generator to a fixed seed before
every test, and pins torch to a single thread so reductions are deterministic.
Tests that set their own ``torch.manual_seed`` do so *after* the fixture and are
unaffected; tests that relied on inherited state now get a deterministic seed
instead. The net effect is that ``pytest tests/`` is order-independent and
reproducible, which is what the determinism readiness gate (#6) and
``VERIFICATION.md`` claim.
"""

import pytest
import torch


@pytest.fixture(autouse=True)
def _deterministic_torch():
    # Single-threaded matmul/autograd: removes non-associative-reduction drift.
    torch.set_num_threads(1)
    # Reset the global generator so no test inherits another test's RNG state.
    torch.manual_seed(0)
    yield
