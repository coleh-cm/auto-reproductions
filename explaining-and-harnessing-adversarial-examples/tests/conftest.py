"""conftest.py — pin determinism for the test suite.

Also restores a clean working tree at import time. test_mutations.py mutates
real source files (train.py, attack.py, models.py, tests/test_invariants.py)
and reverts them in a `finally`; if a previous run was interrupted (kill/timeout)
the reverted file never came back and the repo shipped the planted defect
permanently. We restore at conftest import time — BEFORE pytest collects the
test modules and imports `train`/`models` — so the in-memory modules never see
a leftover defect. This runs before any test, so a poisoned tree from an
interrupted mutation run can never reach the suite.
"""
import json
import os
import subprocess

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _mutation_files():
    """Files the mutation suite may mutate, read from mutations.json."""
    mut_path = os.path.join(REPO, "mutations.json")
    if not os.path.exists(mut_path):
        return []
    try:
        muts = json.load(open(mut_path)).get("mutations", [])
    except Exception:
        return []
    return [m["file"] for m in muts if "file" in m]


def _restore_clean_tree():
    """Reset every mutation-target file from git and drop stray .mutbak backups.
    Idempotent: a no-op on a clean tree. Runs at conftest import (before the
    test modules are collected/imported) so in-memory modules see clean source.
    Skipped when EAE_SKIP_RESTORE is set — the mutation test sets this in its
    subprocess so the deliberately-planted defect survives to be caught."""
    if os.environ.get("EAE_SKIP_RESTORE"):
        return
    files = _mutation_files()
    for f in files:
        bak = os.path.join(REPO, f + ".mutbak")
        if os.path.exists(bak):
            try:
                os.remove(bak)
            except OSError:
                pass
    targets = [f for f in files if os.path.exists(os.path.join(REPO, f))]
    if targets:
        subprocess.run(
            ["git", "checkout", "HEAD", "--"] + targets,
            cwd=REPO, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )


# Runs at conftest import — before pytest collects imports of train/models.
_restore_clean_tree()


@pytest.fixture(autouse=True)
def _deterministic():
    import torch, numpy, random
    torch.set_num_threads(1)
    torch.manual_seed(0)
    numpy.random.seed(0)
    random.seed(0)
    yield
