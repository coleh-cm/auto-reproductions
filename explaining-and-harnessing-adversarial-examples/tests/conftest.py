"""conftest.py — pin determinism for the test suite.

Tree-restore is OPT-IN (default OFF). It used to run on every conftest import
to recover from an interrupted mutation run, but that auto-restore reverts
*any* planted defect at collection time — including the deliberate defects the
external mutation gate plants directly into a source file before running the
must_fail test node. The gate does not set EAE_SKIP_RESTORE (that flag is local
to our own tests/test_mutations.py subprocess), so the auto-restore silently
undid the gate's defect, the must_fail test ran on clean code, PASSED, and the
gate reported the mutation as SURVIVED. Making restore opt-in lets the gate's
planted defect reach the test.

Recovery from an interrupted mutation run is now handled by
tests/test_mutations.py's own `finally: git checkout` (robust to normal
interruption) plus the committed-clean tree. To force a clean-tree reset on
import (e.g. after a hard-killed mutation run), set EAE_AUTO_RESTORE=1.
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

    Opt-in via EAE_AUTO_RESTORE=1 (default OFF). When enabled, runs at conftest
    import (before the test modules are collected/imported) so in-memory modules
    see clean source. Always skipped when EAE_SKIP_RESTORE is set (our own
    tests/test_mutations.py sets that so its deliberately-planted defect survives
    to be caught).

    Why opt-in / default OFF: the external mutation gate plants a defect
    directly in a source file and then runs the must_fail test node WITHOUT
    EAE_SKIP_RESTORE. An unconditional restore here would revert the gate's
    defect at collection time, the must_fail test would run on clean code, PASS,
    and the gate would report the mutation as SURVIVED. Default-off lets the
    gate's planted defect reach the test; the gate reverts via git itself
    between mutations."""
    if os.environ.get("EAE_SKIP_RESTORE"):
        return
    if not os.environ.get("EAE_AUTO_RESTORE"):
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
# Opt-in (see _restore_clean_tree docstring); a no-op unless EAE_AUTO_RESTORE=1.
_restore_clean_tree()


@pytest.fixture(autouse=True)
def _deterministic():
    import torch, numpy, random
    torch.set_num_threads(1)
    torch.manual_seed(0)
    numpy.random.seed(0)
    random.seed(0)
    yield
