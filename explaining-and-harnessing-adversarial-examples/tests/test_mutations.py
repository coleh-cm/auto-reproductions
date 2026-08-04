#!/usr/bin/env python3
"""test_mutations.py — verify each mutation in mutations.json is caught by its
must_fail test. A suite nobody has broken on purpose is not evidence.

For each mutation: apply the find->replace to the file, run the must_fail test
node, assert it FAILS (the defect is caught), then revert. If the must_fail
test PASSES with the defect present, the mutation is not covered — fail.

Robustness notes (learned the hard way):
  - The mutation subprocess MUST NOT write .pyc for the mutated file. The
    revert happens within the same wall-clock second, and on second-granularity
    filesystems Python's mtime-based .pyc check then treats the stale mutated
    bytecode as valid against the reverted (clean) source — later runs execute
    mutated code while the traceback shows clean source. We disable bytecode
    writing in the subprocess and also delete the mutated module's .pyc after
    reverting.
  - Revert via `git checkout` (idempotent) so an interrupted run leaves the
    tree recoverable by the session fixture in conftest.py.
"""
import glob
import json
import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUT_PATH = os.path.join(REPO, "mutations.json")
PY = sys.executable


def _pyc_for(fpath):
    """All __pycache__ .pyc files that could be stale for `fpath`."""
    d = os.path.dirname(fpath)
    base = os.path.splitext(os.path.basename(fpath))[0]
    return glob.glob(os.path.join(d, "__pycache__", base + "*.pyc"))


def _run_test(node):
    """Return True if the test node FAILS (defect caught). Runs in a subprocess
    with bytecode writing disabled so the mutated file never leaves a .pyc, and
    with EAE_SKIP_RESTORE set so conftest does NOT reset the planted defect."""
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["EAE_SKIP_RESTORE"] = "1"
    r = subprocess.run([PY, "-B", "-m", "pytest", "-q", "-x", node, "--no-header",
                        "-p", "no:cacheprovider", "--tb=no"],
                       cwd=REPO, env=env, capture_output=True, text=True, timeout=120)
    return r.returncode != 0  # non-zero => test failed => defect caught


def _restore(fpath):
    """Restore fpath from git (idempotent) and purge any stale .pyc for it."""
    subprocess.run(["git", "checkout", "HEAD", "--", fpath],
                   cwd=REPO, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for p in _pyc_for(fpath):
        try:
            os.remove(p)
        except OSError:
            pass


def test_each_mutation_is_caught():
    muts = json.load(open(MUT_PATH))["mutations"]
    fails = []
    for mut in muts:
        fpath = os.path.join(REPO, mut["file"])
        try:
            src = open(fpath).read()
            if mut["find"] not in src:
                fails.append(f"{mut['id']}: find string not present in {mut['file']}")
                continue
            mutated = src.replace(mut["find"], mut["replace"], 1)
            open(fpath, "w").write(mutated)
            # purge this module's .pyc so the subprocess starts from the mutated source
            for p in _pyc_for(fpath):
                try:
                    os.remove(p)
                except OSError:
                    pass
            caught = _run_test(mut["must_fail"])
            if not caught:
                fails.append(f"{mut['id']}: must_fail test '{mut['must_fail']}' did NOT fail with the defect")
        finally:
            _restore(fpath)
    assert not fails, "mutations not covered:\n  " + "\n  ".join(fails)
