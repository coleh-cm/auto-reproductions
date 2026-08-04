#!/usr/bin/env python3
"""test_mutations.py — verify each mutation in mutations.json is caught by its
must_fail test. A suite nobody has broken on purpose is not evidence.

For each mutation: apply the find->replace to the file, run the must_fail test
node, assert it FAILS (the defect is caught), then revert. If the must_fail
test PASSES with the defect present, the mutation is not covered — fail.
"""
import json
import os
import shutil
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUT_PATH = os.path.join(REPO, "mutations.json")
PY = sys.executable


def _run_test(node):
    """Return True if the test node FAILS (defect caught)."""
    r = subprocess.run([PY, "-m", "pytest", "-q", "-x", node, "--no-header",
                        "-p", "no:cacheprovider", "--tb=no"],
                       cwd=REPO, capture_output=True, text=True, timeout=120)
    return r.returncode != 0  # non-zero => test failed => defect caught


def test_each_mutation_is_caught():
    muts = json.load(open(MUT_PATH))["mutations"]
    fails = []
    for mut in muts:
        fpath = os.path.join(REPO, mut["file"])
        backup = fpath + ".mutbak"
        shutil.copy(fpath, backup)
        try:
            src = open(fpath).read()
            if mut["find"] not in src:
                fails.append(f"{mut['id']}: find string not present in {mut['file']}")
                continue
            mutated = src.replace(mut["find"], mut["replace"], 1)
            open(fpath, "w").write(mutated)
            caught = _run_test(mut["must_fail"])
            if not caught:
                fails.append(f"{mut['id']}: must_fail test '{mut['must_fail']}' did NOT fail with the defect")
        finally:
            shutil.move(backup, fpath)
    assert not fails, "mutations not covered:\n  " + "\n  ".join(fails)
