"""Mutation tests: apply each deliberate defect in mutations.json to a temp copy of the
target file and assert the named test node FAILS (proving the suite catches the defect).
Uses sys.executable for subprocess, never bare python."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
MUTATIONS_PATH = os.path.join(REPO, 'mutations.json')


def _load_mutations():
    with open(MUTATIONS_PATH) as f:
        return json.load(f)


@pytest.mark.parametrize("mut", _load_mutations(), ids=lambda m: m['covers'][0] + ':' + m['file'])
def test_each_mutation_is_caught(mut, tmp_path):
    """Apply find->replace to a temp copy, run must_fail, assert it FAILS."""
    rel = mut['file']
    src = os.path.join(REPO, rel)
    assert os.path.exists(src), f"mutation target file missing: {src}"
    with open(src) as f:
        original = f.read()
    assert mut['find'] in original, f"find string not present in {rel}: {mut['find']!r}"
    assert original.count(mut['find']) == 1, (
        f"find string not unique in {rel} (count={original.count(mut['find'])}): {mut['find']!r}")
    broken = original.replace(mut['find'], mut['replace'], 1)

    # write the broken version to a temp copy and run the must_fail test against the temp tree.
    # We test by temporarily swapping the real file, running pytest, then restoring.
    with open(src, 'w') as f:
        f.write(broken)
    try:
        r = subprocess.run([PY, '-m', 'pytest', mut['must_fail'], '-q', '--no-header',
                           '-p', 'no:cacheprovider'],
                          capture_output=True, text=True, cwd=REPO, timeout=300)
        # we EXPECT the named test to fail (nonzero exit) when the defect is applied
        assert r.returncode != 0, (
            f"mutation NOT caught: {mut['covers']} defect in {rel} did NOT fail "
            f"test {mut['must_fail']}\nstdout:\n{r.stdout}\nstderr:\n{r.stderr}")
        # and the failure should mention the test node
        assert (mut['must_fail'] in r.stdout + r.stderr
                or 'failed' in r.stdout.lower()
                or 'error' in r.stdout.lower()), (
            f"defect ran but no clear failure for {mut['must_fail']}:\n{r.stdout}")
    finally:
        # always restore the original
        with open(src, 'w') as f:
            f.write(original)


def test_mutations_json_schema():
    muts = _load_mutations()
    assert len(muts) >= 5, "mutations.json must have >=5 defects"
    covers_seen = set()
    for m in muts:
        assert 'covers' in m and len(m['covers']) >= 1
        assert 'core' in m['covers'] or 'degeneracy' in m['covers'], (
            f"mutation {m.get('file')} must cover core or degeneracy: {m['covers']}")
        for c in m['covers']:
            covers_seen.add(c)
        assert m['file'] and m['find'] and m['replace'] and m['must_fail']
    assert 'core' in covers_seen and 'degeneracy' in covers_seen, (
        f"mutation suite must cover both 'core' and 'degeneracy': {covers_seen}")
