#!/usr/bin/env python3
"""Apply each mutation in mutations.json, run its must_fail test, confirm it FAILS,
then revert. A mutation that does NOT fail its must_fail test is a broken suite
(reported to stderr, exit nonzero). Uses the venv python via sys.executable.

mutations.json may be a flat list of mutation dicts OR a dict with a 'mutations' key.

Usage: python tests/run_mutations.py
"""
from __future__ import annotations
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable


def _load_mutations():
    with open(os.path.join(REPO, 'mutations.json')) as f:
        d = json.load(f)
    return d['mutations'] if isinstance(d, dict) else d


def _read(path):
    with open(path) as f:
        return f.read()


def _write(path, content):
    with open(path, 'w') as f:
        f.write(content)


def _run_test(node):
    r = subprocess.run([PY, '-m', 'pytest', node, '-q', '--no-header',
                        '--tb=no', '-p', 'no:warnings'],
                       cwd=REPO, capture_output=True, text=True, timeout=180)
    return r.returncode, r.stdout + r.stderr


def main():
    muts = _load_mutations()
    results = []
    for i, m in enumerate(muts):
        mid = m.get('id', f'M{i}')
        fpath = os.path.join(REPO, m['file'])
        original = _read(fpath)
        if m['find'] not in original:
            print(f"  {mid} SKIP: find not in file (already applied or stale)", file=sys.stderr)
            results.append((mid, 'SKIP_FIND_MISSING', None))
            continue
        mutated = original.replace(m['find'], m['replace'], 1)
        if mutated == original:
            print(f"  {mid} SKIP: replace is identical", file=sys.stderr)
            results.append((mid, 'SKIP_NO_CHANGE', None))
            continue
        _write(fpath, mutated)
        try:
            rc, out = _run_test(m['must_fail'])
            broke = rc != 0
            results.append((mid, 'BROKE' if broke else 'DID_NOT_BREAK', m['must_fail']))
            status = 'OK (mutation broke must_fail test)' if broke else 'FAIL (mutation did NOT break must_fail test)'
            print(f"  {mid}: {status}")
            if not broke:
                print(f"     pytest output tail:\n     " + out[-400:].replace('\n', '\n     '), file=sys.stderr)
        finally:
            _write(fpath, original)
    bad = [r for r in results if r[1] == 'DID_NOT_BREAK' or r[1].startswith('SKIP')]
    print(f"\nmutations: {len(results)} total, {sum(1 for r in results if r[1]=='BROKE')} broke their test, "
          f"{len(bad)} did not break / skipped")
    if any(r[1] == 'DID_NOT_BREAK' for r in results):
        sys.exit(1)


if __name__ == '__main__':
    main()
