#!/usr/bin/env python3
"""Verify every mutation in mutations.json: apply find->replace, confirm the
must_fail test FAILS, revert, confirm it PASSES on clean code."""
import json, subprocess, sys, pathlib, shutil

ROOT = pathlib.Path(__file__).resolve().parent
PY = str(ROOT / ".venv/bin/python")
mut_file = ROOT / "mutations.json"
data = json.load(open(mut_file))
defects = data["defects"]

def run_pytest(node):
    r = subprocess.run([PY, "-m", "pytest", node, "-q", "-p", "no:cacheprovider"],
                       cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout, r.stderr

results = []
for d in defects:
    fpath = ROOT / d["file"]
    orig = fpath.read_text()
    find = d["find"]; repl = d["replace"]
    if find not in orig:
        results.append((d["id"], "FIND-MISSING", None)); continue
    # count occurrences -> must be unique
    nocc = orig.count(find)
    if nocc != 1:
        results.append((d["id"], f"NOT-UNIQUE({nocc})", None)); continue
    # apply defect
    fpath.write_text(orig.replace(find, repl, 1))
    rc, out, err = run_pytest(d["must_fail"])
    failed_under_defect = (rc != 0)
    # revert
    fpath.write_text(orig)
    rc2, out2, err2 = run_pytest(d["must_fail"])
    passes_on_clean = (rc2 == 0)
    ok = failed_under_defect and passes_on_clean
    results.append((d["id"], "OK" if ok else "BAD",
                    f"defect_fail={failed_under_defect} clean_pass={passes_on_clean}"))
    # also clear pycache so clean import unaffected
    for p in (ROOT/"src").rglob("__pycache__"): shutil.rmtree(p, ignore_errors=True)

print("=" * 60)
allok = True
for rid, status, det in results:
    print(f"{rid:45s} {status}  {det or ''}")
    if status != "OK": allok = False
print("=" * 60)
print("ALL MUTATIONS VERIFIED" if allok else "SOME MUTATIONS BROKEN")
sys.exit(0 if allok else 1)
