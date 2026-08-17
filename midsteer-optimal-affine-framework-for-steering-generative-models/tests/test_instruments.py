"""Schema and reachability tests for instruments.json.

instruments.json is the registry of every thing that decides whether an
output is correct (grader / scorer / equivalence check / data loader), each
with a name, what_it_decides, a positive_test proving it accepts a known-
correct input, and a negative_test proving it rejects a known-wrong one. If
nothing here judged an output we would say so with not_applicable and a
reason; this reproduction has nine instruments, so not_applicable does not
apply.

These tests guard the SCHEMA (not the behavior -- the behavioral positive/
negative cases live in test_eval_metrics.py, test_data.py, test_invariants.py,
test_claims_eval.py and test_mutations.py). They prevent the regression where
instruments.json is restructured into a shape the gate no longer recognises as
"declaring instruments" (a prior review round rejected a flat named-key
object as "declares no instruments and gives no reason"; the gate expects a
top-level "instruments" array).

Subprocess invocations use sys.executable (never bare python).
"""
import json
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INSTRUMENTS_PATH = os.path.join(REPO, "instruments.json")
PY = sys.executable

REQUIRED_FIELDS = ("name", "what_it_decides", "positive_test", "negative_test")


def _load():
    with open(INSTRUMENTS_PATH) as f:
        return json.load(f)


def test_instruments_json_top_level_is_instruments_array():
    """The gate looks for a top-level "instruments" array. A flat named-key
    object (the prior structure) is not recognised as declaring instruments,
    so it must be an array under that exact key."""
    doc = _load()
    assert "instruments" in doc, (
        "instruments.json must have a top-level 'instruments' array "
        "(or a not_applicable with a reason if nothing judges an output)")
    assert isinstance(doc["instruments"], list), "'instruments' must be a list"
    assert len(doc["instruments"]) >= 1, (
        "instruments.json declares no instruments and gives no reason: "
        "add instruments, or a not_applicable with a reason")


def test_each_instrument_has_required_fields():
    doc = _load()
    for i, inst in enumerate(doc["instruments"]):
        for field in REQUIRED_FIELDS:
            assert field in inst, (
                f"instrument #{i} ({inst.get('name', '?')}) missing field {field!r}")
        assert isinstance(inst["name"], str) and inst["name"], (
            f"instrument #{i} name must be a non-empty string")
        assert isinstance(inst["what_it_decides"], str) and inst["what_it_decides"], (
            f"instrument {inst['name']!r} what_it_decides must be non-empty")
        assert isinstance(inst["positive_test"], str) and inst["positive_test"], (
            f"instrument {inst['name']!r} positive_test must be a non-empty node id")
        assert isinstance(inst["negative_test"], str) and inst["negative_test"], (
            f"instrument {inst['name']!r} negative_test must be a non-empty node id")


def test_not_applicable_only_when_no_instruments():
    """If not_applicable is present with applies=true, there must be no
    instruments; if instruments exist, not_applicable.applies must be false
    with a reason. This catches a contradictory file."""
    doc = _load()
    na = doc.get("not_applicable")
    if na is not None:
        assert "applies" in na and "reason" in na, (
            "not_applicable must have 'applies' and 'reason'")
        if na["applies"]:
            assert len(doc["instruments"]) == 0, (
                "not_applicable.applies=true but instruments are declared")
        else:
            assert na["reason"], (
                "not_applicable.applies=false requires a non-empty reason")


def _collectable_nodes():
    """Return the set of collectible pytest node ids in this repo. Cached on
    the function attribute so we run pytest --collect-only once."""
    if hasattr(_collectable_nodes, "_cache"):
        return _collectable_nodes._cache
    r = subprocess.run(
        [PY, "-m", "pytest", "--collect-only", "-q", "--no-header",
         "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=REPO, timeout=180,
    )
    # collect-only -q prints "<file>::<test>" lines plus a trailing summary.
    nodes = set()
    for line in r.stdout.splitlines():
        line = line.strip()
        if "::" in line and not line.endswith("::"):
            nodes.add(line)
    _collectable_nodes._cache = nodes
    return nodes


def test_instrument_test_nodes_are_collectible():
    """Every positive_test and negative_test node id (the part before any
    trailing parametrisation) must be a real collectible test in this repo --
    an instrument that points at a non-existent test is an instrument that
    cannot be exercised, which is the same as not having one. additional_tests
    are checked too, except for non-test script paths (e.g.
    experiments/run_e1_synth.py, which has no '::' and is run as a script)."""
    nodes = _collectable_nodes()
    doc = _load()

    def _present(node):
        # A node is present if it is collectible as-is, OR (for parametrised
        # tests) some collectible node starts with "<base>[" -- the base name
        # alone is not collected when the test is parametrised.
        base = node.split("[")[0]
        if base in nodes:
            return True
        return any(n.startswith(base + "[") for n in nodes)

    for inst in doc["instruments"]:
        for field in ("positive_test", "negative_test"):
            assert _present(inst[field]), (
                f"instrument {inst['name']!r} {field}={inst[field]!r} is not "
                f"a collectible pytest node in this repo")
        for extra in inst.get("additional_tests", []):
            if "::" not in extra:
                # a script path (e.g. experiments/run_e1_synth.py) -- assert the
                # file exists instead of a pytest node.
                assert os.path.exists(os.path.join(REPO, extra)), (
                    f"instrument {inst['name']!r} additional_test script "
                    f"{extra!r} does not exist")
            else:
                assert _present(extra), (
                    f"instrument {inst['name']!r} additional_test {extra!r} "
                    f"is not a collectible pytest node")


def test_instruments_cover_a_data_loader():
    """A data loader is an instrument; one of the instruments must decide
    whether the paper's own dataset is loaded by fingerprint, so a synthetic
    fallback is caught (the closed-book 65-token-vocabulary failure mode)."""
    doc = _load()
    names = {inst["name"] for inst in doc["instruments"]}
    assert "data_loader" in names, (
        "instruments.json must include a data_loader instrument that "
        "fingerprint-checks the paper's own dataset")


def test_instruments_cover_a_degeneracy_or_invariant_check():
    """The method at its no-op setting must reproduce the baseline exactly
    (degeneracy), and the paper's equation-invariants must be checked. At
    least one instrument must decide a degeneracy/equivalence/invariant
    question (the cheapest real correctness evidence)."""
    doc = _load()
    found = False
    for inst in doc["instruments"]:
        text = (inst["name"] + " " + inst["what_it_decides"]).lower()
        if any(k in text for k in ("degeneracy", "invariant", "equivalence", "leace")):
            found = True
            break
    assert found, (
        "instruments.json must include a degeneracy/equivalence/invariant "
        "instrument (the method at its no-op setting must reproduce the "
        "baseline exactly)")


def test_requires_tools_is_declared_per_instrument():
    """Each instrument must declare whether it requires external tools (CUDA,
    HF_TOKEN, model downloads) so a reader knows which verdicts are blocked
    by this sandbox vs measured. Booleans, strings, or lists are all accepted
    -- the field's presence is what matters."""
    doc = _load()
    for inst in doc["instruments"]:
        assert "requires_tools" in inst, (
            f"instrument {inst['name']!r} must declare requires_tools "
            f"(true/false, a string, or a list of external dependencies)")
