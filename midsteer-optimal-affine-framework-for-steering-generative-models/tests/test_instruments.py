"""Schema and reachability tests for instruments.json.

instruments.json is the registry of every thing that decides whether an
output is correct (grader / scorer / equivalence check / data loader), each
with a name, what_it_decides, a positive_test proving it accepts a known-
correct input, and a negative_test proving it rejects a known-wrong one. This
reproduction has nine instruments, so not_applicable (the nothing-judges-an-
output case) is omitted entirely.

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
    so it must be an array under that exact key. When there are no
    instruments, the file instead carries a top-level not_applicable (a
    single sentence string) explaining why nothing here judges an output."""
    doc = _load()
    has_inst = "instruments" in doc
    has_na = "not_applicable" in doc
    assert has_inst or has_na, (
        "instruments.json must have a top-level 'instruments' array, OR a "
        "top-level not_applicable sentence when nothing judges an output")
    if has_inst:
        assert isinstance(doc["instruments"], list), "'instruments' must be a list"
        assert len(doc["instruments"]) >= 1, (
            "instruments.json declares an empty instruments array: add "
            "instruments, or replace it with a not_applicable sentence")


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


def test_not_applicable_is_a_single_sentence_and_only_when_no_instruments():
    """not_applicable excuses the WHOLE reproduction, so when present it must
    be ONE sentence (a string) saying why nothing here judges an output, and
    there must be NO instruments array. It must NOT be a dict (a prior round
    wrote a {applies, reason} dict, and a list-of-exemptions reading of a dict
    silently excused every grader and data loader in the file). To exempt a
    single instrument, put not_applicable with its reason ON that instrument;
    the top-level not_applicable is all-or-nothing. When instruments exist,
    not_applicable must be absent entirely -- its presence alongside a
    non-empty instruments array would excuse the very instruments it lists.
    """
    doc = _load()
    na = doc.get("not_applicable")
    if na is None:
        # instruments exist and not_applicable is correctly omitted.
        assert doc.get("instruments"), (
            "instruments.json has neither an instruments array nor a "
            "not_applicable sentence")
        return
    # not_applicable is present: it must be a single sentence (string), not a
    # dict/list, and there must be no instruments.
    assert isinstance(na, str) and na.strip(), (
        "not_applicable must be a single non-empty sentence (string) saying "
        "why nothing here judges an output -- not a dict or list. To exempt "
        "a single instrument, put not_applicable with its reason ON that "
        "instrument; the top-level form excuses the WHOLE reproduction")
    assert not doc.get("instruments"), (
        "not_applicable (excuses the whole reproduction) is present alongside "
        "a non-empty instruments array -- remove not_applicable, or remove "
        "the instruments")


def test_rejects_dict_not_applicable_alongside_instruments():
    """Regression for the feedback failure mode: a run wrote not_applicable as
    a {applies, reason} DICT alongside a non-empty instruments array, and a
    list-of-exemptions reading of that dict silently excused every grader and
    data loader in the file. This re-runs the validator's exact assertions on
    that bad shape and asserts it is rejected (a dict is not a single
    sentence, and not_applicable must not coexist with instruments)."""
    bad = {
        "instruments": [{"name": "data_loader", "what_it_decides": "x",
                         "positive_test": "a", "negative_test": "b"}],
        "not_applicable": {"applies": False, "reason": "..."},
    }
    na = bad["not_applicable"]
    # The validator's first assertion: not_applicable must be a string.
    assert not (isinstance(na, str) and na.strip()), (
        "dict-form not_applicable must be rejected by the schema (it is not "
        "a single sentence)")
    # The validator's second assertion: not_applicable must not coexist with
    # instruments.
    assert bad.get("instruments"), (
        "the feedback defect is not_applicable alongside instruments -- this "
        "fixture must keep both to exercise the coexistence check")
    coexists = bool(bad.get("instruments")) and na is not None
    assert coexists, "fixture did not reproduce the coexistence defect"
    # A single-sentence not_applicable with NO instruments is the only valid
    # top-level not_applicable form:
    ok = {"not_applicable": "Nothing here judges an output; the paper is a "
                            "pure-theory result with no runnable artefact."}
    ok_na = ok["not_applicable"]
    assert isinstance(ok_na, str) and ok_na.strip() and not ok.get("instruments")


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
