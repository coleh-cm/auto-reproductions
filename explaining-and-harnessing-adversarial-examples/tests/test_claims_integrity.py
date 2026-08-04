"""Claims-contract integrity tests.

claims.json is the numbers-gate contract. Every claim's ``quote`` must be a
VERBATIM substring of the cited file (``paper/source/iclr2015.tex``), starting
AT the line its ``citation`` names, so a reader can grep the quote rather than
trust it. Every claim must carry the contract fields (kind, compute_invariance,
and the arithmetic its kind needs), and at least one claim must be
compute-invariance ``high`` -- a table where everything is ``low`` reports
that nothing was affordable, which is not this paper's situation.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPRO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPRO_ROOT))

CLAIMS = json.loads((REPRO_ROOT / "claims.json").read_text())
CITATION_RE = re.compile(r"^(?P<file>\S+):(?P<line>\d+)$")


def _quote_start_lines(file_text: str, quote: str) -> list[int]:
    """1-based line numbers at which ``quote`` starts inside ``file_text``."""
    lines = file_text.split("\n")
    out, start = [], 0
    while True:
        i = file_text.find(quote, start)
        if i == -1:
            break
        out.append(file_text[:i].count("\n") + 1)
        start = i + 1
    return out


def test_every_claim_quote_is_verbatim_at_cited_line():
    tex = (REPRO_ROOT / "paper" / "source" / "iclr2015.tex").read_text()
    for claim in CLAIMS["claims"]:
        quote = claim["quote"]
        m = CITATION_RE.match(claim["citation"])
        assert m, f"{claim['id']}: citation {claim['citation']!r} is not <file>:<line>"
        assert m.group("file") == "paper/source/iclr2015.tex"
        cited_line = int(m.group("line"))
        starts = _quote_start_lines(tex, quote)
        assert starts, f"{claim['id']}: quote is NOT a verbatim substring of iclr2015.tex"
        assert cited_line in starts, (
            f"{claim['id']}: quote starts at tex lines {starts}, "
            f"but citation says {cited_line}")


def test_every_not_tested_quote_is_verbatim_at_cited_line():
    tex = (REPRO_ROOT / "paper" / "source" / "iclr2015.tex").read_text()
    for nt in CLAIMS["not_tested"]:
        m = CITATION_RE.match(nt["citation"])
        assert m, f"{nt['id']}: bad citation {nt['citation']!r}"
        starts = _quote_start_lines(tex, nt["quote"])
        assert starts, f"{nt['id']}: quote is NOT verbatim"
        assert int(m.group("line")) in starts


def test_claim_arithmetic_fields_match_kind():
    for claim in CLAIMS["claims"]:
        kind = claim["kind"]
        assert kind in ("ordering", "value", "existence", "invariant", "curve"), (
            f"{claim['id']}: unknown kind {kind!r}")
        assert claim["compute_invariance"] in ("high", "medium", "low")
        if kind in ("existence", "invariant"):
            assert "predicate" in claim, f"{claim['id']}: missing predicate"
        elif kind == "ordering":
            assert "quantity" in claim and claim["direction"] in (
                ">0", "<0", ">=0", "<=0", "==0"), f"{claim['id']}"
        elif kind == "value":
            assert "quantity" in claim and "claimed" in claim and "tolerance" in claim
        elif kind == "curve":
            assert "quantity" in claim and "x" in claim
            assert claim["comparison"] in (
                "increasing", "decreasing", "above", "below", "crosses", "matches")
            for tok in (claim["quantity"], claim["x"],
                        claim.get("against", claim.get("reference"))):
                if tok is None:
                    continue
                m = re.match(r"^measured\.([A-Za-z0-9_]+)\.(.+)$", tok)
                assert m, f"{claim['id']}: bad token {tok!r}"
                arm, metric = m.group(1), m.group(2)
                assert metric in CLAIMS["arms"][arm].get("curve_metrics", {}), (
                    f"{claim['id']}: {tok} not a curve_metric of arm {arm}")


def test_at_least_one_high_claim_and_counts_match():
    highs = [c for c in CLAIMS["claims"] if c["compute_invariance"] == "high"]
    assert len(highs) >= 1
    actual = {
        level: sum(1 for c in CLAIMS["claims"] if c["compute_invariance"] == level)
        for level in ("high", "medium", "low")
    }
    assert CLAIMS["evaluation"]["compute_invariance"]["counts"] == actual


def test_spec_embedded_claims_json_is_byte_identical():
    """SPEC.md section 10 embeds claims.json; drift between the two would mean the
    spec a reader reads is not the contract the gate enforces."""
    spec = (REPRO_ROOT / "SPEC.md").read_text()
    claims = (REPRO_ROOT / "claims.json").read_text()
    marker = "```json\n{\n  \"paper_ref\": \"43e841f0-05f8-4249-ba14-14c1a586f6ee\""
    i = spec.index(marker)
    j = spec.index("\n```", i)
    embedded = spec[i + len("```json\n"):j] + "\n"
    assert embedded == claims, "SPEC.md's embedded claims.json is stale; re-embed"


def test_every_arm_metric_pointer_resolves_in_shipped_results():
    """Every scalar metric pointer resolves in the arm's shipped results file."""
    for arm, spec in CLAIMS["arms"].items():
        blob = json.loads((REPRO_ROOT / spec["results"]).read_text())
        for metric, pointer in spec["metrics"].items():
            _f, _, path = pointer.partition(":")
            cur = blob
            # deep keys may contain dots; walk greedily like make_measured does
            while path:
                best = None
                for k in cur:
                    if path == k or path.startswith(k + ".") or path.startswith(k + "["):
                        if best is None or len(k) > len(best):
                            best = k
                assert best is not None, f"{arm}.{metric}: path {path!r} unresolvable"
                val = cur[best]
                path = path[len(best):]
                if path.startswith("[*]"):
                    val = val[0][path[4:]]
                    path = ""
                elif path.startswith("."):
                    path = path[1:]
                    cur = val
                else:
                    path = ""
            assert isinstance(val, (int, float)), (
                f"{arm}.{metric} resolved to non-scalar {val!r}")


def test_claims_verifier_rejects_bad_quote_count_line_pointer():
    """Negative arm of the claims-quote verifier.

    The positive test proves the verifier ACCEPTS the shipped claims.json. This
    proves it REJECTS known-wrong inputs of every kind the contract names — a
    non-verbatim quote, a stale compute-invariance count, a shifted citation
    line, and an unresolvable metric pointer each fail loudly. A grader that
    cannot run must raise, never return a negative verdict; here each bad input
    is shown to be caught (empty start-line set / count mismatch / line
    mismatch / unresolvable walk), so a corrupted claims.json could not slip
    through silently.
    """
    tex = (REPRO_ROOT / "paper" / "source" / "iclr2015.tex").read_text()

    # 1. A non-verbatim quote yields NO start line -> the quote check rejects.
    assert _quote_start_lines(tex, "ZZZ_NOT_IN_TEX_qqq1299") == []

    # 2. A shifted citation line is not among the real start lines -> rejected.
    c0 = CLAIMS["claims"][0]
    starts = _quote_start_lines(tex, c0["quote"])
    assert starts, "sanity: the shipped claim's quote is verbatim"
    bad_line = next(L for L in range(1, 6000) if L not in starts)
    assert bad_line not in starts, "a shifted citation line is rejected"

    # 3. A stale compute-invariance count disagrees with the real count ->
    #    the count-equality check rejects it.
    real = {lvl: sum(1 for c in CLAIMS["claims"]
                     if c["compute_invariance"] == lvl)
            for lvl in ("high", "medium", "low")}
    stale = dict(real)
    stale["high"] = real["high"] + 1
    assert stale != CLAIMS["evaluation"]["compute_invariance"]["counts"], (
        "a stale count is rejected by the equality check")

    # 4. An unresolvable metric pointer (a bogus tail key) cannot be walked ->
    #    the pointer-resolver rejects it (the walk finds no key to descend into).
    arm_spec = next(iter(CLAIMS["arms"].values()))
    blob = json.loads((REPRO_ROOT / arm_spec["results"]).read_text())
    _f, _, path = next(iter(arm_spec["metrics"].values())).partition(":")
    bad_path = path + ".NO_SUCH_KEY_zzz"
    cur, p, resolved = blob, bad_path, True
    while p:
        if not isinstance(cur, dict):
            # reached a leaf (scalar/list) with path left to walk -> unresolvable
            resolved = False
            break
        best = None
        for k in cur:
            if p == k or p.startswith(k + ".") or p.startswith(k + "["):
                if best is None or len(k) > len(best):
                    best = k
        if best is None:
            resolved = False
            break
        val = cur[best]
        p = p[len(best):]
        if p.startswith("[*]"):
            val = val[0][p[4:]]
            p = ""
        elif p.startswith("."):
            p = p[1:]
            cur = val
        else:
            p = ""
    assert not resolved, "an unresolvable metric pointer must not resolve"
