"""Answer grading for the four reasoning/code benchmarks (SPEC §4.13).

- MATH-500 / GSM8K: math_verify boxed/numeric extraction + equivalence.
- HumanEval: official execution harness (human_eval.execution.check_correctness).
- MBPP (sanitized): execute the generated code against the problem's test_list asserts
  in a sandboxed subprocess with a 10 s timeout (SPEC §5.6).
"""
from __future__ import annotations
import multiprocessing
import os
import signal
import tempfile


# ---------------------------------------------------------------------------
# Math grading
# ---------------------------------------------------------------------------
def grade_math(prediction: str, gold: str) -> bool:
    """MATH-500 / GSM8K: math_verify boxed/numeric extraction + equivalence.

    math_verify's ``parse`` needs the LaTeX wrapper (``\\boxed{...}``) to recognise
    constructs like ``\\dfrac``; we therefore parse the *original* prediction text
    (which contains the boxed answer) rather than the bare extracted content, and
    parse the gold wrapped in ``\\boxed{}``. Falls back to normalised numeric
    string-equality if math_verify cannot parse either side.
    """
    pred_ans = _extract_answer(prediction)
    try:
        from math_verify import parse, verify
        g = parse(f"\\boxed{{{gold}}}")
        # prefer the raw prediction (keeps \boxed wrapper for \dfrac etc.)
        p = parse(prediction) if "\\boxed" in prediction else parse(
            f"\\boxed{{{pred_ans}}}" if pred_ans is not None else prediction)
        if g is None or p is None or not g or not p:
            return _norm(pred_ans) == _norm(gold)
        return bool(verify(g, p))
    except Exception:
        return _norm(pred_ans) == _norm(gold)


def _extract_answer(text: str):
    import re
    m = list(re.finditer(r"\\boxed\{", text))
    if m:
        idx = m[-1].end()
        depth = 1
        j = idx
        while j < len(text) and depth > 0:
            if text[j] == "{":
                depth += 1
            elif text[j] == "}":
                depth -= 1
            j += 1
        return text[idx:j - 1].strip() if depth == 0 else None
    # GSM8K #### style
    if "####" in text:
        return text.split("####")[-1].strip()
    # last number
    nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", text)
    return nums[-1] if nums else None


def _norm(s):
    import re
    if s is None:
        return ""
    s = str(s).strip()
    s = s.replace(",", "").replace("\\%", "").replace("%", "").replace("\\$", "").replace("$", "")
    s = s.rstrip(".")
    try:
        return str(float(s))
    except Exception:
        return s


# ---------------------------------------------------------------------------
# Code grading
# ---------------------------------------------------------------------------
def grade_humaneval(completion: str, problem) -> bool:
    """HumanEval pass@1: append the completion to the prompt and run the official test."""
    try:
        from human_eval.execution import check_correctness
    except Exception as e:
        raise RuntimeError(f"human_eval import failed: {e!r}")
    full = problem.prompt_text + completion
    test = problem.extra["test"]
    entry = problem.extra["entry_point"]
    res = check_correctness(problem.id, {"prompt": problem.prompt_text, "test": test,
                                          "entry_point": entry}, completion,
                            timeout=10.0)
    return res["passed"] if isinstance(res, dict) else (res == "passed")


def grade_mbpp(completion: str, problem) -> bool:
    """MBPP sanitized: run ``test_list`` asserts against the generated code (10 s)."""
    test_list = problem.extra.get("test_list", [])
    imports = problem.extra.get("test_imports", [])
    code = completion
    test_code = "\n".join(imports) + "\n" + code + "\n" + "\n".join(test_list) + "\nprint('OK')"
    return _run_subprocess_ok(test_code, timeout=10)


def _run_subprocess_ok(code: str, timeout: int = 10) -> bool:
    """Run ``code`` in an isolated subprocess; success iff it prints OK within timeout."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        import subprocess
        proc = subprocess.run(
            ["python", path], capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
        return proc.returncode == 0 and "OK" in (proc.stdout or "")
    except Exception:
        return False
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
def grade(benchmark: str, completion: str, problem) -> bool:
    if benchmark == "MATH-500":
        return grade_math(completion, problem.gold)
    if benchmark == "GSM8K":
        return grade_math(completion, problem.gold)
    if benchmark == "HumanEval":
        return grade_humaneval(completion, problem)
    if benchmark == "MBPP":
        return grade_mbpp(completion, problem)
    raise ValueError(f"unknown benchmark {benchmark!r}")
