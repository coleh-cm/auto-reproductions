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
import subprocess
import sys
import tempfile


# ---------------------------------------------------------------------------
# Math grading
# ---------------------------------------------------------------------------
def grade_math(prediction: str, gold: str) -> bool:
    """MATH-500 / GSM8K: math_verify boxed/numeric extraction + equivalence.

    Standard MATH grading convention: score the LAST boxed answer in the
    prediction (a model's CoT often emits intermediate ``\\boxed{}`` results
    before the final answer). ``_extract_answer`` isolates that last boxed
    content; we wrap it back in ``\\boxed{}`` (so math_verify recognises
    constructs like ``\\dfrac``) and parse it. If the prediction has no boxed
    marker, fall back to parsing the raw prediction. Falls back to normalised
    numeric string-equality if math_verify cannot parse either side.

    NOTE: parsing the WHOLE raw prediction when multiple ``\\boxed{}`` markers
    are present is a bug -- math_verify collapses every boxed value into a set
    and the single gold then fails the set-size check, marking a correct final
    answer wrong (depressing headline accuracy and corrupting fit-time trace
    labels for the MATH-500 manifold). Always parse only the last boxed answer.
    """
    pred_ans = _extract_answer(prediction)
    try:
        from math_verify import parse, verify
        g = parse(f"\\boxed{{{gold}}}")
        if pred_ans is not None:
            p = parse(f"\\boxed{{{pred_ans}}}")
        else:
            p = parse(prediction)
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
    test = problem.extra["test"]
    entry = problem.extra["entry_point"]
    # human_eval 1.0.3: check_correctness(problem: Dict, completion: str, timeout: float,
    #   completion_id=None) -> Dict. The problem dict MUST carry `task_id` (read for the
    #   return value) plus the prompt/test/entry_point the harness executes. Passing
    #   positional args out of order (as the prior code did) raises
    #   TypeError: multiple values for argument 'timeout', making every HumanEval grade
    #   crash (HumanEval is one of four headline benchmarks, tex:L710-713).
    problem_dict = {"task_id": problem.id, "prompt": problem.prompt_text,
                    "test": test, "entry_point": entry}
    res = check_correctness(problem_dict, completion, timeout=10.0)
    return res["passed"] if isinstance(res, dict) else (res == "passed")


def grade_mbpp(completion: str, problem) -> bool:
    """MBPP sanitized: run ``test_list`` asserts against the generated code (10 s)."""
    test_list = problem.extra.get("test_list", [])
    imports = problem.extra.get("test_imports", [])
    code = completion
    test_code = "\n".join(imports) + "\n" + code + "\n" + "\n".join(test_list) + "\nprint('OK')"
    return _run_subprocess_ok(test_code, timeout=10)


def grade_apps(completion: str, problem) -> bool:
    """APPS trace grading (SPEC §5, tex:L399 keep-if-both): execute the generated
    solution against the problem's ``input_output`` test cases (stdin/stdout match).

    APPS rows store ``input_output`` as a JSON ``{"inputs":[...], "outputs":[...]}``.
    We feed each input to the generated program on stdin and require an exact stdout
    match (after rstrip). A trace is correct iff ALL provided cases pass within a
    combined 20 s budget. This is the grader used during contrastive-trace collection
    for HumanEval/MBPP manifolds (the contrastive source is APPS, tex:L398)."""
    import json as _json
    io = problem.extra.get("input_output")
    if not io:
        return False
    try:
        cases = _json.loads(io)
    except Exception:
        return False
    inputs = cases.get("inputs", [])
    outputs = cases.get("outputs", [])
    if not inputs:
        return False
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        starter = problem.extra.get("starter_code") or ""
        f.write(starter + "\n" + completion)
        path = f.name
    try:
        for inp, exp in zip(inputs, outputs):
            # sys.executable, and OSError raises: see _run_subprocess_ok. This grader
            # labels the contrastive traces, so a swallowed launch failure here does not
            # merely score zero -- it empties the correct class and no manifold is fit.
            try:
                proc = subprocess.run(
                    [sys.executable, path], input=inp, capture_output=True, text=True,
                    timeout=20 / max(len(inputs), 1), env={**os.environ},
                )
            except subprocess.TimeoutExpired:
                return False
            except OSError as exc:
                raise RuntimeError(
                    f"grader could not execute {sys.executable!r}: {exc!r}") from exc
            if proc.returncode != 0:
                return False
            if (proc.stdout or "").rstrip() != str(exp).rstrip():
                return False
        return True
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


def _run_subprocess_ok(code: str, timeout: int = 10) -> bool:
    """Run ``code`` in an isolated subprocess; success iff it prints OK within timeout.

    Uses ``sys.executable``, not ``"python"``: an environment with only ``python3`` on
    PATH -- the default on Debian, and on macOS without a shim -- raises
    FileNotFoundError, and returning False for that reports every generated solution as
    wrong. That silence propagates: APPS grading labels the contrastive traces, so
    all-incorrect labels leave no problem with both classes, every head's fit is
    degenerate, and the fit step reports 0 heads selected while exiting 0.

    A timeout IS a verdict (a solution that hangs has failed). A missing or broken
    interpreter is not: it raises, because a grader that cannot run has no opinion
    about correctness.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        path = f.name
    try:
        proc = subprocess.run(
            [sys.executable, path], capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        )
    except subprocess.TimeoutExpired:
        return False
    except OSError as exc:
        raise RuntimeError(f"grader could not execute {sys.executable!r}: {exc!r}") from exc
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return proc.returncode == 0 and "OK" in (proc.stdout or "")


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
    if benchmark == "APPS-train":
        return grade_apps(completion, problem)
    # MATH-500-train: the MathInstruct contrastive-trace source (SPEC §4.13,
    # load_mathinstruct) tags its problems `MATH-500-train`; its gold is a
    # \\boxed{} answer extracted from the MathInstruct `output`, identical in
    # format to MATH-500, so it grades with the same math grader. This case is
    # reached by mags.fit when labelling contrastive traces for the MATH-500
    # manifold (fit grades against the SOURCE problem's gold, tex:L399); without
    # it the fit CLI raises `unknown benchmark 'MATH-500-train'` and the whole
    # MATH-500 manifold fit crashes on a GPU host.
    if benchmark == "MATH-500-train":
        return grade_math(completion, problem.gold)
    raise ValueError(f"unknown benchmark {benchmark!r}")
