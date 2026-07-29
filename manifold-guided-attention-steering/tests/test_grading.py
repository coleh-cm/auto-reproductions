"""Grading tests: math boxed/numeric equivalence, GSM8K #### extraction, MBPP
subprocess execution, HumanEval harness. These verify the answer-comparison logic
without needing any model."""
from mags.grading import grade, grade_math, _extract_answer, _norm, _run_subprocess_ok
from mags.data.loaders import Problem


def test_extract_boxed_last():
    s = r"Some work and $\boxed{42}$ then more $\boxed{\dfrac{1}{2}}$."
    assert _extract_answer(s).strip() == r"\dfrac{1}{2}"


def test_extract_gsm8k_hash():
    s = "Some reasoning.\n#### 42"
    assert _extract_answer(s) == "42"


def test_math_equivalent_numbers():
    assert grade_math(r"The answer is $\boxed{42}$", "42")
    assert grade_math(r"$\boxed{\dfrac{1}{2}}$", "1/2")
    assert grade_math(r"$\boxed{0.5}$", "1/2")


def test_math_wrong_answer():
    assert not grade_math(r"$\boxed{7}$", "42")


def test_gsm8k_numeric():
    assert grade_math("The total is 100 apples.\n#### 100", "100")
    assert not grade_math("#### 99", "100")


def test_mbpp_pass():
    prob = Problem(id="t1", benchmark="MBPP", prompt_text="",
                   gold="",
                   extra={"test_list": ["assert add(1,2)==3", "assert add(0,0)==0"],
                          "test_imports": []})
    completion = "def add(a,b):\n    return a+b\n"
    assert grade("MBPP", completion, prob)


def test_mbpp_fail():
    prob = Problem(id="t2", benchmark="MBPP", prompt_text="", gold="",
                   extra={"test_list": ["assert add(1,2)==4"], "test_imports": []})
    completion = "def add(a,b):\n    return a+b\n"
    assert not grade("MBPP", completion, prob)


def test_subprocess_timeout_safety():
    # infinite loop must be caught and return False, not hang
    assert not _run_subprocess_ok("while True:\n    pass\nprint('OK')", timeout=3)


def test_apps_grader_correct_and_wrong():
    """APPS trace grading (SPEC §5, tex:L399): execute generated code against
    input_output test cases. A correct solution passes; a wrong one fails."""
    import json as _json
    from mags.data.loaders import Problem
    io = _json.dumps({"inputs": ["2 3\n", "4 5\n"], "outputs": ["5\n", "9\n"]})
    prob = Problem(id="apps-1", benchmark="APPS-train", prompt_text="", gold="",
                   extra={"input_output": io, "starter_code": ""})
    correct = "a,b=map(int,input().split());print(a+b)\n"
    from mags.grading import grade_apps
    assert grade_apps(correct, prob)
    wrong = "print(0)\n"
    assert not grade_apps(wrong, prob)
    assert grade_apps("", prob) is False  # no code -> fails


def test_humaneval_pass_and_fail():
    """HumanEval (N=164, tex:L710-713) via human_eval.execution.check_correctness.
    Regression guard: the prior call passed arguments out of order (problem.id as
    the problem dict, the dict as completion, completion as timeout, timeout=10.0
    again) which raised TypeError: multiple values for argument 'timeout' and made
    every HumanEval grade crash. The problem dict MUST carry task_id (read for the
    return value) plus prompt/test/entry_point; the harness appends completion to
    the prompt and calls check(entry_point), so the test field must define check()."""
    from mags.data.loaders import Problem
    test = ("from typing import List\n\n"
            "def check(candidate):\n    assert candidate(1, 2) == 3\n"
            "    assert candidate(0, 0) == 0\n    assert candidate(10, -5) == 5\n")
    prob = Problem(id="HumanEval/0", benchmark="HumanEval",
                   prompt_text="def add(a, b):\n    ", gold="",
                   extra={"test": test, "entry_point": "add"})
    assert grade("HumanEval", "    return a + b\n", prob), \
        "correct completion must pass"
    assert not grade("HumanEval", "    return a - b\n", prob), \
        "wrong completion must fail"
    # dispatch + the raw grader agree
    from mags.grading import grade_humaneval
    assert grade_humaneval("    return a + b\n", prob)
    assert not grade_humaneval("    return a * b\n", prob)
