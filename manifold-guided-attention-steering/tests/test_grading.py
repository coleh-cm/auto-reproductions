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
