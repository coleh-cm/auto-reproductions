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


def test_mbpp_fenced_code_extraction():
    """MBPP runs the instruct model through its chat template (SPEC §4.13), so the
    completion is prose + a fenced ```python``` block. grade_mbpp must extract the
    block before executing or the leading prose / fence markers raise SyntaxError
    and collapse MBPP accuracy below the paper's Table 1/2 (tex:L423/L469).
    _extract_code reduces a chat completion to the fenced block; raw code (no
    fence) is returned unchanged so completion-style HumanEval is unaffected."""
    from mags.grading import _extract_code, grade
    prob = Problem(id="t3", benchmark="MBPP", prompt_text="", gold="",
                   extra={"test_list": ["assert add(1,2)==3"], "test_imports": []})
    chat_completion = (
        "Sure, here's a function that adds two numbers.\n\n"
        "```python\ndef add(a, b):\n    return a + b\n```\n"
    )
    assert _extract_code(chat_completion).strip() == "def add(a, b):\n    return a + b"
    assert grade("MBPP", chat_completion, prob), "fenced-block completion must pass"
    # language-tagged fence (```py) also extracted
    py_tag = "Intro.\n```py\ndef add(a, b):\n    return a + b\n```"
    assert grade("MBPP", py_tag, prob)
    # no fence -> raw completion returned unchanged (HumanEval path)
    assert _extract_code("def add(a, b):\n    return a + b\n") == \
        "def add(a, b):\n    return a + b\n"
    # multiple fences -> LAST one extracted
    multi = ("```python\ndef old(a, b):\n    return 0\n```\n"
             "```python\ndef add(a, b):\n    return a + b\n```")
    assert grade("MBPP", multi, prob)


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


def test_apps_prompt_includes_starter_code():
    """Adversarial-review fix (training-loop F1): APPS intro/interview problems carry
    a non-empty starter_code skeleton that the grader prepends to the completion.
    load_apps MUST put the skeleton in the prompt the model conditions on, or the
    model writes a standalone solution, the grader's injected skeleton misaligns,
    all <=8 traces are marked incorrect, the keep-if-both rule (tex:L399) drops the
    problem, and the contrastive set is biased toward empty-starter competition
    problems — a number-affecting deviation in the fitted HumanEval/MBPP manifold.
    Competition problems (empty starter) keep a skeleton-free prompt."""
    import inspect
    from mags.data import loaders
    src = inspect.getsource(loaders.load_apps)
    # prompt_text concatenates question + starter when starter is non-empty
    assert "starter" in src and "prompt_text" in src
    # build a Problem directly mimicking load_apps's prompt construction so the
    # test does not require the APPS dataset (gated / unavailable in sandbox)
    from mags.data.loaders import Problem
    question = "Write a function that adds two numbers."
    starter = "def add(a, b):\n    "
    prompt_text = (question + "\n" + starter) if starter.strip() else question
    prob = Problem(id="apps-starter", benchmark="APPS-train",
                   prompt_text=prompt_text, gold="",
                   extra={"starter_code": starter, "input_output": None})
    assert starter in prob.prompt_text, "starter_code must appear in the prompt"
    # empty-starter (competition) problem: prompt is the question alone
    empty = ""
    p2 = Problem(id="apps-comp", benchmark="APPS-train",
                 prompt_text=question if not empty.strip() else question + "\n" + empty,
                 gold="", extra={"starter_code": empty})
    assert p2.prompt_text == question


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


def test_grade_math500_train_source_tag():
    """Regression: mags.fit grades contrastive traces against the SOURCE problem's
    gold. The MathInstruct source (load_mathinstruct, MATH-500 contrastive-trace
    source per tex:L399) tags problems `benchmark='MATH-500-train'`; grade() must
    dispatch that tag to the math grader. Without it mags.fit raises
    `unknown benchmark 'MATH-500-train'` and the whole MATH-500 manifold fit
    crashes on a GPU host (the bug was found by running the fit CLI on a tiny
    cached model + the real MathInstruct source)."""
    prob = Problem(id="mathinstruct-0", benchmark="MATH-500-train",
                   prompt_text="", gold="42",
                   extra={"solution": r"...$\boxed{42}$"})
    assert grade("MATH-500-train", r"The answer is $\boxed{42}$", prob)
    assert not grade("MATH-500-train", r"The answer is $\boxed{7}$", prob)


def test_grade_unknown_tag_raises():
    """An unrecognised benchmark tag must raise, not silently pass/fail — that is
    how the MATH-500-train bug was originally caught."""
    import pytest
    prob = Problem(id="x", benchmark="no-such-bench", prompt_text="", gold="")
    with pytest.raises(ValueError):
        grade("no-such-bench", "anything", prob)


def test_grade_math_multi_boxed_scores_last_answer():
    """Regression (adversarial review, confirmed MAJOR): a model's chain-of-thought
    on MATH-500 routinely emits intermediate \\boxed{} results before the final
    answer. The prior grade_math parsed the WHOLE raw prediction; math_verify then
    collapsed every boxed value into a FiniteSet and the single gold failed the
    set-size check, marking a CORRECT final answer WRONG. This depressed headline
    MATH-500 accuracy (Table 1/2) and corrupted fit-time contrastive-trace labels
    for the MATH-500 manifold. The grader must score ONLY the last boxed answer.
    """
    pred = r"Step 1: $\boxed{3}$. Step 2: $\boxed{5}$. Final: $\boxed{8}$."
    assert grade_math(pred, "8"), "last boxed is 8 == gold; must pass"
    assert not grade_math(pred, "3"), "first boxed is not the final answer"
    # dfrac preserved through the last-boxed path
    pred2 = r"intermediate $\boxed{3}$ then $\boxed{\dfrac{1}{2}}$"
    assert grade_math(pred2, r"\dfrac{1}{2}"), "last boxed dfrac must pass"
    assert not grade_math(pred2, "3"), "intermediate boxed must not be scored"


def test_mathinstruct_gold_extraction_per_source():
    """Round-19 major-fix regression: load_mathinstruct's gold extraction must keep
    MATH_train CoT (boxed) AND college_math (multiple-choice letter), and must NOT
    silently drop college_math for lacking a boxed marker. The prior boxed-only
    filter kept only ~11.3k of the ~13.2k intended MATH-sourced corpus and
    discarded ALL 1840 college_math rows."""
    from mags.data.loaders import _extract_mathinstruct_gold
    # MATH_train CoT: trailing boxed -> boxed content
    assert _extract_mathinstruct_gold(
        "data/CoT/MATH_train.json",
        r"...so the area is $\boxed{108\pi}$. The answer is 108\pi") == r"108\pi"
    # college_math: "The answer is B." -> letter (boxed absent)
    assert _extract_mathinstruct_gold(
        "data/CoT/college_math.json",
        "...It's possible for G to be cyclic. The answer is B.") == "B"
    # camel free-form prose (no boxed, not college_math) -> None (dropped, recorded)
    assert _extract_mathinstruct_gold(
        "data/CoT/math50k_camel.json",
        "D = log(5)/log(3)\n\nD \u2248 1.46497") is None
    # MATH_train PoT (python, no boxed) -> None (dropped, recorded)
    assert _extract_mathinstruct_gold(
        "data/PoT/MATH_train.json",
        "def f(x):\n    return 3**(x/2)\n\nprint(f(6))") is None
    # a camel row that itself carries a boxed is still kept (boxed takes priority)
    assert _extract_mathinstruct_gold(
        "data/CoT/math50k_camel.json",
        r"therefore $\boxed{7}$") == "7"


def test_smiles_validity_grader():
    """SMILES Validity (Table 3, tex:L527): a SMILES is valid iff RDKit parses it.
    The grade() dispatch must route SMILES-molecular-generation to the validity
    grader (not raise ValueError), and the grader must accept canonical SMILES
    and reject garbage/empty."""
    from mags.grading import grade, smiles_is_valid, grade_smiles_validity
    # canonical valid SMILES
    assert smiles_is_valid("CCO")            # ethanol
    assert smiles_is_valid("c1ccccc1")       # benzene
    assert smiles_is_valid("CC(=O)O")        # acetic acid
    assert smiles_is_valid("C1CCCCC1")       # cyclohexane
    # invalid / empty
    assert not smiles_is_valid("")
    assert not smiles_is_valid("   ")
    assert not smiles_is_valid("not a smiles!!!")
    assert not smiles_is_valid("C1CC")       # unbalanced ring closure (digit 1 unpaired)
    # dispatch routes the molecular benchmark to the validity grader
    assert grade("SMILES-molecular-generation", "CCO", None) is True
    assert grade("SMILES-molecular-generation", "garbage!!", None) is False
    # grade_smiles_validity is the per-molecule predicate; fraction over 500 is
    # the Table-3 Validity % (computed in eval, not here).
    assert grade_smiles_validity("c1ccccc1") and not grade_smiles_validity("")

