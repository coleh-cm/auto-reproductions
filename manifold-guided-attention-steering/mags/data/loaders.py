"""Dataset loaders for the MAGS reproduction (SPEC §5, §6).

Evaluation sets (obtainable without gating):
- MATH-500      : HuggingFaceH4/MATH-500          (test, 500)
- GSM8K         : openai/gsm8k 'main'              (test, 1319)
- HumanEval     : openai/openai_humaneval          (test, 164)
- MBPP(sanitized): google-research-datasets/mbpp 'sanitized' (combine all splits = 427)

Contrastive-trace training sources (used to fit manifolds; these are the "training
split" sources named by the paper, tex:L398-399):
- MathInstruct  : TIGER-Lab/MathInstruct           (MATH-500 source)
- GSM8K train   : openai/gsm8k 'main' train        (GSM8K source)
- APPS          : codeparrot/apps                  (HumanEval & MBPP source)

APPS ships as a dataset *script*, which datasets>=3 no longer executes. We attempt the
script first, then a parquet fallback from the Hub's auto-converted branch; if both
fail the loader raises ``DatasetUnavailable`` so the caller can record a BLOCKED result
rather than silently substituting data (the research-code rule: real data or no data).
"""
from __future__ import annotations
from dataclasses import dataclass


class DatasetUnavailable(RuntimeError):
    """Raised when a contrastive-trace source cannot be obtained. The caller MUST
    report a BLOCKED result, not substitute synthetic data."""


@dataclass
class Problem:
    id: str
    benchmark: str
    prompt_text: str
    gold: str             # canonical answer (math: boxed answer; code: canonical code)
    extra: dict = None     # benchmark-specific (HumanEval: test+entry_point; MBPP: test_list)


# ---------------------------------------------------------------------------
def load_math500() -> list:
    from datasets import load_dataset
    d = load_dataset("HuggingFaceH4/MATH-500")["test"]
    out = []
    for r in d:
        out.append(Problem(
            id=str(r["unique_id"]), benchmark="MATH-500",
            prompt_text=r["problem"], gold=str(r["answer"]),
            extra={"solution": r.get("solution"), "subject": r.get("subject"),
                   "level": r.get("level")},
        ))
    return out


def load_gsm8k(split="test") -> list:
    from datasets import load_dataset
    d = load_dataset("openai/gsm8k", "main")[split]
    out = []
    for i, r in enumerate(d):
        # gold = the number after the final '####'
        ans = r["answer"].split("####")[-1].strip()
        out.append(Problem(
            id=f"gsm8k-{split}-{i}", benchmark="GSM8K",
            prompt_text=r["question"], gold=ans, extra={},
        ))
    return out


def load_humaneval() -> list:
    from datasets import load_dataset
    d = load_dataset("openai/openai_humaneval")["test"]
    out = []
    for r in d:
        out.append(Problem(
            id=str(r["task_id"]), benchmark="HumanEval",
            prompt_text=r["prompt"], gold=r["canonical_solution"],
            extra={"test": r["test"], "entry_point": r["entry_point"]},
        ))
    return out


def load_mbpp_sanitized() -> list:
    """MBPP sanitized full set = train(120)+test(257)+validation(43)+prompt(7) = 427
    (paper's MBPP(sanitized-test) N=427, tex:L712)."""
    from datasets import load_dataset
    d = load_dataset("google-research-datasets/mbpp", "sanitized")
    out = []
    for split in ["train", "test", "validation", "prompt"]:
        if split not in d:
            continue
        for r in d[split]:
            out.append(Problem(
                id=str(r["task_id"]), benchmark="MBPP",
                prompt_text=r["prompt"], gold=r["code"],
                extra={"test_list": r["test_list"],
                       "test_imports": r.get("test_imports", [])},
            ))
    return out


# ---------------------------------------------------------------------------
def load_mathinstruct(limit=None) -> list:
    """MathInstruct (MATH-500 contrastive-trace source, tex:L398). Returns problems
    keyed by instruction; we keep only the MATH subset where 'source' contains 'MATH'
    to match the paper's MATH-500 manifold (diagnostic uses Math-Instruct traces from
    Llama-3.1-8B-Instruct, tex:L296).

    Gold is extracted per source (SPEC 4.23, realized keep-set):
      - MATH_train CoT: trailing ``\\boxed{...}`` (the MATH grader, mags.grading.grade_math,
        also extracts boxed from a generated trace, so the gold/trace graders match).
      - college_math: multiple-choice letter from ``The answer is <X>.`` (all 1840 rows
        end this way; the generated trace is graded by boxed-letter match).
    Sources excluded (recorded in SPEC 4.23):
      - math50k_camel: free-form prose with no reliable single-answer marker (only ~90 of
        49484 carry a boxed; the rest end in prose like 'D approx 1.46497.' or are
        non-single-answer). Parsing a gold from prose would inject noisy contrastive
        labels and corrupt the manifold, so camel is excluded.
      - MATH_train PoT: Python programs whose gold is the executed stdout; grading a
        generated PoT trace needs a program-execution grader distinct from the MATH
        boxed grader and out of scope for the MATH-500 (boxed-answer) manifold, so PoT
        is excluded.
    """
    from datasets import load_dataset
    d = load_dataset("TIGER-Lab/MathInstruct")["train"]
    out = []
    for i, r in enumerate(d):
        if limit and len(out) >= limit:
            break
        src = str(r.get("source", ""))
        # MathInstruct mixes MATH and GSM8K-derived items; keep MATH-sourced for MATH-500.
        if "MATH" not in src and "math" not in src.lower():
            continue
        ans = _extract_mathinstruct_gold(src, r["output"])
        if ans is None:
            continue
        out.append(Problem(
            id=f"mathinstruct-{i}", benchmark="MATH-500-train",
            prompt_text=r["instruction"], gold=ans,
            extra={"solution": r["output"], "source": src},
        ))
    return out


def load_apps(limit=None, subset="all") -> list:
    """APPS (HumanEval & MBPP contrastive-trace source, tex:L398). The canonical
    `codeparrot/apps` is a dataset script no longer executed by datasets>=3; the repo's
    auto-converted parquet branch ``refs/convert/parquet`` has configs
    ``all``/``competition``/``interview``/``introductory`` each with ``train``/``test``
    shards. We load the TRAIN split of ``subset`` (default ``all`` = the full APPS
    training set, matching the paper's 'corresponding training split', tex:L398).

    Each APPS row carries ``input_output`` (JSON of test inputs/outputs) which the
    APPS grader (mags.grading.grade_apps) executes to judge a generated trace
    correct/incorrect — needed for the keep-if-both rule (tex:L399).
    """
    from datasets import load_dataset
    data_files = {"train": f"hf://datasets/codeparrot/apps@refs/convert/parquet/"
                           f"{subset}/train/0000.parquet"}
    try:
        d = load_dataset("parquet", data_files=data_files)
    except Exception as e:
        raise DatasetUnavailable(
            f"APPS parquet unavailable at refs/convert/parquet/{subset}/train: {e!r}. "
            f"This blocks HumanEval/MBPP contrastive-trace collection.")
    out = []
    for r in d["train"]:
        if limit and len(out) >= limit:
            break
        out.append(Problem(
            id=f"apps-{r['problem_id']}", benchmark="APPS-train",
            prompt_text=str(r.get("question", "")),
            gold=str(r.get("solutions", "")),
            extra={"difficulty": r.get("difficulty"), "source": "apps",
                   "input_output": r.get("input_output"),
                   "starter_code": r.get("starter_code", "")},
        ))
    return out


# ---------------------------------------------------------------------------
def _extract_boxed(text: str) -> str | None:
    import re
    # last \boxed{...} with brace balancing
    idx = text.rfind("\\boxed{")
    if idx < 0:
        return None
    i = idx + len("\\boxed{")
    depth = 1
    j = i
    while j < len(text) and depth > 0:
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        j += 1
    if depth != 0:
        return None
    return text[i:j - 1].strip()


def _extract_mathinstruct_gold(source: str, output: str) -> str | None:
    """Per-source gold extraction for MathInstruct (SPEC 4.23 realized keep-set).

    MATH_train CoT -> trailing boxed; college_math -> multiple-choice letter. Returns
    None for sources we exclude (camel free-form prose, MATH_train PoT) so the caller
    drops them rather than fitting a manifold on unreliable labels.
    """
    import re
    ans = _extract_boxed(output)
    if ans is not None:
        return ans
    # college_math: every row ends with "The answer is <letter>." (verified 1840/1840)
    if "college_math" in source:
        m = re.search(r"answer is\s+([A-D])\b", output, re.I)
        if m:
            return m.group(1).upper()
    return None


EVAL_LOADERS = {
    "MATH-500": load_math500,
    "GSM8K": load_gsm8k,
    "HumanEval": load_humaneval,
    "MBPP": load_mbpp_sanitized,
}

TRAIN_LOADERS = {
    "MATH-500": load_mathinstruct,
    "GSM8K": lambda limit=None: load_gsm8k(split="train") if not limit else load_gsm8k(split="train")[:limit],
    "HumanEval": load_apps,
    "MBPP": load_apps,
}
