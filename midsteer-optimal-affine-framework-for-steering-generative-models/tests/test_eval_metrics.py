"""Instrument tests for the eval scorers: mock scoring logic (positive/negative) on
known inputs, plus a BlockedException assertion on the real backbone path here.

Every scorer's positive/negative test exercises the SCORING LOGIC (the part that decides
a number), not the blocked backbone. Each subprocess uses sys.executable, never bare python.
"""
import subprocess
import sys
import torch

from midsteer_core.data import BlockedException
from midsteer_core.eval import judge_cs, clip_cs, fid, detoxify_score, armorm_score, bertscore_mmlu


def test_subprocess_uses_sys_executable_not_bare_python():
    """A grader that shells out to bare `python` on a python3-only host scores everything
    wrong. This test asserts our subprocess convention works on THIS host."""
    r = subprocess.run([sys.executable, '-c', 'print(1+1)'], capture_output=True, text=True)
    assert r.returncode == 0
    assert r.stdout.strip() == '2'


def test_judge_cs_argmax_positive():
    # logits peak at token id for '7' (index 7 in the 0..10 list) => score 7.0
    token_ids = list(range(11))
    logits = torch.full((20,), -10.0)
    logits[7] = 10.0
    assert judge_cs._argmax_score_from_logits(logits, token_ids) == 7.0


def test_judge_cs_argmax_negative():
    token_ids = list(range(11))
    logits = torch.full((20,), -10.0)
    logits[1] = 10.0
    assert judge_cs._argmax_score_from_logits(logits, token_ids) == 1.0


def test_judge_cs_blocked_here():
    try:
        judge_cs.judge_cs([judge_cs.POSITIVE_TEXT], judge_cs.POSITIVE_CONCEPT)
    except BlockedException:
        return
    raise AssertionError("judge_cs did not raise BlockedException in this sandbox")


def test_clip_cs_scale100_positive():
    assert abs(clip_cs._scale100(1.0) - 100.0) < 1e-9
    assert abs(clip_cs._scale100(0.5) - 50.0) < 1e-9


def test_clip_cs_blocked_here():
    import pytest
    with pytest.raises(BlockedException):
        clip_cs.clip_cs([None], 'horse')


def test_fid_identical_is_zero_positive():
    mu = torch.zeros(4, dtype=torch.float64)
    sig = torch.eye(4, dtype=torch.float64) * 2.0
    assert abs(fid._fid_from_stats(mu, sig, mu, sig)) < 1e-6


def test_fid_disjoint_is_large_negative():
    mu_a = torch.zeros(4, dtype=torch.float64)
    mu_b = torch.tensor([10.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    sig = torch.eye(4, dtype=torch.float64)
    val = fid._fid_from_stats(mu_a, sig, mu_b, sig)
    assert val > 50.0  # far apart


def test_fid_blocked_here():
    import pytest
    with pytest.raises(BlockedException):
        fid.fid([None], [None])


def test_detoxify_mean_positive():
    assert abs(detoxify_score._mean_toxicity([0.9, 0.8]) - 0.85) < 1e-9


def test_detoxify_mean_raises_on_empty():
    import pytest
    with pytest.raises(ValueError):
        detoxify_score._mean_toxicity([])


def test_detoxify_blocked_here():
    import pytest
    with pytest.raises(BlockedException):
        detoxify_score.detoxify_toxicity(['x'])


def test_armorm_mean_positive():
    assert abs(armorm_score._mean_helpfulness([0.4, 0.6]) - 0.5) < 1e-9


def test_armorm_blocked_here():
    import pytest
    with pytest.raises(BlockedException):
        armorm_score.armorm_helpfulness(['q'], ['a'])


def test_bertscore_aggregate_positive():
    assert abs(bertscore_mmlu._bertscore_aggregate([1.0, 1.0]) - 1.0) < 1e-9


def test_bertscore_aggregate_raises_on_empty():
    import pytest
    with pytest.raises(ValueError):
        bertscore_mmlu._bertscore_aggregate([])


def test_bertscore_blocked_here():
    import pytest
    with pytest.raises(BlockedException):
        bertscore_mmlu.bertscore_p(['a'], ['b'])
