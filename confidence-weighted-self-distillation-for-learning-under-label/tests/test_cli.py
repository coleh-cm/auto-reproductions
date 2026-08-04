"""CLI / output-contract tests (SPEC §5: exactly one line `FINAL accuracy=<float>`)."""

import subprocess
import sys

import run_experiment as r


def _run(args):
    cmd = [sys.executable, "run_experiment.py"] + args
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    return out.stdout, out.stderr


def test_cli_baseline_output_format():
    out, err = _run(["--lambda", "0.0", "--steps", "50"])
    lines = out.strip().splitlines()
    assert len(lines) == 1, f"expected one stdout line, got {lines!r}"
    line = lines[0]
    assert line.startswith("FINAL accuracy=")
    val = float(line.split("=", 1)[1])
    assert 0.0 <= val <= 1.0


def test_cli_cwsd_output_format():
    out, _ = _run(["--lambda", "1.0", "--steps", "50"])
    lines = out.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("FINAL accuracy=")
    float(lines[0].split("=", 1)[1])  # parses


def test_cli_rejects_lambda_out_of_range():
    cmd = [sys.executable, "run_experiment.py", "--lambda", "1.5", "--steps", "1"]
    out = subprocess.run(cmd, capture_output=True, text=True)
    assert out.returncode != 0
    # SPEC §5: all diagnostics go to stderr, nothing to stdout
    assert out.stdout.strip() == "", out.stdout
    assert out.returncode == 2


def test_cli_accepts_all_documented_flags():
    """Smoke test that every SPEC §5 flag parses (does not assert accuracy)."""
    out, _ = _run([
        "--lambda", "1.0", "--steps", "5",
        "--grad-mode", "literal",
        "--s", "0.15", "--tau", "0.9", "--temperature", "2.0",
        "--seed", "0", "--lr", "0.1", "--batch-size", "64",
        "--init", "he", "--noise-mode", "uniform-all",
        "--batch-mode", "epoch-permutation", "--rng-layout", "init-first",
    ])
    assert out.strip().startswith("FINAL accuracy=")
