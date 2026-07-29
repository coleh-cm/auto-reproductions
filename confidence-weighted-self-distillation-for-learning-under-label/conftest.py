"""Pytest config: make the repo root importable from tests/.

The reproduction keeps everything in one file (``run_experiment.py``) at the
repo root per the SPEC §5 frozen interface. Test files live under ``tests/``;
without this conftest pytest would put ``tests/`` (not the repo root) on
``sys.path`` and ``import run_experiment`` would fail.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
