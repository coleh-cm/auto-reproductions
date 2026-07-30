"""pytest path bootstrap.

Make the repo root importable so ``gdr`` resolves when pytest is invoked as
``.venv/bin/pytest -q`` (without ``python -m``).  Without this the test modules
fail collection with ``ModuleNotFoundError: No module named 'gdr'`` because the
console-script entry point does not add the cwd to ``sys.path``.
"""
import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
