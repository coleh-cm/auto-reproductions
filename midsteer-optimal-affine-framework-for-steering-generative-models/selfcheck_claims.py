"""Self-check evaluator — the TOP-LEVEL agent's own redundant check of its claims.
Writes selfcheck.json (a DIFFERENT filename from claims_result.json so the gate can
tell them apart). Same logic as evaluate_claims.py with generated_by='selfcheck'."""
from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_claims

REPO = os.path.dirname(os.path.abspath(__file__))


def main():
    result = evaluate_claims.evaluate(os.path.join(REPO, 'claims.json'),
                                      os.path.join(REPO, 'measured.json'),
                                      os.path.join(REPO, 'results', 'e1_synth.json'))
    result['generated_by'] = 'selfcheck'
    out = os.path.join(REPO, 'selfcheck.json')
    with open(out, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"selfcheck.json: pass={result['summary']['pass']} "
          f"fail={result['summary']['fail']} blocked={result['summary']['blocked']}")


if __name__ == '__main__':
    main()
