"""Evaluation metrics for the MidSteer reproduction (SPEC section 4.4).

Each scorer is an INSTRUMENT: it decides whether an output is correct/scored.
Every scorer:
  - imports cleanly CPU-side (backbones are lazy-loaded inside the call);
  - raises BlockedException (from midsteer_core.data) when its backbone is unavailable
    in this CPU sandbox — it NEVER fabricates a plausible-looking number;
  - exposes pure scoring-logic helpers (`_...`) that are unit-tested on known inputs.
"""
