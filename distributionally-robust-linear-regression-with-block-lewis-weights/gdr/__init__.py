"""gdr — reproduction package for
"Distributionally Robust Linear Regression With Block Lewis Weights"
(Manoj & Patel, arXiv:2607.00252, ICLR 2026).

This is a minimal package marker; units are imported explicitly from their
submodules (``gdr.data``, ``gdr.objectives``, ``gdr.lewis``, ...).  Importing
``gdr`` alone must NOT download any data and must NOT pull heavy optional
dependencies (folktables/pandas are imported lazily inside ``gdr.data``).
"""
