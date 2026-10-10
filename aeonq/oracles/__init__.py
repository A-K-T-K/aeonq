"""Test oracles that share no code with the semantic core.

Everything in this package is written from the problem statements alone: it
imports NumPy, the standard library and the SDKs, and never aeonq's IR,
synthesis, reference simulator or adapters (``tests/unit/test_oracle_independence.py``
enforces this). Comparisons against the semantic core live in
``aeonq.conformance.independent``.

* ``cases``: analytic amplitudes, good-state predicates and gate angles.
* ``numpy_sim``: a statevector simulator built on basis-index arithmetic.
* ``handwritten``: A, Q and controlled Q written natively in each SDK.
"""
