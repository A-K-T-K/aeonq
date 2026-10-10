"""Exact semantic Grover expansion; no cross-iteration cancellation."""

from math import pi

from aeonq.ir.nodes import Adjoint, GlobalPhase, GroverOperator, Sequence


def synthesize_grover(node: GroverOperator) -> Sequence:
    """Execute Sχ, A†, S0, A, -I to realize Q = -A S0 A† Sχ."""
    n = node.n_qubits
    return Sequence(
        (
            node.objective_reflection,
            Adjoint(node.preparation, n),
            node.zero_reflection,
            node.preparation,
            GlobalPhase(pi, n),
        ),
        n,
    )
