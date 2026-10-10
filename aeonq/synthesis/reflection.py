"""Exact phase-oracle synthesis into gates and semantic multi-controls.

A predicate χ is decomposed into disjoint *cubes* (partial assignments of
qubit values). Each cube becomes one phase flip, and because the cubes are
disjoint their phases multiply to exactly (-1)^χ(x), global phase included.
Structured predicates use closed forms: ``bit`` is one Z, ``equal`` is one
multi-controlled Z on its own qubits, and ``x >= K`` needs at most m+1 cubes.
Only ``marked`` and Boolean combinations enumerate basis states, over the
predicate's support alone, which is restricted to tractable widths.

The multi-controlled Z remains first-class until backend-aware decomposition.
Results are immutable IR, so they are cached: lowering Q^k re-requests the same
reflections k times.
"""

from functools import lru_cache
from math import pi

import numpy as np

from aeonq.ir.nodes import (
    ControlledOperator,
    Gate,
    GlobalPhase,
    ObjectiveReflection,
    Operator,
    Sequence,
    ZeroReflection,
)
from aeonq.ir.predicates import ObjectivePredicate

# Sorted (qubit, required bit) pairs; qubits outside the cube are unconstrained.
Cube = tuple[tuple[int, int], ...]


def _phase_on_cube(n_qubits: int, cube: Cube) -> Operator:
    """Multiply by -1 exactly on basis states matching every entry of ``cube``."""
    if not cube:
        return GlobalPhase(pi, n_qubits)
    flips = tuple(Gate("X", (q,), n_qubits) for q, bit in cube if bit == 0)
    qubits = tuple(q for q, _ in cube)
    if len(qubits) == 1:
        phase: Operator = Gate("Z", qubits, n_qubits)
    else:
        phase = ControlledOperator(
            Gate("Z", (0,), 1),
            qubits[:-1],
            (1,) * (len(qubits) - 1),
            (qubits[-1],),
            n_qubits,
        )
    return Sequence(flips + (phase,) + tuple(reversed(flips)), n_qubits)


def _full_cube(n_qubits: int, basis: int) -> Cube:
    return tuple((q, (basis >> q) & 1) for q in range(n_qubits))


def _intersect(left: Cube, right: Cube) -> Cube | None:
    """The cube matching both, or None when they fix a qubit differently."""
    fixed = dict(left)
    for q, bit in right:
        if fixed.setdefault(q, bit) != bit:
            return None
    return tuple(sorted(fixed.items()))


def _support(predicate: ObjectivePredicate, n_qubits: int) -> tuple[int, ...]:
    if predicate.kind == "marked":
        return tuple(range(n_qubits))
    if predicate.kind in ("and", "or", "not"):
        return tuple(
            sorted(set().union(*(_support(c, n_qubits) for c in predicate.children)))
        )
    return tuple(sorted(predicate.qubits))


def threshold_cubes(qubits: tuple[int, ...], value: int) -> list[Cube]:
    """Disjoint cubes for x >= value, where qubits[i] carries bit i of x."""
    if value == 0:
        return [()]
    m = len(qubits)
    # Trailing zero bits of the threshold never constrain x >= value.
    low = (value & -value).bit_length() - 1
    cubes: list[Cube] = []
    for i in range(m - 1, low - 1, -1):
        if not (value >> i) & 1:
            # Bits above i equal the threshold's and bit i exceeds it.
            fixed = [(qubits[j], (value >> j) & 1) for j in range(i + 1, m)]
            cubes.append(tuple(sorted(fixed + [(qubits[i], 1)])))
    cubes.append(tuple(sorted((qubits[j], (value >> j) & 1) for j in range(low, m))))
    return cubes


def _enumerated_cubes(
    predicate: ObjectivePredicate, n_qubits: int, max_qubits: int
) -> tuple[list[Cube], bool]:
    support = _support(predicate, n_qubits)
    if len(support) > max_qubits:
        raise ValueError("predicate enumeration limit exceeded")
    local = np.arange(2 ** len(support), dtype=np.int64)
    basis = np.zeros_like(local)
    for i, q in enumerate(support):
        basis |= ((local >> i) & 1) << q
    truth = predicate.evaluate_many(basis, n_qubits)
    # Marking the smaller of the true and false sets is exact: the rest is -I.
    negate = bool(2 * int(truth.sum()) > truth.size)
    selected = np.flatnonzero(~truth if negate else truth)
    cubes = [
        tuple((q, (int(r) >> i) & 1) for i, q in enumerate(support)) for r in selected
    ]
    return cubes, negate


def _predicate_cubes(
    predicate: ObjectivePredicate, n_qubits: int, max_qubits: int
) -> tuple[list[Cube], bool]:
    """Disjoint cubes and a flag with χ(x) = flag XOR [x lies in a cube]."""
    if predicate.kind == "bit":
        return [((predicate.qubits[0], 1),)], False
    if predicate.kind == "equal":
        assert predicate.value is not None
        cube = tuple(
            sorted(
                (q, (predicate.value >> i) & 1) for i, q in enumerate(predicate.qubits)
            )
        )
        return [cube], False
    if predicate.kind == "ge":
        assert predicate.value is not None
        return threshold_cubes(predicate.qubits, predicate.value), False
    if predicate.kind == "not":
        cubes, negate = _predicate_cubes(predicate.children[0], n_qubits, max_qubits)
        return cubes, not negate
    if predicate.kind == "and":
        left, left_negated = _predicate_cubes(
            predicate.children[0], n_qubits, max_qubits
        )
        right, right_negated = _predicate_cubes(
            predicate.children[1], n_qubits, max_qubits
        )
        if not left_negated and not right_negated:
            # Pairwise intersections of two disjoint cube sets stay disjoint.
            merged = (_intersect(a, b) for a in left for b in right)
            return [cube for cube in merged if cube is not None], False
    if predicate.kind == "marked" and 2 * len(set(predicate.marked)) <= 2**n_qubits:
        # One full-register cube per marked basis state, in ascending order.
        return [_full_cube(n_qubits, b) for b in sorted(set(predicate.marked))], False
    return _enumerated_cubes(predicate, n_qubits, max_qubits)


@lru_cache(maxsize=256)
def synthesize_zero_reflection(node: ZeroReflection) -> Sequence:
    """S0 flips the sign exactly where every listed state qubit is zero."""
    cube = tuple(sorted((q, 0) for q in node.state_qubits))
    phase = _phase_on_cube(node.n_qubits, cube)
    return phase if isinstance(phase, Sequence) else Sequence((phase,), node.n_qubits)


@lru_cache(maxsize=256)
def synthesize_objective_reflection(
    node: ObjectiveReflection, *, max_qubits: int = 10
) -> Sequence:
    node.predicate.validate(node.n_qubits)
    cubes, negate = _predicate_cubes(node.predicate, node.n_qubits, max_qubits)
    children = [_phase_on_cube(node.n_qubits, cube) for cube in cubes]
    if negate:
        children.append(GlobalPhase(pi, node.n_qubits))
    return Sequence(tuple(children), node.n_qubits)
