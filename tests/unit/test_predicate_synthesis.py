"""Structured objective-reflection synthesis and objective-register problems."""

from itertools import combinations
from math import asin, sin, sqrt
import random

import numpy as np
import pytest

from aeonq import AEProblem, estimate
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.benchmarks.problems import lognormal_tail, strike_for_index
from aeonq.ir import (
    ControlledOperator,
    Gate,
    ObjectivePredicate,
    ObjectiveReflection,
    Sequence,
    StatePreparation,
    ZeroReflection,
)
from aeonq.reference import ideal_success_probability, matrix, reference_amplitude
from aeonq.synthesis import (
    synthesize_objective_reflection,
    synthesize_zero_reflection,
    threshold_cubes,
)


def _random_predicate(rng: random.Random, n: int, depth: int = 0) -> ObjectivePredicate:
    kinds = ["bit", "equal", "ge", "marked"] + (
        ["and", "or", "not"] if depth < 2 else []
    )
    kind = rng.choice(kinds)
    if kind == "bit":
        return ObjectivePredicate("bit", qubits=(rng.randrange(n),))
    if kind in ("equal", "ge"):
        qubits = tuple(rng.sample(range(n), rng.randint(1, n)))
        return ObjectivePredicate(
            kind, qubits=qubits, value=rng.randrange(2 ** len(qubits))
        )
    if kind == "marked":
        count = rng.randint(0, 2**n)
        return ObjectivePredicate(
            "marked", marked=tuple(rng.sample(range(2**n), count))
        )
    if kind == "not":
        return ObjectivePredicate(
            "not", children=(_random_predicate(rng, n, depth + 1),)
        )
    children = (
        _random_predicate(rng, n, depth + 1),
        _random_predicate(rng, n, depth + 1),
    )
    return ObjectivePredicate(kind, children=children)


def test_random_predicate_reflections_are_exact_including_global_phase():
    rng = random.Random(20260929)
    for _ in range(400):
        n = rng.randint(1, 5)
        node = ObjectiveReflection(n, _random_predicate(rng, n))
        # Exact equality: a global-phase error would be observable once controlled.
        assert np.allclose(
            matrix(synthesize_objective_reflection(node)), matrix(node), atol=1e-12
        )


@pytest.mark.parametrize("n", range(1, 9))
def test_threshold_cubes_are_disjoint_exact_and_linear(n):
    qubits = tuple(range(n))
    for value in range(2**n):
        cubes = threshold_cubes(qubits, value)
        assert len(cubes) <= n + 1
        hits = np.zeros(2**n, dtype=int)
        for cube in cubes:
            for x in range(2**n):
                hits[x] += all(((x >> q) & 1) == bit for q, bit in cube)
        assert hits.tolist() == [int(x >= value) for x in range(2**n)]


def test_structured_predicates_do_not_enumerate_the_register():
    width = 12  # beyond the 10-qubit enumeration limit
    bit = synthesize_objective_reflection(
        ObjectiveReflection(width, ObjectivePredicate("bit", qubits=(7,)))
    )
    assert bit.children[0].children == (Gate("Z", (7,), width),)
    threshold = ObjectivePredicate("ge", qubits=tuple(range(width)), value=1500)
    cubes = synthesize_objective_reflection(ObjectiveReflection(width, threshold))
    assert len(cubes.children) <= width + 1
    # A Boolean combination involving a full-register mark must still enumerate.
    unstructured = ObjectivePredicate(
        "or",
        children=(
            ObjectivePredicate("marked", marked=(1,)),
            ObjectivePredicate("bit", qubits=(0,)),
        ),
    )
    with pytest.raises(ValueError, match="enumeration limit"):
        synthesize_objective_reflection(ObjectiveReflection(width, unstructured))


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_zero_reflection_on_every_subregister(n):
    for size in range(1, n + 1):
        for qubits in combinations(range(n), size):
            node = ZeroReflection(n, qubits)
            assert np.allclose(matrix(synthesize_zero_reflection(node)), matrix(node))


def _flag_problem(p0: float, p1: float) -> AEProblem:
    """State qubits 0,1; the preparation computes flag = x0 AND x1 into qubit 2."""
    operations = (
        Gate("RY", (0,), 3, 2 * asin(sqrt(p0))),
        Gate("RY", (1,), 3, 2 * asin(sqrt(p1))),
        ControlledOperator(Gate("X", (0,), 1), (0, 1), (1, 1), (2,), 3),
    )
    return AEProblem(
        StatePreparation(Sequence(operations, 3), 3),
        ObjectivePredicate("bit", qubits=(2,)),
        (0, 1),
        objective_register=(2,),
    )


def test_objective_register_problem_follows_amplification_law():
    problem = _flag_problem(0.3, 0.6)
    a = 0.3 * 0.6
    assert reference_amplitude(problem) == pytest.approx(a, abs=1e-12)
    theta = asin(sqrt(a))
    for k in range(6):
        assert ideal_success_probability(problem, k) == pytest.approx(
            sin((2 * k + 1) * theta) ** 2, abs=1e-12
        )


@pytest.mark.parametrize("algorithm", ["iqae", "mlqae", "faster_ae", "canonical_qae"])
def test_every_estimator_accepts_an_objective_register(algorithm):
    options = {"evaluation_qubits": 4} if algorithm == "canonical_qae" else {}
    result = estimate(
        _flag_problem(0.3, 0.6),
        algorithm=algorithm,
        backend=ReferenceAdapter(),
        epsilon=0.05,
        seed=11,
        **options,
    )
    assert abs(result.estimate - 0.18) < 0.1


@pytest.mark.parametrize("flag", [False, True])
@pytest.mark.parametrize("n", [2, 3, 4, 5])
def test_lognormal_tail_matches_classical_mass_and_amplification(n, flag):
    index = max(1, int(0.6 * 2**n))
    case = lognormal_tail(n, strike_for_index(n, index), objective_ancilla=flag)
    assert case.problem.state_preparation.n_qubits == n + int(flag)
    theta = asin(sqrt(case.a_true))
    for k in range(4):
        assert ideal_success_probability(case.problem, k) == pytest.approx(
            sin((2 * k + 1) * theta) ** 2, abs=1e-12
        )
