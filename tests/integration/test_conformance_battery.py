"""The conformance battery, its mutation analysis and non-ideal sampling."""

import pytest
from qiskit_aer.noise import NoiseModel, depolarizing_error

from aeonq import estimate
from aeonq.adapters.cirq import CirqAdapter
from aeonq.adapters.pennylane import PennyLaneAdapter
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.qsharp import QSharpAdapter
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.algorithms.base import NON_IDEAL_SAMPLING_WARNING
from aeonq.benchmarks.problems import bernoulli
from aeonq.conformance.battery import run_battery
from aeonq.conformance.mutation import MUTANTS, MutantAdapter


@pytest.mark.parametrize(
    "factory",
    [ReferenceAdapter, QiskitAdapter, CirqAdapter, PennyLaneAdapter, QSharpAdapter],
)
def test_every_adapter_passes_the_full_battery(factory):
    failures = [r for r in run_battery(factory()) if not r.passed]
    assert failures == []


@pytest.mark.parametrize("mutant", sorted(MUTANTS))
def test_battery_kills_every_seeded_adapter_fault(mutant):
    results = run_battery(MutantAdapter(CirqAdapter(), mutant))
    assert any(not r.passed for r in results)


@pytest.mark.parametrize(
    "mutant",
    ["drop_global_phase", "drop_phase_under_control", "objective_reflection_negated"],
)
def test_phase_faults_are_invisible_to_shots_but_caught_by_exact_operators(mutant):
    # Such faults leave every A·Q^k distribution unchanged, so comparing counts
    # across SDKs cannot find them; they still break canonical QAE.
    results = run_battery(MutantAdapter(CirqAdapter(), mutant))
    failed = {r.level for r in results if not r.passed}
    assert failed == {"operator"}


def _depolarizing(rate: float) -> NoiseModel:
    model = NoiseModel()
    model.add_all_qubit_quantum_error(
        depolarizing_error(rate, 1), ["u", "ry", "x", "h"]
    )
    model.add_all_qubit_quantum_error(depolarizing_error(10 * rate, 2), ["cx"])
    return model


def test_noisy_backend_withholds_theoretical_guarantee():
    noisy = QiskitAdapter(noise_model=_depolarizing(1e-3))
    assert not noisy.capabilities().ideal_sampling
    assert QiskitAdapter().capabilities().ideal_sampling
    problem = bernoulli(0.25).problem
    ideal = estimate(
        problem, algorithm="iqae", backend=QiskitAdapter(), epsilon=0.05, seed=3
    )
    result = estimate(problem, algorithm="iqae", backend=noisy, epsilon=0.05, seed=3)
    assert ideal.guarantee_status == "THEORETICAL_GUARANTEE_SATISFIED"
    assert ideal.warnings == ()
    assert result.convergence_status == "CONVERGED"
    assert result.guarantee_status == "THEORETICAL_GUARANTEE_NOT_APPLICABLE"
    assert result.warnings == (NON_IDEAL_SAMPLING_WARNING,)
    canonical = estimate(problem, algorithm="canonical_qae", backend=noisy, seed=3)
    assert canonical.warnings == (NON_IDEAL_SAMPLING_WARNING,)


def test_missing_matrix_export_is_unsupported_never_passed():
    class NoMatrices(ReferenceAdapter):
        def unitary(self, ir):
            raise NotImplementedError

    results = run_battery(NoMatrices())
    unsupported = [r for r in results if r.status == "unsupported"]
    assert len(unsupported) == 43
    assert all(r.passed is None and r.level == "operator" for r in unsupported)
    assert sum(r.passed is True for r in results) == 23


def test_asymmetric_rotations_expose_sign_error_in_sampling():
    results = run_battery(MutantAdapter(CirqAdapter(), "rotation_sign"))
    failures = [r for r in results if r.level == "sampling" and r.passed is False]
    assert any(
        r.case == "asymmetric_rotations_n1" and r.check == "A_Q^0" for r in failures
    )
