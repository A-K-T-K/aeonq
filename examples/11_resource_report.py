"""Required resources and circuit metrics before running, then the resources used."""

from aeonq import estimate
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.benchmarks.problems import lognormal_tail, strike_for_index
from aeonq.resources import circuit_metrics, estimate_resources, format_circuit_metrics

case = lognormal_tail(3, strike_for_index(3, 5), objective_ancilla=True)
backend = QiskitAdapter()

print("Compiled circuits A Q^k:")
print(format_circuit_metrics(circuit_metrics(case.problem, backend, (0, 1, 2, 4))))

for algorithm in ("mlqae", "iqae"):
    print()
    print(estimate_resources(case.problem, algorithm=algorithm, backend=backend))

print()
result = estimate(case.problem, algorithm="iqae", backend=backend, seed=5)
print(result.summary())
print(f"(true value {case.a_true:.6g})")
