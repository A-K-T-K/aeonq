"""Compare the result fields each estimator reports under small local budgets."""

from aeonq import estimate
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.benchmarks.problems import bernoulli


problem = bernoulli(0.25).problem
backend = ReferenceAdapter()
choices = {
    "iqae": {"shots_per_round": 64, "max_rounds": 3, "max_power": 2},
    "mlqae": {"powers": (0, 1, 2), "shots_per_power": 64},
    "canonical_qae": {"evaluation_qubits": 2, "shots": 64},
    "faster_ae": {"max_iterations": 1, "max_total_shots": 50_000},
}
for method, options in choices.items():
    result = estimate(
        problem,
        algorithm=method,
        backend=backend,
        epsilon=0.1,
        seed=21,
        **options,
    )
    print(method, result.estimate, result.interval_type, result.convergence_status)
