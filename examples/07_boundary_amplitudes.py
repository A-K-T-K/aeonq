"""Check exact and near-endpoint results without assuming convergence."""

from aeonq import estimate
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.benchmarks.problems import bernoulli


for amplitude in (0.0, 1e-9, 1 - 1e-9, 1.0):
    result = estimate(
        bernoulli(amplitude).problem,
        algorithm="mlqae",
        backend=ReferenceAdapter(),
        powers=(0, 1),
        shots_per_power=64,
        epsilon=0.1,
        seed=52,
    )
    print(amplitude, result.estimate, result.confidence_or_credible_interval)
