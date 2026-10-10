"""Estimate the probability of a uniform two-bit value meeting a threshold."""

from aeonq import estimate
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.benchmarks.problems import independent_threshold


case = independent_threshold(2, 2)
result = estimate(
    case.problem,
    algorithm="mlqae",
    backend=ReferenceAdapter(),
    powers=(0, 1, 2),
    shots_per_power=64,
    epsilon=0.1,
    seed=17,
)
print("classical threshold mass:", case.a_true)
print("estimate:", result.estimate)
print("interval type:", result.interval_type)
print("interval:", result.confidence_or_credible_interval)
