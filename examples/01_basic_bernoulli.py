"""Estimate a known one-qubit success probability with local IQAE."""

from aeonq import estimate
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.benchmarks.problems import bernoulli


case = bernoulli(0.25)
result = estimate(
    case.problem,
    algorithm="iqae",
    backend=ReferenceAdapter(),
    epsilon=0.1,
    alpha=0.05,
    shots_per_round=64,
    max_rounds=4,
    max_power=4,
    seed=41,
)
print("classical truth:", case.a_true)
print("estimate:", result.estimate)
print("interval:", result.confidence_or_credible_interval)
print("status:", result.convergence_status, result.guarantee_status)
