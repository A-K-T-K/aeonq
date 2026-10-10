"""Run IQAE with a Fisher-information schedule."""

from aeonq import estimate
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.benchmarks.problems import bernoulli
from aeonq.scheduling import FisherInformationSchedule


result = estimate(
    bernoulli(0.35).problem,
    algorithm="iqae",
    backend=ReferenceAdapter(),
    epsilon=0.08,
    shots_per_round=96,
    max_rounds=4,
    max_power=8,
    scheduler=FisherInformationSchedule(),
    seed=8,
)
print("estimate:", result.estimate)
print("chosen powers:", result.powers_used)
print("oracle queries:", result.total_oracle_queries)
