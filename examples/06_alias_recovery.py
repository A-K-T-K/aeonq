"""Show why a large power is unsafe before a branch has been localized."""

from math import pi

from aeonq import estimate
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.benchmarks.problems import bernoulli
from aeonq.inference.alias import candidate_branches, unambiguous_branch


initial_phase = (0.0, pi / 2)
print("power 8 candidate branches:", candidate_branches(initial_phase, 8))
print("power 8 safe branch:", unambiguous_branch(initial_phase, 8))
result = estimate(
    bernoulli(0.25).problem,
    algorithm="iqae",
    backend=ReferenceAdapter(),
    epsilon=0.08,
    shots_per_round=96,
    max_rounds=4,
    max_power=8,
    seed=31,
)
print("selected powers:", result.powers_used)
print("rejected alias count:", result.diagnostics["alias_risk"])
print("resampling triggered:", result.diagnostics["resampling_triggered"])
