"""Resume IQAE from its versioned local checkpoint."""

from pathlib import Path
from tempfile import TemporaryDirectory

from aeonq.adapters.reference import ReferenceAdapter
from aeonq.algorithms.iqae import IQAE
from aeonq.benchmarks.problems import bernoulli


with TemporaryDirectory() as directory:
    checkpoint = Path(directory) / "iqae.json"
    solver = IQAE(
        epsilon=0.01,
        shots_per_round=64,
        max_rounds=3,
        max_power=4,
        seed=12,
        checkpoint=checkpoint,
    )
    problem = bernoulli(0.25).problem
    first = solver.run(problem, ReferenceAdapter())
    resumed = solver.run(problem, ReferenceAdapter())
    assert first.estimate == resumed.estimate
    print("checkpoint schema and state restored from:", checkpoint.name)
    print("estimate:", resumed.estimate, "shots:", resumed.total_shots)
