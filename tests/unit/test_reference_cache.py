from concurrent.futures import ThreadPoolExecutor

import numpy as np

from aeonq.adapters.reference import ReferenceAdapter, _probability_vector
from aeonq.benchmarks.problems import independent_threshold
from aeonq.ir import GroverPower, Sequence
from aeonq.reference import grover_for, matrix


def test_repeated_and_concurrent_reference_sampling_preserves_seeded_counts():
    problem = independent_threshold(3, 3).problem
    q = grover_for(problem)
    ir = Sequence((problem.state_preparation, GroverPower(q, 8, 3)), 3)
    amplitudes = matrix(ir)[:, 0]
    probabilities = np.abs(amplitudes) ** 2
    probabilities /= probabilities.sum()
    adapter = ReferenceAdapter()
    _probability_vector.cache_clear()

    def sample(seed: int) -> dict[str, int]:
        return adapter.execute(ir, 128, seed=seed)

    with ThreadPoolExecutor(max_workers=4) as pool:
        observed = list(pool.map(sample, range(20)))
    expected = []
    for seed in range(20):
        counts = np.random.default_rng(seed).multinomial(128, probabilities)
        expected.append(
            {f"{i:03b}": int(count) for i, count in enumerate(counts) if count}
        )
    assert observed == expected
    assert _probability_vector.cache_info().hits >= 1
    assert not _probability_vector(ir).flags.writeable
