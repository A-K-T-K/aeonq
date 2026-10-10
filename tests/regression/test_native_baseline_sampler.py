import sys
from pathlib import Path

from qiskit import QuantumCircuit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.native_baselines import CountingSampler  # noqa: E402


def test_counting_sampler_draws_independent_streams_per_circuit():
    # An int seed makes StatevectorSampler reseed every circuit identically,
    # which would give all circuits of one qiskit-algorithms run the same
    # random stream. The counting sampler keeps one stream across circuits.
    circuit = QuantumCircuit(1)
    circuit.ry(1.0, 0)
    circuit.measure_all()
    sampler = CountingSampler(seed=7)
    first = sampler.run([circuit, circuit, circuit]).result()
    second = sampler.run([circuit]).result()
    counts = [r.data.meas.get_counts().get("1", 0) for r in [*first, *second]]
    assert len(set(counts)) > 1
    assert sampler.total_shots == 4 * 512


def test_counting_sampler_is_reproducible_for_a_seed():
    circuit = QuantumCircuit(1)
    circuit.ry(1.0, 0)
    circuit.measure_all()
    runs = [
        [
            r.data.meas.get_counts().get("1", 0)
            for r in CountingSampler(seed=7).run([circuit, circuit]).result()
        ]
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
