"""Execute the same semantic problem on four local framework targets."""

from aeonq import estimate
from aeonq.adapters.cirq import CirqAdapter
from aeonq.adapters.pennylane import PennyLaneAdapter
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.qsharp import QSharpAdapter
from aeonq.benchmarks.problems import bernoulli


problem = bernoulli(0.25).problem
for adapter in (QiskitAdapter(), CirqAdapter(), PennyLaneAdapter(), QSharpAdapter()):
    result = estimate(
        problem,
        algorithm="mlqae",
        backend=adapter,
        powers=(0,),
        shots_per_power=32,
        epsilon=0.2,
        seed=13,
    )
    print(type(adapter).__name__, result.estimate, result.total_shots)
