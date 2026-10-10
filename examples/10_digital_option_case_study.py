"""Case study: probability that a log-normal asset price ends above a strike.

This is the undiscounted price of a digital (cash-or-nothing) option, or a
tail-exceedance probability. The price distribution is loaded exactly on three
qubits, and the preparation computes the flag [price >= strike] into a fourth
(objective) qubit, the usual finance-library pattern. The same semantic
problem then runs on every local SDK, and a noisy backend shows the
guarantee being withheld.
"""

from qiskit_aer.noise import NoiseModel, depolarizing_error

from aeonq import estimate
from aeonq.adapters.cirq import CirqAdapter
from aeonq.adapters.pennylane import PennyLaneAdapter
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.qsharp import QSharpAdapter
from aeonq.benchmarks.problems import lognormal_grid, lognormal_tail, strike_for_index

strike = strike_for_index(3, 5)
case = lognormal_tail(3, strike, objective_ancilla=True)
print(f"price grid: {[round(p, 3) for p in lognormal_grid(3).points]}")
print(f"strike {strike:.3f}; exact P(S_T >= strike) = {case.a_true:.6f}")

for adapter in (QiskitAdapter(), CirqAdapter(), PennyLaneAdapter(), QSharpAdapter()):
    result = estimate(
        case.problem,
        algorithm="iqae",
        backend=adapter,
        epsilon=0.02,
        max_power=8,
        seed=29,
    )
    low, high = result.confidence_or_credible_interval
    print(
        f"{type(adapter).__name__:17s} estimate {result.estimate:.4f} "
        f"interval [{low:.4f}, {high:.4f}] queries {result.total_oracle_queries} "
        f"{result.guarantee_status}"
    )

noise = NoiseModel()
noise.add_all_qubit_quantum_error(depolarizing_error(1e-3, 1), ["u", "ry", "x"])
noise.add_all_qubit_quantum_error(depolarizing_error(1e-2, 2), ["cx"])
noisy = estimate(
    case.problem,
    algorithm="iqae",
    backend=QiskitAdapter(noise_model=noise),
    epsilon=0.02,
    max_power=8,
    seed=29,
)
print(f"noisy Qiskit: estimate {noisy.estimate:.4f} {noisy.guarantee_status}")
print(f"warning: {noisy.warnings[0]}")
