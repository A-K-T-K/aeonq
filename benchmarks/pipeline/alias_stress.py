"""Generate a reproducible incompatible high-power observation trace."""

from math import asin, sqrt

from aeonq import AEProblem
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.algorithms.iqae import IQAE
from aeonq.ir import Gate, ObjectivePredicate, StatePreparation


class InjectedQiskit(QiskitAdapter):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def execute(self, compiled_program, shots, *, seed=None):
        self.calls += 1
        if self.calls == 2:
            return {"1": shots}
        return super().execute(compiled_program, shots, seed=seed)


def generate() -> dict:
    amplitude = 0.1
    problem = AEProblem(
        StatePreparation(Gate("RY", (0,), 1, 2 * asin(sqrt(amplitude))), 1),
        ObjectivePredicate("bit", qubits=(0,)),
        (0,),
    )
    solver = IQAE(
        epsilon=0.01,
        shots_per_round=512,
        max_rounds=3,
        max_power=5,
        max_resamples_per_round=1,
        seed=19,
    )
    result = solver.run(problem, InjectedQiskit())
    return {
        "a_true": amplitude,
        "injection": "all success on second circuit evaluation",
        "seed": 19,
        "estimate": result.estimate,
        "interval": result.confidence_or_credible_interval,
        "powers_used": result.powers_used,
        "observations": result.diagnostics["observation_history"],
        "alias_history": result.diagnostics["alias_history"],
        "resampling_triggered": result.diagnostics["resampling_triggered"],
        "status": result.guarantee_status,
    }
