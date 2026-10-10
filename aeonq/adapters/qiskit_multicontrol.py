"""Exact Qiskit multi-control strategies with measured resource selection.

Qiskit's three synthesis routines use no ancilla, k-2 clean ancillas, or k-2
dirty ancillas. The dirty routine is called with relative_phase=False and
action_only=False. Only the available workspace is considered, and the
selection is made among these three routines.
"""

from dataclasses import dataclass
from typing import Literal, cast

from qiskit import QuantumCircuit, transpile
from qiskit.synthesis import (
    synth_mcx_n_clean_m15,
    synth_mcx_n_dirty_i15,
    synth_mcx_noaux_v24,
)

from aeonq.compiler.resource_model import CompilerCost


@dataclass(frozen=True)
class MultiControlCandidate:
    strategy: Literal["ancilla_free", "clean_ladder", "dirty_ladder"]
    circuit: QuantumCircuit
    ancillas: int
    metrics: dict[str, int | float | None]
    cost: float


@dataclass(frozen=True)
class MultiControlSelection:
    selected: MultiControlCandidate
    candidates: tuple[MultiControlCandidate, ...]


def _metrics(circuit: QuantumCircuit, ancillas: int) -> dict[str, int | float | None]:
    lowered = transpile(
        circuit, basis_gates=["u", "cx"], optimization_level=0, seed_transpiler=0
    )
    counts = lowered.count_ops()
    return {
        "qubits": circuit.num_qubits,
        "ancillas": ancillas,
        "one_qubit_gates": counts.get("u", 0),
        "two_qubit_gates": counts.get("cx", 0),
        "multi_qubit_gates": 0,
        "depth": lowered.depth(),
        "t_count": None,
    }


def select_multi_control(
    num_controls: int,
    *,
    phase: bool = True,
    clean_available: int = 0,
    dirty_available: int = 0,
    cost_model: CompilerCost = CompilerCost(),
) -> MultiControlSelection:
    """Select minimum measured cost among feasible exact MCX/MCZ syntheses."""
    if not isinstance(num_controls, int) or num_controls < 3:
        raise ValueError("multi-control selection requires at least 3 controls")
    if clean_available < 0 or dirty_available < 0:
        raise ValueError("workspace counts must be nonnegative")
    strategy_specs = [("ancilla_free", synth_mcx_noaux_v24)]
    if clean_available >= num_controls - 2:
        strategy_specs.append(("clean_ladder", synth_mcx_n_clean_m15))
    if dirty_available >= num_controls - 2:
        strategy_specs.append(("dirty_ladder", synth_mcx_n_dirty_i15))
    candidates: list[MultiControlCandidate] = []
    for name, synth in strategy_specs:
        raw = synth(num_controls)
        ancillas = raw.num_qubits - num_controls - 1
        if phase:
            circuit = QuantumCircuit(raw.num_qubits)
            circuit.h(num_controls)
            circuit.compose(raw, inplace=True)
            circuit.h(num_controls)
        else:
            circuit = raw
        metrics = _metrics(circuit, ancillas)
        cost = cost_model.evaluate(metrics, num_controls + 1)
        candidates.append(
            MultiControlCandidate(
                cast(Literal["ancilla_free", "clean_ladder", "dirty_ladder"], name),
                circuit,
                ancillas,
                metrics,
                cost,
            )
        )
    selected = min(candidates, key=lambda candidate: candidate.cost)
    return MultiControlSelection(selected, tuple(candidates))
