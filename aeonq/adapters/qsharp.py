"""Lowering to Q# source, evaluated and executed through the Microsoft QDK."""

from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from typing import Any

import numpy as np
from qdk import qsharp

from aeonq.adapters.base import BackendAdapter, BackendCapabilities, NormalizedResult
from aeonq.ir.nodes import (
    Adjoint,
    ControlledOperator,
    EmbeddedOperator,
    Gate,
    GlobalPhase,
    GroverOperator,
    GroverPower,
    ObjectiveReflection,
    Operator,
    Sequence,
    StatePreparation,
    ZeroReflection,
)
from aeonq.ir.registers import RegisterMap
from aeonq.ir.serialization import dumps
from aeonq.synthesis.reflection import (
    synthesize_objective_reflection,
    synthesize_zero_reflection,
)


# Sources already evaluated in the process-wide Q# interpreter. Operation names
# are derived from a hash of the IR, so an identical source is evaluated only
# once. Keyed by the full source so that a hash-prefix collision cannot skip a
# needed evaluation.
_EVALUATED_SOURCES: set[str] = set()


def qdk_seed(seed: int | None) -> int | None:
    """Spread an estimator seed over QDK's seed space.

    QDK seeds shot i of a run with seed + i, so runs with nearby seeds reuse
    almost all of each other's random draws. Estimators give successive
    rounds consecutive seeds, so passing them through unchanged would
    correlate those rounds. Hashing keeps runs reproducible while making any
    overlap between two runs' shot seeds negligible. QDK accepts seeds below
    2^63.
    """
    if seed is None:
        return None
    return int(np.random.SeedSequence(seed).generate_state(1, np.uint64)[0] >> 1)


def clear_evaluation_cache() -> None:
    """Force the next compile of every program to evaluate it again.

    Used by timing studies so that each measured compile includes the
    evaluation.
    """
    _EVALUATED_SOURCES.clear()


@dataclass(frozen=True)
class QSharpProgram:
    source: str
    entry: str
    n_qubits: int
    primitive_count: int


class QSharpAdapter(BackendAdapter):
    # dump_operation prints about six significant decimals per matrix entry, so
    # entries of a correct lowering differ from the exact reference by ~5e-7.
    unitary_precision = 1e-6

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            True,
            True,
            True,
            False,
            False,
            False,
            False,
            False,
            12,
            None,
            native_gate_set=frozenset(
                {"H", "X", "Y", "Z", "Rx", "Ry", "Rz", "CNOT", "R"}
            ),
            metadata={"qdk": version("qdk")},
        )

    def validate(self, ir: Operator) -> None:
        self.compile(self.lower(ir))

    def lower(self, ir: Operator) -> QSharpProgram:
        if ir.n_qubits > 12:
            raise ValueError("local Q# adapter limits semantic lowering to 12 qubits")
        prefix = "Aeonq" + sha256(dumps(ir).encode("utf-8")).hexdigest()[:16]
        definitions: list[str] = []
        primitive_count = 0

        def emit(node: Operator) -> str:
            nonlocal primitive_count
            name = f"{prefix}Op{len(definitions)}"
            # Reserve a slot before traversing children so names remain unique.
            slot = len(definitions)
            definitions.append("")
            if isinstance(node, Gate):
                if node.name in ("RX", "RY", "RZ"):
                    if not isinstance(node.parameter, (int, float)):
                        raise ValueError("bind symbolic parameters before lowering")
                    body = f"{node.name.title()}({float(node.parameter)!r}, qs[{node.targets[0]}]);"
                elif node.name == "CX":
                    body = f"CNOT(qs[{node.targets[0]}], qs[{node.targets[1]}]);"
                else:
                    body = f"{node.name}(qs[{node.targets[0]}]);"
                primitive_count += 1
            elif isinstance(node, Sequence):
                body = "\n".join(f"{emit(child)}(qs);" for child in node.children)
            elif isinstance(node, StatePreparation):
                body = f"{emit(node.operation)}(qs);"
            elif isinstance(node, Adjoint):
                body = f"Adjoint {emit(node.operation)}(qs);"
            elif isinstance(node, GlobalPhase):
                body = f"R(PauliI, {-2 * node.radians!r}, qs[0]);"
                primitive_count += 1
            elif isinstance(node, ZeroReflection):
                body = f"{emit(synthesize_zero_reflection(node))}(qs);"
            elif isinstance(node, ObjectiveReflection):
                body = f"{emit(synthesize_objective_reflection(node))}(qs);"
            elif isinstance(node, GroverOperator):
                from aeonq.synthesis.grover import synthesize_grover

                body = f"{emit(synthesize_grover(node))}(qs);"
            elif isinstance(node, GroverPower):
                child = emit(node.grover)
                body = "\n".join(f"{child}(qs);" for _ in range(node.power))
            elif isinstance(node, ControlledOperator):
                child = emit(node.operation)
                controls = ", ".join(f"qs[{q}]" for q in node.controls)
                targets = ", ".join(f"qs[{q}]" for q in node.targets)
                body = f"Controlled {child}([{controls}], [{targets}]);"
                zero_controls = [
                    q for q, v in zip(node.controls, node.control_values) if v == 0
                ]
                if zero_controls:
                    # Condition on |0> by X-conjugating those controls; Q#
                    # derives the adjoint and controlled forms of within/apply.
                    flips = " ".join(f"X(qs[{q}]);" for q in zero_controls)
                    body = f"within {{ {flips} }} apply {{ {body} }}"
            elif isinstance(node, EmbeddedOperator):
                child = emit(node.operation)
                targets = ", ".join(f"qs[{q}]" for q in node.targets)
                body = f"{child}([{targets}]);"
            else:
                raise TypeError(f"Q# lowering does not support {type(node).__name__}")
            definitions[slot] = (
                f"operation {name}(qs : Qubit[]) : Unit is Adj + Ctl {{\n{body}\n}}"
            )
            return name

        entry = emit(ir)
        measurements = ", ".join(f"M(qs[{i}])" for i in range(ir.n_qubits))
        run = (
            f"operation {entry}Run() : Result[] {{\n"
            f"use qs = Qubit[{ir.n_qubits}];\n{entry}(qs);\n"
            f"let results = [{measurements}];\nResetAll(qs);\nreturn results;\n}}"
        )
        return QSharpProgram(
            "\n".join(definitions + [run]), entry, ir.n_qubits, primitive_count
        )

    def compile(
        self, native_program: QSharpProgram, options: dict[str, Any] | None = None
    ) -> QSharpProgram:
        if options:
            raise ValueError("Q# compilation options are not implemented")
        if native_program.source not in _EVALUATED_SOURCES:
            qsharp.eval(native_program.source)
            _EVALUATED_SOURCES.add(native_program.source)
        return native_program

    def execute(
        self, compiled_program: QSharpProgram, shots: int, *, seed: int | None = None
    ) -> Any:
        if shots < 1:
            raise ValueError("shots must be positive")
        results = qsharp.run(
            f"{compiled_program.entry}Run()", shots=shots, seed=qdk_seed(seed)
        )
        raw: dict[str, int] = {}
        for row in results:
            if len(row) != compiled_program.n_qubits:
                raise ValueError("Q# returned unexpected result width")
            bitstring = "".join(
                "1" if str(bit) == "One" else "0" if str(bit) == "Zero" else "?"
                for bit in reversed(row)
            )
            if "?" in bitstring:
                raise ValueError("Q# returned malformed result")
            raw[bitstring] = raw.get(bitstring, 0) + 1
        return raw

    def normalize_result(self, raw: Any, register_map: RegisterMap) -> NormalizedResult:
        if not isinstance(raw, dict):
            raise ValueError("Q# counts must be a mapping")
        counts: dict[tuple[int, ...], int] = {}
        for bitstring, count in raw.items():
            if (
                not isinstance(bitstring, str)
                or not isinstance(count, int)
                or count < 0
            ):
                raise ValueError("malformed count entry")
            bits = register_map.normalize_bitstring(bitstring)
            counts[bits] = counts.get(bits, 0) + count
        return NormalizedResult(counts, sum(counts.values()))

    def resource_metrics(
        self, compiled_program: QSharpProgram
    ) -> dict[str, int | float | None]:
        return {
            "qubits": compiled_program.n_qubits,
            "depth": None,
            "two_qubit_gates": None,
            "one_qubit_gates": None,
            "primitive_statements": compiled_program.primitive_count,
            "t_count": None,
        }

    def unitary(self, ir: Operator) -> np.ndarray:
        program = self.compile(self.lower(ir))
        native = np.asarray(
            qsharp.dump_operation(program.entry, program.n_qubits), dtype=complex
        )
        n = program.n_qubits
        # QDK dump_operation uses q0 as the most significant bit.
        permutation = [int(f"{i:0{n}b}"[::-1], 2) for i in range(2**n)]
        return native[np.ix_(permutation, permutation)]
