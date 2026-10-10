"""Translator fidelity: do existing SDK converters preserve AE operators?

Each amplitude-estimation operator is lowered natively in a source SDK and
passed through an existing third-party conversion path: the OpenQASM 2 and 3
exporters and importers shipped with Qiskit, Cirq, PennyLane and the QDK, and
pytket's Qiskit and Cirq extensions. aeonq takes no part in the conversion.
The program that comes out is evaluated only with the target SDK's own
facilities and compared with the program that went in:

* exact equality of the unitary, entry by entry;
* equality up to one global phase, and the size of that phase;
* equality once the translated program is controlled inside the target SDK,
  where a dropped global phase becomes an observable relative phase;
* the qubit permutation, if any, under which the operators agree;
* whether each data-carrying rotation angle survived, and how precisely;
* a count-based check on the same program (shots of U|0...0>, G-test).

A row whose counts pass while an operator check fails is a translation fault
that count-based testing cannot see. Converters are optional dependencies; a
path whose modules are missing is reported as unavailable, not skipped.
"""

from ast import (
    BinOp,
    Constant,
    Expression,
    Name,
    UnaryOp,
    Add,
    Div,
    Mult,
    Pow,
    Sub,
    USub,
    UAdd,
    parse,
)
from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from itertools import permutations
from math import pi
import re
from typing import Any

import numpy as np

from aeonq.conformance.battery import (
    SAMPLING_P_THRESHOLD,
    _goodness_of_fit,
    battery_cases,
)
from aeonq.ir import (
    Adjoint,
    ControlledOperator,
    GlobalPhase,
    GroverPower,
    Sequence,
)
from aeonq.ir.nodes import Operator
from aeonq.problem import AEProblem
from aeonq.reference import grover_for, matrix

COUNT_SHOTS = 4000
# Angles that are multiples of pi/4 come from Clifford+T structure and are
# legitimately re-expressed by converters; other angles carry problem data.
_STRUCTURAL_ANGLE = pi / 4
_ANGLE_SURVIVES = 1e-6
_CONTROL = "aeonq_control"


# ---------------------------------------------------------------- programs


@dataclass(frozen=True)
class Translated:
    """A program produced by a converter, in the target SDK's own types.

    ``labels[i]`` is the target-SDK qubit that the converter names as source
    qubit ``i`` (register ``q``, index ``i``); the study never renumbers.
    """

    sdk: str
    program: Any
    labels: tuple[Any, ...]
    intermediate: Any


def _source_program(sdk: str, node: Operator) -> tuple[Any, np.ndarray]:
    """Native program and its matrix, from the adapter verified by the battery."""
    if sdk == "qiskit":
        from aeonq.adapters.qiskit import QiskitAdapter

        adapter: Any = QiskitAdapter()
    elif sdk == "cirq":
        from aeonq.adapters.cirq import CirqAdapter

        adapter = CirqAdapter()
    elif sdk == "pennylane":
        from aeonq.adapters.pennylane import PennyLaneAdapter

        adapter = PennyLaneAdapter()
    else:
        raise ValueError(f"unknown source SDK {sdk}")
    return adapter.lower(node), adapter.unitary(node)


# --------------------------------------------------------------- exporters


def _qiskit_qasm2(program: Any, n: int) -> str:
    from qiskit import qasm2

    return qasm2.dumps(program)


def _qiskit_qasm3(program: Any, n: int) -> str:
    from qiskit import qasm3

    return qasm3.dumps(program)


def _cirq_qasm2(program: Any, n: int) -> str:
    return program.to_qasm()


def _cirq_qasm3(program: Any, n: int) -> str:
    return program.to_qasm(version="3.0")


def _pennylane_qasm2(program: Any, n: int) -> str:
    import pennylane as qml

    return qml.to_openqasm(program, wires=list(range(n)), measure_all=False)


def _qiskit_to_tk(program: Any, n: int) -> Any:
    from pytket.extensions.qiskit import qiskit_to_tk

    return qiskit_to_tk(program)


def _cirq_to_tk(program: Any, n: int) -> Any:
    from pytket.extensions.cirq import cirq_to_tk

    return cirq_to_tk(program)


# --------------------------------------------------------------- importers


def _into_qiskit_qasm2(text: str, n: int) -> Translated:
    from qiskit import qasm2

    # Qiskit documents LEGACY_CUSTOM_INSTRUCTIONS for reading its own exports.
    circuit = qasm2.loads(text, custom_instructions=qasm2.LEGACY_CUSTOM_INSTRUCTIONS)
    return Translated("qiskit", circuit, tuple(range(n)), text)


def _into_qiskit_qasm3(text: str, n: int) -> Translated:
    from qiskit import qasm3

    return Translated("qiskit", qasm3.loads(text), tuple(range(n)), text)


def _into_cirq_qasm(text: str, n: int) -> Translated:
    import cirq
    from cirq.contrib.qasm_import import circuit_from_qasm

    labels = tuple(cirq.NamedQubit(f"q_{i}") for i in range(n))
    return Translated("cirq", circuit_from_qasm(text), labels, text)


def _into_pennylane_qasm3(text: str, n: int) -> Translated:
    import pennylane as qml

    return Translated(
        "pennylane", qml.from_qasm3(text), tuple(f"q[{i}]" for i in range(n)), text
    )


_STD_INCLUDE = re.compile(r'^\s*include\s+"stdgates\.inc"\s*;\s*$', re.MULTILINE)


def _into_pennylane_qasm3_without_include(text: str, n: int) -> Translated:
    # PennyLane's interpreter rejects the include statement that every QASM 3
    # exporter writes; deleting it by hand is the workaround a user would try.
    return _into_pennylane_qasm3(_STD_INCLUDE.sub("", text), n)


_QSHARP_IMPORTS = 0


def _into_qsharp(text: str, n: int) -> Translated:
    import qdk.openqasm as openqasm

    global _QSHARP_IMPORTS
    _QSHARP_IMPORTS += 1
    name = f"AeonqTranslated{_QSHARP_IMPORTS}"
    openqasm.import_openqasm(text, name=name)
    return Translated("qsharp", name, tuple(range(n)), text)


def _tk_into_cirq(circuit: Any, n: int) -> Translated:
    import cirq
    from pytket.extensions.cirq import tk_to_cirq

    labels = tuple(cirq.LineQubit(i) for i in range(n))
    return Translated("cirq", tk_to_cirq(circuit), labels, circuit)


def _tk_into_qiskit(circuit: Any, n: int) -> Translated:
    from pytket.extensions.qiskit import tk_to_qiskit

    return Translated("qiskit", tk_to_qiskit(circuit), tuple(range(n)), circuit)


@dataclass(frozen=True)
class TranslationPath:
    name: str
    source: str
    intermediate: str
    target: str
    export: Callable[[Any, int], Any]
    load: Callable[[Any, int], Translated]
    requires: tuple[str, ...]
    note: str = ""


PATHS: tuple[TranslationPath, ...] = (
    TranslationPath(
        "qiskit>qasm2>qiskit",
        "qiskit",
        "OpenQASM 2",
        "qiskit",
        _qiskit_qasm2,
        _into_qiskit_qasm2,
        ("qiskit.qasm2",),
        "round trip",
    ),
    TranslationPath(
        "qiskit>qasm3>qiskit",
        "qiskit",
        "OpenQASM 3",
        "qiskit",
        _qiskit_qasm3,
        _into_qiskit_qasm3,
        ("qiskit.qasm3", "qiskit_qasm3_import"),
        "round trip",
    ),
    TranslationPath(
        "qiskit>qasm2>cirq",
        "qiskit",
        "OpenQASM 2",
        "cirq",
        _qiskit_qasm2,
        _into_cirq_qasm,
        ("qiskit.qasm2", "cirq.contrib.qasm_import"),
    ),
    TranslationPath(
        "qiskit>qasm3>cirq",
        "qiskit",
        "OpenQASM 3",
        "cirq",
        _qiskit_qasm3,
        _into_cirq_qasm,
        ("qiskit.qasm3", "cirq.contrib.qasm_import"),
    ),
    TranslationPath(
        "qiskit>qasm3>pennylane",
        "qiskit",
        "OpenQASM 3",
        "pennylane",
        _qiskit_qasm3,
        _into_pennylane_qasm3,
        ("qiskit.qasm3", "pennylane", "openqasm3"),
    ),
    TranslationPath(
        "qiskit>qasm3-noinclude>pennylane",
        "qiskit",
        "OpenQASM 3 (include line deleted)",
        "pennylane",
        _qiskit_qasm3,
        _into_pennylane_qasm3_without_include,
        ("qiskit.qasm3", "pennylane", "openqasm3"),
        "manual workaround",
    ),
    TranslationPath(
        "qiskit>qasm2>qsharp",
        "qiskit",
        "OpenQASM 2",
        "qsharp",
        _qiskit_qasm2,
        _into_qsharp,
        ("qiskit.qasm2", "qdk.openqasm"),
    ),
    TranslationPath(
        "qiskit>qasm3>qsharp",
        "qiskit",
        "OpenQASM 3",
        "qsharp",
        _qiskit_qasm3,
        _into_qsharp,
        ("qiskit.qasm3", "qdk.openqasm"),
    ),
    TranslationPath(
        "qiskit>pytket>cirq",
        "qiskit",
        "pytket Circuit",
        "cirq",
        _qiskit_to_tk,
        _tk_into_cirq,
        ("pytket.extensions.qiskit", "pytket.extensions.cirq"),
    ),
    TranslationPath(
        "cirq>qasm2>qiskit",
        "cirq",
        "OpenQASM 2",
        "qiskit",
        _cirq_qasm2,
        _into_qiskit_qasm2,
        ("qiskit.qasm2",),
    ),
    TranslationPath(
        "cirq>qasm3>qiskit",
        "cirq",
        "OpenQASM 3",
        "qiskit",
        _cirq_qasm3,
        _into_qiskit_qasm3,
        ("qiskit.qasm3", "qiskit_qasm3_import"),
    ),
    TranslationPath(
        "cirq>qasm2>cirq",
        "cirq",
        "OpenQASM 2",
        "cirq",
        _cirq_qasm2,
        _into_cirq_qasm,
        ("cirq.contrib.qasm_import",),
        "round trip",
    ),
    TranslationPath(
        "cirq>qasm3>qsharp",
        "cirq",
        "OpenQASM 3",
        "qsharp",
        _cirq_qasm3,
        _into_qsharp,
        ("qdk.openqasm",),
    ),
    TranslationPath(
        "cirq>pytket>qiskit",
        "cirq",
        "pytket Circuit",
        "qiskit",
        _cirq_to_tk,
        _tk_into_qiskit,
        ("pytket.extensions.qiskit", "pytket.extensions.cirq"),
    ),
    TranslationPath(
        "pennylane>qasm2>qiskit",
        "pennylane",
        "OpenQASM 2",
        "qiskit",
        _pennylane_qasm2,
        _into_qiskit_qasm2,
        ("qiskit.qasm2",),
    ),
    TranslationPath(
        "pennylane>qasm2>cirq",
        "pennylane",
        "OpenQASM 2",
        "cirq",
        _pennylane_qasm2,
        _into_cirq_qasm,
        ("cirq.contrib.qasm_import",),
    ),
    TranslationPath(
        "pennylane>qasm2>qsharp",
        "pennylane",
        "OpenQASM 2",
        "qsharp",
        _pennylane_qasm2,
        _into_qsharp,
        ("qdk.openqasm",),
    ),
)

# Conversion routes that exist but cannot be installed next to the pinned SDKs.
UNAVAILABLE_PATHS: dict[str, str] = {
    "qiskit>pennylane-qiskit>pennylane": (
        "qml.from_qiskit / qml.from_qasm need pennylane-qiskit 0.45.0, which "
        "requires qiskit 2.3.0; the pinned stack uses qiskit 2.5.2"
    ),
}

CONVERTER_DISTRIBUTIONS = (
    "qiskit",
    "qiskit-qasm3-import",
    "openqasm3",
    "cirq-core",
    "ply",
    "pennylane",
    "qdk",
    "pytket",
    "pytket-qiskit",
    "pytket-cirq",
    "pennylane-qiskit",
)


def converter_versions() -> dict[str, str | None]:
    found: dict[str, str | None] = {}
    for name in CONVERTER_DISTRIBUTIONS:
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            found[name] = None
    return found


def missing_modules(path: TranslationPath) -> list[str]:
    return [module for module in path.requires if not _importable(module)]


# ------------------------------------------------- target-SDK evaluation


def _little_endian_from_msb(native: np.ndarray, n: int) -> np.ndarray:
    permutation = [int(f"{i:0{n}b}"[::-1], 2) for i in range(2**n)]
    return native[np.ix_(permutation, permutation)]


class _Qiskit:
    precision = 1e-9

    @staticmethod
    def _checked(t: Translated) -> Any:
        circuit = t.program
        if circuit.num_qubits != len(t.labels):
            raise ValueError(
                f"qubit count changed from {len(t.labels)} to {circuit.num_qubits}"
            )
        return circuit

    def unitary(self, t: Translated) -> np.ndarray:
        from qiskit.quantum_info import Operator as QiskitOperator

        return np.asarray(QiskitOperator(self._checked(t)).data)

    def controlled_unitary(self, t: Translated) -> np.ndarray:
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Operator as QiskitOperator

        n = len(t.labels)
        circuit = QuantumCircuit(n + 1)
        circuit.append(self._checked(t).to_gate().control(1), [n, *range(n)])
        return np.asarray(QiskitOperator(circuit).data)

    def sample(self, t: Translated, shots: int, seed: int) -> dict[int, int]:
        from qiskit import transpile
        from qiskit_aer import AerSimulator

        circuit = self._checked(t).copy()
        circuit.measure_all()
        # Aer runs only its own instruction set; transpiling first is the
        # documented way to run a circuit containing imported custom gates.
        simulator = AerSimulator()
        circuit = transpile(circuit, simulator, optimization_level=0)
        raw = simulator.run(circuit, shots=shots, seed_simulator=seed).result()
        return {int(k.replace(" ", ""), 2): v for k, v in raw.get_counts().items()}

    def parameters(self, t: Translated) -> list[float]:
        return qiskit_parameters(t.program)


def qiskit_parameters(circuit: Any) -> list[float]:
    from qiskit.circuit.library import get_standard_gate_name_mapping

    standard = set(get_standard_gate_name_mapping())
    found = [float(circuit.global_phase)] if circuit.global_phase else []
    for item in circuit.data:
        operation = item.operation
        found.extend(float(p) for p in operation.params if _is_number(p))
        definition = getattr(operation, "definition", None)
        if operation.name not in standard and definition is not None:
            found.extend(qiskit_parameters(definition))
    return found


class _Cirq:
    precision = 1e-9

    def unitary(self, t: Translated) -> np.ndarray:
        self._check_qubits(t)
        order = list(reversed(t.labels))
        return np.asarray(
            t.program.unitary(qubit_order=order, qubits_that_should_be_present=order)
        )

    def controlled_unitary(self, t: Translated) -> np.ndarray:
        import cirq

        self._check_qubits(t)
        control = cirq.NamedQubit(_CONTROL)
        operation = cirq.CircuitOperation(t.program.freeze()).controlled_by(control)
        order = [control, *reversed(t.labels)]
        return np.asarray(
            cirq.Circuit(operation).unitary(
                qubit_order=order, qubits_that_should_be_present=order
            )
        )

    def sample(self, t: Translated, shots: int, seed: int) -> dict[int, int]:
        import cirq

        self._check_qubits(t)
        circuit = t.program + cirq.Circuit(cirq.measure(*t.labels, key="m"))
        random = np.random.RandomState(np.random.SeedSequence(seed).generate_state(4))
        rows = cirq.Simulator(seed=random).run(circuit, repetitions=shots)
        counts: dict[int, int] = {}
        for row in rows.measurements["m"]:
            x = sum(int(bit) << i for i, bit in enumerate(row))
            counts[x] = counts.get(x, 0) + 1
        return counts

    def parameters(self, t: Translated) -> list[float]:
        return cirq_parameters(t.program)

    @staticmethod
    def _check_qubits(t: Translated) -> None:
        unknown = set(t.program.all_qubits()) - set(t.labels)
        if unknown:
            raise ValueError(f"converter introduced unmapped qubits {sorted(unknown)}")


def cirq_parameters(circuit: Any) -> list[float]:
    import cirq

    found: list[float] = []

    def gate_angles(gate: Any) -> None:
        if isinstance(gate, cirq.ControlledGate):
            gate_angles(gate.sub_gate)
        elif isinstance(gate, cirq.GlobalPhaseGate):
            found.append(float(np.angle(complex(gate.coefficient))))
        elif hasattr(gate, "_rads") and _is_number(gate._rads):
            found.append(float(gate._rads))
        elif isinstance(gate, cirq.EigenGate) and _is_number(gate.exponent):
            found.append(float(gate.exponent) * pi)

    def walk(operations: Any) -> None:
        for operation in operations:
            while isinstance(operation, cirq.TaggedOperation):
                operation = operation.sub_operation
            if isinstance(operation, cirq.CircuitOperation):
                walk(operation.mapped_circuit(deep=False).all_operations())
            elif isinstance(operation, cirq.ControlledOperation):
                walk([operation.sub_operation])
            elif operation.gate is not None:
                gate_angles(operation.gate)

    walk(circuit.all_operations())
    return found


class _PennyLane:
    precision = 1e-9

    def unitary(self, t: Translated) -> np.ndarray:
        import pennylane as qml

        order = list(reversed(t.labels))
        return np.asarray(qml.matrix(t.program, wire_order=order)())

    def controlled_unitary(self, t: Translated) -> np.ndarray:
        import pennylane as qml

        order = [_CONTROL, *reversed(t.labels)]
        return np.asarray(
            qml.matrix(qml.ctrl(t.program, control=_CONTROL), wire_order=order)()
        )

    def sample(self, t: Translated, shots: int, seed: int) -> dict[int, int]:
        import pennylane as qml

        operations = self._operations(t)
        labels = list(t.labels)
        device = qml.device("default.qubit", wires=labels, seed=seed)
        tape = qml.tape.QuantumScript(
            operations, [qml.sample(wires=labels)], shots=shots
        )
        samples = np.asarray(qml.execute([tape], device)[0]).reshape(shots, -1)
        counts: dict[int, int] = {}
        for row in samples:
            x = sum(int(bit) << i for i, bit in enumerate(row))
            counts[x] = counts.get(x, 0) + 1
        return counts

    def parameters(self, t: Translated) -> list[float]:
        return pennylane_parameters(self._operations(t))

    @staticmethod
    def _operations(t: Translated) -> list[Any]:
        import pennylane as qml

        with qml.queuing.AnnotatedQueue() as queue:
            t.program()
        return list(qml.tape.QuantumScript.from_queue(queue).operations)


def pennylane_parameters(operations: Any) -> list[float]:
    return [float(p) for op in operations for p in op.parameters if _is_number(p)]


class _QSharp:
    # dump_operation prints about six decimals per entry (see QSharpAdapter).
    precision = 1e-6

    def unitary(self, t: Translated) -> np.ndarray:
        from qdk import qsharp

        n = len(t.labels)
        native = np.asarray(qsharp.dump_operation(t.program, n), dtype=complex)
        return _little_endian_from_msb(native, n)

    def controlled_unitary(self, t: Translated) -> np.ndarray:
        from qdk import qsharp

        n = len(t.labels)
        wrapper = f"{t.program}Controlled"
        qsharp.eval(
            f"operation {wrapper}(qs : Qubit[]) : Unit {{ "
            f"Controlled {t.program}([qs[{n}]], qs[0..{n - 1}]); }}"
        )
        native = np.asarray(qsharp.dump_operation(wrapper, n + 1), dtype=complex)
        return _little_endian_from_msb(native, n + 1)

    def sample(self, t: Translated, shots: int, seed: int) -> dict[int, int]:
        from qdk import qsharp

        from aeonq.adapters.qsharp import qdk_seed

        n = len(t.labels)
        rows = qsharp.run(
            f"{{ use qs = Qubit[{n}]; {t.program}(qs); MResetEachZ(qs) }}",
            shots=shots,
            seed=qdk_seed(seed),
        )
        counts: dict[int, int] = {}
        for row in rows:
            x = sum((str(bit) == "One") << i for i, bit in enumerate(row))
            counts[x] = counts.get(x, 0) + 1
        return counts

    def parameters(self, t: Translated) -> list[float]:
        # The QDK exposes no gate list for an imported operation; the angles
        # are read from the OpenQASM text it was given.
        return qasm_parameters(t.intermediate)


TARGETS: dict[str, Any] = {
    "qiskit": _Qiskit(),
    "cirq": _Cirq(),
    "pennylane": _PennyLane(),
    "qsharp": _QSharp(),
}


# ------------------------------------------------------------- parameters


def _is_number(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


_ALLOWED_OPERATORS = {Add: np.add, Sub: np.subtract, Mult: np.multiply, Div: np.divide}


def _evaluate(node: Any) -> float:
    if isinstance(node, Expression):
        return _evaluate(node.body)
    if isinstance(node, Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, Name) and node.id in ("pi", "π"):
        return pi
    if isinstance(node, UnaryOp) and isinstance(node.op, (USub, UAdd)):
        value = _evaluate(node.operand)
        return -value if isinstance(node.op, USub) else value
    if isinstance(node, BinOp):
        if isinstance(node.op, Pow):
            return float(_evaluate(node.left) ** _evaluate(node.right))
        operation = _ALLOWED_OPERATORS.get(type(node.op))
        if operation is not None:
            return float(operation(_evaluate(node.left), _evaluate(node.right)))
    raise ValueError("not a numeric OpenQASM expression")


_ARGUMENTS = re.compile(
    r"^\s*(?!gate\b)[A-Za-z_][\w]*\s*\(([^()]*(?:\([^()]*\))*[^()]*)\)"
)


def qasm_parameters(text: str) -> list[float]:
    """Numeric gate arguments in OpenQASM text, in order of appearance."""
    found: list[float] = []
    for line in text.replace("{", "\n").replace(";", ";\n").splitlines():
        match = _ARGUMENTS.match(line)
        if not match:
            continue
        for argument in match.group(1).split(","):
            try:
                found.append(_evaluate(parse(argument.strip(), mode="eval")))
            except (SyntaxError, ValueError):
                continue
    return found


def source_parameters(sdk: str, program: Any) -> list[float]:
    if sdk == "qiskit":
        return qiskit_parameters(program)
    if sdk == "cirq":
        return cirq_parameters(program)
    return pennylane_parameters(program.operations)


def data_angles(angles: list[float]) -> list[float]:
    """Angles that are not multiples of pi/4, i.e. carry problem data."""
    out = []
    for angle in angles:
        remainder = abs(angle) % _STRUCTURAL_ANGLE
        if min(remainder, _STRUCTURAL_ANGLE - remainder) > 1e-9:
            out.append(angle)
    return out


def angle_survival(
    source: list[float], translated: list[float]
) -> tuple[int, float | None, int]:
    """(data angles checked, worst residual, angles not found within 1e-6).

    Each data-carrying source angle is matched with the nearest angle anywhere
    in the translated program, modulo 4*pi (the period of a rotation gate
    including its phase). An angle that is not found was re-expressed by a
    decomposition or rebase. Angles are compared as stored, so a converter
    that keeps a number but changes its convention passes here and fails only
    the operator checks.
    """
    wanted = data_angles(source)
    if not wanted:
        return 0, None, 0
    if not translated:
        return len(wanted), None, len(wanted)
    have = np.asarray(translated)
    period = 4 * pi
    residuals = [
        float(np.min(np.abs((have - angle + period / 2) % period - period / 2)))
        for angle in wanted
    ]
    return (
        len(wanted),
        max(residuals),
        sum(r > _ANGLE_SURVIVES for r in residuals),
    )


# ------------------------------------------------------------ comparisons


def _phase_alignment(reference: np.ndarray, other: np.ndarray) -> tuple[float, float]:
    """(max entry error after the best global phase, that phase in radians)."""
    overlap = np.vdot(reference, other)
    phase = float(np.angle(overlap)) if abs(overlap) > 0 else 0.0
    error = float(np.max(np.abs(other - np.exp(1j * phase) * reference)))
    return error, phase


def _permute(unitary: np.ndarray, order: tuple[int, ...]) -> np.ndarray:
    """Relabel so that logical qubit i of the result is qubit order[i]."""
    n = len(order)
    index = np.array(
        [sum(((x >> i) & 1) << order[i] for i in range(n)) for x in range(2**n)]
    )
    return unitary[np.ix_(index, index)]


def qubit_order(reference: np.ndarray, other: np.ndarray, tolerance: float) -> str:
    n = int(np.log2(reference.shape[0]))
    identity = tuple(range(n))
    for order in permutations(range(n)):
        if _phase_alignment(reference, _permute(other, order))[0] <= tolerance:
            if order == identity:
                return "identity"
            if order == tuple(reversed(identity)):
                return "reversed"
            return "permuted" + "".join(map(str, order))
    return "none"


CANONICAL_EVALUATION_QUBITS = 5


def canonical_qae_estimate(
    q: np.ndarray, state: np.ndarray, m: int = CANONICAL_EVALUATION_QUBITS
) -> float:
    """Most likely canonical-QAE estimate sin^2(pi y / 2^m) for ``state``.

    Ideal phase estimation on controlled powers of Q reads the eigenphases of
    Q, global phase included. The outcome distribution is computed exactly
    from an orthonormal eigenbasis (complex Schur form of the unitary).
    """
    from scipy.linalg import schur

    triangular, basis = schur(q, output="complex")
    phases = np.angle(np.diag(triangular)) / (2 * np.pi)
    weights = np.abs(basis.conj().T @ state) ** 2
    size = 2**m
    k = np.arange(size)
    offsets = phases[:, None] - np.arange(size)[None, :] / size
    kernel = np.abs(np.exp(2j * np.pi * offsets[:, :, None] * k).mean(axis=2)) ** 2
    y = int(np.argmax(weights @ kernel))
    return float(np.sin(np.pi * y / size) ** 2)


def _controlled(unitary: np.ndarray) -> np.ndarray:
    """|0><0| (x) I + |1><1| (x) U with the control as the most significant bit."""
    dim = unitary.shape[0]
    out = np.eye(2 * dim, dtype=complex)
    out[dim:, dim:] = unitary
    return out


# ------------------------------------------------------------------ study


def study_nodes(problem: AEProblem) -> dict[str, Operator]:
    """The operators an AE implementation is built from, plus measured circuits."""
    q = grover_for(problem)
    n = q.n_qubits
    nodes: dict[str, Operator] = {
        "A": problem.state_preparation,
        "A_adjoint": Adjoint(problem.state_preparation, n),
        "S0": q.zero_reflection,
        "Schi": q.objective_reflection,
        "Q": q,
        "Q_power_2": GroverPower(q, 2, n),
        "controlled_Q": ControlledOperator(q, (n,), (1,), tuple(range(n)), n + 1),
    }
    for power in (0, 1, 2):
        nodes[f"A_Q^{power}"] = Sequence(
            (problem.state_preparation, GroverPower(q, power, n)), n
        )
    return nodes


def study_cases() -> list[tuple[str, dict[str, Operator]]]:
    cases = [(name, study_nodes(problem)) for name, problem in battery_cases()]
    cases.append(
        (
            "phase",
            {
                "controlled_global_phase": ControlledOperator(
                    GlobalPhase(pi / 3, 1), (1,), (1,), (0,), 2
                )
            },
        )
    )
    return cases


def _native_unitary(sdk: str, program: Any) -> np.ndarray:
    if sdk != "qiskit":
        raise ValueError("native sources are implemented for Qiskit only")
    from qiskit.quantum_info import Operator as QiskitOperator

    return np.asarray(QiskitOperator(program).data)


def native_qiskit_sources() -> list[tuple[str, str, Any, Operator]]:
    """Grover operators as a Qiskit user writes them, without aeonq.

    (case, program name, Qiskit circuit, IR node it must equal). Q comes from
    Qiskit's ``grover_operator`` with hand-written A and oracle and, where the
    objective is one qubit, from ``qiskit-algorithms``' ``EstimationProblem``.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import grover_operator

    from aeonq.oracles.cases import CASES
    from aeonq.oracles.handwritten import qiskit_objective, qiskit_preparation

    battery = dict(battery_cases())
    objective_qubit = {"bernoulli_0.25": 0, "lognormal_tail_n2_k2_flag": 2}
    sources: list[tuple[str, str, Any, Operator]] = []
    for make in CASES:
        case = make()
        n = case.n_qubits
        nodes = study_nodes(battery[case.name])
        prep = qiskit_preparation(case)
        q = grover_operator(qiskit_objective(case), prep)
        controlled = QuantumCircuit(n + 1)
        controlled.append(q.to_gate().control(1), [n, *range(n)])
        measured = prep.compose(q)
        sources += [
            (case.name, "Q (grover_operator)", q, nodes["Q"]),
            (case.name, "Q^2 (grover_operator)", q.compose(q), nodes["Q_power_2"]),
            (
                case.name,
                "controlled Q (grover_operator)",
                controlled,
                nodes["controlled_Q"],
            ),
            (case.name, "A Q (grover_operator)", measured, nodes["A_Q^1"]),
        ]
        if case.name in objective_qubit:
            try:
                from qiskit_algorithms import EstimationProblem
            except ImportError:
                continue
            problem = EstimationProblem(prep, [objective_qubit[case.name]])
            sources.append(
                (
                    case.name,
                    "Q (qiskit-algorithms)",
                    problem.grover_operator,
                    nodes["Q"],
                )
            )
    return sources


def _failure(row: dict[str, Any], status: str, exc: BaseException) -> dict[str, Any]:
    message = " ".join(str(exc).split())
    row.update(status=status, error=f"{type(exc).__name__}: {message[:240]}")
    return row


def evaluate_translation(
    path: TranslationPath,
    case: str,
    node_name: str,
    node: Operator,
    *,
    seed: int,
    shots: int = COUNT_SHOTS,
    prepared_state: np.ndarray | None = None,
    source_program: Any = None,
) -> dict[str, Any]:
    """Translate one program along ``path`` and compare it with its source.

    The source is the adapter's lowering of ``node`` unless ``source_program``
    gives a native program written without aeonq (``node`` then only supplies
    the reference semantics it must match). For a Grover operator,
    ``prepared_state`` (A|0...0>) adds the canonical QAE estimate that the
    source and the translated operator would yield.
    """
    n = node.n_qubits
    row: dict[str, Any] = {
        "case": case,
        "node": node_name,
        "qubits": n,
        "path": path.name,
        "source": path.source,
        "intermediate": path.intermediate,
        "target": path.target,
        "note": path.note,
        "status": "evaluated",
        "error": "",
    }
    missing = missing_modules(path)
    if missing:
        row.update(status="unavailable", error="missing " + ", ".join(missing))
        return row
    reference = matrix(node)
    try:
        if source_program is None:
            program, source_unitary = _source_program(path.source, node)
        else:
            program = source_program
            source_unitary = _native_unitary(path.source, program)
    except Exception as exc:
        return _failure(row, "source_failed", exc)
    row["source_vs_ir_max_error"] = float(np.max(np.abs(source_unitary - reference)))
    try:
        intermediate = path.export(program, n)
    except Exception as exc:
        return _failure(row, "export_failed", exc)
    try:
        translated = path.load(intermediate, n)
    except Exception as exc:
        return _failure(row, "import_failed", exc)
    target = TARGETS[path.target]
    tolerance = target.precision
    row["tolerance"] = tolerance
    try:
        unitary = target.unitary(translated)
    except Exception as exc:
        return _failure(row, "unitary_failed", exc)
    if unitary.shape != source_unitary.shape:
        row.update(status="unitary_failed", error=f"shape {unitary.shape}")
        return row

    exact_error = float(np.max(np.abs(unitary - source_unitary)))
    aligned_error, phase = _phase_alignment(source_unitary, unitary)
    row.update(
        max_abs_error=exact_error,
        equal_exact=exact_error <= tolerance,
        phase_aligned_error=aligned_error,
        equal_up_to_global_phase=aligned_error <= tolerance,
        global_phase_offset=phase if aligned_error <= tolerance else None,
        qubit_order=qubit_order(source_unitary, unitary, tolerance),
    )

    try:
        controlled = target.controlled_unitary(translated)
        controlled_error = float(
            np.max(np.abs(controlled - _controlled(source_unitary)))
        )
        row.update(
            controlled_max_abs_error=controlled_error,
            equal_when_controlled=controlled_error <= tolerance,
        )
    except Exception as exc:
        message = " ".join(str(exc).split())
        row.update(
            equal_when_controlled=None,
            controlled_error=f"{type(exc).__name__}: {message[:160]}",
        )

    try:
        checked, residual, reexpressed = angle_survival(
            source_parameters(path.source, program), target.parameters(translated)
        )
        row.update(
            data_angles_checked=checked,
            data_angle_max_residual=residual,
            data_angles_reexpressed=reexpressed,
        )
    except Exception as exc:
        row["parameter_error"] = f"{type(exc).__name__}: {exc}"[:160]

    expected = np.abs(source_unitary[:, 0]) ** 2
    row["distribution_tvd"] = float(
        0.5 * np.sum(np.abs(np.abs(unitary[:, 0]) ** 2 - expected))
    )
    try:
        counts = target.sample(translated, shots, seed)
        p_value, detail = _goodness_of_fit(counts, expected)
        row.update(
            count_shots=sum(counts.values()),
            count_p_value=p_value,
            count_pass=sum(counts.values()) == shots
            and p_value >= SAMPLING_P_THRESHOLD,
            count_detail=detail,
        )
    except Exception as exc:
        row.update(count_pass=None, count_detail=f"{type(exc).__name__}: {exc}"[:160])

    if prepared_state is not None:
        row["canonical_qae_source"] = canonical_qae_estimate(
            source_unitary, prepared_state
        )
        row["canonical_qae_translated"] = canonical_qae_estimate(
            unitary, prepared_state
        )

    operator_pass = row["equal_exact"] and row.get("equal_when_controlled") is not False
    row["operator_pass"] = operator_pass
    row["count_invisible_failure"] = bool(row.get("count_pass")) and not operator_pass
    return row


# ------------------------------------------------- importer gate semantics


def _rotation(axis: str, theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    if axis == "x":
        return np.array([[c, -1j * s], [-1j * s, c]])
    if axis == "y":
        return np.array([[c, -s], [s, c]], complex)
    return np.diag([np.exp(-1j * theta / 2), np.exp(1j * theta / 2)])


def _controlled_on_q0(u: np.ndarray) -> np.ndarray:
    """Two-qubit little-endian matrix: control q0, target q1."""
    out = np.eye(4, dtype=complex)
    # Indices with q0 = 1 are 1 (q1 = 0) and 3 (q1 = 1).
    out[np.ix_([1, 3], [1, 3])] = u
    return out


def gate_probe_cases() -> list[tuple[str, int, np.ndarray, bool]]:
    """(statement, qubits, matrix from the gate's textbook formula, QASM 3 only).

    Rotation angles outside [0, 2*pi) separate implementations that keep the
    real-valued angle from ones that reduce it modulo 2*pi; the reduction
    multiplies Rx, Ry and Rz by -1.
    """
    h = np.array([[1, 1], [1, -1]], complex) / np.sqrt(2)
    cases: list[tuple[str, int, np.ndarray, bool]] = [
        ("h q[0];", 1, h, False),
        ("s q[0];", 1, np.diag([1, 1j]), False),
        ("t q[0];", 1, np.diag([1, np.exp(1j * pi / 4)]), False),
        ("cx q[0], q[1];", 2, _controlled_on_q0(np.array([[0, 1], [1, 0]])), False),
        ("cz q[0], q[1];", 2, _controlled_on_q0(np.diag([1, -1])), False),
    ]
    for theta, text in ((0.7, "0.7"), (-0.7, "-0.7"), (7.0, "7.0")):
        for axis in "xyz":
            cases.append((f"r{axis}({text}) q[0];", 1, _rotation(axis, theta), False))
        cases.append(
            (
                f"cry({text}) q[0], q[1];",
                2,
                _controlled_on_q0(_rotation("y", theta)),
                False,
            )
        )
        cases.append(
            (
                f"crz({text}) q[0], q[1];",
                2,
                _controlled_on_q0(_rotation("z", theta)),
                False,
            )
        )
    phase = np.diag([1, np.exp(0.7j)])
    cases += [
        ("p(0.7) q[0];", 1, phase, True),
        ("u1(0.7) q[0];", 1, phase, False),
        ("cp(0.7) q[0], q[1];", 2, _controlled_on_q0(phase), True),
        ("gphase(0.7);", 1, np.exp(0.7j) * np.eye(2), True),
    ]
    return cases


GATE_PROBE_IMPORTERS: dict[
    str, tuple[str, Callable[[str, int], Translated], tuple[str, ...]]
] = {
    "qiskit.qasm2.loads": ("2", _into_qiskit_qasm2, ("qiskit.qasm2",)),
    "qiskit.qasm3.loads": ("3", _into_qiskit_qasm3, ("qiskit_qasm3_import",)),
    "cirq circuit_from_qasm (2)": ("2", _into_cirq_qasm, ("cirq.contrib.qasm_import",)),
    "cirq circuit_from_qasm (3)": ("3", _into_cirq_qasm, ("cirq.contrib.qasm_import",)),
    "pennylane from_qasm3": (
        "3",
        _into_pennylane_qasm3_without_include,
        ("pennylane", "openqasm3"),
    ),
    "qdk import_openqasm (2)": ("2", _into_qsharp, ("qdk.openqasm",)),
    "qdk import_openqasm (3)": ("3", _into_qsharp, ("qdk.openqasm",)),
}


def importer_gate_probe() -> list[dict[str, Any]]:
    """Each importer's matrix for one-gate programs against the formula."""
    rows: list[dict[str, Any]] = []
    for importer, (qasm, load, requires) in GATE_PROBE_IMPORTERS.items():
        unavailable = [m for m in requires if not _importable(m)]
        for statement, n, expected, qasm3_only in gate_probe_cases():
            row: dict[str, Any] = {
                "importer": importer,
                "openqasm": qasm,
                "statement": statement,
            }
            if qasm == "2" and (qasm3_only or statement.startswith("gphase")):
                continue
            if unavailable:
                rows.append({**row, "status": "unavailable"})
                continue
            header = (
                f'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[{n}];\n'
                if qasm == "2"
                else f'OPENQASM 3.0;\ninclude "stdgates.inc";\nqubit[{n}] q;\n'
            )
            try:
                translated = load(header + statement + "\n", n)
                unitary = TARGETS[translated.sdk].unitary(translated)
            except Exception as exc:
                message = " ".join(str(exc).split())
                rows.append(
                    {
                        **row,
                        "status": "rejected",
                        "error": f"{type(exc).__name__}: {message[:160]}",
                    }
                )
                continue
            tolerance = TARGETS[translated.sdk].precision
            exact = float(np.max(np.abs(unitary - expected)))
            aligned, offset = _phase_alignment(expected, unitary)
            rows.append(
                {
                    **row,
                    "status": "imported",
                    "max_abs_error": exact,
                    "equal_exact": exact <= tolerance,
                    "equal_up_to_global_phase": aligned <= tolerance,
                    "global_phase_offset": offset if aligned <= tolerance else None,
                    "tolerance": tolerance,
                }
            )
    return rows


def _importable(module: str) -> bool:
    try:
        import_module(module)
    except Exception:
        return False
    return True
