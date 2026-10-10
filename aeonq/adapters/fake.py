"""Local fault-injection adapter for runtime resilience tests."""

from collections.abc import Iterable
import time
from typing import Any

from aeonq.adapters.base import BackendAdapter, BackendCapabilities, NormalizedResult
from aeonq.ir.nodes import Operator
from aeonq.ir.registers import RegisterMap


class BackendQuotaError(RuntimeError):
    pass


class UnsupportedCircuitError(RuntimeError):
    pass


class FakeBackendAdapter(BackendAdapter):
    def __init__(
        self, faults: Iterable[str] = (), *, delay_seconds: float = 0.1
    ) -> None:
        self.faults = list(faults)
        self.delay_seconds = delay_seconds
        self.calls = 0

    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(
            False,
            True,
            False,
            False,
            False,
            False,
            False,
            False,
            20,
            None,
            metadata={"fake": "1"},
        )

    def validate(self, ir: Operator) -> None:
        if ir.n_qubits > 20:
            raise UnsupportedCircuitError("qubit limit exceeded")

    def lower(self, ir: Operator) -> Operator:
        self.validate(ir)
        return ir

    def compile(
        self, native_program: Operator, options: dict[str, Any] | None = None
    ) -> Operator:
        if self.faults and self.faults[0] == "compilation_failure":
            self.faults.pop(0)
            raise RuntimeError("injected compilation failure")
        return native_program

    def execute(
        self, compiled_program: Operator, shots: int, *, seed: int | None = None
    ) -> Any:
        self.calls += 1
        fault = self.faults.pop(0) if self.faults else "none"
        if fault == "timeout":
            time.sleep(self.delay_seconds)
        elif fault == "delayed_result":
            time.sleep(self.delay_seconds)
        elif fault == "dropped_connection":
            raise ConnectionError("injected dropped connection")
        elif fault == "job_cancellation":
            raise RuntimeError("injected job cancellation")
        elif fault == "interrupted_job":
            raise InterruptedError("injected interrupted job")
        elif fault == "backend_quota":
            raise BackendQuotaError("injected quota rejection")
        elif fault == "unsupported_circuit":
            raise UnsupportedCircuitError("injected unsupported circuit")
        elif fault == "malformed_result":
            return {"2": shots}
        elif fault in ("partial_result", "missing_shots"):
            return {"0": max(0, shots - 1)}
        elif fault not in ("none", "timeout", "delayed_result"):
            raise ValueError(f"unknown fault mode: {fault}")
        return {"0": shots}

    def normalize_result(self, raw: Any, register_map: RegisterMap) -> NormalizedResult:
        if not isinstance(raw, dict):
            raise ValueError("fake result must be a count mapping")
        counts: dict[tuple[int, ...], int] = {}
        for bitstring, count in raw.items():
            if not isinstance(count, int) or count < 0:
                raise ValueError("invalid count")
            bits = register_map.normalize_bitstring(bitstring)
            counts[bits] = counts.get(bits, 0) + count
        return NormalizedResult(counts, sum(counts.values()))

    def resource_metrics(
        self, compiled_program: Operator
    ) -> dict[str, int | float | None]:
        return {
            "qubits": compiled_program.n_qubits,
            "depth": None,
            "one_qubit_gates": None,
            "two_qubit_gates": None,
            "t_count": None,
        }
