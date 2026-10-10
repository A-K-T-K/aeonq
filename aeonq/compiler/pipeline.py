"""Compare available native compile configurations after exact IR rewrites."""

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from aeonq.adapters.base import BackendAdapter
from aeonq.compiler.resource_model import CompilerCost
from aeonq.compiler.rewrite import RewriteRecord, rewrite
from aeonq.ir.nodes import Operator


@dataclass(frozen=True)
class CompiledProgram:
    semantic_ir: Operator
    native_program: Any
    compiled_program: Any
    rewrite_records: tuple[RewriteRecord, ...]
    options: dict[str, Any]
    metrics: dict[str, int | float | None]
    selected_cost: float
    candidate_costs: tuple[tuple[dict[str, Any], float], ...]
    compile_time: float


@dataclass(frozen=True)
class AECompiler:
    cost: CompilerCost = CompilerCost()
    qiskit_optimization_levels: tuple[int, ...] = (0, 1, 2, 3)

    def compile(self, ir: Operator, backend: BackendAdapter) -> CompiledProgram:
        start = perf_counter()
        rewritten, records = rewrite(ir)
        backend.validate(rewritten)
        native = backend.lower(rewritten)
        if type(backend).__name__ == "QiskitAdapter":
            options = tuple(
                {
                    "optimization_level": level,
                    "seed_transpiler": 0,
                    "basis_gates": ["u", "cx"],
                }
                for level in self.qiskit_optimization_levels
            )
        else:
            options = ({},)
        if not options:
            raise ValueError("no compiler configurations available")
        candidate_costs: list[tuple[dict[str, Any], float]] = []
        best: (
            tuple[float, Any, dict[str, Any], dict[str, int | float | None]] | None
        ) = None
        for option in options:
            compiled = backend.compile(native, option)
            metrics = backend.resource_metrics(compiled)
            cost = self.cost.evaluate(metrics, rewritten.n_qubits)
            candidate_costs.append((option, cost))
            if best is None or cost < best[0]:
                best = (cost, compiled, option, metrics)
        assert best is not None
        return CompiledProgram(
            rewritten,
            native,
            best[1],
            records,
            best[2],
            best[3],
            best[0],
            tuple(candidate_costs),
            perf_counter() - start,
        )
