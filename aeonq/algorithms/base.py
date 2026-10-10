"""Shared validated execution path for semantic Grover experiments."""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import numpy as np

from aeonq.adapters.base import BackendAdapter
from aeonq.ir.nodes import GroverPower, Sequence
from aeonq.ir.registers import RegisterMap
from aeonq.problem import AEProblem
from aeonq.reference import grover_for


@dataclass
class Experiment:
    problem: AEProblem
    backend: BackendAdapter
    compile_time: float = 0.0
    execution_time: float = 0.0
    evaluations: int = 0
    oracle_queries: int = 0
    shots: int = 0
    resource_metrics: dict[str, Any] | None = None
    _compiled_by_power: dict[int, Any] = field(
        default_factory=dict, init=False, repr=False
    )
    _metrics_by_power: dict[int, dict[str, Any]] = field(
        default_factory=dict, init=False, repr=False
    )

    def _compiled(self, power: int) -> tuple[Any, dict[str, Any]]:
        """Compile Q^power once; later calls reuse the program and metrics."""
        if power not in self._compiled_by_power:
            q = grover_for(self.problem)
            circuit = Sequence(
                (self.problem.state_preparation, GroverPower(q, power, q.n_qubits)),
                q.n_qubits,
            )
            start = perf_counter()
            native = self.backend.lower(circuit)
            compiled = self.backend.compile(native)
            metrics = self.backend.resource_metrics(compiled)
            self.compile_time += perf_counter() - start
            self._compiled_by_power[power] = compiled
            self._metrics_by_power[power] = metrics
        return self._compiled_by_power[power], self._metrics_by_power[power]

    def metrics(self, power: int) -> dict[str, Any]:
        """Compiled resource metrics for Q^power, without executing it."""
        if power < 0:
            raise ValueError("power must be nonnegative")
        return self._compiled(power)[1]

    def sample(self, power: int, shots: int, seed: int | None = None) -> int:
        if power < 0 or shots < 1:
            raise ValueError("power and shots must be nonnegative and positive")
        n = self.problem.state_preparation.n_qubits
        compiled, metrics = self._compiled(power)
        start = perf_counter()
        raw = self.backend.execute(compiled, shots, seed=seed)
        self.execution_time += perf_counter() - start
        mapping = RegisterMap(tuple(range(n)), tuple(range(n)))
        result = self.backend.normalize_result(raw, mapping)
        if result.shots != shots:
            raise ValueError("backend returned wrong shot count")
        indices: list[int] = []
        weights: list[int] = []
        for bits, count in result.counts.items():
            if len(bits) != n:
                raise ValueError("backend returned wrong logical width")
            indices.append(sum(bit << i for i, bit in enumerate(bits)))
            weights.append(count)
        good = self.problem.objective.evaluate_many(
            np.asarray(indices, dtype=np.int64), n
        )
        successes = int(np.asarray(weights, dtype=np.int64)[good].sum())
        self.evaluations += 1
        self.shots += shots
        previous = self.resource_metrics or {}
        aggregate = dict(metrics)
        for key, output_key in (
            ("depth", "total_compiled_depth"),
            ("two_qubit_gates", "total_two_qubit_gates"),
        ):
            prior = previous.get(output_key, 0)
            current = metrics.get(key)
            aggregate[output_key] = (
                None if prior is None or current is None else int(prior) + int(current)
            )
        self.resource_metrics = aggregate
        # State-preparation oracle convention: Q^k A uses 2k+1 calls to A/A†.
        self.oracle_queries += shots * (2 * power + 1)
        return successes


NON_IDEAL_SAMPLING_WARNING = (
    "backend does not sample ideally (for example a noise model or hardware); "
    "the reported interval assumes ideal independent Bernoulli shots and its "
    "coverage is not guaranteed"
)


def qualified_guarantee(
    backend: BackendAdapter, status: str
) -> tuple[str, tuple[str, ...]]:
    """Withhold a theoretical guarantee whose sampling assumption is violated."""
    if backend.capabilities().ideal_sampling:
        return status, ()
    if status == "THEORETICAL_GUARANTEE_SATISFIED":
        status = "THEORETICAL_GUARANTEE_NOT_APPLICABLE"
    return status, (NON_IDEAL_SAMPLING_WARNING,)


class CompiledMetrics(Mapping[int, dict[str, Any]]):
    """Per-power metrics for schedulers; looking up a power compiles it on demand."""

    def __init__(self, experiment: Experiment) -> None:
        self._experiment = experiment

    def __getitem__(self, power: int) -> dict[str, Any]:
        if not isinstance(power, int) or power < 0:
            raise KeyError(power)
        return self._experiment.metrics(power)

    def __iter__(self) -> Iterator[int]:
        return iter(tuple(self._experiment._metrics_by_power))

    def __len__(self) -> int:
        return len(self._experiment._metrics_by_power)
