"""Resources an estimator will need, and compiled circuit metrics, before running.

Nothing here executes a circuit. Circuits are built and compiled through the
same ``Experiment`` path the estimators use, so the metrics are those of the
programs that would run. Fixed-schedule estimators (MLQAE, canonical QAE) have
exact budgets. Adaptive estimators (IQAE, Faster AE) choose powers from the
data, so their budgets are worst-case upper bounds over every path the
algorithm can take, and their circuit lists cover every power it may use.
"""

from dataclasses import asdict, dataclass, field
from math import pi
from typing import Any

from aeonq.adapters.base import BackendAdapter
from aeonq.algorithms.base import Experiment
from aeonq.problem import AEProblem
from aeonq.runtime.result import AEResult

# Column order for printed tables: (metric key, header).
METRIC_COLUMNS = (
    ("qubits", "qubits"),
    ("depth", "depth"),
    ("one_qubit_gates", "1q gates"),
    ("two_qubit_gates", "2q gates"),
    ("multi_qubit_gates", ">2q gates"),
    ("t_count", "T-count"),
    ("primitive_statements", "statements"),
)


@dataclass(frozen=True)
class CircuitResources:
    """Compiled metrics of one circuit; ``None`` means the SDK does not report it."""

    label: str
    power: int | None
    logical_qubits: int
    metrics: dict[str, Any]


@dataclass(frozen=True)
class ResourceEstimate:
    algorithm: str
    backend: str
    exact: bool
    shots: int
    oracle_queries: int
    circuit_evaluations: int
    max_power: int
    circuits: tuple[CircuitResources, ...]
    notes: tuple[str, ...] = field(default_factory=tuple)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        kind = "exact" if self.exact else "worst-case upper bound"
        lines = [
            f"Resources for {self.algorithm} on {self.backend} ({kind})",
            f"  shots               {self.shots:,}",
            f"  oracle queries      {self.oracle_queries:,}  (calls to A or A^dagger)",
            f"  circuit evaluations {self.circuit_evaluations:,}",
            f"  max Grover power    {self.max_power}",
            "",
            format_circuit_metrics(self.circuits),
        ]
        lines += [f"Note: {note}" for note in self.notes]
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


def _value(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:g}"
    return f"{value:,}" if isinstance(value, int) else str(value)


def format_circuit_metrics(circuits: tuple[CircuitResources, ...]) -> str:
    """A plain-text table of compiled circuit metrics, one row per circuit."""
    present = [
        (key, header)
        for key, header in METRIC_COLUMNS
        if any(key in c.metrics for c in circuits)
    ]
    header = ["circuit", "logical qubits", *(h for _, h in present)]
    rows = [
        [
            c.label,
            str(c.logical_qubits),
            *(_value(c.metrics.get(k)) for k, _ in present),
        ]
        for c in circuits
    ]
    widths = [max(len(row[i]) for row in [header, *rows]) for i in range(len(header))]

    def line(cells: list[str]) -> str:
        return "  ".join(
            cell.ljust(width) if i == 0 else cell.rjust(width)
            for i, (cell, width) in enumerate(zip(cells, widths))
        )

    rule = "  ".join("-" * width for width in widths)
    return "\n".join([line(header), rule, *(line(row) for row in rows)])


def _power_circuits(
    problem: AEProblem, backend: BackendAdapter, powers: list[int], prefix: str = ""
) -> tuple[CircuitResources, ...]:
    experiment = Experiment(problem, backend)
    n = problem.state_preparation.n_qubits
    return tuple(
        CircuitResources(
            f"{prefix}A Q^{k}" if k else f"{prefix}A",
            k,
            n,
            dict(experiment.metrics(k)),
        )
        for k in sorted(set(powers))
    )


def circuit_metrics(
    problem: AEProblem,
    backend: BackendAdapter,
    powers: tuple[int, ...] = (0, 1, 2, 4, 8),
) -> tuple[CircuitResources, ...]:
    """Compile A Q^k for each power and return its metrics, without executing."""
    if not powers or any(not isinstance(k, int) or k < 0 for k in powers):
        raise ValueError("powers must be nonnegative integers")
    return _power_circuits(problem, backend, list(powers))


def _iqae(solver: Any, problem: AEProblem, backend: BackendAdapter) -> ResourceEstimate:
    evaluations = solver.max_rounds * (solver.max_resamples_per_round + 1)
    round_shots = max(solver.shot_options or (solver.shots_per_round,))
    shots = evaluations * round_shots
    return ResourceEstimate(
        "iqae",
        type(backend).__name__,
        False,
        shots,
        shots * (2 * solver.max_power + 1),
        evaluations,
        solver.max_power,
        _power_circuits(problem, backend, [0, solver.max_power]),
        (
            "IQAE picks each round's power from the data; the bound assumes every "
            "round is resampled at the maximum power. Only the smallest and "
            "largest possible circuits are listed.",
        ),
    )


def _mlqae(
    solver: Any, problem: AEProblem, backend: BackendAdapter
) -> ResourceEstimate:
    powers = list(solver.powers)
    shots = solver.shots_per_power * len(powers)
    return ResourceEstimate(
        "mlqae",
        type(backend).__name__,
        True,
        shots,
        sum(solver.shots_per_power * (2 * k + 1) for k in powers),
        len(powers),
        max(powers),
        _power_circuits(problem, backend, powers),
    )


def _canonical(
    solver: Any, problem: AEProblem, backend: BackendAdapter
) -> ResourceEstimate:
    circuit = solver.circuit(problem)
    metrics = dict(backend.resource_metrics(backend.compile(backend.lower(circuit))))
    modulus = 2**solver.evaluation_qubits
    return ResourceEstimate(
        "canonical_qae",
        type(backend).__name__,
        True,
        solver.shots,
        solver.shots * (2 * (modulus - 1) + 1),
        1,
        2 ** (solver.evaluation_qubits - 1),
        (
            CircuitResources(
                f"QPE, {solver.evaluation_qubits} evaluation qubits",
                None,
                circuit.n_qubits,
                metrics,
            ),
        ),
    )


def _faster_ae(
    solver: Any, problem: AEProblem, backend: BackendAdapter
) -> ResourceEstimate:
    from aeonq.algorithms.faster_ae import THETA_MAX, attenuated_problem

    n_first, n_second = solver.required_shots
    iterations = solver.max_iterations
    worst = {"shots": 0, "queries": 0, "evaluations": 0}
    powers: set[int] = set()
    # The stage switch happens after some first-stage iteration j0 < L, or never;
    # follow every path through the same budget check the estimator applies.
    # The switch needs 2^(j0+1) * theta_upper >= 3*pi/8, and theta never exceeds
    # THETA_MAX, so earlier switch points are unreachable and are skipped.
    reachable = [
        j for j in range(1, iterations) if 2 ** (j + 1) * THETA_MAX >= 3 * pi / 8
    ]
    for switch in [None, *reachable]:
        shots = queries = evaluations = 0
        for j in range(1, iterations + 1):
            main = 2 ** (j - 1)
            first = switch is None or j <= switch
            if first:
                circuits = [(main, n_first)]
            else:
                assert switch is not None
                circuits = [(main, n_second), (main + 2 ** (switch - 1), n_second)]
            needed = sum(s for _, s in circuits)
            if shots + needed > solver.max_total_shots:
                break
            for power, count in circuits:
                powers.add(power)
                shots += count
                queries += count * (2 * power + 1)
                evaluations += 1
        worst["shots"] = max(worst["shots"], shots)
        worst["queries"] = max(worst["queries"], queries)
        worst["evaluations"] = max(worst["evaluations"], evaluations)
    adapted = attenuated_problem(problem)
    return ResourceEstimate(
        "faster_ae",
        type(backend).__name__,
        False,
        worst["shots"],
        worst["queries"],
        worst["evaluations"],
        max(powers, default=0),
        _power_circuits(adapted, backend, sorted(powers)),
        (
            "Faster AE stops early once it converges; the bound is the largest "
            "budget over every stage-switch path. Circuits include the extra "
            "attenuation qubit.",
        ),
    )


ESTIMATORS = {
    "iqae": _iqae,
    "mlqae": _mlqae,
    "canonical_qae": _canonical,
    "faster_ae": _faster_ae,
}


def estimate_resources(
    problem: AEProblem,
    *,
    algorithm: str,
    backend: BackendAdapter,
    epsilon: float = 0.01,
    alpha: float = 0.05,
    **options: Any,
) -> ResourceEstimate:
    """Shots, oracle queries and compiled circuits ``estimate()`` would use.

    Takes the same arguments as :func:`aeonq.estimate` and validates them the
    same way, but compiles circuits without executing any of them.
    """
    from aeonq.algorithms.canonical_qae import CanonicalQAE
    from aeonq.algorithms.faster_ae import FasterAE
    from aeonq.algorithms.iqae import IQAE
    from aeonq.algorithms.mlqae import MLQAE

    classes = {
        "iqae": IQAE,
        "mlqae": MLQAE,
        "canonical_qae": CanonicalQAE,
        "faster_ae": FasterAE,
    }
    if algorithm not in classes:
        raise ValueError(
            f"unsupported algorithm {algorithm!r}; available: {', '.join(classes)}"
        )
    solver = classes[algorithm](epsilon=epsilon, alpha=alpha, **options)
    return ESTIMATORS[algorithm](solver, problem, backend)


def result_summary(result: AEResult) -> str:
    """Estimate, interval, guarantee and the resources a finished run used."""
    low, high = result.confidence_or_credible_interval
    metrics = result.resource_metrics or {}
    lines = [
        f"Estimate {result.estimate:.6g}  interval [{low:.6g}, {high:.6g}]",
        f"  interval type       {result.interval_type}",
        f"  convergence         {result.convergence_status}",
        f"  guarantee           {result.guarantee_status}",
        f"  backend             {result.backend} ({result.sdk} {result.sdk_version})",
        "Resources used",
        f"  shots               {result.total_shots:,}",
        f"  oracle queries      {result.total_oracle_queries:,}  (calls to A or A^dagger)",
        f"  circuit evaluations {result.total_circuit_evaluations:,}",
        f"  powers used         {', '.join(map(str, sorted(set(result.powers_used))))}",
        f"  max Grover power    {result.max_power}",
        f"  compile / execute   {result.compile_time:.3g} s / "
        f"{result.execution_time:.3g} s (wall {result.wall_clock_time:.3g} s)",
    ]
    for key, label in (
        ("total_compiled_depth", "summed depth"),
        ("total_two_qubit_gates", "summed 2q gates"),
    ):
        if key in metrics:
            lines.append(f"  {label:<19} {_value(metrics[key])}")
    lines += [f"Warning: {warning}" for warning in result.warnings]
    return "\n".join(lines)
