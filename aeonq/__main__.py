"""Command line: required resources, circuit metrics, and single estimator runs.

    python -m aeonq resources --problem bernoulli:0.25 --algorithm mlqae --backend qiskit
    python -m aeonq circuits --problem lognormal:3:5:flag --backend cirq --powers 0,1,2,4
    python -m aeonq run --problem threshold:3:5 --algorithm iqae --backend reference

Problems: ``bernoulli:A``, ``threshold:N:K`` (uniform N-qubit state, x >= K),
``lognormal:N:INDEX[:flag]`` (discretized log-normal tail; ``flag`` computes
[x >= K] into an extra objective qubit). Estimator options are passed as
``--set name=value`` with Python literal values, for example
``--set shots_per_power=512 --set powers=(0,1,2,4,8)``.
"""

import argparse
import ast
from importlib import import_module
import json
import sys
from typing import Any

from aeonq.problem import AEProblem

BACKENDS = {
    "reference": ("aeonq.adapters.reference", "ReferenceAdapter"),
    "qiskit": ("aeonq.adapters.qiskit", "QiskitAdapter"),
    "cirq": ("aeonq.adapters.cirq", "CirqAdapter"),
    "pennylane": ("aeonq.adapters.pennylane", "PennyLaneAdapter"),
    "qsharp": ("aeonq.adapters.qsharp", "QSharpAdapter"),
}
ALGORITHMS = ("iqae", "mlqae", "canonical_qae", "faster_ae")


def parse_problem(spec: str) -> AEProblem:
    from aeonq.benchmarks import problems

    kind, *fields = spec.split(":")
    try:
        if kind == "bernoulli" and len(fields) == 1:
            return problems.bernoulli(float(fields[0])).problem
        if kind == "threshold" and len(fields) == 2:
            return problems.independent_threshold(
                int(fields[0]), int(fields[1])
            ).problem
        if kind == "lognormal" and len(fields) in (2, 3):
            n, index = int(fields[0]), int(fields[1])
            if len(fields) == 3 and fields[2] != "flag":
                raise ValueError(f"unknown lognormal option {fields[2]!r}")
            return problems.lognormal_tail(
                n,
                problems.strike_for_index(n, index),
                objective_ancilla=len(fields) == 3,
            ).problem
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid problem {spec!r}: {exc}") from exc
    raise argparse.ArgumentTypeError(
        f"invalid problem {spec!r}; use bernoulli:A, threshold:N:K or "
        "lognormal:N:INDEX[:flag]"
    )


def make_backend(name: str) -> Any:
    module, cls = BACKENDS[name]
    try:
        return getattr(import_module(module), cls)()
    except ImportError as exc:
        raise SystemExit(
            f"backend {name!r} needs its SDK: pip install -e '.[{name}]' ({exc})"
        ) from exc


def parse_options(pairs: list[str]) -> dict[str, Any]:
    options: dict[str, Any] = {}
    for pair in pairs:
        name, sep, text = pair.partition("=")
        if not sep or not name:
            raise SystemExit(f"--set expects name=value, got {pair!r}")
        try:
            options[name] = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            options[name] = text
    return options


def _json(value: Any) -> str:
    return json.dumps(value, indent=2, default=str)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m aeonq", description=__doc__.split("\n")[0]
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def common(sub: argparse.ArgumentParser, algorithm: bool) -> None:
        sub.add_argument("--problem", required=True, type=parse_problem)
        sub.add_argument("--backend", choices=BACKENDS, default="reference")
        if algorithm:
            sub.add_argument("--algorithm", choices=ALGORITHMS, required=True)
            sub.add_argument("--epsilon", type=float, default=0.01)
            sub.add_argument("--alpha", type=float, default=0.05)
            sub.add_argument("--set", action="append", default=[], metavar="NAME=VALUE")
        sub.add_argument("--json", action="store_true", help="machine-readable output")

    common(
        commands.add_parser("resources", help="budget and circuits, without running"),
        algorithm=True,
    )
    circuits = commands.add_parser("circuits", help="compiled metrics of A Q^k")
    common(circuits, algorithm=False)
    circuits.add_argument("--powers", default="0,1,2,4,8", help="comma-separated k")
    common(commands.add_parser("run", help="run once and print the result"), True)

    arguments = parser.parse_args(argv)
    backend = make_backend(arguments.backend)

    if arguments.command == "circuits":
        from aeonq.resources import circuit_metrics, format_circuit_metrics

        powers = tuple(int(k) for k in arguments.powers.split(","))
        table = circuit_metrics(arguments.problem, backend, powers)
        print(
            _json([c.__dict__ for c in table])
            if arguments.json
            else format_circuit_metrics(table)
        )
        return 0

    options = parse_options(arguments.set)
    if arguments.command == "resources":
        from aeonq.resources import estimate_resources

        report = estimate_resources(
            arguments.problem,
            algorithm=arguments.algorithm,
            backend=backend,
            epsilon=arguments.epsilon,
            alpha=arguments.alpha,
            **options,
        )
        print(_json(report.as_dict()) if arguments.json else report.summary())
        return 0

    from aeonq import estimate

    result = estimate(
        arguments.problem,
        algorithm=arguments.algorithm,
        backend=backend,
        epsilon=arguments.epsilon,
        alpha=arguments.alpha,
        **options,
    )
    print(_json(result.__dict__) if arguments.json else result.summary())
    return 0


if __name__ == "__main__":
    sys.exit(main())
