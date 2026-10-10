"""Size, protocol surface and conformance results of each backend adapter.

For each adapter module this records its size (non-blank, non-comment source
lines), the abstract methods it had to implement, the optional capabilities
it adds, and its result on the shared conformance battery, which runs
unchanged against every adapter.
"""

import ast
from collections import Counter
import csv
import inspect
from pathlib import Path

from aeonq.adapters.base import BackendAdapter
from aeonq.adapters.cirq import CirqAdapter
from aeonq.adapters.pennylane import PennyLaneAdapter
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.qsharp import QSharpAdapter
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.conformance.battery import run_battery

OPTIONAL = ("unitary", "statevector", "execute_async")


def source_lines(path: Path) -> int:
    """Physical lines holding code, excluding blanks, comments and docstrings."""
    text = path.read_text(encoding="utf-8")
    docstrings = set()
    for node in ast.walk(ast.parse(text)):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            body = node.body
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
            ):
                docstrings.update(range(body[0].lineno, body[0].end_lineno + 1))
    return sum(
        1
        for number, line in enumerate(text.splitlines(), start=1)
        if line.strip()
        and not line.strip().startswith("#")
        and number not in docstrings
    )


def run(output: Path) -> list[dict]:
    required = sorted(BackendAdapter.__abstractmethods__)
    rows = []
    for name, cls in (
        ("reference", ReferenceAdapter),
        ("qiskit", QiskitAdapter),
        ("cirq", CirqAdapter),
        ("pennylane", PennyLaneAdapter),
        ("qsharp", QSharpAdapter),
    ):
        results = run_battery(cls())
        outcome = Counter((r.level, r.passed) for r in results)
        rows.append(
            {
                "adapter": name,
                "module": Path(inspect.getsourcefile(cls))
                .relative_to(Path(__file__).resolve().parents[1])
                .as_posix(),
                "source_lines": source_lines(Path(inspect.getsourcefile(cls))),
                "required_methods": len(required),
                "required_method_names": " ".join(required),
                "optional_methods_added": " ".join(
                    m
                    for m in OPTIONAL
                    if getattr(cls, m, None) is not None
                    and getattr(cls, m) is not getattr(BackendAdapter, m, None)
                ),
                "declared_unitary_precision": cls.unitary_precision,
                "battery_checks": len(results),
                "battery_passed": sum(r.passed is True for r in results),
                "battery_failed": sum(r.passed is False for r in results),
                "battery_unsupported": sum(r.passed is None for r in results),
                **{
                    f"{level}_passed": f"{outcome[(level, True)]}/{outcome[(level, True)] + outcome[(level, False)]}"
                    for level in ("operator", "measurement", "sampling")
                },
            }
        )
        print(rows[-1], flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == "__main__":
    run(Path("benchmarks/results/raw/extensibility.csv"))
