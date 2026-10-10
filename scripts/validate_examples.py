"""Run every documented example with the active Python environment."""

from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def validate() -> list[str]:
    failures = []
    for source in sorted((ROOT / "examples").glob("[0-9][0-9]_*.py")):
        result = subprocess.run(
            [sys.executable, str(source)],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            failures.append(f"{source.name}: {result.stderr.strip()}")
    return failures


if __name__ == "__main__":
    failed = validate()
    if failed:
        raise SystemExit("\n".join(failed))
    count = len(list((ROOT / "examples").glob("[0-9][0-9]_*.py")))
    print(f"All {count} executable examples passed")
