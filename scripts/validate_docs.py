"""Validate local Markdown links as the repository's static documentation build."""

from pathlib import Path
import re
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"(?<!!)\[[^]]+\]\(([^)]+)\)")


def validate() -> list[str]:
    errors: list[str] = []
    sources = [ROOT / "README.md"]
    sources += list((ROOT / "docs").glob("*.md"))
    sources += list((ROOT / "benchmarks").glob("*.md"))
    sources += list((ROOT / "examples").glob("*.md"))
    for source in sources:
        for target in LINK.findall(source.read_text(encoding="utf-8")):
            path_part = unquote(target.split("#", 1)[0])
            if not path_part or "://" in path_part or path_part.startswith("mailto:"):
                continue
            if not (source.parent / path_part).exists():
                errors.append(f"{source.relative_to(ROOT)}: missing {target}")
    return errors


if __name__ == "__main__":
    missing = validate()
    if missing:
        raise SystemExit("\n".join(missing))
    print("Local Markdown links valid")
