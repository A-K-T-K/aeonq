"""Deterministic condition-specific seeds and content-addressed source evidence."""

from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

SEED_SCHEME = "sha256-condition-v1"
ROOT = Path(__file__).resolve().parents[2]


def trial_seed(
    experiment: str,
    case: str,
    method: str,
    backend: str,
    trial: int,
    base_seed: int = 0,
    condition: object = None,
) -> int:
    """63-bit entropy keyed by the complete trial identity, independent of order.

    Estimators derive evaluation seeds from this seed, so distinct conditions
    never share a random stream.
    """
    if trial < 0:
        raise ValueError("trial must be nonnegative")
    identity = [
        SEED_SCHEME,
        base_seed,
        experiment,
        case,
        method,
        backend,
        trial,
        condition,
    ]
    digest = sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).digest()
    return int.from_bytes(digest[:8], "big") >> 1


@lru_cache(maxsize=1)
def source_snapshot() -> dict[str, str]:
    """Archive exact experiment code; data/output/editor files are excluded.

    The digest hashes a sorted manifest of paths and file-byte hashes, rather
    than ZIP metadata. The archive contains those exact bytes and manifest.
    Cached for the lifetime of the process.
    """
    paths = []
    for directory in ("aeonq", "benchmarks", "tests"):
        for path in (ROOT / directory).rglob("*.py"):
            if not {"__pycache__", "archives"}.intersection(path.parts):
                paths.append(path)
    paths.extend(
        ROOT / name
        for name in (
            "pyproject.toml",
            "requirements-lock.txt",
            "Dockerfile",
            "benchmarks/experiment_manifest.yaml",
        )
    )
    contents = {
        p.relative_to(ROOT).as_posix(): p.read_bytes() for p in sorted(set(paths))
    }
    manifest = {name: sha256(data).hexdigest() for name, data in contents.items()}
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    digest = sha256(encoded).hexdigest()
    relative = f"benchmarks/results/archives/source-{digest}.zip"
    destination = ROOT / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        # Independent workers can compute the same digest. Replace only a
        # complete ZIP; never expose a partial archive to readers.
        import os

        temporary = destination.with_suffix(f".{os.getpid()}.tmp")
        with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
            for name, data in contents.items():
                archive.writestr(name, data)
            archive.writestr("SOURCE_MANIFEST.json", encoded)
        temporary.replace(destination)
    return {
        "source_snapshot_sha256": digest,
        "source_archive": relative,
        "seed_scheme": SEED_SCHEME,
    }


def verify_source_archive(environment: dict) -> None:
    digest = environment.get("source_snapshot_sha256")
    relative = environment.get("source_archive")
    if not digest or relative != f"benchmarks/results/archives/source-{digest}.zip":
        raise ValueError("missing or invalid source provenance")
    with ZipFile(ROOT / relative) as archive:
        encoded = archive.read("SOURCE_MANIFEST.json")
        if sha256(encoded).hexdigest() != digest:
            raise ValueError("source manifest hash mismatch")
        for name, expected in json.loads(encoded).items():
            if sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"source archive mismatch: {name}")
