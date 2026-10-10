"""Versioned atomic iteration checkpoints with strict identity checks."""

from hashlib import sha256
import json
import os
from pathlib import Path
from uuid import uuid4
from typing import Any

from aeonq.ir.serialization import dumps
from aeonq.problem import AEProblem
from aeonq.adapters.base import BackendAdapter


CHECKPOINT_SCHEMA_VERSION = 1


class IncompatibleCheckpoint(ValueError):
    """A checkpoint belongs to a different problem, configuration or backend."""


def identity(
    problem: AEProblem,
    backend: BackendAdapter,
    algorithm: str,
    configuration: dict[str, Any],
) -> dict[str, str]:
    problem_hash = sha256(dumps(problem).encode("utf-8")).hexdigest()
    config_hash = sha256(
        json.dumps(configuration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    backend_id = f"{type(backend).__module__}.{type(backend).__name__}:" + json.dumps(
        backend.capabilities().metadata, sort_keys=True, default=str
    )
    return {
        "problem": problem_hash,
        "backend": backend_id,
        "algorithm": algorithm,
        "configuration": config_hash,
    }


def save_checkpoint(
    path: str | Path, identity_fields: dict[str, str], state: dict[str, Any]
) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "identity": identity_fields,
        "state": state,
    }
    encoded = json.dumps(
        document, allow_nan=False, sort_keys=True, separators=(",", ":")
    )
    temporary = target.with_name(target.name + f".{uuid4().hex}.tmp")
    try:
        temporary.write_text(encoded, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_checkpoint(
    path: str | Path, expected_identity: dict[str, str]
) -> dict[str, Any]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IncompatibleCheckpoint("checkpoint cannot be read or parsed") from exc
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != CHECKPOINT_SCHEMA_VERSION
    ):
        raise IncompatibleCheckpoint("unsupported checkpoint schema version")
    if document.get("identity") != expected_identity:
        raise IncompatibleCheckpoint(
            "checkpoint problem, backend, algorithm or configuration differs"
        )
    state = document.get("state")
    if not isinstance(state, dict):
        raise IncompatibleCheckpoint("checkpoint state is malformed")
    return state
