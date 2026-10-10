"""Versioned JSON serialization of the supported semantic IR subset."""

from dataclasses import fields, is_dataclass
import json
from typing import Any

from aeonq.ir.nodes import (
    Adjoint,
    ControlledOperator,
    EmbeddedOperator,
    Gate,
    GlobalPhase,
    GroverOperator,
    GroverPower,
    Measurement,
    ObjectiveReflection,
    Parameter,
    Sequence,
    StatePreparation,
    ZeroReflection,
)
from aeonq.ir.predicates import ObjectivePredicate
from aeonq.ir.registers import RegisterMap
from aeonq.problem import AEProblem

SCHEMA_VERSION = 1
_KINDS = {
    cls.__name__: cls
    for cls in (
        Adjoint,
        ControlledOperator,
        EmbeddedOperator,
        Gate,
        GlobalPhase,
        GroverOperator,
        GroverPower,
        Measurement,
        ObjectiveReflection,
        Parameter,
        Sequence,
        StatePreparation,
        ZeroReflection,
        ObjectivePredicate,
        RegisterMap,
        AEProblem,
    )
}


def _encode(value: Any) -> Any:
    if is_dataclass(value):
        return {
            "__type__": type(value).__name__,
            **{f.name: _encode(getattr(value, f.name)) for f in fields(value)},
        }
    if isinstance(value, tuple):
        return [_encode(item) for item in value]
    if isinstance(value, dict):
        if any(not isinstance(key, str) or key == "__type__" for key in value):
            raise TypeError(
                "serialized metadata keys must be strings other than __type__"
            )
        return {key: _encode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode(item) for item in value]
    if isinstance(value, (int, float, str, type(None))):
        return value
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _decode(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(_decode(item) for item in value)
    if isinstance(value, dict):
        if "__type__" not in value:
            return {key: _decode(item) for key, item in value.items()}
        kind = value.get("__type__")
        if kind not in _KINDS:
            raise ValueError(f"unsupported IR kind: {kind}")
        cls = _KINDS[kind]
        allowed = {f.name for f in fields(cls)}
        if set(value) != allowed | {"__type__"}:
            raise ValueError(f"invalid fields for {kind}")
        return cls(
            **{key: _decode(item) for key, item in value.items() if key != "__type__"}
        )
    return value


def dumps(node: Any) -> str:
    return json.dumps(
        {"schema_version": SCHEMA_VERSION, "node": _encode(node)},
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def loads(source: str) -> Any:
    document = json.loads(source)
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != SCHEMA_VERSION
        or set(document) != {"schema_version", "node"}
    ):
        raise ValueError("unsupported or malformed IR schema")
    return _decode(document["node"])
