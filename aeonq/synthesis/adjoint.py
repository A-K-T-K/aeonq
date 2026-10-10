"""Exact symbolic adjoint rules in left-to-right execution order."""

from collections.abc import Mapping

from aeonq.ir.nodes import (
    Adjoint,
    ControlledOperator,
    EmbeddedOperator,
    Gate,
    GlobalPhase,
    GroverOperator,
    GroverPower,
    ObjectiveReflection,
    Operator,
    Parameter,
    Sequence,
    StatePreparation,
    ZeroReflection,
)


def _negative(value: float | Parameter) -> float | Parameter:
    return (
        Parameter(value.name, -value.scale) if isinstance(value, Parameter) else -value
    )


def adjoint(node: Operator) -> Operator:
    """Return symbolic adjoint, preserving exact phase and reversed order."""
    if isinstance(node, Gate):
        if node.name in ("RX", "RY", "RZ") and node.parameter is None:
            raise ValueError("rotation parameter is missing")
        return Gate(
            node.name,
            node.targets,
            node.n_qubits,
            _negative(node.parameter) if node.parameter is not None else None,
        )
    if isinstance(node, Sequence):
        return Sequence(
            tuple(adjoint(child) for child in reversed(node.children)), node.n_qubits
        )
    if isinstance(node, StatePreparation):
        return adjoint(node.operation)
    if isinstance(node, Adjoint):
        return node.operation
    if isinstance(node, GlobalPhase):
        return GlobalPhase(-node.radians, node.n_qubits)
    if isinstance(node, (ZeroReflection, ObjectiveReflection)):
        return node
    if isinstance(node, ControlledOperator):
        return ControlledOperator(
            adjoint(node.operation),
            node.controls,
            node.control_values,
            node.targets,
            node.n_qubits,
        )
    if isinstance(node, EmbeddedOperator):
        return EmbeddedOperator(adjoint(node.operation), node.targets, node.n_qubits)
    if isinstance(node, (GroverOperator, GroverPower)):
        # Exact semantic inversion; expansion can be performed separately.
        return Adjoint(node, node.n_qubits)
    raise TypeError(f"unsupported adjoint: {type(node).__name__}")


def bind_parameters(node: Operator, values: Mapping[str, float]) -> Operator:
    """Bind only declared symbolic rotation parameters; reject missing names."""
    if isinstance(node, Gate):
        parameter = node.parameter
        if isinstance(parameter, Parameter):
            if parameter.name not in values:
                raise ValueError(f"missing parameter: {parameter.name}")
            parameter = parameter.scale * values[parameter.name]
        return Gate(node.name, node.targets, node.n_qubits, parameter)
    if isinstance(node, Sequence):
        return Sequence(
            tuple(bind_parameters(child, values) for child in node.children),
            node.n_qubits,
        )
    if isinstance(node, StatePreparation):
        return StatePreparation(bind_parameters(node.operation, values), node.n_qubits)
    if isinstance(node, Adjoint):
        return Adjoint(bind_parameters(node.operation, values), node.n_qubits)
    if isinstance(node, ControlledOperator):
        return ControlledOperator(
            bind_parameters(node.operation, values),
            node.controls,
            node.control_values,
            node.targets,
            node.n_qubits,
        )
    if isinstance(node, EmbeddedOperator):
        return EmbeddedOperator(
            bind_parameters(node.operation, values), node.targets, node.n_qubits
        )
    if isinstance(node, GroverOperator):
        preparation = bind_parameters(node.preparation, values)
        assert isinstance(preparation, StatePreparation)
        return GroverOperator(
            preparation,
            node.zero_reflection,
            node.objective_reflection,
            node.n_qubits,
        )
    if isinstance(node, GroverPower):
        grover = bind_parameters(node.grover, values)
        assert isinstance(grover, GroverOperator)
        return GroverPower(grover, node.power, node.n_qubits)
    return node
