"""Exact local rewrite rules; Grover powers remain symbolic."""

from dataclasses import dataclass

from aeonq.ir.nodes import (
    Adjoint,
    ControlledOperator,
    EmbeddedOperator,
    Gate,
    GlobalPhase,
    GroverOperator,
    GroverPower,
    Operator,
    Parameter,
    Sequence,
    StatePreparation,
)


@dataclass(frozen=True)
class RewriteRecord:
    rule: str
    precondition: str
    before: tuple[Operator, ...]
    after: tuple[Operator, ...]


SELF_INVERSE = frozenset({"H", "X", "Y", "Z", "CX"})
ROTATIONS = frozenset({"RX", "RY", "RZ"})


def _sum_parameters(
    left: float | Parameter, right: float | Parameter
) -> float | Parameter | None:
    if (
        isinstance(left, Parameter)
        and isinstance(right, Parameter)
        and left.name == right.name
    ):
        return Parameter(left.name, left.scale + right.scale)
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return float(left + right)
    return None


def _is_zero(value: float | Parameter) -> bool:
    return value.scale == 0 if isinstance(value, Parameter) else value == 0


def _rewrite_children(node: Operator, records: list[RewriteRecord]) -> Operator:
    if isinstance(node, Sequence):
        return Sequence(
            tuple(_rewrite_children(child, records) for child in node.children),
            node.n_qubits,
        )
    if isinstance(node, StatePreparation):
        return StatePreparation(
            _rewrite_children(node.operation, records), node.n_qubits
        )
    if isinstance(node, Adjoint):
        return Adjoint(_rewrite_children(node.operation, records), node.n_qubits)
    if isinstance(node, ControlledOperator):
        return ControlledOperator(
            _rewrite_children(node.operation, records),
            node.controls,
            node.control_values,
            node.targets,
            node.n_qubits,
        )
    if isinstance(node, EmbeddedOperator):
        return EmbeddedOperator(
            _rewrite_children(node.operation, records), node.targets, node.n_qubits
        )
    if isinstance(node, GroverOperator):
        preparation = _rewrite_children(node.preparation, records)
        assert isinstance(preparation, StatePreparation)
        return GroverOperator(
            preparation,
            node.zero_reflection,
            node.objective_reflection,
            node.n_qubits,
        )
    if isinstance(node, GroverPower):
        grover = _rewrite_children(node.grover, records)
        assert isinstance(grover, GroverOperator)
        return GroverPower(grover, node.power, node.n_qubits)
    return node


def rewrite(node: Operator) -> tuple[Operator, tuple[RewriteRecord, ...]]:
    """Apply only adjacent exact-unitary rules and return auditable records."""
    records: list[RewriteRecord] = []
    node = _rewrite_children(node, records)
    if not isinstance(node, Sequence):
        return node, tuple(records)
    stack: list[Operator] = []
    for child in node.children:
        if isinstance(child, GroverPower) and child.power == 0:
            records.append(RewriteRecord("zero_power", "Q^0 = I exactly", (child,), ()))
            continue
        if stack:
            previous = stack[-1]
            if isinstance(previous, Adjoint) and previous.operation == child:
                stack.pop()
                records.append(
                    RewriteRecord(
                        "adjoint_left", "U†U = I for unitary U", (previous, child), ()
                    )
                )
                continue
            if isinstance(child, Adjoint) and child.operation == previous:
                stack.pop()
                records.append(
                    RewriteRecord(
                        "adjoint_right", "UU† = I for unitary U", (previous, child), ()
                    )
                )
                continue
            if (
                isinstance(previous, Gate)
                and isinstance(child, Gate)
                and previous.name == child.name
                and previous.targets == child.targets
            ):
                if previous.name in SELF_INVERSE:
                    stack.pop()
                    records.append(
                        RewriteRecord(
                            "self_inverse_pair",
                            "G² = I for H,X,Y,Z,CX",
                            (previous, child),
                            (),
                        )
                    )
                    continue
                if previous.name in ROTATIONS:
                    assert (
                        previous.parameter is not None and child.parameter is not None
                    )
                    total = _sum_parameters(previous.parameter, child.parameter)
                    if total is not None:
                        stack.pop()
                        replacement: tuple[Operator, ...] = (
                            ()
                            if _is_zero(total)
                            else (
                                Gate(
                                    previous.name,
                                    previous.targets,
                                    previous.n_qubits,
                                    total,
                                ),
                            )
                        )
                        stack.extend(replacement)
                        records.append(
                            RewriteRecord(
                                "same_axis_rotation_fusion",
                                "same axis and target; R(t)R(s)=R(t+s) exactly",
                                (previous, child),
                                replacement,
                            )
                        )
                        continue
            if isinstance(previous, GlobalPhase) and isinstance(child, GlobalPhase):
                stack.pop()
                replacement = (
                    GlobalPhase(previous.radians + child.radians, node.n_qubits),
                )
                stack.extend(replacement)
                records.append(
                    RewriteRecord(
                        "phase_fusion",
                        "scalar phases commute and add exactly",
                        (previous, child),
                        replacement,
                    )
                )
                continue
            if (
                isinstance(previous, GroverPower)
                and isinstance(child, GroverPower)
                and previous.grover == child.grover
            ):
                stack.pop()
                replacement = (
                    GroverPower(
                        previous.grover, previous.power + child.power, node.n_qubits
                    ),
                )
                stack.extend(replacement)
                records.append(
                    RewriteRecord(
                        "same_grover_power_fusion",
                        "Q^j Q^k = Q^(j+k) for identical Q",
                        (previous, child),
                        replacement,
                    )
                )
                continue
        stack.append(child)
    flat: list[Operator] = []
    flattened = False
    for child in stack:
        if isinstance(child, Sequence):
            flattened = True
            flat.extend(child.children)
            records.append(
                RewriteRecord(
                    "flatten_sequence",
                    "same logical width and execution order",
                    (child,),
                    child.children,
                )
            )
        else:
            flat.append(child)
    if flattened:
        reduced, additional = rewrite(Sequence(tuple(flat), node.n_qubits))
        return reduced, tuple(records) + additional
    return Sequence(tuple(stack), node.n_qubits), tuple(records)
