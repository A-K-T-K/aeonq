"""Logical-to-physical qubit and classical result mappings."""

from dataclasses import dataclass


@dataclass(frozen=True)
class RegisterMap:
    logical_to_physical: tuple[int, ...]
    logical_to_classical: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        for mapping in (self.logical_to_physical, self.logical_to_classical):
            if len(set(mapping)) != len(mapping) or any(x < 0 for x in mapping):
                raise ValueError(
                    "register mapping must contain unique nonnegative positions"
                )
        if self.logical_to_classical and len(self.logical_to_classical) != len(
            self.logical_to_physical
        ):
            raise ValueError("classical mapping must cover all logical qubits")
        if set(self.logical_to_physical) != set(range(len(self.logical_to_physical))):
            raise ValueError("Physical map must be a permutation")
        if self.logical_to_classical and set(self.logical_to_classical) != set(
            range(len(self.logical_to_classical))
        ):
            raise ValueError("Classical map must be a permutation")

    def logical_index(self, physical_index: int) -> int:
        if physical_index < 0 or physical_index >= 2 ** len(self.logical_to_physical):
            raise ValueError("physical basis state out of range")
        return sum(
            ((physical_index >> p) & 1) << i
            for i, p in enumerate(self.logical_to_physical)
        )

    def physical_index(self, logical_index: int) -> int:
        if logical_index < 0 or logical_index >= 2 ** len(self.logical_to_physical):
            raise ValueError("logical basis state out of range")
        return sum(
            ((logical_index >> i) & 1) << p
            for i, p in enumerate(self.logical_to_physical)
        )

    def normalize_bitstring(self, raw: str) -> tuple[int, ...]:
        """Raw string is highest classical index on the left, as declared by adapter."""
        mapping = self.logical_to_classical
        if (
            not mapping
            or len(raw) <= max(mapping)
            or any(bit not in "01" for bit in raw)
        ):
            raise ValueError("invalid raw bitstring or missing classical map")
        return tuple(int(raw[-1 - p]) for p in mapping)
