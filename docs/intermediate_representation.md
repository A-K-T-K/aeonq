# Semantic Intermediate Representation & Synthesis

The core abstraction of `aeonq` is a semantic intermediate representation (IR) that decouples problem definition from SDK-specific compilation and gate lowering.

---

## 1. Declarative Problem Architecture

An amplitude estimation instance is represented by the immutable `AEProblem` dataclass:

```python
from aeonq import AEProblem
from aeonq.ir import StatePreparation, ObjectivePredicate

problem = AEProblem(
    state_preparation=StatePreparation(...),
    objective=ObjectivePredicate(...),
    state_register=(0, 1, 2),
    work_register=(),
    objective_register=(3,),
)
```

### Register Partitioning
Logical qubits are partitioned into three disjoint roles:
1. **State Register**: Encodes the primary computational state or distribution.
2. **Work Register**: Auxiliary ancilla qubits used during state preparation.
3. **Objective Register**: Optional flag qubit(s) computed during preparation.

Registers strictly adhere to **little-endian** indexing (qubit 0 is the least significant bit).

---

## 2. Semantic IR Node Hierarchy

Unlike gate-level DAGs or syntactic OpenQASM ASTs, `aeonq` IR retains composite algorithmic structures:

```text
Operator (Base class)
├── Gate (H, X, Y, Z, CX, RX, RY, RZ)
├── Sequence (Ordered composition of operators)
├── StatePreparation (Initial operator A)
├── Adjoint (Formal symbolic adjoint A†)
├── ControlledOperator (Coherently conditioned operator)
├── GlobalPhase (Explicit global phase exp(i * radians))
├── ZeroReflection (S0: Reflection about the zero state)
├── ObjectiveReflection (Sχ: Phase oracle for objective predicate)
├── GroverOperator (Q = -A S0 A† Sχ)
└── GroverPower (Symbolic Q^k deferred until lowering)
```

### Key Semantic Invariants
- **Non-Cancellation**: Successive Grover operators $Q = -A S_0 A^\dagger S_\chi$ do not simplify via internal $A^\dagger A$ cancellation because reflections separate the preparation and adjoint factors across iterations.
- **Whole-Register Zero Reflection $S_0$**: $S_0 = I - 2|0\rangle\langle 0|$ reflects about the zero state across the entire register allocation (state, work, and objective qubits). This eliminates the requirement that auxiliary work qubits be uncomputed to $|0\rangle$ following state preparation.
- **Explicit Negative Sign**: The leading minus sign in $Q = -A S_0 A^\dagger S_\chi$ is maintained explicitly via `GlobalPhase(pi, n)`, ensuring correct relative phase behavior when $Q$ is coherently controlled.

---

## 3. Structured Phase-Oracle Synthesis

Phase oracles $S_\chi$ invert the phase of basis states satisfying a Boolean predicate $\chi(x) = 1$:
$$S_\chi|x\rangle = (-1)^{\chi(x)}|x\rangle$$

Directly constructing $S_\chi$ by enumerating truth tables scales exponentially ($O(2^m)$). `aeonq` avoids this through **disjoint-cube synthesis**.

### Disjoint-Cube Decomposition
A Boolean predicate is decomposed into a set of disjoint cubes (partial bit assignments). Because all cubes are pairwise disjoint:
1. Every input basis state matches at most one cube.
2. Each cube corresponds directly to a single multi-controlled phase flip.
3. The product of these independent phase flips implements $(-1)^{\chi(x)}$ exactly, with no interference or double-counting.

### Threshold Predicates ($x \ge K$)
For an integer threshold $K$ on an $m$-qubit register, `aeonq` synthesizes the predicate into **at most $m + 1$ disjoint cubes**:

```python
from aeonq.synthesis import threshold_cubes

# Disjoint cubes for x >= 5 on 3 qubits
cubes = threshold_cubes(qubits=(0, 1, 2), value=5)
print(cubes)
# Output: [((0, 1), (1, 1), (2, 1)), ((0, 0), (1, 1), (2, 1)), ((0, 1), (1, 0), (2, 1))]
```

This ensures linear scaling in register width for arithmetic threshold objectives (commonly used in financial tail-risk quantification and option pricing).

