# Capabilities & Backend Adapters

`aeonq` achieves cross-framework portability by establishing a formal interface contract between high-level estimation semantics and target SDK compilation pipelines.

---

## 1. Adapter Capabilities Matrix

Each adapter wraps a specific target SDK and declares its operational capabilities and numerical tolerances:

| Adapter | SDK Version | Declared Tolerance | Optional Capabilities | Max Qubits | Conformance Passed |
| :--- | :--- | :---: | :--- | :---: | :---: |
| **ReferenceAdapter** | Native NumPy | $10^{-9}$ | `unitary` | 10 | 66/66 |
| **QiskitAdapter** | Qiskit 2.5.2 | $10^{-9}$ | `unitary`, `statevector` | Unbounded | 66/66 |
| **CirqAdapter** | Cirq 1.7.0 | $10^{-9}$ | `unitary`, `statevector` | Unbounded | 66/66 |
| **PennyLaneAdapter** | PennyLane 0.45.1 | $10^{-9}$ | `unitary`, `statevector` | Unbounded | 66/66 |
| **QSharpAdapter** | QDK 1.32.3 | $10^{-6}$ | `unitary` | 12 | 66/66 |

---

## 2. Standardized Adapter Contract

Every backend adapter inherits from `BackendAdapter` and implements seven required interface methods:

```python
from abc import ABC, abstractmethod

class BackendAdapter(ABC):
    unitary_precision: float = 1e-9

    @abstractmethod
    def capabilities(self) -> BackendCapabilities:
        """Declare supported features, gate sets, and sampling characteristics."""

    @abstractmethod
    def validate(self, ir: Operator) -> None:
        """Validate that the intermediate representation can be lowered."""

    @abstractmethod
    def lower(self, ir: Operator) -> Any:
        """Lower the semantic IR to a native target circuit or program."""

    @abstractmethod
    def compile(self, native_program: Any, options: dict | None = None) -> Any:
        """Execute platform-specific compilation or transpilation passes."""

    @abstractmethod
    def execute(self, compiled_program: Any, shots: int, *, seed: int | None = None) -> Any:
        """Sample finite measurement shots from the compiled circuit."""

    @abstractmethod
    def normalize_result(self, raw: Any, register_map: RegisterMap) -> NormalizedResult:
        """Convert native bitstrings/counts into normalized logical tuples."""

    @abstractmethod
    def resource_metrics(self, compiled_program: Any) -> dict[str, int | float | None]:
        """Extract compiled circuit depth, gate counts, and qubit metrics."""
```

Optional methods include `unitary(ir)` (small-system exact matrix export), `statevector(ir)`, and `execute_async(...)`.

---

## 3. Platform Convention Discrepancies Resolved

Different SDKs diverge across syntactic and operational conventions that break semantic equivalence if treated as simple API translations:

### Qubit Indexing & Register Endianness
- All logical registers in `aeonq` adhere strictly to **little-endian** indexing (qubit 0 is the least significant bit).
- When lowering to platforms or reading measurement records, adapters map logical qubit indices to native hardware/simulator registers via explicit `RegisterMap` definitions.

### Global Phase Orientations
- `aeonq` IR defines global phase rotations as $e^{i\varphi}$.
- **PennyLane** uses the opposite sign convention, $e^{-i\varphi}$. The PennyLane adapter automatically negates phase arguments (`qml.GlobalPhase(-ir.radians)`).
- Because relative phases under coherent control (such as controlled Grover powers $Q^k$) depend on global phase offsets, preserving this orientation is strictly necessary for correct phase estimation.

### Output Formatting & Tolerances
- **Microsoft Q\# (QDK)** outputs matrix entries formatted to approximately six significant decimal digits in `dump_operation`. The Q# adapter declares a numerical tolerance of $10^{-6}$, whereas other simulators achieve $10^{-9}$.
- QDK orders matrix rows with qubit 0 as the most significant bit; the Q# adapter explicitly permutes matrix rows and columns to little-endian ordering.
- Zero-valued controls in Q# are synthesized using Pauli-$X$ conjugation within `within { ... } apply { ... }` blocks, allowing Q# to automatically derive adjoint and controlled variants.

---

## 4. Runtime Guarantee Contracts

Theoretical guarantees (such as finite-sample Clopper-Pearson coverage in iterative estimation) rely on the assumption of independent, identically distributed Bernoulli sampling:

```python
# Capability declaration on backend
capabilities = BackendCapabilities(
    ...,
    ideal_sampling=True  # Set to False for noisy simulators or hardware
)
```

When a backend declares `ideal_sampling=False`:
1. The inference engine replaces satisfied algorithmic guarantees with `THEORETICAL_GUARANTEE_NOT_APPLICABLE`.
2. A formal runtime warning is attached to the returned `AEResult`:
   > *"backend does not sample ideally (for example a noise model or hardware); the reported interval assumes ideal independent Bernoulli shots and its coverage is not guaranteed"*
3. Users and downstream pipelines are prevented from mistaking empirical sampling noise for calibrated algorithmic confidence.

