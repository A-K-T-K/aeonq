# Conformance & Mutation Testing

Counts from an uncontrolled circuit cannot distinguish operators differing only by global phase. Coherently controlled or interferometric measurement tests can expose that difference. The conformance battery therefore combines operator comparisons with measurement and sampling checks, and reports which evaluated checks detect each fault.

`aeonq` introduces a rigorous conformance testing battery and mutation analysis suite specifically designed to validate lowering layers and detect convention-induced semantic defects.

---

## 1. Conformance Battery (66 Checks)

The shared battery consists of **66 standardized conformance checks** evaluated against exact reference semantics:

- **43 Operator Checks**: Compare lowering unitaries entry-wise against exact state-tensor reference matrices for operators $A$, $A^\dagger$, $S_0$, $S_\chi$, $Q$, $Q^2$, and controlled $Q$.
- **5 Measurement Checks**: Validate deterministic bitstring normalization and multi-qubit register maps across basic and asymmetric state preparations.
- **18 Sampling Checks**: Use 4,000 shots per circuit and a likelihood-ratio G-test against the exact outcome distribution, with a passing threshold of p >= 0.001 and an exact shot-count check. This differs from the separate cross-SDK equivalence study, which uses 200,000 shots and a distribution-free TVD upper bound.

### Running the Conformance Battery on an Adapter

You can evaluate any adapter (including custom implementations) using `run_battery`:

```python
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.conformance.battery import run_battery

adapter = QiskitAdapter()
results = run_battery(adapter)

passed = sum(r.passed is True for r in results)
print(f"Passed: {passed} / {len(results)} conformance checks")
```

---

## 2. Mutation Analysis (15 Seeded Faults)

To evaluate fault-detection sensitivity, `aeonq` implements fifteen mutation operators representing common cross-platform porting bugs:

| Lowering Fault | Category | Operator Failures | Measurement Failures | Sampling Failures | Count Invisible? |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Drop global phase** | IR Rewrite | 13 | 0 | 0 | **Yes** |
| **Invert global phase sign** | IR Rewrite | 1 | 0 | 0 | **Yes** |
| **Omit phase under control** | IR Rewrite | 7 | 0 | 0 | **Yes** |
| **Negate rotation parameters** | IR Rewrite | 24 | 0 | 3 | No |
| **Halve rotation angles** | IR Rewrite | 25 | 0 | 15 | No |
| **Lower $R_Z$ as phase gate** | IR Rewrite | 4 | 0 | 0 | **Yes** |
| **Swap CNOT direction** | IR Rewrite | 5 | 0 | 3 | No |
| **Invert control polarity** | IR Rewrite | 26 | 0 | 9 | No |
| **Omit adjoint** | IR Rewrite | 20 | 0 | 9 | No |
| **Shift power index ($Q^{k+1}$)** | IR Rewrite | 6 | 0 | 8 | No |
| **Reflect about $\|1\ldots1\rangle$** | IR Rewrite | 22 | 0 | 7 | No |
| **Negate objective reflection $S_\chi$** | IR Rewrite | 18 | 0 | 0 | **Yes** |
| **Reverse qubit order** | IR Rewrite | 21 | 4 | 8 | No |
| **Reverse readout bit order** | Readout Alteration | 0 | 4 | 8 | No |
| **Omit one shot record** | Readout Alteration | 0 | 5 | 18 | No |

### The 5 Count-Invisible Faults
Five faults (*Drop global phase*, *Invert global phase sign*, *Omit phase under control*, *Lower $R_Z$ as phase gate*, and *Negate objective reflection $S_\chi$*) yield **zero measurement and zero sampling failures** because they preserve the projective probability distribution of $Q^k A|0\rangle$. Within the evaluated battery, they are detected by entry-wise operator comparisons.

The same fault classes occur in real converters: see [Translator Fidelity & Independent Oracles](translator_fidelity.md) for what the OpenQASM exporters and importers and pytket's Qiskit/Cirq bridges do to these operators, and for checks that do not rely on aeonq's own reference semantics.

---

## 3. Testing Custom Backend Adapters

When implementing a new backend adapter for a new quantum SDK or hardware platform, wrap the adapter in `MutantAdapter` to verify that your testing battery can detect injected faults:

```python
from aeonq.conformance.battery import run_battery
from aeonq.conformance.mutation import MutantAdapter, MUTANTS
from aeonq.adapters.reference import ReferenceAdapter

base_adapter = ReferenceAdapter()

for mutant_name in MUTANTS:
    mutated_adapter = MutantAdapter(base_adapter, mutant=mutant_name)
    results = run_battery(mutated_adapter)
    failed = sum(r.passed is False for r in results)
    print(f"Mutant [{mutant_name:28s}]: {failed} failing checks")
```

