# Translator Fidelity & Independent Oracles

The conformance battery and mutation analysis ([Conformance Testing](conformance_testing.md)) show that *seeded* convention faults are invisible to count-based tests. Two further studies ask whether this happens with **real converters**, and whether aeonq's own checks can be trusted **without** relying on aeonq's own reference semantics.

---

## 1. Translator-Fidelity Study

`benchmarks/translator_fidelity.py` lowers every amplitude-estimation operator of the battery cases ($A$, $A^\dagger$, $S_0$, $S_\chi$, $Q$, $Q^2$, controlled $Q$, and the measured circuits $A Q^k$ for $k = 0, 1, 2$), plus a controlled global phase, natively in Qiskit, Cirq or PennyLane. It then passes each program through an **existing third-party conversion path**. aeonq takes no part in the conversion:

| Source | Conversion | Target |
|:---|:---|:---|
| Qiskit | `qasm2.dumps` / `qasm3.dumps` | Qiskit (`qasm2.loads`, `qasm3.loads`), Cirq (`circuit_from_qasm`), PennyLane (`qml.from_qasm3`), QDK (`import_openqasm`) |
| Qiskit | pytket `qiskit_to_tk` → `tk_to_cirq` | Cirq |
| Cirq | `Circuit.to_qasm` (2.0 and 3.0) | Qiskit, Cirq, QDK |
| Cirq | pytket `cirq_to_tk` → `tk_to_qiskit` | Qiskit |
| PennyLane | `qml.to_openqasm` | Qiskit, Cirq, QDK |

The translated program is evaluated **only with the target SDK** (its own matrix export, its own control operation, its own simulator) and compared with the source program:

- **equal exactly**: every unitary entry within the target's precision ($10^{-9}$, or $10^{-6}$ for QDK's `dump_operation`);
- **equal up to global phase**, and the size of that phase;
- **equal when controlled**: the translated program is controlled *inside the target SDK*, where a dropped global phase becomes a relative phase;
- **qubit order**: the permutation, if any, under which the operators agree;
- **parameters**: whether each data-carrying rotation angle survived, and how precisely;
- **count-based check**: 4000 shots of $U|0\ldots0\rangle$ against the source distribution (G-test, $p \ge 10^{-3}$);
- **canonical QAE consequence**: for $Q$, the most likely canonical-QAE estimate from the eigenphases of the source and translated operators.

A row whose counts pass while an operator check fails is a **count-invisible translation fault**. A one-gate probe of every OpenQASM importer then locates the gates responsible.

### Observed results

The full tables are regenerated into `benchmarks/results/processed/translator_fidelity.md`. In summary, of 702 programs that translated and could be evaluated:

- 542 were exactly equal; 145 were equal only up to a global phase; 15 were not equal even up to phase;
- 118 failed once controlled in the target SDK;
- **152 passed the count check while failing an operator check.**

Qubit order was preserved wherever the operators agreed up to phase, and data-carrying angles survived to $\approx10^{-15}$ (Cirq's exporter: $1.4\times10^{-10}$). The faults trace to five converter behaviours:

| Behaviour | Where | Visible in counts? |
|:---|:---|:---|
| The circuit's global phase is dropped on export (Cirq warns that "the global phase does not affect the measurement results"). | Qiskit `qasm2.dumps`, `qasm3.dumps`; Cirq `to_qasm` | No |
| Power gates are written as rotations: `Z**t` becomes `rz(πt)`, dropping $e^{i\pi t/2}$. Angles are printed to 10 significant digits ($\approx10^{-10}$ error). | Cirq `to_qasm` | No |
| `Ry(θ)` becomes `cirq.YPowGate(θ/π)`, which differs by a phase of $e^{i\theta/2}$. | pytket `tk_to_cirq` | No |
| Rotation angles are reduced modulo $2\pi$ on import, multiplying `rx`/`ry`/`rz` by $-1$ for negative or $>2\pi$ angles. This is relative under control (`cry`, `crz`). | QDK `import_openqasm` | Uncontrolled: no. Controlled: yes |
| `gphase(γ)` is read as `GlobalPhase(γ)`, i.e. $e^{-i\gamma}$ rather than $e^{i\gamma}$. | PennyLane `qml.from_qasm3` | No |

**The omitted $-1$ is not an aeonq artifact.** Grover operators built by Qiskit's own `grover_operator` and by `qiskit-algorithms`' `EstimationProblem` carry it too, and 29 of their 33 translations lost it while passing the count check; the other four reached the QDK, where the angle reduction restored the sign by coincidence. (`grover_operator` itself also discards any global phase of the oracle circuit, so an oracle written as $-S_0$ loses its sign before any conversion.)

39 of the 58 translated Grover operators change the canonical-QAE estimate. In the Bernoulli case ($a = 0.25$), canonical QAE run on the controlled powers of $Q$ after a Qiskit QASM round trip reports $\hat a \approx 0.78$ instead of $0.22$ (5 evaluation qubits). The dropped $-1$ shifts every eigenphase by $\pi$ and turns $a$ into $1 - a$.

::: info Unavailable path
`qml.from_qiskit` / `qml.from_qasm` need `pennylane-qiskit` 0.45.0, which requires Qiskit 2.3.0, while the pinned stack uses 2.5.2. It is reported as unavailable rather than silently skipped. PennyLane's native `qml.from_qasm3` rejects the `include "stdgates.inc"` line that every exporter writes, so a second path deletes that line, as a user would.
:::

### Running the study

```bash
pip install -r requirements-lock.txt -r requirements-translators.txt
python benchmarks/translator_fidelity.py
```

`requirements-translators.txt` adds the converters (pytket and its Qiskit/Cirq extensions, `ply`, `qiskit-qasm3-import`) without changing any pin in `requirements-lock.txt`.

---

## 2. Independent Oracle Checks

The battery compares adapters with `aeonq.reference`, which evaluates the same IR that the adapters lower. A defect shared by the IR, the synthesis and the reference simulator could therefore cancel. The `aeonq.oracles` package is written **from the problem statements alone**: it imports NumPy and the SDKs, never aeonq's IR, synthesis, reference or adapters. `tests/unit/test_oracle_independence.py` enforces this.

| Oracle | What it is |
|:---|:---|
| **Analytic amplitudes** (`aeonq.oracles.cases`) | $a$, the good-state predicate and every rotation angle derived from the problem's probabilities, including an independent re-derivation of the log-normal grid. |
| **Hand-written circuits** (`aeonq.oracles.handwritten`) | $A$, $Q$, $Q^2$ and controlled $Q$ written natively in Qiskit, Cirq, PennyLane and Q#, with deliberately different constructions where an exact one exists (e.g. the $x\in\{1,2\}$ oracle as $Z\otimes Z$, Q#'s $-1$ as $ZXZX$). |
| **Second simulator** (`aeonq.oracles.numpy_sim`, `sdk_import`) | A NumPy statevector simulator on basis-index arithmetic with textbook gate matrices. It also reads each adapter's *native program* gate by gate (Qiskit via `{u, cx}`, Cirq and PennyLane via their own decompositions), bypassing every SDK simulator. |

Every source is held to closed forms, with the global phase of $Q$ not free:

- amplitude $\lVert\Pi_\text{good} A|0\rangle\rVert^2 = a$;
- $Q$ acts on the good/bad plane as the rotation by $2\theta$, $\sin^2\theta = a$, and leaves the plane invariant; $Q^2$ rotates by $4\theta$;
- controlled $Q$ is the identity when the control is 0 and the rotation when it is 1;
- $P_\text{good}(Q^k A|0\rangle) = \sin^2((2k+1)\theta)$ for $k = 0..4$.

Every source is also compared entry by entry with the NumPy oracle. **All 460 checks pass**, for the IR reference, all four adapters, all four hand-written SDK circuits and the second simulator on all three adapter programs. Negative controls ($Q$ without its $-1$, qubit-reversed $Q$, $A$ with negated angles) are all rejected. $Q$ without its $-1$ still satisfies the $Q^2$ and amplification laws; only the rotation and controlled checks reject it. At $a = 1/2$ an angle sign flip in $A$ is invisible to every closed form and is caught only by the entrywise comparison.

```bash
python benchmarks/independent_oracles.py
```
