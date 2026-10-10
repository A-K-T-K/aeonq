# Translator fidelity on amplitude-estimation circuits

Generated 2026-10-10T08:55:50.988067+00:00 by `benchmarks/translator_fidelity.py`. Every program was lowered by a verified aeonq adapter and then converted only by third-party code; the translated program was evaluated only with the target SDK. Operator comparisons are against the source program's own matrix, entry by entry; counts use 4000 shots of U|0...0> and a G-test at p >= 1e-3.

* programs attempted: 1098 (61 on unavailable paths)
* programs translated and evaluated: 702
* exactly equal: 542
* equal only up to global phase: 145
* not equal even up to global phase: 15
* failed once controlled in the target SDK: 118
* count check passed while an operator check failed: 152

## Per path

| path | evaluated / attempted | exact | phase only | not equal | controlled failed | counts pass | count-invisible |
|---|---|---|---|---|---|---|---|
| `qiskit>qasm2>qiskit` | 61 / 61 | 49 | 12 | 0 | 12 | 61 | 12 |
| `qiskit>qasm3>qiskit` | 61 / 61 | 49 | 12 | 0 | 12 | 61 | 12 |
| `qiskit>qasm2>cirq` | 42 / 61 | 34 | 8 | 0 | 8 | 42 | 8 |
| `qiskit>qasm3>cirq` | 42 / 61 | 34 | 8 | 0 | 8 | 42 | 8 |
| `qiskit>qasm3>pennylane` | 0 / 61 | 0 | 0 | 0 | 0 | 0 | 0 |
| `qiskit>qasm3-noinclude>pennylane` | 25 / 61 | 21 | 4 | 0 | 4 | 25 | 4 |
| `qiskit>qasm2>qsharp` | 40 / 61 | 35 | 5 | 0 | 0 | 40 | 5 |
| `qiskit>qasm3>qsharp` | 61 / 61 | 47 | 9 | 5 | 0 | 61 | 14 |
| `qiskit>pytket>cirq` | 25 / 61 | 10 | 15 | 0 | 15 | 25 | 15 |
| `cirq>qasm2>qiskit` | 61 / 61 | 37 | 24 | 0 | 24 | 61 | 24 |
| `cirq>qasm3>qiskit` | 41 / 61 | 30 | 11 | 0 | 11 | 41 | 11 |
| `cirq>qasm2>cirq` | 61 / 61 | 37 | 24 | 0 | 24 | 61 | 24 |
| `cirq>qasm3>qsharp` | 61 / 61 | 41 | 10 | 10 | 0 | 53 | 12 |
| `cirq>pytket>qiskit` | 25 / 61 | 25 | 0 | 0 | 0 | 25 | 0 |
| `pennylane>qasm2>qiskit` | 35 / 61 | 35 | 0 | 0 | 0 | 35 | 0 |
| `pennylane>qasm2>cirq` | 35 / 61 | 35 | 0 | 0 | 0 | 35 | 0 |
| `pennylane>qasm2>qsharp` | 26 / 61 | 23 | 3 | 0 | 0 | 26 | 3 |
| `qiskit>pennylane-qiskit>pennylane` | 0 / 61 | 0 | 0 | 0 | 0 | 0 | 0 |

## Conversion failures (first message per path)

* `qiskit>pytket>cirq` export_failed: RuntimeError: Bad operation type: CircBox
* `cirq>pytket>qiskit` export_failed: NotImplementedError: Operation not supported by tket: None
* `pennylane>qasm2>qiskit` import_failed: QASM2ParseError: "<input>:11,0: 'gphase' is not defined in this scope"
* `pennylane>qasm2>cirq` import_failed: QasmException: Syntax error: ';' ...793) ; ^ at line 11, column 28
* `pennylane>qasm2>qsharp` import_failed: QSharpError: Qdk.Qasm.Lowerer.UndefinedSymbol x undefined symbol: gphase ,-[./program.qasm:11:1] 10 | ry(1.0471975511965979) q[0]; 11 | gphase(-3.141592653589793) ; : ^^^^^^ `---- Qdk.Qasm.Lowerer.CannotCallNonGate x cannot call a gate that is not a gat
* `qiskit>qasm2>cirq` import_failed: QasmException: cu takes 3 parameter(s), got: 4, at line 3
* `qiskit>qasm3>cirq` import_failed: QasmException: cu takes 3 parameter(s), got: 4, at line 7
* `qiskit>qasm2>qsharp` import_failed: QSharpError: Qdk.Qasm.Lowerer.UndefinedSymbol x undefined symbol: cu ,-[./program.qasm:3:50] 2 | include "qelib1.inc"; 3 | gate ccircuit_1140 q0,q1 { h q1; cx q0,q1; h q1; cu(-pi/3,0,0,0) q0,q1; cx q0,q1; h q1; cx q0,q1; h q1; cx q0,q1; cu(pi/3,0,0,0) q
* `cirq>qasm3>qiskit` import_failed: QASM3ImporterError: "27,0: gate 'sxdg' is not defined."
* `qiskit>qasm3>pennylane` unitary_failed: NotImplementedError: An unsupported QASM instruction Include was encountered on line 2, in global.
* `qiskit>qasm3-noinclude>pennylane` unitary_failed: TransformError: Wires in circuit ['_gate_q_1', '_gate_q_0'] are inconsistent with those in wire_order ['q[1]', 'q[0]']
* `qiskit>pennylane-qiskit>pennylane` unavailable: qml.from_qiskit / qml.from_qasm need pennylane-qiskit 0.45.0, which requires qiskit 2.3.0; the pinned stack uses qiskit 2.5.2

## Count-invisible failures

| case | node | path | max entry error | global phase offset | controlled equal |
|---|---|---|---|---|---|
| bernoulli_0.25 | A | `qiskit>pytket>cirq` | 0.448 | +0.5236 | False |
| bernoulli_0.25 | A_adjoint | `qiskit>qasm2>qsharp` | 1.73 | +3.1416 | None |
| bernoulli_0.25 | A_adjoint | `qiskit>qasm3>qsharp` | 1.73 | +3.1416 | None |
| bernoulli_0.25 | A_adjoint | `qiskit>pytket>cirq` | 0.448 | -0.5236 | False |
| bernoulli_0.25 | A_adjoint | `cirq>qasm3>qsharp` | 1.73 | +3.1416 | None |
| bernoulli_0.25 | A_adjoint | `pennylane>qasm2>qsharp` | 1.73 | +3.1416 | None |
| bernoulli_0.25 | Q | `qiskit>qasm2>qiskit` | 1.73 | -3.1416 | False |
| bernoulli_0.25 | Q | `qiskit>qasm3>qiskit` | 1.73 | -3.1416 | False |
| bernoulli_0.25 | Q | `qiskit>qasm2>cirq` | 1.73 | -3.1416 | False |
| bernoulli_0.25 | Q | `qiskit>qasm3>cirq` | 1.73 | -3.1416 | False |
| bernoulli_0.25 | Q | `qiskit>qasm3-noinclude>pennylane` | 1.73 | -3.1416 | False |
| bernoulli_0.25 | Q | `qiskit>pytket>cirq` | 1.73 | +3.1416 | False |
| bernoulli_0.25 | Q | `cirq>qasm2>qiskit` | 1.73 | -3.1416 | False |
| bernoulli_0.25 | Q | `cirq>qasm3>qiskit` | 1.73 | -3.1416 | False |
| bernoulli_0.25 | Q | `cirq>qasm2>cirq` | 1.73 | -3.1416 | False |
| bernoulli_0.25 | controlled_Q | `qiskit>qasm3>qsharp` | 1.73 |  | None |
| bernoulli_0.25 | A_Q^0 | `qiskit>pytket>cirq` | 0.448 | +0.5236 | False |
| bernoulli_0.25 | A_Q^1 | `qiskit>qasm2>qiskit` | 2 | -3.1416 | False |
| bernoulli_0.25 | A_Q^1 | `qiskit>qasm3>qiskit` | 2 | -3.1416 | False |
| bernoulli_0.25 | A_Q^1 | `qiskit>qasm2>cirq` | 2 | -3.1416 | False |
| bernoulli_0.25 | A_Q^1 | `qiskit>qasm3>cirq` | 2 | -3.1416 | False |
| bernoulli_0.25 | A_Q^1 | `qiskit>qasm3-noinclude>pennylane` | 2 | -3.1416 | False |
| bernoulli_0.25 | A_Q^1 | `qiskit>pytket>cirq` | 1.93 | -2.6180 | False |
| bernoulli_0.25 | A_Q^1 | `cirq>qasm2>qiskit` | 2 | -3.1416 | False |
| bernoulli_0.25 | A_Q^1 | `cirq>qasm3>qiskit` | 2 | -3.1416 | False |
| bernoulli_0.25 | A_Q^1 | `cirq>qasm2>cirq` | 2 | -3.1416 | False |
| bernoulli_0.25 | A_Q^2 | `qiskit>pytket>cirq` | 0.448 | +0.5236 | False |
| uniform_threshold_n2_k1 | A_adjoint | `cirq>qasm2>qiskit` | 1 | -3.1416 | False |
| uniform_threshold_n2_k1 | A_adjoint | `cirq>qasm3>qiskit` | 1 | -3.1416 | False |
| uniform_threshold_n2_k1 | A_adjoint | `cirq>qasm2>cirq` | 1 | +3.1416 | False |
| uniform_threshold_n2_k1 | Q | `qiskit>qasm2>qiskit` | 1 | -3.1416 | False |
| uniform_threshold_n2_k1 | Q | `qiskit>qasm3>qiskit` | 1 | -3.1416 | False |
| uniform_threshold_n2_k1 | Q | `qiskit>qasm2>cirq` | 1 | -3.1416 | False |
| uniform_threshold_n2_k1 | Q | `qiskit>qasm3>cirq` | 1 | -3.1416 | False |
| uniform_threshold_n2_k1 | Q | `qiskit>qasm2>qsharp` | 1 | -3.1416 | None |
| uniform_threshold_n2_k1 | Q | `qiskit>qasm3>qsharp` | 1 | -3.1416 | None |
| uniform_threshold_n2_k1 | Q | `cirq>qasm3>qsharp` | 1 |  | None |
| uniform_threshold_n2_k1 | A_Q^1 | `qiskit>qasm2>qiskit` | 2 | -3.1416 | False |
| uniform_threshold_n2_k1 | A_Q^1 | `qiskit>qasm3>qiskit` | 2 | -3.1416 | False |
| uniform_threshold_n2_k1 | A_Q^1 | `qiskit>qasm2>cirq` | 2 | -3.1416 | False |
| uniform_threshold_n2_k1 | A_Q^1 | `qiskit>qasm3>cirq` | 2 | -3.1416 | False |
| uniform_threshold_n2_k1 | A_Q^1 | `qiskit>qasm2>qsharp` | 2 | -3.1416 | None |
| uniform_threshold_n2_k1 | A_Q^1 | `qiskit>qasm3>qsharp` | 2 | -3.1416 | None |
| uniform_threshold_n2_k1 | A_Q^2 | `cirq>qasm3>qsharp` | 1 |  | None |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A | `cirq>qasm2>qiskit` | 0.617 | +0.9553 | False |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A | `cirq>qasm2>cirq` | 0.617 | +0.9553 | False |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A_adjoint | `cirq>qasm2>qiskit` | 0.617 | -0.9553 | False |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A_adjoint | `cirq>qasm2>cirq` | 0.617 | -0.9553 | False |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q | `qiskit>qasm2>qiskit` | 1.6 | -3.1416 | False |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q | `qiskit>qasm3>qiskit` | 1.6 | -3.1416 | False |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q | `qiskit>qasm3>qsharp` | 1.6 | -3.1416 | None |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q | `cirq>qasm2>qiskit` | 1.6 | +3.1416 | False |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q | `cirq>qasm2>cirq` | 1.6 | +3.1416 | False |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q | `cirq>qasm3>qsharp` | 1.6 | -3.1416 | None |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | controlled_Q | `qiskit>qasm3>qsharp` | 1.6 |  | None |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | controlled_Q | `cirq>qasm3>qsharp` | 2 | -3.1416 | None |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A_Q^0 | `cirq>qasm2>qiskit` | 0.617 | +0.9553 | False |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A_Q^0 | `cirq>qasm2>cirq` | 0.617 | +0.9553 | False |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A_Q^1 | `qiskit>qasm2>qiskit` | 1.51 | -3.1416 | False |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A_Q^1 | `qiskit>qasm3>qiskit` | 1.51 | -3.1416 | False |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A_Q^1 | `qiskit>qasm3>qsharp` | 1.51 | -3.1416 | None |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A_Q^1 | `cirq>qasm2>qiskit` | 1.34 | -2.1863 | False |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A_Q^1 | `cirq>qasm2>cirq` | 1.34 | -2.1863 | False |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A_Q^1 | `cirq>qasm3>qsharp` | 1.51 | -3.1416 | None |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A_Q^2 | `cirq>qasm2>qiskit` | 0.695 | +0.9553 | False |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A_Q^2 | `cirq>qasm2>cirq` | 0.695 | +0.9553 | False |
| lognormal_tail_n2_k2_flag | A | `cirq>qasm2>qiskit` | 0.156 | +0.2105 | False |
| lognormal_tail_n2_k2_flag | A | `cirq>qasm2>cirq` | 0.156 | +0.2105 | False |
| lognormal_tail_n2_k2_flag | A_adjoint | `cirq>qasm2>qiskit` | 0.156 | -0.2105 | False |
| lognormal_tail_n2_k2_flag | A_adjoint | `cirq>qasm2>cirq` | 0.156 | -0.2105 | False |
| lognormal_tail_n2_k2_flag | Q | `qiskit>qasm2>qiskit` | 2 | -3.1416 | False |
| lognormal_tail_n2_k2_flag | Q | `qiskit>qasm3>qiskit` | 2 | -3.1416 | False |
| lognormal_tail_n2_k2_flag | Q | `qiskit>qasm3>qsharp` | 2 | -3.1416 | None |
| lognormal_tail_n2_k2_flag | Q | `cirq>qasm2>qiskit` | 2 | +3.1416 | False |
| lognormal_tail_n2_k2_flag | Q | `cirq>qasm2>cirq` | 2 | +3.1416 | False |
| lognormal_tail_n2_k2_flag | Q | `cirq>qasm3>qsharp` | 2 | -3.1416 | None |
| lognormal_tail_n2_k2_flag | controlled_Q | `qiskit>qasm3>qsharp` | 2 |  | None |
| lognormal_tail_n2_k2_flag | controlled_Q | `cirq>qasm2>qiskit` | 0.196 | -0.1963 | False |
| lognormal_tail_n2_k2_flag | controlled_Q | `cirq>qasm2>cirq` | 0.196 | -0.1963 | False |
| lognormal_tail_n2_k2_flag | controlled_Q | `cirq>qasm3>qsharp` | 0.196 | -0.1963 | None |
| lognormal_tail_n2_k2_flag | A_Q^0 | `cirq>qasm2>qiskit` | 0.156 | +0.2105 | False |
| lognormal_tail_n2_k2_flag | A_Q^0 | `cirq>qasm2>cirq` | 0.156 | +0.2105 | False |
| lognormal_tail_n2_k2_flag | A_Q^1 | `qiskit>qasm2>qiskit` | 1.63 | -3.1416 | False |
| lognormal_tail_n2_k2_flag | A_Q^1 | `qiskit>qasm3>qiskit` | 1.63 | -3.1416 | False |
| lognormal_tail_n2_k2_flag | A_Q^1 | `qiskit>qasm3>qsharp` | 1.63 | -3.1416 | None |
| lognormal_tail_n2_k2_flag | A_Q^1 | `cirq>qasm2>qiskit` | 1.62 | -2.9311 | False |
| lognormal_tail_n2_k2_flag | A_Q^1 | `cirq>qasm2>cirq` | 1.62 | -2.9311 | False |
| lognormal_tail_n2_k2_flag | A_Q^1 | `cirq>qasm3>qsharp` | 1.63 | -3.1416 | None |
| lognormal_tail_n2_k2_flag | A_Q^2 | `cirq>qasm2>qiskit` | 0.183 | +0.2105 | False |
| lognormal_tail_n2_k2_flag | A_Q^2 | `cirq>qasm2>cirq` | 0.183 | +0.2105 | False |
| entangled_cx_n2 | A | `qiskit>pytket>cirq` | 0.492 | +0.8000 | False |
| entangled_cx_n2 | A_adjoint | `qiskit>qasm2>qsharp` | 1.26 | +3.1416 | None |
| entangled_cx_n2 | A_adjoint | `qiskit>qasm3>qsharp` | 1.26 | +3.1416 | None |
| entangled_cx_n2 | A_adjoint | `qiskit>pytket>cirq` | 0.492 | -0.8000 | False |
| entangled_cx_n2 | A_adjoint | `cirq>qasm2>qiskit` | 0.893 | +1.5708 | False |
| entangled_cx_n2 | A_adjoint | `cirq>qasm3>qiskit` | 0.893 | +1.5708 | False |
| entangled_cx_n2 | A_adjoint | `cirq>qasm2>cirq` | 0.893 | +1.5708 | False |
| entangled_cx_n2 | A_adjoint | `pennylane>qasm2>qsharp` | 1.26 | +3.1416 | None |
| entangled_cx_n2 | Q | `qiskit>qasm2>qiskit` | 1.59 | -3.1416 | False |
| entangled_cx_n2 | Q | `qiskit>qasm3>qiskit` | 1.59 | -3.1416 | False |
| entangled_cx_n2 | Q | `qiskit>qasm2>cirq` | 1.59 | -3.1416 | False |
| entangled_cx_n2 | Q | `qiskit>qasm3>cirq` | 1.59 | -3.1416 | False |
| entangled_cx_n2 | Q | `cirq>qasm2>qiskit` | 1.13 | -1.5708 | False |
| entangled_cx_n2 | Q | `cirq>qasm3>qiskit` | 1.13 | -1.5708 | False |
| entangled_cx_n2 | Q | `cirq>qasm2>cirq` | 1.13 | -1.5708 | False |
| entangled_cx_n2 | Q_power_2 | `cirq>qasm2>qiskit` | 1.84 | -3.1416 | False |
| entangled_cx_n2 | Q_power_2 | `cirq>qasm3>qiskit` | 1.84 | -3.1416 | False |
| entangled_cx_n2 | Q_power_2 | `cirq>qasm2>cirq` | 1.84 | -3.1416 | False |
| entangled_cx_n2 | controlled_Q | `qiskit>qasm3>qsharp` | 1.59 |  | None |
| entangled_cx_n2 | controlled_Q | `cirq>qasm3>qsharp` | 2 | -3.1416 | None |
| entangled_cx_n2 | A_Q^0 | `qiskit>pytket>cirq` | 0.492 | +0.8000 | False |
| entangled_cx_n2 | A_Q^1 | `qiskit>qasm2>qiskit` | 1.78 | -3.1416 | False |
| entangled_cx_n2 | A_Q^1 | `qiskit>qasm3>qiskit` | 1.78 | -3.1416 | False |
| entangled_cx_n2 | A_Q^1 | `qiskit>qasm2>cirq` | 1.78 | -3.1416 | False |
| entangled_cx_n2 | A_Q^1 | `qiskit>qasm3>cirq` | 1.78 | -3.1416 | False |
| entangled_cx_n2 | A_Q^1 | `cirq>qasm2>qiskit` | 1.26 | -1.5708 | False |
| entangled_cx_n2 | A_Q^1 | `cirq>qasm3>qiskit` | 1.26 | -1.5708 | False |
| entangled_cx_n2 | A_Q^1 | `cirq>qasm2>cirq` | 1.26 | -1.5708 | False |
| entangled_cx_n2 | A_Q^2 | `cirq>qasm2>qiskit` | 2 | -3.1416 | False |
| entangled_cx_n2 | A_Q^2 | `cirq>qasm3>qiskit` | 2 | -3.1416 | False |
| entangled_cx_n2 | A_Q^2 | `cirq>qasm2>cirq` | 2 | -3.1416 | False |
| asymmetric_rotations_n1 | A | `qiskit>pytket>cirq` | 0.651 | +0.7000 | False |
| asymmetric_rotations_n1 | A_adjoint | `qiskit>qasm2>qsharp` | 1.9 | +3.1416 | None |
| asymmetric_rotations_n1 | A_adjoint | `qiskit>qasm3>qsharp` | 1.9 | +3.1416 | None |
| asymmetric_rotations_n1 | A_adjoint | `qiskit>pytket>cirq` | 0.651 | -0.7000 | False |
| asymmetric_rotations_n1 | A_adjoint | `cirq>qasm3>qsharp` | 1.9 | +3.1416 | None |
| asymmetric_rotations_n1 | A_adjoint | `pennylane>qasm2>qsharp` | 1.9 | +3.1416 | None |
| asymmetric_rotations_n1 | Q | `qiskit>qasm2>qiskit` | 1.61 | -3.1416 | False |
| asymmetric_rotations_n1 | Q | `qiskit>qasm3>qiskit` | 1.61 | -3.1416 | False |
| asymmetric_rotations_n1 | Q | `qiskit>qasm2>cirq` | 1.61 | -3.1416 | False |
| asymmetric_rotations_n1 | Q | `qiskit>qasm3>cirq` | 1.61 | -3.1416 | False |
| asymmetric_rotations_n1 | Q | `qiskit>qasm3-noinclude>pennylane` | 1.61 | -3.1416 | False |
| asymmetric_rotations_n1 | Q | `qiskit>pytket>cirq` | 1.61 | +3.1416 | False |
| asymmetric_rotations_n1 | Q | `cirq>qasm2>qiskit` | 1.61 | -3.1416 | False |
| asymmetric_rotations_n1 | Q | `cirq>qasm3>qiskit` | 1.61 | -3.1416 | False |
| asymmetric_rotations_n1 | Q | `cirq>qasm2>cirq` | 1.61 | -3.1416 | False |
| asymmetric_rotations_n1 | controlled_Q | `qiskit>qasm3>qsharp` | 1.61 |  | None |
| asymmetric_rotations_n1 | A_Q^0 | `qiskit>pytket>cirq` | 0.651 | +0.7000 | False |
| asymmetric_rotations_n1 | A_Q^1 | `qiskit>qasm2>qiskit` | 1.63 | -3.1416 | False |
| asymmetric_rotations_n1 | A_Q^1 | `qiskit>qasm3>qiskit` | 1.63 | -3.1416 | False |
| asymmetric_rotations_n1 | A_Q^1 | `qiskit>qasm2>cirq` | 1.63 | -3.1416 | False |
| asymmetric_rotations_n1 | A_Q^1 | `qiskit>qasm3>cirq` | 1.63 | -3.1416 | False |
| asymmetric_rotations_n1 | A_Q^1 | `qiskit>qasm3-noinclude>pennylane` | 1.63 | -3.1416 | False |
| asymmetric_rotations_n1 | A_Q^1 | `qiskit>pytket>cirq` | 1.53 | -2.4416 | False |
| asymmetric_rotations_n1 | A_Q^1 | `cirq>qasm2>qiskit` | 1.63 | -3.1416 | False |
| asymmetric_rotations_n1 | A_Q^1 | `cirq>qasm3>qiskit` | 1.63 | -3.1416 | False |
| asymmetric_rotations_n1 | A_Q^1 | `cirq>qasm2>cirq` | 1.63 | -3.1416 | False |
| asymmetric_rotations_n1 | A_Q^2 | `qiskit>pytket>cirq` | 0.686 | +0.7000 | False |
| phase | controlled_global_phase | `cirq>qasm2>qiskit` | 0.518 | -0.5236 | False |
| phase | controlled_global_phase | `cirq>qasm3>qiskit` | 0.518 | -0.5236 | False |
| phase | controlled_global_phase | `cirq>qasm2>cirq` | 0.518 | -0.5236 | False |
| phase | controlled_global_phase | `cirq>qasm3>qsharp` | 0.518 | -0.5236 | None |

## Consequence for canonical QAE

Most likely canonical-QAE estimate (5 evaluation qubits, exact phase estimation on the eigenphases of Q, global phase included) from the source Q and from the translated Q. Rows where they differ:

| case | path | estimate from source Q | estimate from translated Q |
|---|---|---|---|
| bernoulli_0.25 | `qiskit>qasm2>qiskit` | 0.2222 | 0.7778 |
| bernoulli_0.25 | `qiskit>qasm3>qiskit` | 0.2222 | 0.7778 |
| bernoulli_0.25 | `qiskit>qasm2>cirq` | 0.2222 | 0.7778 |
| bernoulli_0.25 | `qiskit>qasm3>cirq` | 0.2222 | 0.7778 |
| bernoulli_0.25 | `qiskit>qasm3-noinclude>pennylane` | 0.2222 | 0.7778 |
| bernoulli_0.25 | `qiskit>pytket>cirq` | 0.2222 | 0.7778 |
| bernoulli_0.25 | `cirq>qasm2>qiskit` | 0.2222 | 0.7778 |
| bernoulli_0.25 | `cirq>qasm3>qiskit` | 0.2222 | 0.7778 |
| bernoulli_0.25 | `cirq>qasm2>cirq` | 0.2222 | 0.7778 |
| uniform_threshold_n2_k1 | `qiskit>qasm2>qiskit` | 0.7778 | 0.2222 |
| uniform_threshold_n2_k1 | `qiskit>qasm3>qiskit` | 0.7778 | 0.2222 |
| uniform_threshold_n2_k1 | `qiskit>qasm2>cirq` | 0.7778 | 0.2222 |
| uniform_threshold_n2_k1 | `qiskit>qasm3>cirq` | 0.7778 | 0.2222 |
| uniform_threshold_n2_k1 | `qiskit>qasm2>qsharp` | 0.7778 | 0.2222 |
| uniform_threshold_n2_k1 | `qiskit>qasm3>qsharp` | 0.7778 | 0.2222 |
| uniform_threshold_n2_k1 | `cirq>qasm3>qsharp` | 0.7778 | 1.0000 |
| lognormal_tail_n2_k2_flag | `qiskit>qasm2>qiskit` | 0.4025 | 0.5975 |
| lognormal_tail_n2_k2_flag | `qiskit>qasm3>qiskit` | 0.4025 | 0.5975 |
| lognormal_tail_n2_k2_flag | `qiskit>qasm3>qsharp` | 0.4025 | 0.5975 |
| lognormal_tail_n2_k2_flag | `cirq>qasm2>qiskit` | 0.4025 | 0.5975 |
| lognormal_tail_n2_k2_flag | `cirq>qasm2>cirq` | 0.4025 | 0.5975 |
| lognormal_tail_n2_k2_flag | `cirq>qasm3>qsharp` | 0.4025 | 0.5975 |
| entangled_cx_n2 | `qiskit>qasm2>qiskit` | 0.0843 | 0.9157 |
| entangled_cx_n2 | `qiskit>qasm3>qiskit` | 0.0843 | 0.9157 |
| entangled_cx_n2 | `qiskit>qasm2>cirq` | 0.0843 | 0.9157 |
| entangled_cx_n2 | `qiskit>qasm3>cirq` | 0.0843 | 0.9157 |
| entangled_cx_n2 | `cirq>qasm2>qiskit` | 0.0843 | 0.2222 |
| entangled_cx_n2 | `cirq>qasm3>qiskit` | 0.0843 | 0.2222 |
| entangled_cx_n2 | `cirq>qasm2>cirq` | 0.0843 | 0.2222 |
| entangled_cx_n2 | `cirq>qasm3>qsharp` | 0.0843 | 0.1464 |
| asymmetric_rotations_n1 | `qiskit>qasm2>qiskit` | 0.0843 | 0.9157 |
| asymmetric_rotations_n1 | `qiskit>qasm3>qiskit` | 0.0843 | 0.9157 |
| asymmetric_rotations_n1 | `qiskit>qasm2>cirq` | 0.0843 | 0.9157 |
| asymmetric_rotations_n1 | `qiskit>qasm3>cirq` | 0.0843 | 0.9157 |
| asymmetric_rotations_n1 | `qiskit>qasm3-noinclude>pennylane` | 0.0843 | 0.9157 |
| asymmetric_rotations_n1 | `qiskit>pytket>cirq` | 0.0843 | 0.9157 |
| asymmetric_rotations_n1 | `cirq>qasm2>qiskit` | 0.0843 | 0.9157 |
| asymmetric_rotations_n1 | `cirq>qasm3>qiskit` | 0.0843 | 0.9157 |
| asymmetric_rotations_n1 | `cirq>qasm2>cirq` | 0.0843 | 0.9157 |

## Qiskit-native Grover operators

Grover operators built by Qiskit's `grover_operator` (hand-written A and oracle) and by `qiskit-algorithms`' `EstimationProblem`, with no aeonq lowering, sent through the Qiskit-source paths. Evaluated 99 of 162; phase only 53; failed once controlled 44; count-invisible 56. Sources that already differ from the reference before conversion: uniform_threshold_n2_k1 A Q (grover_operator), uniform_threshold_n2_k1 Q (grover_operator), uniform_threshold_n2_k1 controlled Q (grover_operator).

| case | program | path | status | exact | phase only | controlled equal | counts pass |
|---|---|---|---|---|---|---|---|
| bernoulli_0.25 | Q (grover_operator) | `qiskit>qasm2>qiskit` | evaluated | False | True | False | True |
| bernoulli_0.25 | Q (grover_operator) | `qiskit>qasm3>qiskit` | evaluated | False | True | False | True |
| bernoulli_0.25 | Q (grover_operator) | `qiskit>qasm2>cirq` | evaluated | False | True | False | True |
| bernoulli_0.25 | Q (grover_operator) | `qiskit>qasm3>cirq` | evaluated | False | True | False | True |
| bernoulli_0.25 | Q (grover_operator) | `qiskit>qasm3>pennylane` | unitary_failed |  | False |  |  |
| bernoulli_0.25 | Q (grover_operator) | `qiskit>qasm3-noinclude>pennylane` | evaluated | False | True | False | True |
| bernoulli_0.25 | Q (grover_operator) | `qiskit>qasm2>qsharp` | evaluated | True | False | None | True |
| bernoulli_0.25 | Q (grover_operator) | `qiskit>qasm3>qsharp` | evaluated | True | False | None | True |
| bernoulli_0.25 | Q (grover_operator) | `qiskit>pytket>cirq` | evaluated | False | True | False | True |
| bernoulli_0.25 | Q^2 (grover_operator) | `qiskit>qasm2>qiskit` | evaluated | True | False | True | True |
| bernoulli_0.25 | Q^2 (grover_operator) | `qiskit>qasm3>qiskit` | evaluated | True | False | True | True |
| bernoulli_0.25 | Q^2 (grover_operator) | `qiskit>qasm2>cirq` | evaluated | True | False | True | True |
| bernoulli_0.25 | Q^2 (grover_operator) | `qiskit>qasm3>cirq` | evaluated | True | False | True | True |
| bernoulli_0.25 | Q^2 (grover_operator) | `qiskit>qasm3>pennylane` | unitary_failed |  | False |  |  |
| bernoulli_0.25 | Q^2 (grover_operator) | `qiskit>qasm3-noinclude>pennylane` | evaluated | True | False | True | True |
| bernoulli_0.25 | Q^2 (grover_operator) | `qiskit>qasm2>qsharp` | evaluated | True | False | None | True |
| bernoulli_0.25 | Q^2 (grover_operator) | `qiskit>qasm3>qsharp` | evaluated | True | False | None | True |
| bernoulli_0.25 | Q^2 (grover_operator) | `qiskit>pytket>cirq` | evaluated | True | False | True | True |
| bernoulli_0.25 | controlled Q (grover_operator) | `qiskit>qasm2>qiskit` | evaluated | True | False | True | True |
| bernoulli_0.25 | controlled Q (grover_operator) | `qiskit>qasm3>qiskit` | evaluated | True | False | True | True |
| bernoulli_0.25 | controlled Q (grover_operator) | `qiskit>qasm2>cirq` | import_failed |  | False |  |  |
| bernoulli_0.25 | controlled Q (grover_operator) | `qiskit>qasm3>cirq` | import_failed |  | False |  |  |
| bernoulli_0.25 | controlled Q (grover_operator) | `qiskit>qasm3>pennylane` | unitary_failed |  | False |  |  |
| bernoulli_0.25 | controlled Q (grover_operator) | `qiskit>qasm3-noinclude>pennylane` | unitary_failed |  | False |  |  |
| bernoulli_0.25 | controlled Q (grover_operator) | `qiskit>qasm2>qsharp` | import_failed |  | False |  |  |
| bernoulli_0.25 | controlled Q (grover_operator) | `qiskit>qasm3>qsharp` | evaluated | False | False | None | True |
| bernoulli_0.25 | controlled Q (grover_operator) | `qiskit>pytket>cirq` | export_failed |  | False |  |  |
| bernoulli_0.25 | A Q (grover_operator) | `qiskit>qasm2>qiskit` | evaluated | False | True | False | True |
| bernoulli_0.25 | A Q (grover_operator) | `qiskit>qasm3>qiskit` | evaluated | False | True | False | True |
| bernoulli_0.25 | A Q (grover_operator) | `qiskit>qasm2>cirq` | evaluated | False | True | False | True |
| bernoulli_0.25 | A Q (grover_operator) | `qiskit>qasm3>cirq` | evaluated | False | True | False | True |
| bernoulli_0.25 | A Q (grover_operator) | `qiskit>qasm3>pennylane` | unitary_failed |  | False |  |  |
| bernoulli_0.25 | A Q (grover_operator) | `qiskit>qasm3-noinclude>pennylane` | evaluated | False | True | False | True |
| bernoulli_0.25 | A Q (grover_operator) | `qiskit>qasm2>qsharp` | evaluated | True | False | None | True |
| bernoulli_0.25 | A Q (grover_operator) | `qiskit>qasm3>qsharp` | evaluated | True | False | None | True |
| bernoulli_0.25 | A Q (grover_operator) | `qiskit>pytket>cirq` | evaluated | False | True | False | True |
| bernoulli_0.25 | Q (qiskit-algorithms) | `qiskit>qasm2>qiskit` | evaluated | False | True | False | True |
| bernoulli_0.25 | Q (qiskit-algorithms) | `qiskit>qasm3>qiskit` | evaluated | False | True | False | True |
| bernoulli_0.25 | Q (qiskit-algorithms) | `qiskit>qasm2>cirq` | unitary_failed |  | False |  |  |
| bernoulli_0.25 | Q (qiskit-algorithms) | `qiskit>qasm3>cirq` | unitary_failed |  | False |  |  |
| bernoulli_0.25 | Q (qiskit-algorithms) | `qiskit>qasm3>pennylane` | unitary_failed |  | False |  |  |
| bernoulli_0.25 | Q (qiskit-algorithms) | `qiskit>qasm3-noinclude>pennylane` | unitary_failed |  | False |  |  |
| bernoulli_0.25 | Q (qiskit-algorithms) | `qiskit>qasm2>qsharp` | evaluated | True | False | None | True |
| bernoulli_0.25 | Q (qiskit-algorithms) | `qiskit>qasm3>qsharp` | evaluated | True | False | None | True |
| bernoulli_0.25 | Q (qiskit-algorithms) | `qiskit>pytket>cirq` | import_failed |  | False |  |  |
| uniform_threshold_n2_k1 | Q (grover_operator) | `qiskit>qasm2>qiskit` | evaluated | False | True | False | True |
| uniform_threshold_n2_k1 | Q (grover_operator) | `qiskit>qasm3>qiskit` | evaluated | False | True | False | True |
| uniform_threshold_n2_k1 | Q (grover_operator) | `qiskit>qasm2>cirq` | evaluated | False | True | False | True |
| uniform_threshold_n2_k1 | Q (grover_operator) | `qiskit>qasm3>cirq` | evaluated | False | True | False | True |
| uniform_threshold_n2_k1 | Q (grover_operator) | `qiskit>qasm3>pennylane` | unitary_failed |  | False |  |  |
| uniform_threshold_n2_k1 | Q (grover_operator) | `qiskit>qasm3-noinclude>pennylane` | evaluated | False | True | False | True |
| uniform_threshold_n2_k1 | Q (grover_operator) | `qiskit>qasm2>qsharp` | evaluated | False | True | None | True |
| uniform_threshold_n2_k1 | Q (grover_operator) | `qiskit>qasm3>qsharp` | evaluated | False | True | None | True |
| uniform_threshold_n2_k1 | Q (grover_operator) | `qiskit>pytket>cirq` | evaluated | False | True | False | True |
| uniform_threshold_n2_k1 | Q^2 (grover_operator) | `qiskit>qasm2>qiskit` | evaluated | True | False | True | True |
| uniform_threshold_n2_k1 | Q^2 (grover_operator) | `qiskit>qasm3>qiskit` | evaluated | True | False | True | True |
| uniform_threshold_n2_k1 | Q^2 (grover_operator) | `qiskit>qasm2>cirq` | evaluated | True | False | True | True |
| uniform_threshold_n2_k1 | Q^2 (grover_operator) | `qiskit>qasm3>cirq` | evaluated | True | False | True | True |
| uniform_threshold_n2_k1 | Q^2 (grover_operator) | `qiskit>qasm3>pennylane` | unitary_failed |  | False |  |  |
| uniform_threshold_n2_k1 | Q^2 (grover_operator) | `qiskit>qasm3-noinclude>pennylane` | evaluated | True | False | True | True |
| uniform_threshold_n2_k1 | Q^2 (grover_operator) | `qiskit>qasm2>qsharp` | evaluated | True | False | None | True |
| uniform_threshold_n2_k1 | Q^2 (grover_operator) | `qiskit>qasm3>qsharp` | evaluated | True | False | None | True |
| uniform_threshold_n2_k1 | Q^2 (grover_operator) | `qiskit>pytket>cirq` | evaluated | True | False | True | True |
| uniform_threshold_n2_k1 | controlled Q (grover_operator) | `qiskit>qasm2>qiskit` | evaluated | True | False | True | True |
| uniform_threshold_n2_k1 | controlled Q (grover_operator) | `qiskit>qasm3>qiskit` | evaluated | True | False | True | True |
| uniform_threshold_n2_k1 | controlled Q (grover_operator) | `qiskit>qasm2>cirq` | evaluated | True | False | True | True |
| uniform_threshold_n2_k1 | controlled Q (grover_operator) | `qiskit>qasm3>cirq` | evaluated | True | False | True | True |
| uniform_threshold_n2_k1 | controlled Q (grover_operator) | `qiskit>qasm3>pennylane` | unitary_failed |  | False |  |  |
| uniform_threshold_n2_k1 | controlled Q (grover_operator) | `qiskit>qasm3-noinclude>pennylane` | unitary_failed |  | False |  |  |
| uniform_threshold_n2_k1 | controlled Q (grover_operator) | `qiskit>qasm2>qsharp` | import_failed |  | False |  |  |
| uniform_threshold_n2_k1 | controlled Q (grover_operator) | `qiskit>qasm3>qsharp` | evaluated | True | False | None | True |
| uniform_threshold_n2_k1 | controlled Q (grover_operator) | `qiskit>pytket>cirq` | export_failed |  | False |  |  |
| uniform_threshold_n2_k1 | A Q (grover_operator) | `qiskit>qasm2>qiskit` | evaluated | False | True | False | True |
| uniform_threshold_n2_k1 | A Q (grover_operator) | `qiskit>qasm3>qiskit` | evaluated | False | True | False | True |
| uniform_threshold_n2_k1 | A Q (grover_operator) | `qiskit>qasm2>cirq` | evaluated | False | True | False | True |
| uniform_threshold_n2_k1 | A Q (grover_operator) | `qiskit>qasm3>cirq` | evaluated | False | True | False | True |
| uniform_threshold_n2_k1 | A Q (grover_operator) | `qiskit>qasm3>pennylane` | unitary_failed |  | False |  |  |
| uniform_threshold_n2_k1 | A Q (grover_operator) | `qiskit>qasm3-noinclude>pennylane` | evaluated | False | True | False | True |
| uniform_threshold_n2_k1 | A Q (grover_operator) | `qiskit>qasm2>qsharp` | evaluated | False | True | None | True |
| uniform_threshold_n2_k1 | A Q (grover_operator) | `qiskit>qasm3>qsharp` | evaluated | False | True | None | True |
| uniform_threshold_n2_k1 | A Q (grover_operator) | `qiskit>pytket>cirq` | evaluated | False | True | False | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q (grover_operator) | `qiskit>qasm2>qiskit` | evaluated | False | True | False | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q (grover_operator) | `qiskit>qasm3>qiskit` | evaluated | False | True | False | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q (grover_operator) | `qiskit>qasm2>cirq` | evaluated | False | True | False | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q (grover_operator) | `qiskit>qasm3>cirq` | evaluated | False | True | False | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q (grover_operator) | `qiskit>qasm3>pennylane` | unitary_failed |  | False |  |  |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q (grover_operator) | `qiskit>qasm3-noinclude>pennylane` | unitary_failed |  | False |  |  |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q (grover_operator) | `qiskit>qasm2>qsharp` | import_failed |  | False |  |  |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q (grover_operator) | `qiskit>qasm3>qsharp` | evaluated | False | True | None | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q (grover_operator) | `qiskit>pytket>cirq` | import_failed |  | False |  |  |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q^2 (grover_operator) | `qiskit>qasm2>qiskit` | evaluated | True | False | True | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q^2 (grover_operator) | `qiskit>qasm3>qiskit` | evaluated | True | False | True | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q^2 (grover_operator) | `qiskit>qasm2>cirq` | evaluated | True | False | True | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q^2 (grover_operator) | `qiskit>qasm3>cirq` | evaluated | True | False | True | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q^2 (grover_operator) | `qiskit>qasm3>pennylane` | unitary_failed |  | False |  |  |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q^2 (grover_operator) | `qiskit>qasm3-noinclude>pennylane` | unitary_failed |  | False |  |  |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q^2 (grover_operator) | `qiskit>qasm2>qsharp` | import_failed |  | False |  |  |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q^2 (grover_operator) | `qiskit>qasm3>qsharp` | evaluated | True | False | None | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q^2 (grover_operator) | `qiskit>pytket>cirq` | import_failed |  | False |  |  |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | controlled Q (grover_operator) | `qiskit>qasm2>qiskit` | evaluated | True | False | True | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | controlled Q (grover_operator) | `qiskit>qasm3>qiskit` | evaluated | True | False | True | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | controlled Q (grover_operator) | `qiskit>qasm2>cirq` | import_failed |  | False |  |  |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | controlled Q (grover_operator) | `qiskit>qasm3>cirq` | import_failed |  | False |  |  |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | controlled Q (grover_operator) | `qiskit>qasm3>pennylane` | unitary_failed |  | False |  |  |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | controlled Q (grover_operator) | `qiskit>qasm3-noinclude>pennylane` | unitary_failed |  | False |  |  |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | controlled Q (grover_operator) | `qiskit>qasm2>qsharp` | import_failed |  | False |  |  |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | controlled Q (grover_operator) | `qiskit>qasm3>qsharp` | evaluated | False | False | None | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | controlled Q (grover_operator) | `qiskit>pytket>cirq` | export_failed |  | False |  |  |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A Q (grover_operator) | `qiskit>qasm2>qiskit` | evaluated | False | True | False | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A Q (grover_operator) | `qiskit>qasm3>qiskit` | evaluated | False | True | False | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A Q (grover_operator) | `qiskit>qasm2>cirq` | evaluated | False | True | False | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A Q (grover_operator) | `qiskit>qasm3>cirq` | evaluated | False | True | False | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A Q (grover_operator) | `qiskit>qasm3>pennylane` | unitary_failed |  | False |  |  |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A Q (grover_operator) | `qiskit>qasm3-noinclude>pennylane` | unitary_failed |  | False |  |  |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A Q (grover_operator) | `qiskit>qasm2>qsharp` | import_failed |  | False |  |  |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A Q (grover_operator) | `qiskit>qasm3>qsharp` | evaluated | False | True | None | True |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A Q (grover_operator) | `qiskit>pytket>cirq` | import_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | Q (grover_operator) | `qiskit>qasm2>qiskit` | evaluated | False | True | False | True |
| lognormal_tail_n2_k2_flag | Q (grover_operator) | `qiskit>qasm3>qiskit` | evaluated | False | True | False | True |
| lognormal_tail_n2_k2_flag | Q (grover_operator) | `qiskit>qasm2>cirq` | evaluated | False | True | False | True |
| lognormal_tail_n2_k2_flag | Q (grover_operator) | `qiskit>qasm3>cirq` | evaluated | False | True | False | True |
| lognormal_tail_n2_k2_flag | Q (grover_operator) | `qiskit>qasm3>pennylane` | unitary_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | Q (grover_operator) | `qiskit>qasm3-noinclude>pennylane` | unitary_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | Q (grover_operator) | `qiskit>qasm2>qsharp` | import_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | Q (grover_operator) | `qiskit>qasm3>qsharp` | evaluated | False | True | None | True |
| lognormal_tail_n2_k2_flag | Q (grover_operator) | `qiskit>pytket>cirq` | import_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | Q^2 (grover_operator) | `qiskit>qasm2>qiskit` | evaluated | True | False | True | True |
| lognormal_tail_n2_k2_flag | Q^2 (grover_operator) | `qiskit>qasm3>qiskit` | evaluated | True | False | True | True |
| lognormal_tail_n2_k2_flag | Q^2 (grover_operator) | `qiskit>qasm2>cirq` | evaluated | True | False | True | True |
| lognormal_tail_n2_k2_flag | Q^2 (grover_operator) | `qiskit>qasm3>cirq` | evaluated | True | False | True | True |
| lognormal_tail_n2_k2_flag | Q^2 (grover_operator) | `qiskit>qasm3>pennylane` | unitary_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | Q^2 (grover_operator) | `qiskit>qasm3-noinclude>pennylane` | unitary_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | Q^2 (grover_operator) | `qiskit>qasm2>qsharp` | import_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | Q^2 (grover_operator) | `qiskit>qasm3>qsharp` | evaluated | True | False | None | True |
| lognormal_tail_n2_k2_flag | Q^2 (grover_operator) | `qiskit>pytket>cirq` | import_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | controlled Q (grover_operator) | `qiskit>qasm2>qiskit` | evaluated | True | False | True | True |
| lognormal_tail_n2_k2_flag | controlled Q (grover_operator) | `qiskit>qasm3>qiskit` | evaluated | True | False | True | True |
| lognormal_tail_n2_k2_flag | controlled Q (grover_operator) | `qiskit>qasm2>cirq` | import_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | controlled Q (grover_operator) | `qiskit>qasm3>cirq` | import_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | controlled Q (grover_operator) | `qiskit>qasm3>pennylane` | unitary_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | controlled Q (grover_operator) | `qiskit>qasm3-noinclude>pennylane` | unitary_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | controlled Q (grover_operator) | `qiskit>qasm2>qsharp` | import_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | controlled Q (grover_operator) | `qiskit>qasm3>qsharp` | evaluated | False | False | None | True |
| lognormal_tail_n2_k2_flag | controlled Q (grover_operator) | `qiskit>pytket>cirq` | export_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | A Q (grover_operator) | `qiskit>qasm2>qiskit` | evaluated | False | True | False | True |
| lognormal_tail_n2_k2_flag | A Q (grover_operator) | `qiskit>qasm3>qiskit` | evaluated | False | True | False | True |
| lognormal_tail_n2_k2_flag | A Q (grover_operator) | `qiskit>qasm2>cirq` | evaluated | False | True | False | True |
| lognormal_tail_n2_k2_flag | A Q (grover_operator) | `qiskit>qasm3>cirq` | evaluated | False | True | False | True |
| lognormal_tail_n2_k2_flag | A Q (grover_operator) | `qiskit>qasm3>pennylane` | unitary_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | A Q (grover_operator) | `qiskit>qasm3-noinclude>pennylane` | unitary_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | A Q (grover_operator) | `qiskit>qasm2>qsharp` | import_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | A Q (grover_operator) | `qiskit>qasm3>qsharp` | evaluated | False | True | None | True |
| lognormal_tail_n2_k2_flag | A Q (grover_operator) | `qiskit>pytket>cirq` | import_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | Q (qiskit-algorithms) | `qiskit>qasm2>qiskit` | evaluated | False | True | False | True |
| lognormal_tail_n2_k2_flag | Q (qiskit-algorithms) | `qiskit>qasm3>qiskit` | evaluated | False | True | False | True |
| lognormal_tail_n2_k2_flag | Q (qiskit-algorithms) | `qiskit>qasm2>cirq` | unitary_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | Q (qiskit-algorithms) | `qiskit>qasm3>cirq` | unitary_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | Q (qiskit-algorithms) | `qiskit>qasm3>pennylane` | unitary_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | Q (qiskit-algorithms) | `qiskit>qasm3-noinclude>pennylane` | unitary_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | Q (qiskit-algorithms) | `qiskit>qasm2>qsharp` | import_failed |  | False |  |  |
| lognormal_tail_n2_k2_flag | Q (qiskit-algorithms) | `qiskit>qasm3>qsharp` | evaluated | False | True | None | True |
| lognormal_tail_n2_k2_flag | Q (qiskit-algorithms) | `qiskit>pytket>cirq` | import_failed |  | False |  |  |

## Importer gate probe: deviations from the gate formula

159 one-gate programs; 23 deviate or are rejected. All others match the formula exactly.

| importer | statement | status | equal up to phase | phase offset |
|---|---|---|---|---|
| cirq circuit_from_qasm (3) | `gphase(0.7);` | rejected |  |  |
| pennylane from_qasm3 | `gphase(0.7);` | imported | True | -1.4000 |
| qdk import_openqasm (2) | `cry(0.7) q[0], q[1];` | rejected |  |  |
| qdk import_openqasm (2) | `rx(-0.7) q[0];` | imported | True | +3.1416 |
| qdk import_openqasm (2) | `ry(-0.7) q[0];` | imported | True | +3.1416 |
| qdk import_openqasm (2) | `rz(-0.7) q[0];` | imported | True | +3.1416 |
| qdk import_openqasm (2) | `cry(-0.7) q[0], q[1];` | rejected |  |  |
| qdk import_openqasm (2) | `crz(-0.7) q[0], q[1];` | imported | False |  |
| qdk import_openqasm (2) | `rx(7.0) q[0];` | imported | True | +3.1416 |
| qdk import_openqasm (2) | `ry(7.0) q[0];` | imported | True | +3.1416 |
| qdk import_openqasm (2) | `rz(7.0) q[0];` | imported | True | +3.1416 |
| qdk import_openqasm (2) | `cry(7.0) q[0], q[1];` | rejected |  |  |
| qdk import_openqasm (2) | `crz(7.0) q[0], q[1];` | imported | False |  |
| qdk import_openqasm (3) | `rx(-0.7) q[0];` | imported | True | +3.1416 |
| qdk import_openqasm (3) | `ry(-0.7) q[0];` | imported | True | +3.1416 |
| qdk import_openqasm (3) | `rz(-0.7) q[0];` | imported | True | +3.1416 |
| qdk import_openqasm (3) | `cry(-0.7) q[0], q[1];` | imported | False |  |
| qdk import_openqasm (3) | `crz(-0.7) q[0], q[1];` | imported | False |  |
| qdk import_openqasm (3) | `rx(7.0) q[0];` | imported | True | +3.1416 |
| qdk import_openqasm (3) | `ry(7.0) q[0];` | imported | True | +3.1416 |
| qdk import_openqasm (3) | `rz(7.0) q[0];` | imported | True | +3.1416 |
| qdk import_openqasm (3) | `cry(7.0) q[0], q[1];` | imported | False |  |
| qdk import_openqasm (3) | `crz(7.0) q[0], q[1];` | imported | False |  |
