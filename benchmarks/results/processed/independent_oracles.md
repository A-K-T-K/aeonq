# Independent oracle checks

Generated 2026-10-16T06:42:59.405039+00:00 by `benchmarks/independent_oracles.py`. Oracles come from `aeonq.oracles`, which imports no aeonq IR, synthesis, reference or adapter code. Closed forms: amplitude a, Q as a rotation by 2θ on the good/bad plane (global phase included), Q² by 4θ, controlled Q as identity ⊕ rotation, and P_good(Q^k A|0>) = sin²((2k+1)θ) for k = 0..4.

| source | closed-form passed | entrywise vs NumPy oracle passed | worst closed-form error | worst entrywise error |
|---|---|---|---|---|
| `aeonq problem builder a_true` | 4 / 4 | n/a | 0.00e+00 |  |
| `ir:reference amplitude` | 4 / 4 | n/a | 3.33e-16 |  |
| `numpy:handwritten` | 20 / 20 | (is the oracle) | 2.00e-15 |  |
| `ir:reference` | 20 / 20 | 16 / 16 | 2.00e-15 | 4.11e-16 |
| `qiskit:handwritten` | 20 / 20 | 16 / 16 | 3.00e-15 | 2.00e-15 |
| `cirq:handwritten` | 20 / 20 | 16 / 16 | 3.77e-15 | 1.01e-15 |
| `pennylane:handwritten` | 20 / 20 | 16 / 16 | 2.00e-15 | 5.84e-16 |
| `qsharp:handwritten` | 20 / 20 | 16 / 16 | 3.50e-06 | 4.68e-07 |
| `qiskit:adapter` | 20 / 20 | 16 / 16 | 3.00e-15 | 2.22e-15 |
| `numpy:qiskit-program` | 20 / 20 | 16 / 16 | 8.51e-15 | 9.27e-15 |
| `cirq:adapter` | 20 / 20 | 16 / 16 | 3.77e-15 | 9.99e-16 |
| `numpy:cirq-program` | 20 / 20 | 16 / 16 | 2.00e-15 | 2.09e-15 |
| `pennylane:adapter` | 20 / 20 | 16 / 16 | 2.00e-15 | 3.33e-16 |
| `numpy:pennylane-program` | 20 / 20 | 16 / 16 | 2.00e-15 | 0.00e+00 |
| `qsharp:adapter` | 20 / 20 | 16 / 16 | 3.50e-06 | 4.68e-07 |

## Negative controls

9 faulty variants of the oracle (variants identical to the oracle for a case are omitted). Rejected by some check: 9; rejected by a closed form alone: 8.

| case | variant | check | result |
|---|---|---|---|
| bernoulli_0.25 | Q_without_global_phase | rotation_2theta (Q) | rejected |
| bernoulli_0.25 | Q_without_global_phase | rotation_4theta (Q_power_2) | passed |
| bernoulli_0.25 | Q_without_global_phase | identity_then_rotation (controlled_Q) | rejected |
| bernoulli_0.25 | Q_without_global_phase | amplification_sin2_(2k+1)theta (A,Q) | passed |
| bernoulli_0.25 | Q_without_global_phase | entrywise_exact (Q) | rejected |
| bernoulli_0.25 | Q_without_global_phase | entrywise_exact (Q_power_2) | passed |
| bernoulli_0.25 | Q_without_global_phase | entrywise_exact (controlled_Q) | rejected |
| bernoulli_0.25 | A_angle_negated | amplitude (A) | passed |
| bernoulli_0.25 | A_angle_negated | amplification_sin2_(2k+1)theta (A,Q) | rejected |
| bernoulli_0.25 | A_angle_negated | entrywise_exact (A) | rejected |
| uniform_threshold_n2_k1 | Q_without_global_phase | rotation_2theta (Q) | rejected |
| uniform_threshold_n2_k1 | Q_without_global_phase | rotation_4theta (Q_power_2) | passed |
| uniform_threshold_n2_k1 | Q_without_global_phase | identity_then_rotation (controlled_Q) | rejected |
| uniform_threshold_n2_k1 | Q_without_global_phase | amplification_sin2_(2k+1)theta (A,Q) | passed |
| uniform_threshold_n2_k1 | Q_without_global_phase | entrywise_exact (Q) | rejected |
| uniform_threshold_n2_k1 | Q_without_global_phase | entrywise_exact (Q_power_2) | passed |
| uniform_threshold_n2_k1 | Q_without_global_phase | entrywise_exact (controlled_Q) | rejected |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q_without_global_phase | rotation_2theta (Q) | rejected |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q_without_global_phase | rotation_4theta (Q_power_2) | passed |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q_without_global_phase | identity_then_rotation (controlled_Q) | rejected |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q_without_global_phase | amplification_sin2_(2k+1)theta (A,Q) | passed |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q_without_global_phase | entrywise_exact (Q) | rejected |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q_without_global_phase | entrywise_exact (Q_power_2) | passed |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q_without_global_phase | entrywise_exact (controlled_Q) | rejected |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q_qubits_reversed | rotation_2theta (Q) | rejected |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q_qubits_reversed | rotation_4theta (Q_power_2) | rejected |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q_qubits_reversed | amplification_sin2_(2k+1)theta (A,Q) | rejected |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q_qubits_reversed | entrywise_exact (Q) | rejected |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | Q_qubits_reversed | entrywise_exact (Q_power_2) | rejected |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A_angle_negated | amplitude (A) | passed |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A_angle_negated | amplification_sin2_(2k+1)theta (A,Q) | passed |
| distribution_0.1_0.2_0.3_0.4_marked_1_2 | A_angle_negated | entrywise_exact (A) | rejected |
| lognormal_tail_n2_k2_flag | Q_without_global_phase | rotation_2theta (Q) | rejected |
| lognormal_tail_n2_k2_flag | Q_without_global_phase | rotation_4theta (Q_power_2) | passed |
| lognormal_tail_n2_k2_flag | Q_without_global_phase | identity_then_rotation (controlled_Q) | rejected |
| lognormal_tail_n2_k2_flag | Q_without_global_phase | amplification_sin2_(2k+1)theta (A,Q) | passed |
| lognormal_tail_n2_k2_flag | Q_without_global_phase | entrywise_exact (Q) | rejected |
| lognormal_tail_n2_k2_flag | Q_without_global_phase | entrywise_exact (Q_power_2) | passed |
| lognormal_tail_n2_k2_flag | Q_without_global_phase | entrywise_exact (controlled_Q) | rejected |
| lognormal_tail_n2_k2_flag | Q_qubits_reversed | rotation_2theta (Q) | rejected |
| lognormal_tail_n2_k2_flag | Q_qubits_reversed | rotation_4theta (Q_power_2) | rejected |
| lognormal_tail_n2_k2_flag | Q_qubits_reversed | amplification_sin2_(2k+1)theta (A,Q) | rejected |
| lognormal_tail_n2_k2_flag | Q_qubits_reversed | entrywise_exact (Q) | rejected |
| lognormal_tail_n2_k2_flag | Q_qubits_reversed | entrywise_exact (Q_power_2) | rejected |
| lognormal_tail_n2_k2_flag | A_angle_negated | amplitude (A) | passed |
| lognormal_tail_n2_k2_flag | A_angle_negated | amplification_sin2_(2k+1)theta (A,Q) | rejected |
| lognormal_tail_n2_k2_flag | A_angle_negated | entrywise_exact (A) | rejected |

Variants that no closed form rejects: negative:A_angle_negated on distribution_0.1_0.2_0.3_0.4_marked_1_2. The amplitude and amplification laws depend on A only through P_good, and at a = 1/2 every sin²((2k±1)π/4) equals 1/2, so a sign change in A is invisible to them there; the entrywise comparison with the hand-written oracle rejects it.
