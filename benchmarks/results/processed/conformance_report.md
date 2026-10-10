# Local operator conformance report

The complete measured matrix distances are in `operator_conformance.csv`.
The QDK matrix-dump tolerance is 1e-6; other SDK tolerances are 1e-9.

| Backend | Passing rows | Maximum exact matrix distance |
| --- | ---: | ---: |
| cirq | 14 / 14 | 4.03e-15 |
| pennylane | 14 / 14 | 4.9e-16 |
| qiskit | 14 / 14 | 6.74e-16 |
| qsharp | 14 / 14 | 5.71e-07 |
