"""Inspect measured Qiskit compile options for a powered Grover experiment."""

from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.benchmarks.problems import bernoulli
from aeonq.compiler import AECompiler
from aeonq.ir import GroverPower, Sequence
from aeonq.reference import grover_for


case = bernoulli(0.25)
q = grover_for(case.problem)
ir = Sequence(
    (case.problem.state_preparation, GroverPower(q, 2, q.n_qubits)), q.n_qubits
)
compiled = AECompiler().compile(ir, QiskitAdapter())
print("selected options:", compiled.options)
print("selected cost:", compiled.selected_cost)
print("native metrics:", compiled.metrics)
