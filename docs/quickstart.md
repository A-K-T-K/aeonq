# Getting Started with aeonq

This guide covers installing `aeonq`, constructing amplitude estimation problems, and running estimators across multiple quantum development kits.

---

## 1. Installation

`aeonq` requires **Python 3.13+**. Pinned dependencies are maintained in `requirements-lock.txt`.

```bash
# Clone the repository
git clone https://github.com/<your-username>/aeonq.git
cd aeonq

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install locked dependencies
pip install -r requirements-lock.txt

# Install aeonq in editable mode
pip install --no-build-isolation -e .
```

---

## 2. Basic Example: Single-Qubit Bernoulli Estimation

Here is how to estimate an unknown success probability $a \in [0, 1]$ using Iterative Quantum Amplitude Estimation (IQAE) on the exact reference simulator:

```python
from aeonq import estimate
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.benchmarks.problems import bernoulli

# Create a Bernoulli problem with true amplitude a = 0.25
case = bernoulli(0.25)

# Run IQAE with target error epsilon = 0.01 and significance alpha = 0.05
result = estimate(
    case.problem,
    algorithm="iqae",
    backend=ReferenceAdapter(),
    epsilon=0.01,
    alpha=0.05,
    shots_per_round=128,
    seed=42,
)

print(f"True amplitude:      {case.a_true}")
print(f"Estimated amplitude: {result.estimate:.5f}")
print(f"Confidence interval: [{result.confidence_or_credible_interval[0]:.5f}, "
      f"{result.confidence_or_credible_interval[1]:.5f}]")
print(f"Interval type:       {result.interval_type}")
print(f"Convergence status:  {result.convergence_status}")
print(f"Guarantee status:    {result.guarantee_status}")
print(f"Oracle queries:      {result.total_oracle_queries}")
```

---

## 3. Multi-Qubit Problem with Structured Threshold Predicate

`aeonq` allows defining objective predicates directly on multi-qubit registers, such as evaluating whether an $m$-qubit value meets or exceeds an integer threshold ($x \ge K$):

```python
from aeonq import estimate
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.benchmarks.problems import independent_threshold

# Uniform 3-qubit state where marked states satisfy x >= 5
case = independent_threshold(n_qubits=3, threshold=5)

# Estimate with Maximum-Likelihood Amplitude Estimation (MLQAE)
result = estimate(
    case.problem,
    algorithm="mlqae",
    backend=ReferenceAdapter(),
    powers=(0, 1, 2, 4),
    shots_per_power=256,
    epsilon=0.05,
    seed=101,
)

print(result.summary())
```

---

## 4. Running Across Multiple Quantum SDKs

A single semantic problem can be executed seamlessly on **Qiskit**, **Cirq**, **PennyLane**, and **Microsoft Q#** without rewriting the circuit or manually adjusting endianness and phase conventions:

```python
from aeonq import estimate
from aeonq.adapters.reference import ReferenceAdapter
from aeonq.adapters.qiskit import QiskitAdapter
from aeonq.adapters.cirq import CirqAdapter
from aeonq.adapters.pennylane import PennyLaneAdapter
from aeonq.adapters.qsharp import QSharpAdapter
from aeonq.benchmarks.problems import bernoulli

problem = bernoulli(0.30).problem

backends = [
    ReferenceAdapter(),
    QiskitAdapter(),
    CirqAdapter(),
    PennyLaneAdapter(),
    QSharpAdapter(),
]

for backend in backends:
    res = estimate(
        problem,
        algorithm="mlqae",
        backend=backend,
        powers=(0, 1),
        shots_per_power=128,
        epsilon=0.1,
        seed=7,
    )
    backend_name = type(backend).__name__
    print(f"[{backend_name:18s}] Estimate: {res.estimate:.4f} | Total Shots: {res.total_shots}")
```

---

## 5. Inspecting Estimation Results

All estimators return an `AEResult` object providing comprehensive inference diagnostics and resource metadata:

```python
result = estimate(...)

# Point estimate and intervals
result.estimate                           # float: point estimate
result.confidence_or_credible_interval    # tuple[float, float]: confidence bounds
result.interval_type                      # str: description of statistical interval

# Convergence & Guarantees
result.convergence_status                 # "CONVERGED", "INSUFFICIENT_BUDGET", "INCONCLUSIVE"
result.guarantee_status                   # "THEORETICAL_GUARANTEE_SATISFIED", "NOT_APPLICABLE"
result.assumptions                        # tuple[str]: theoretical assumptions verified

# Resource Accounting
result.total_shots                        # int: total physical circuit shots executed
result.total_oracle_queries               # int: standardized 2k+1 query count
result.powers_used                        # tuple[int]: schedule of powers evaluated
result.compile_time                       # float: seconds in lowering and native compilation
result.execution_time                     # float: seconds in simulation/sampling

# Formatted human-readable summary
print(result.summary())
```

