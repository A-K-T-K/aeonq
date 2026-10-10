# Estimation Algorithms & Statistical Guarantees

`aeonq` implements four core quantum amplitude estimation algorithms decoupled from any specific SDK runtime.

---

## 1. Overview of Supported Estimators

| Algorithm | Method Key | Coherent Control? | Statistical Interval Basis | Convergence Criterion |
| :--- | :--- | :---: | :--- | :--- |
| **Canonical QAE** | `"canonical_qae"` | Yes (QPE) | Asymptotic likelihood-ratio ($\chi^2_1$) | Fixed evaluation qubits |
| **Iterative QAE (IQAE)** | `"iqae"` | No | Exact Clopper--Pearson + union bound | Full width $\le 2\epsilon$ |
| **Maximum Likelihood (MLQAE)** | `"mlqae"` | No | Exact Clopper--Pearson ($k=0$ shots) | Full width $\le \epsilon$ |
| **Faster AE (FAE)** | `"faster_ae"` | No | Multi-stage trigonometric intersection | Stage budget / width $\le 2\epsilon$ |

---

## 2. Algorithm Details & Options

### Iterative Amplitude Estimation (IQAE)
IQAE eliminates quantum phase estimation by querying powers $Q^k$ sequentially across rounds. In each round, feasible confidence intervals are contracted via exact Clopper--Pearson binomial bounds.
- **Monotonicity**: The Grover power index $k$ advances only when the feasible confidence interval is guaranteed to lie within a monotonic branch of $\sin^2((2k+1)\theta)$.
- **Error Budget**: Error probabilities are distributed across rounds via the union bound.
- **Midpoint Return**: Upon converging to an amplitude interval width of at most $2\epsilon$, the interval midpoint is returned.

```python
result = estimate(
    problem,
    algorithm="iqae",
    backend=backend,
    epsilon=0.01,         # Target accuracy
    alpha=0.05,           # Significance level (95% confidence)
    shots_per_round=128,  # Measurement shots per round
    seed=1,
)
```

### Maximum-Likelihood Amplitude Estimation (MLQAE)
MLQAE samples from a predefined, non-adaptive power schedule (e.g., $k \in \{0, 1, 2, 4, 8\}$):
- **Point Estimate**: Computed via numerical optimization over the complete phase domain $[0, \pi/2]$ to avoid restricting the estimator inside an incorrect branch.
- **Interval Guarantee**: In contrast to asymptotic Fisher information approximations (which can collapse near boundary probabilities $a \approx 0$ or $a \approx 1$), `aeonq` constructs an exact Clopper--Pearson interval derived from the unamplified ($k=0$) evaluation shots.
- **Convergence**: Reports `CONVERGED` when the full $k=0$ interval width is at most $\epsilon$; otherwise reports `INSUFFICIENT_BUDGET`.

```python
result = estimate(
    problem,
    algorithm="mlqae",
    backend=backend,
    epsilon=0.05,
    alpha=0.05,
    powers=(0, 1, 2, 4, 8),
    shots_per_power=256,
    seed=1,
)
```

### Canonical QAE (QPE-based)
Canonical estimation couples powers of $Q$ with quantum phase estimation over $m$ evaluation qubits followed by an inverse Quantum Fourier Transform (QFT).
- **Point Estimate**: Maximum-likelihood amplitude under the exact phase estimation outcome distribution (modeled as a mixture of Fejér kernels).
- **Interval**: Likelihood-ratio confidence interval calibrated by the $\chi^2_1$ approximation.
- **Guarantees**: Annotated as asymptotic (carrying no finite-sample coverage guarantees).

```python
result = estimate(
    problem,
    algorithm="canonical_qae",
    backend=backend,
    evaluation_qubits=4,
    shots=512,
    seed=1,
)
```

### Faster Amplitude Estimation (FAE)
Implements the two-stage cosine reconstruction of Nakaji (2020):
- **Amplitude Attenuation**: Uses an additional ancilla qubit to attenuate the amplitude by a factor of 5 ($a' = a / 5$), ensuring that the effective phase $\theta \le \arcsin(1/5) < 0.252$ falls inside the required initial domain.
- **Two-Stage Cosine Reconstruction**: Evaluates first-stage and second-stage cosine circuits to achieve quadratic quantum speedup with minimal constant factors.

```python
result = estimate(
    problem,
    algorithm="faster_ae",
    backend=backend,
    epsilon=0.01,
    alpha=0.05,
    max_iterations=4,
    seed=1,
)
```

---

## 3. Standardized Oracle Query Accounting

Different SDKs account oracle queries inconsistently:
- Some report only the Grover power exponent $k$.
- Canonical implementations often report only $2^m - 1$ controlled iterations.

In `aeonq`, query complexity is standardized to the **total calls to state preparation and its adjoint ($A$ and $A^\dagger$)**:
$$\text{Queries per shot at power } k = 2k + 1$$

Because each Grover iterate $Q = -A S_0 A^\dagger S_\chi$ applies both $A^\dagger$ and $A$, evaluated at power $k$ this requires $2k$ calls, plus the initial state preparation $A$, yielding exactly $2k + 1$ queries per shot.

