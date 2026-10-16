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
- **Pooling**: Consecutive rounds at the same power, shot count and branch form a segment, and each round's interval uses every count in its segment. Without pooling, a repeated power (when `max_power` binds or a scheduler prefers it) can only intersect intervals no narrower than a single round's, and the estimate stalls however many rounds are allowed.
- **Error Budget**: With at most $B$ = `max_rounds` $\times$ (`max_resamples_per_round` + 1) batches and $B>1$, the union bound spends $\alpha/(2B)$ on each fresh interval and $\alpha/(B(B-1))$ on each pooled one, at most $\alpha$ in total. For $B=1$, the single interval uses failure probability $\alpha$. A segment's power and shot count are fixed before its data arrive, so each pooled interval is an exact Clopper--Pearson interval for a fixed number of fresh shots.
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
- **Convergence**: Reports `CONVERGED` when the full $k=0$ interval width is at most $\epsilon$ and the point estimate lies inside it, and `INSUFFICIENT_BUDGET` while the width is not yet reached. The width criterion is $\epsilon$ rather than $2\epsilon$ because the point estimate is the MLE, not the midpoint: an estimate inside a width-$\epsilon$ interval is within $\epsilon$ of $a$ whenever the interval covers. An MLE outside the interval means the interval missed or the MLE aliased, so that run reports `INCONCLUSIVE` and no guarantee.

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
- **Interval**: Hull of the likelihood-ratio set calibrated by the $\chi^2_1$ approximation. Candidate likelihood modes are refined before locating the outer threshold crossings, so accepted peaks narrower than the initial grid spacing still contribute to the interval.
- **Numerical Failure**: If the fit cannot resolve a valid interval, the result is `INCONCLUSIVE`, carries a warning, and reports the full amplitude range $[0,1]$.
- **Guarantees**: Annotated as asymptotic (carrying no finite-sample coverage guarantees). With few evaluation qubits and shots the coverage can fall below nominal: on the exact reference backend, $m=3$ and 256 shots covered $a=0.1$ in 91.7% of 1000 runs at $\alpha=0.05$. Use IQAE or Faster AE when a finite-sample interval is required.

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
- **Amplitude Attenuation**: Uses an additional ancilla qubit to attenuate the good-state amplitude $\sqrt{a}$ by a factor of 5, so the success probability becomes $a'=a/25$. This ensures that the effective phase $\theta \le \arcsin(1/5) < 0.252$ falls inside the required initial domain.
- **Two-Stage Cosine Reconstruction**: Evaluates first-stage and second-stage cosine circuits to achieve quadratic quantum speedup with minimal constant factors.
- **Iteration Count**: By default `max_iterations` is derived from $\epsilon$ as the fewest iterations whose final second-stage interval is at most $2\epsilon$ wide in the worst case. Iteration $l$ leaves a phase window of $(2\pi/3)/(2^{l+1}+2)$, and the attenuation stretches phase widths by up to $25\sin(2\arcsin(1/5)) = 2\sqrt{24}$ in amplitude, so $\epsilon = 0.01$ needs $l = 10$. Passing a smaller `max_iterations` caps circuit depth at the cost of `INSUFFICIENT_BUDGET` for larger amplitudes.

```python
result = estimate(
    problem,
    algorithm="faster_ae",
    backend=backend,
    epsilon=0.01,
    alpha=0.05,
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

