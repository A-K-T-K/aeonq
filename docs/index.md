---
layout: home

hero:
  name: "aeonq"
  text: "Quantum Amplitude Estimation"
  tagline: Semantics-preserving amplitude estimation across Qiskit, Cirq, PennyLane, and Microsoft Q#.
  image:
    src: /logo.png
    alt: aeonq Logo
  actions:
    - theme: brand
      text: Get Started
      link: /quickstart
    - theme: alt
      text: Platform Capabilities
      link: /capabilities
    - theme: alt
      text: View on GitHub
      link: https://github.com/username/aeonq

features:
  - icon: ⚛️
    title: Cross-SDK Portability
    details: Execute seamlessly across Qiskit, Cirq, PennyLane, Microsoft Q# (QDK), and exact reference simulators with zero circuit rewrite.
  - icon: 📐
    title: Semantic Intermediate Representation
    details: Preserves composite algorithmic boundaries (A, S₀, S_χ, Q^k), enforces uniform little-endian indexing, and exact symbolic adjoints.
  - icon: 🎯
    title: Disjoint-Cube Synthesis
    details: Synthesizes arithmetic threshold predicates (x ≥ K) into at most m+1 orthogonal multi-controlled phase operations with zero ancilla overhead.
  - icon: 🛡️
    title: 66-Check Conformance Battery
    details: 43 operator, 5 measurement, and 18 sampling checks systematically uncover silent convention mismatches and 15 mutation operators.
  - icon: 📊
    title: Decoupled Statistical Inference
    details: Canonical QAE, IQAE, MLQAE, and Faster AE with standardized 2k+1 query accounting and dynamic runtime guarantee qualification.
  - icon: 📜
    title: Formal Adapter Contracts
    details: Unified 7-method interface contract separating problem formulation, lowering, native compilation, execution, and normalization.
---

## Overview

Welcome to the documentation for **aeonq**, a quantum software engineering library for semantics-preserving quantum amplitude estimation across quantum SDKs.

## Architecture Overview

`aeonq` consists of decoupled layers separating declarative problem formulation, semantic intermediate representation, cross-SDK lowering, conformance testing, and statistical inference:

![aeonq Architecture](./images/figure_01_architecture.png)

## Core Design Principles

1. **Semantics over Syntactic Lowering**: Preserves algorithmic abstractions (state preparation $A$, reflection $S_0$, oracle $S_\chi$, and Grover iterate $Q = -A S_0 A^\dagger S_\chi$) across SDK boundaries.
2. **Unified Operational Conventions**: Enforces uniform little-endian qubit indexing, consistent phase sign orientations, exact adjoints, and standardized oracle query accounting ($2k+1$ calls per shot).
3. **Decoupled Architecture**: Estimators and inference engines contain zero SDK imports, guaranteeing identical inference rules across all execution targets.
4. **Guarantees as First-Class Runtime Contracts**: Theoretical guarantees are validated against declared backend sampling capabilities, withholding idealized claims on noisy backends.

## Documentation Structure

- **[Getting Started & Quickstart](./quickstart.md)**: Installation, basic usage, creating problem specifications, and running estimation algorithms across different backends.
- **[Capabilities & Backend Adapters](./capabilities.md)**: Framework capabilities, supported SDKs (Qiskit, Cirq, PennyLane, Microsoft Q#), precision tolerances, and runtime interface contracts.
- **[Algorithms & Statistical Guarantees](./algorithms.md)**: Canonical QAE, Iterative (IQAE), Maximum Likelihood (MLQAE), Faster AE (FAE), statistical confidence intervals, and runtime guarantee qualification.
- **[Semantic Intermediate Representation & Synthesis](./intermediate_representation.md)**: Declarative IR nodes, register semantics, disjoint-cube oracle synthesis for threshold predicates, and symbolic adjoints.
- **[Conformance & Mutation Testing](./conformance_testing.md)**: The 66-check conformance battery, 15 lowering and readout mutation operators, and testing custom backend adapters.
- **[Translator Fidelity & Independent Oracles](./translator_fidelity.md)**: Real SDK converters (OpenQASM 2/3, pytket, QDK) on amplitude-estimation operators, and oracles that share no code with the semantic core.
