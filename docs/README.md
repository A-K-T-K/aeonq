# aeonq Documentation

The documentation for **aeonq** is powered by [VitePress](https://vitepress.dev).

## Local Development & Build

To develop or build the documentation locally:

```bash
# Install dependencies
npm install

# Start local development server with hot module reloading
npm run docs:dev

# Build production static site
npm run docs:build

# Preview production build locally
npm run docs:preview
```

## Documentation Contents

- **[Getting Started & Quickstart](quickstart.md)**: Installation, basic usage, creating problem specifications, and running estimation algorithms across different backends.
- **[Capabilities & Backend Adapters](capabilities.md)**: Framework capabilities, supported SDKs (Qiskit, Cirq, PennyLane, Microsoft Q#), precision tolerances, and runtime interface contracts.
- **[Algorithms & Statistical Guarantees](algorithms.md)**: Canonical QAE, Iterative (IQAE), Maximum Likelihood (MLQAE), Faster AE (FAE), statistical confidence intervals, and runtime guarantee qualification.
- **[Semantic Intermediate Representation & Synthesis](intermediate_representation.md)**: Declarative IR nodes, register semantics, disjoint-cube oracle synthesis for threshold predicates, and symbolic adjoints.
- **[Conformance & Mutation Testing](conformance_testing.md)**: The 66-check conformance battery, 15 lowering and readout mutation operators, and testing custom backend adapters.
- **[Translator Fidelity & Independent Oracles](translator_fidelity.md)**: Real SDK converters (OpenQASM 2/3, pytket, QDK) on amplitude-estimation operators, and oracles that share no code with the semantic core.
