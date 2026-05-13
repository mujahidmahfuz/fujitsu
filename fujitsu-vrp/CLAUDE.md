# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Fujitsu Quantum Simulator Challenge 2026** - Logistic routing for Tokyo SME delivery.

This project develops a quantum-enhanced Vehicle Routing Problem (VRP) solver using a novel **QLNRS (Quantum Large Neighborhood Risk Search)** algorithm that combines:
- Chaotic operator selection with Lyapunov-adaptive control
- Quantum annealing/Ising machine for repair subproblems
- Risk-aware objective function for robust solutions

See `PLANS.md` for detailed implementation roadmap and `QLNRS_Complete_Paper (1).pdf` for the algorithm specification.

## Technical Stack

- **Language**: Python 3.10+
- **Classical Optimization**: Google OR-Tools
- **Quantum (Gate-based)**: Qiskit (prototyping) → Fujitsu Quantum Simulator
- **Quantum Annealing**: D-Wave neal, Fujitsu Digital Annealer
- **Constraints**: 40-qubit limit on Fujitsu simulator

## Development Commands

TODO: Add build, test, and lint commands once the project is initialized.

## Architecture

```
src/fujitsu_vrp/
├── data/           # Tokyo SME synthetic data generation
├── classical/      # OR-Tools baseline, classical LNS
├── quantum/
│   ├── qubo/       # VRP → QUBO encoding (40-qubit constraint)
│   ├── solvers/    # QAOA, VQE, Simulated QA, Ising machine
│   ├── backends/   # Qiskit, Fujitsu backends
│   └── qlnrs/      # Main algorithm: chaotic ops, Lyapunov, repair
├── analysis/       # Risk metrics, stability analysis
└── experiments/    # Benchmark framework
```

## Key Algorithm: QLNRS

1. **Destroy**: Risk-weighted or chaotic customer removal
2. **Quantum Repair**: Formulate as QUBO, solve with quantum annealing
3. **Risk Evaluation**: Multi-component risk metric
4. **Accept**: Simulated annealing criterion
5. **Lyapunov Adapt**: Adjust exploration based on solution trajectory stability