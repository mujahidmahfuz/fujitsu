# Project Plans and Updates

## Fujitsu Quantum Simulator Challenge 2026
**Topic**: Logistic routing for Tokyo SME delivery

---

## Current Plan (March 2026)

### Project Overview

Develop a quantum-enhanced VRP solver using novel **QLNRS (Quantum Large Neighborhood Risk Search)** algorithm:

1. **Chaotic Operator Selection** - Using Logistic/Tent/Chebyshev maps for destroy operator selection
2. **Lyapunov-Adaptive Control** - Dynamically adjusting exploration based on solution trajectory stability
3. **Quantum-Annealing-Based Repair** - Formulating repair as QUBO for quantum annealer/Ising machine
4. **Risk-Aware Objective** - Multi-component risk metric for robust solutions

### Technical Stack

- **Language**: Python 3.10+
- **Classical Solver**: Google OR-Tools
- **Quantum (Gate-based)**: Qiskit (prototyping) → Fujitsu Quantum Simulator
- **Quantum Annealing**: D-Wave neal (SA), Fujitsu Digital Annealer
- **Data**: Synthetic Tokyo SME delivery scenarios

### Constraints

- **40-qubit limit** on Fujitsu Quantum Simulator
- **1-2 month timeline** (solo developer)

---

## QLNRS Algorithm

```
Algorithm: Quantum Large Neighborhood Risk Search (QLNRS)
─────────────────────────────────────────────────────────
Input: VRP instance I, max iterations T, initial solution S₀
Output: Best solution S*

1. Initialize: S ← S₀ (from OR-Tools or greedy)
2. For t = 1 to T:
   a. Chaotic Operator Selection:
      - Select destroy operator using chaotic map
      - Map selected based on Lyapunov exponent λₜ₋₁

   b. Destroy:
      - Remove n_remove customers using selected operator
      - Methods: random, worst, related, risk-weighted

   c. Quantum Repair:
      - Formulate repair as QUBO
      - Solve using quantum annealing / Ising machine
      - Sample K solutions from quantum distribution

   d. Risk-Aware Evaluation:
      - Compute R(S) = cost(S) + α·TW_risk + β·Cap_risk + γ·Op_risk

   e. Accept:
      - Apply simulated annealing acceptance criterion

   f. Lyapunov Update:
      - Compute λₜ from solution trajectory
      - Adjust chaotic map parameters

   g. Adapt:
      - If λ < λ_min: increase exploration (higher-MLE map)
      - If λ > λ_max: increase exploitation (lower-MLE map)

3. Return best solution found
```

---

## Project Structure

```
fujitsu-vrp/
├── pyproject.toml
├── CLAUDE.md
├── PLANS.md                      # This file
├── config/
├── data/
├── notebooks/
├── src/fujitsu_vrp/
│   ├── data/                     # Tokyo SME data generation
│   ├── classical/                # OR-Tools, LNS baseline
│   ├── quantum/
│   │   ├── qubo/                 # QUBO encoding
│   │   ├── solvers/              # QAOA, VQE, QA, Ising
│   │   ├── backends/             # Qiskit, Fujitsu
│   │   └── qlnrs/                # QLNRS algorithm
│   ├── analysis/                 # Risk metrics, stability
│   └── experiments/              # Benchmarks
└── tests/
```

---

## Implementation Phases

| Phase | Week | Focus | Deliverables |
|-------|------|-------|--------------|
| 1 | 1 | Foundation | Project setup, data generator, OR-Tools baseline |
| 2 | 2 | Classical LNS | Destroy/repair operators, SA acceptance |
| 3 | 3 | Chaotic & Lyapunov | Chaotic maps, Lyapunov analysis, adaptive control |
| 4 | 4-5 | Quantum Core | QUBO encoder, QAOA/VQE, Simulated QA |
| 5 | 6-7 | QLNRS Integration | Full algorithm, Fujitsu backend, testing |
| 6 | 8-10 | Experiments | Benchmarks, ablation studies, final report |

---

## Key Technical Decisions

### Qubit-Efficient Encoding

| Encoding | Max Nodes (40 qubits) | Use Case |
|----------|----------------------|----------|
| Edge-based | 6 | Small benchmarks |
| Slot-based | 13 | Medium subproblems |
| Decomposed | 20+ | Realistic problems via QLNS |

### Quantum Approaches

1. **Gate-based (Qiskit/Fujitsu)**: QAOA, VQE for small subproblems
2. **Simulated Quantum Annealing**: Larger subproblems, diverse solutions
3. **Fujitsu Digital Annealer**: Main target for competition

### Chaotic Maps (MLE = Maximum Lyapunov Exponent)

| Map | Formula | MLE | Use Case |
|-----|---------|-----|----------|
| Logistic | x_{n+1} = r·x_n·(1-x_n) | ~0.693 | Exploration |
| Tent | x_{n+1} = 1-2\|x_n-0.5\| | ~0.693 | Exploration |
| Chebyshev | x_{n+1} = cos(k·arccos(x_n)) | ln(k) | Tunable |

---

## Dependencies

```toml
# Core
numpy, scipy, pandas

# Classical
ortools

# Quantum Gate-based
qiskit, qiskit-algorithms, qiskit-optimization
qulacs

# Quantum Annealing
dwave-neal, dimod

# Fujitsu
fujitsu-quantum

# Geospatial
geopy, shapely

# Visualization
matplotlib, seaborn

# Dev
pytest, black, ruff, mypy, jupyter
```

---

## Updates Log

### March 4, 2026
- Initial plan created
- Incorporated QLNRS paper concepts
- Added quantum annealing and Ising machine approaches
- Defined project structure and implementation phases