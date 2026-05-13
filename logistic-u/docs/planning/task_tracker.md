# Fujitsu Quantum Simulator Challenge — Tokyo SME VRP

## Planning
- [x] Read competition page & rules
- [x] Read QARP features PDF
- [x] Analyze 2024 winners for competitive intelligence
- [x] Write refined implementation plan
- [x] Get user approval on plan

## Phase 0: Foundation & QUBO Formulation
- [x] Set up project structure
- [x] Install dependencies (qiskit, ortools, networkx, etc.)
- [x] Implement clean QUBO formulation for VRP
- [x] ARM64 Docker compatibility check (Dockerfile created)

## Phase 1: Baby Steps (5-qubit)
- [x] 2-stop TSP with QAOA on numpy simulator
- [x] Classical OR-Tools baseline benchmark
- [x] Compare quantum vs classical results (via benchmark suite)

## Phase 2: Tokyo Data + Traffic Dynamics + Hybrid Solver
- [x] Tokyo data generator (`tokyo_generator.py`)
- [x] Traffic dynamics with Lyapunov exponents (`traffic_dynamics.py`)
- [x] Traffic simulation / disruption events (`traffic_sim.py`)
- [x] Hybrid classical-quantum solver (`hybrid_solver.py`)
- [x] Tests for all new modules (21/21 passing)
- [x] Penalty tuning & success probability analysis (`penalty_tuning.py`)

## Phase 3: Grover Solver + Notebooks
- [x] Grover Adaptive Search module (`grover_solver.py`)
- [x] Jupyter demo notebooks (01-02)
- [x] 4-stop full VRP with all constraints on Tokyo data
- [ ] MPS tensor-network simulator benchmarks — 🟡 **Needs Fujitsu QARP**

## Phase 4: Re-routing + 40-Qubit Scaling
- [x] Real-time re-routing engine (`rerouter.py`)
- [x] Circuit cutting for 40-qubit simulation (`circuit_cutting.py`)
- [x] QARP SDK integration layer (`qarp_integration.py`)
- [ ] 8-stop VRP via divide-and-conquer — 🟡 **Needs Fujitsu QARP (40 qubits)**

## Phase 5: Dashboard, Benchmarks & Polish
- [x] Streamlit dashboard with Tokyo map (`dashboard.py`)
- [x] Benchmark suite (`benchmark.py`)
- [x] Tests for all modules (81/81 passing)
- [x] QARP feedback document (`docs/qarp_feedback.md`)
- [x] Final presentation materials (`docs/final_presentation.md`)
