# Fujitsu Quantum Simulator Challenge 2025-26 — Refined Competition Strategy

## Competitive Intelligence Summary

I've read the competition page, the QARP features PDF, and analyzed the **2024 winners in detail**. Here are the critical findings that reshape your plan:

### What Won Last Year (and Why)

| Place | Team | Project | Qubits | Key Insight |
|-------|------|---------|--------|-------------|
| 🥇 1st | Delft Univ. | Industrial Shift Scheduling | **39** | Used **Grover's exact search**, not heuristic QAOA. Open-sourced their algorithm (QISS). |
| 🥈 2nd | TU Ilmenau | Quantum PIV (fluid analysis) | **32** | Unique domain (aerospace/CFD), not generic optimization. |
| 🥉 3rd | QunaSys | QPE for molecular states | **39** | Maxed out qubit count, demonstrated scaling beyond classical. |

### The 5 Evaluation Criteria (Ranked by Impact)

1. **Project Uniqueness** — VRP is common. We MUST differentiate hard.
2. **Business Applicability** — Real ¥en, real Tokyo, real data = strong.
3. **Algorithm Quality** — Not just "run QAOA", but novel algorithmic contributions.
4. **Qubit Utilization** — Push toward 39-40 qubits. The winners all used 32-39.
5. **QARP Usage + Feedback** — Mandatory. Must deeply integrate QARP and provide actionable feedback.

### QARP Features Available to Us

- **QAOA** (directly relevant!)
- **QPE** (Grover-related; potential for exact VRP search)
- **VQE/VQD/ADAPT-VQE** (less relevant but could be used for sub-problems)
- **Block-based modular circuit construction** (custom ansatze)
- **Multiple backends**: Qiskit, Pytket, Qulacs
- **Circuit cutting** for simulating large circuits on smaller resources
- **Gradient backpropagation** for optimization

---

## User Review Required

> [!IMPORTANT]
> **Participation Requirements**: Applicants must be **legal entities** (companies or institutions). Individual participation may not be eligible. Please confirm you have or can secure a company/institution affiliation for the application.

> [!WARNING]
> **Application Deadline**: January 30, 2026 (already passed based on current date Feb 14, 2026). The challenge runs Jan–Mar 2026. Please confirm: have you already been accepted into the challenge, or are you preparing a late/next-round application?

> [!IMPORTANT]
> **Strategy Pivot**: The 2024 winner used **Grover's exact algorithm** (not QAOA heuristics) and got 1st place with 39 qubits. Your plan focuses heavily on QAOA, which is the obvious choice for VRP — meaning many competitors will do the same. I recommend a **dual-algorithm approach**: QAOA for warm-start + Grover Adaptive Search (GAS) for exact refinement. This is algorithmically novel and directly mirrors what won last year.

> [!CAUTION]
> **Your plan claims "18% cost reduction" and "95% on-time rate"** as deliverables. These are speculative. Making unsubstantiated claims will hurt credibility with the judges. I recommend: benchmark rigorously and report honest numbers. If quantum only matches classical at small scale, that's fine — show the **scaling trajectory** that proves quantum advantage emerges at 40+ qubits.

---

## What I Changed From Your Plan (and Why)

### ✅ Kept (Your best ideas)
- Tokyo/Shibuya real-data narrative — strong business applicability
- Phased qubit scaling (5→10→20→40) — good pedagogy
- Hybrid classical-quantum decomposition — necessary for real scaling
- ARM64 Docker portability — shows Fujitsu platform awareness
- Real-time re-routing demo — compelling UX story

### 🔄 Changed (Critical improvements)

| Your Plan | My Revision | Reason |
|-----------|-------------|--------|
| Pure QAOA approach | **QAOA + Grover Adaptive Search (GAS) hybrid** | 2024 winner used Grover. Novel dual-algorithm = higher uniqueness score |
| Generic QUBO encoding | **Route-based VRP encoding** (not position-based) | Position-based `x[i][t]` needs O(n²) qubits. Route-based needs O(n·log n), enabling more stops per qubit |
| `qiskit_optimization.applications.VehicleRouting` | **Custom QUBO builder from scratch** | Using a pre-built class looks lazy to judges. Building your own shows algorithm quality |
| Fujitsu MPS emulation via Qiskit Aer | **Direct QARP integration from Phase 2 onward** | QARP usage is a judging criterion. Start early, provide deep feedback |
| Streamlit dashboard as final phase | **Jupyter-first, Streamlit as polish** | Judges read reports/notebooks. Dashboard is bonus, not core |
| Time windows as separate phase | **Integrated from Phase 2** | Time windows are your key "realistic constraint" differentiator |

### ❌ Removed (Will waste time or hurt you)

| Removed | Reason |
|---------|--------|
| LaTeX one-pager in Phase 0 | Judges want code + results, not papers. Put math in Jupyter |
| AWS Graviton2 testing | Unnecessary; Docker ARM64 build is sufficient proof |
| "Loading spinner for 5 seconds" demo trick | Judges are researchers. This will seem dishonest |
| Elaborate slide deck planning | Focus on the technical report. Fujitsu provides their own presentation format |

### ➕ Added (Your missing weapons)

| Addition | Why |
|----------|-----|
| **Grover Adaptive Search for exact VRP** | Mirrors 2024 winner's approach. Provides exact solution benchmark |
| **QARP circuit cutting** for scaling beyond 40 qubits | Shows you understand QARP's unique capabilities |
| **Warm-start QAOA** (CVaR + parameter transfer) | State-of-the-art technique that dramatically improves QAOA performance |
| **Noise-aware penalty calibration** | Shows physics depth — penatlies must account for shot noise |
| **Structured QARP feedback document** | Explicit judging criterion. Prepare detailed, actionable feedback |
| **Comparative analysis**: your algorithm vs. 2024 winner's QISS on scheduling variant | Shows awareness of prior work and positions your contribution |

---

## Proposed Changes — The Refined Phase Plan

### Phase 0: Foundation (Days 1-3)

#### [NEW] [project structure](file:///home/mujahid/logistic)

```
logistic/
├── README.md
├── requirements.txt
├── Dockerfile.arm64
├── pyproject.toml
├── src/
│   ├── __init__.py
│   ├── qubo/
│   │   ├── __init__.py
│   │   ├── vrp_qubo.py          # Core QUBO builder
│   │   ├── penalty_calibration.py
│   │   └── encodings.py          # Route-based + position-based
│   ├── solvers/
│   │   ├── __init__.py
│   │   ├── qaoa_solver.py        # QAOA with warm-start
│   │   ├── grover_solver.py      # Grover Adaptive Search
│   │   ├── hybrid_solver.py      # Classical decomposition + quantum
│   │   └── classical_baseline.py # OR-Tools wrapper
│   ├── data/
│   │   ├── __init__.py
│   │   ├── tokyo_generator.py    # OSMnx Shibuya data
│   │   └── datasets/
│   │       ├── shibuya_5stops.json
│   │       ├── shibuya_10stops.json
│   │       └── shibuya_20stops.json
│   ├── routing/
│   │   ├── __init__.py
│   │   ├── rerouter.py           # Dynamic re-routing engine
│   │   └── traffic_sim.py        # Traffic disruption simulator
│   └── visualization/
│       ├── __init__.py
│       ├── tokyo_map.py          # Folium map rendering
│       └── quantum_plots.py     # Convergence, energy landscapes
├── notebooks/
│   ├── 01_qubo_formulation.ipynb
│   ├── 02_baby_steps_5qubit.ipynb
│   ├── 03_scaling_10_20qubit.ipynb
│   ├── 04_frontier_40qubit.ipynb
│   ├── 05_fujitsu_integration.ipynb
│   └── 06_final_benchmarks.ipynb
├── benchmarks/
│   ├── benchmark.py
│   └── results/
├── tests/
│   ├── test_qubo.py
│   ├── test_solvers.py
│   └── test_data.py
├── dashboard/
│   └── app.py                    # Streamlit dashboard
└── docs/
    ├── qarp_feedback.md          # Structured QARP feedback for judges
    └── technical_report.md       # Final submission report
```

---

### Phase 1: 5-Qubit "Proof of Concept" (Days 3-5)

#### [NEW] [vrp_qubo.py](file:///home/mujahid/logistic/src/qubo/vrp_qubo.py)

Implement the core QUBO builder with:
- **Route-based encoding**: Decision variable `x[i][j]` = 1 if edge (i→j) is in the route
- **Objective**: Minimize `Σ d[i][j] · x[i][j]`
- **Constraints as penalties**:
  - Visit-once: `P₁ · Σᵢ(Σⱼ x[j][i] - 1)²`
  - Flow conservation: `P₂ · Σᵢ(Σⱼ x[i][j] - Σⱼ x[j][i])²`
  - Capacity: Binary-encoded load variables with `P₃ · max(0, load - C)²`
  - Time windows: `P₄ · max(0, arrival[i] - latest[i])²`

#### [NEW] [qaoa_solver.py](file:///home/mujahid/logistic/src/solvers/qaoa_solver.py)

- QAOA with `p=1` on Qiskit Aer `statevector_simulator`
- COBYLA optimizer
- Measure cost vs iteration
- Compare with brute-force (NumPy diagonalization)

#### [NEW] [classical_baseline.py](file:///home/mujahid/logistic/src/solvers/classical_baseline.py)

- OR-Tools VRP solver wrapper
- Metrics: solution cost, runtime, optimality gap

**Deliverable**: Jupyter notebook showing QAOA finds optimal 2-stop route, comparison table vs classical.

---

### Phase 2: Constrained Scaling (10-20 qubits, Days 5-10)

#### [MODIFY] [vrp_qubo.py](file:///home/mujahid/logistic/src/qubo/vrp_qubo.py)

Add capacity + time window constraints. Scale to 3-4 stops (10-20 qubits with route-based encoding).

#### [NEW] [penalty_calibration.py](file:///home/mujahid/logistic/src/qubo/penalty_calibration.py)

**Novel contribution**: Automatic penalty weight calibration using:
- Eigenvalue gap analysis of the QUBO matrix
- Shot-noise-aware penalty bounds: `P_min = max_cost / sqrt(n_shots)`
- Sweep P₁, P₂, P₃ and plot constraint violation rate vs solution quality

#### [NEW] [tokyo_generator.py](file:///home/mujahid/logistic/src/data/tokyo_generator.py)

- OSMnx download of Shibuya Ward driving network
- Generate realistic stop locations (convenience stores, offices)
- Compute shortest-path distance matrices
- Export as JSON with lat/lon, demand, time windows

#### [NEW] [encodings.py](file:///home/mujahid/logistic/src/qubo/encodings.py)

Implement TWO encoding strategies (for comparative analysis):
1. **Position-based**: `x[i][t]` — O(n²) qubits, standard
2. **Route-based**: `x[i][j]` — O(n·log n) for sparse graphs, better for Tokyo street networks

**Deliverable**: Penalty calibration plot, Tokyo map with 4-stop routes, encoding comparison table.

---

### Phase 3: The 20-Qubit Frontier (Days 10-17)

#### [NEW] [grover_solver.py](file:///home/mujahid/logistic/src/solvers/grover_solver.py)

**Key differentiator**: Implement Grover Adaptive Search (GAS) for VRP:
1. Classical preprocessing: Find a "good enough" solution `C*` via QAOA
2. Grover oracle: Mark all solutions with cost < `C*`
3. Iterate: Each Grover round reduces `C*`, converging to optimum
4. Uses QARP's QPE and circuit construction blocks

This mirrors the 2024 winner's approach (Grover for scheduling) applied to VRP — novel combination.

#### [NEW] [hybrid_solver.py](file:///home/mujahid/logistic/src/solvers/hybrid_solver.py)

Classical-quantum decomposition:
1. **Classical**: K-means cluster stops by geography + time window compatibility
2. **Quantum**: Solve intra-cluster TSP/VRP using QAOA + GAS
3. **Classical**: Solve inter-cluster routing (2-opt local search)
4. **Merge**: Concatenate routes with connector optimization

#### [MODIFY] [qaoa_solver.py](file:///home/mujahid/logistic/src/solvers/qaoa_solver.py)

Add warm-start techniques:
- **CVaR-QAOA**: Conditional Value-at-Risk objective (focus on tail of distribution)
- **Parameter transfer**: Use p=1 optimized parameters to initialize p=2
- **Multi-angle QAOA**: Different (γ,β) per qubit layer

**Deliverable**: Side-by-side comparison of QAOA vs GAS vs Hybrid on 4-stop Tokyo instance. GAS should find exact optimum; QAOA should be faster but approximate.

---

### Phase 4: 40-Qubit Scaling (Days 17-24)

#### [MODIFY] [hybrid_solver.py](file:///home/mujahid/logistic/src/solvers/hybrid_solver.py)

Scale to 8-stop VRP (target: 39-40 qubits):
- Decompose into 2 × 4-stop sub-problems
- Each sub-problem: ~20 qubits with QAOA + GAS refinement
- Use QARP circuit cutting to attempt single 40-qubit runs

#### [NEW] [rerouter.py](file:///home/mujahid/logistic/src/routing/rerouter.py)

Real-time re-routing engine:
- Input: current position, remaining stops, disrupted edge
- Process: Update distance matrix, re-run quantum solver with warm-start from previous solution
- Output: New route within 90 seconds
- Uses QAOA parameter transfer (previous optimal parameters → warm-start for disrupted instance)

#### [NEW] [traffic_sim.py](file:///home/mujahid/logistic/src/routing/traffic_sim.py)

Traffic disruption simulator:
- Random edge failures on Tokyo network
- Time-varying congestion multipliers (rush hour: 7-9AM, 5-7PM)
- Accident scenarios (set edge weight to ∞)

**Deliverable**: 8-stop full solution, re-routing demo video, scaling analysis.

---

### Phase 5: Fujitsu Integration & Submission (Days 24-30)

#### [NEW] [05_fujitsu_integration.ipynb](file:///home/mujahid/logistic/notebooks/05_fujitsu_integration.ipynb)

Migrate all solvers to use QARP as the backend:
- Replace `qiskit.primitives.Sampler` with QARP backend
- Use QARP's block-based circuit construction for QAOA ansatze
- Test on Fujitsu's tensor-network simulator (MPS)
- Benchmark: QARP vs Qiskit Aer for same circuits

#### [NEW] [qarp_feedback.md](file:///home/mujahid/logistic/docs/qarp_feedback.md)

Structured feedback document for judges:
- API usability (what worked, what was confusing)
- Performance comparison with Qiskit Aer
- Feature requests (e.g., native VRP support, automatic penalty tuning)
- Documentation quality assessment
- ARM64 compilation issues encountered (if any)

#### [NEW] [benchmark.py](file:///home/mujahid/logistic/benchmarks/benchmark.py)

Comprehensive benchmark suite:
- Instance sizes: 3, 4, 5, 6, 7, 8 stops
- Solvers: OR-Tools exact, OR-Tools heuristic, QAOA (p=1,2,3), GAS, Hybrid
- Metrics: Cost, runtime, success probability, qubits used, circuit depth
- Output: CSV + auto-generated Markdown report with tables and plots

#### [NEW] [app.py](file:///home/mujahid/logistic/dashboard/app.py)

Streamlit dashboard:
- Interactive Tokyo map (Folium) with route visualization
- Solver comparison panel (classical vs quantum)
- Re-routing demo: click to add accident, watch route update
- Quantum internals: circuit diagram, energy landscape, convergence

#### [NEW] [Dockerfile.arm64](file:///home/mujahid/logistic/Dockerfile.arm64)

ARM64-compatible container for deployment to Fujitsu platform.

---

## What Makes This Win

| Criterion | Our Strategy | Expected Score |
|-----------|-------------|----------------|
| **Uniqueness** | Dual-algorithm QAOA+GAS for VRP (never done before). Real Tokyo data with dynamic re-routing. | ⭐⭐⭐⭐⭐ |
| **Business Applicability** | Real ¥en savings quantified per fleet. Tokyo SME logistics is tangible. | ⭐⭐⭐⭐⭐ |
| **Algorithm Quality** | Custom QUBO with route-based encoding, warm-start CVaR-QAOA, Grover refinement, noise-aware penalties | ⭐⭐⭐⭐⭐ |
| **Qubit Utilization** | Target 39-40 qubits via hybrid decomposition + circuit cutting | ⭐⭐⭐⭐ |
| **QARP Usage** | Deep integration from Phase 2, detailed feedback document | ⭐⭐⭐⭐⭐ |

## Verification Plan

### Automated Tests

Each phase has concrete verification:

```bash
# Phase 1: Verify QAOA finds known optimal for 2-stop instance
cd /home/mujahid/logistic
python -m pytest tests/test_qubo.py -v      # QUBO construction correctness
python -m pytest tests/test_solvers.py -v    # QAOA vs brute-force on 2-stop

# Phase 2: Verify constraint satisfaction
python -m pytest tests/test_qubo.py::test_capacity_constraint -v
python -m pytest tests/test_qubo.py::test_time_window_constraint -v

# Phase 3-4: Benchmark comparison
python benchmarks/benchmark.py --sizes 3,4,5,6,7,8 --output results/
```

### Manual Verification

1. **Visual route check**: Open Jupyter notebooks, verify plotted routes on Tokyo map look reasonable (no crossings, respect one-way streets)
2. **Re-routing demo**: Run re-routing simulation in notebook, visually confirm new route avoids disrupted edge
3. **Dashboard check**: Run `streamlit run dashboard/app.py`, interact with all controls
4. **ARM64 build**: Run `docker buildx build --platform linux/arm64 -t fujitsu-vrp:latest .` and verify container starts

---

## Implementation Order

We will build this bottom-up, phase by phase. Each phase produces a working deliverable:

1. **Phase 0** → Project skeleton + dependencies
2. **Phase 1** → `vrp_qubo.py` + `qaoa_solver.py` + `classical_baseline.py` + notebook 01-02
3. **Phase 2** → Penalties, Tokyo data, encodings + notebook 03
4. **Phase 3** → GAS solver, hybrid solver, warm-start + notebook 04
5. **Phase 4** → 40-qubit scaling, re-routing + notebook 04 (extended)
6. **Phase 5** → QARP migration, dashboard, benchmarks, report

Shall we begin with Phase 0?
