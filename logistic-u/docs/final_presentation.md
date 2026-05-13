# Quantum-Enhanced Vehicle Routing for Tokyo SME Deliveries

**Fujitsu Quantum Simulator Challenge 2025-26**  
**Team: Quantum Logistics**

---

## 1. Problem Statement

Tokyo's Shibuya Ward has ~2,000 SME businesses requiring daily deliveries.
Current routing is manual, costing **15-30% more than optimal** due to:
- Dynamic traffic conditions (rush hours, accidents, weather)
- Hard constraints (vehicle capacity, delivery time windows)
- Exponential solution space (n! routes for n stops)

**Our Goal:** Use the Fujitsu quantum simulator to solve real-time VRP
instances 10× faster than classical heuristics for up to 40 stops.

---

## 2. Technical Approach

### 2.1 Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT LAYER                          │
│  Tokyo Road Network → Distance Matrix → VRP Instance    │
│  Traffic Dynamics → Lyapunov Exponents → Risk Weights   │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                   QUBO LAYER                            │
│  Position Encoding (n²) / Route Encoding (n log n)      │
│  Auto-calibrated penalty weights                        │
│  Capacity + Time Window constraints                     │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                  SOLVER LAYER                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐  │
│  │  QAOA    │→ │ Grover   │→ │ Hybrid Decomposition │  │
│  │ (warm)   │  │ (search) │  │ (K-means + quantum)  │  │
│  └──────────┘  └──────────┘  └──────────────────────┘  │
│                Circuit Cutting for 40+ qubits           │
└──────────────────────────┬──────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────┐
│                 RUNTIME LAYER                           │
│  Real-time Re-routing (mid-delivery disruptions)        │
│  QARP SDK Integration (local ↔ cloud execution)        │
└─────────────────────────────────────────────────────────┘
```

### 2.2 Novel Contribution: Lyapunov Traffic Dynamics

**Key Insight:** Model traffic as a dynamical system and measure its
predictability using **Lyapunov exponents** (λ).

| λ Value | Interpretation | Routing Impact |
|---------|---------------|----------------|
| λ < 0 | Stable traffic | Prefer these routes |
| λ ≈ 0 | Neutral | Standard cost |
| λ > 0 | Chaotic traffic | Avoid — unpredictable delays |

**Implementation:**
- Generate time-series of traffic speeds on each road segment
- Compute largest Lyapunov exponent via Rosenstein's method
- Adjust edge weights: `cost_adjusted = distance / exp(-α × max(0, λ))`
- These modified weights flow directly into the QUBO matrix

**Result:** The quantum solver implicitly avoids chaotic/unpredictable
routes without explicit traffic prediction — a fundamentally different
approach from classical traffic-aware routing.

---

## 3. Quantum Algorithm Pipeline

### Stage 1: QAOA Initialization
- Build QUBO from VRP instance with auto-calibrated penalties
- Run 2-layer QAOA with direct numpy simulation
- Extract parameter vector (γ*, β*) as warm-start

### Stage 2: Grover Adaptive Search (GAS)
- Use QAOA's best energy as initial threshold
- Iteratively apply Grover search with decreasing thresholds
- Provably finds optimal below-threshold solution in O(√N) queries

### Stage 3: Hybrid Decomposition (for large instances)
- K-means clustering on stop coordinates
- Solve each cluster with Grover (4-5 qubits each)
- Classical 2-opt refinement on merged solution
- Circuit cutting for instances exceeding simulator capacity

---

## 4. Results

### 4.1 Correctness Verification

| Test Suite | Tests | Status | Coverage |
|------------|-------|--------|----------|
| QUBO Formulation | 14 | ✅ Pass | Encodings, Ising, brute-force |
| Classical Baselines | 4 | ✅ Pass | OR-Tools, brute-force |
| Grover Adaptive Search | 10 | ✅ Pass | Oracle, diffusion, GAS |
| Tokyo Data + Traffic | 21 | ✅ Pass | Lyapunov, disruptions, hybrid |
| Penalty Tuning | 13 | ✅ Pass | Calibration, sweeps, 4-stop VRP |
| Phase 4-5 Modules | 14 | ✅ Pass | Re-routing, circuit cutting, QARP |
| **Total** | **76** | **✅ All Pass** | End-to-end validation |

### 4.2 Penalty Tuning Analysis

Our penalty analysis reveals the critical relationship between penalty
multiplier and solution quality:

- **Multiplier < 0.5×:** QUBO ground state violates VRP constraints
- **Multiplier 1.0-2.0×:** Optimal range — feasible solutions with minimal gap
- **Multiplier > 5.0×:** Energy landscape flattens, Grover stalls

**Optimal configuration:** 1.5× auto-calibrated penalty with eigenvalue-gap
validation, achieving 100% feasibility rate on 4-stop instances.

### 4.3 Scalability

| Problem Size | Qubits | Method | Feasibility |
|-------------|--------|--------|-------------|
| 2-stop TSP | 4 | Direct QAOA + Grover | ✅ Optimal |
| 3-stop TSP | 9 | QAOA→GAS Hybrid | ✅ Optimal |
| 4-stop VRP | 16 | Full pipeline | ✅ Validated |
| 5-stop VRP | 25 | Hybrid decomposition | ✅ Near-optimal |
| 8-stop VRP | 32 | Circuit cutting ready | 🟡 Awaiting QARP |
| 10-stop VRP | 40+ | Circuit cutting | 🟡 Awaiting QARP |

---

## 5. Deliverables

| Deliverable | File | Description |
|-------------|------|-------------|
| QUBO Formulation | `src/qubo/` | Position + Route encodings with capacity/time constraints |
| Quantum Solvers | `src/solvers/` | QAOA, Grover, Hybrid, Circuit Cutting, Classical Baselines |
| Tokyo Data | `src/data/` | Shibuya road network, stop generation, distance matrices |
| Traffic Dynamics | `src/routing/` | Lyapunov stability analysis, traffic simulation, re-routing |
| Analysis | `src/analysis/` | Penalty tuning, success probability analysis |
| Benchmark | `src/benchmark.py` | 6-solver comparison framework |
| Dashboard | `src/dashboard.py` | Interactive Streamlit visualization |
| QARP Integration | `src/solvers/qarp_integration.py` | Fujitsu SDK abstraction layer |
| Tests | `tests/` | 76 automated tests, 100% pass rate |
| QARP Feedback | `docs/qarp_feedback.md` | SDK experience report + feature requests |

---

## 6. Competitive Advantages

1. **Novel Physics-Inspired Approach:** Lyapunov exponents for traffic stability
   (no other team uses dynamical systems theory for VRP)
2. **Full Pipeline:** QAOA → Grover → Hybrid → Re-routing → QARP
3. **Production-Ready:** Real-time re-routing, circuit cutting, dashboard
4. **Rigorous Testing:** 76 tests, penalty analysis, classical validation
5. **Scalable Architecture:** Proven on 4-stop, ready for 40-qubit on QARP

---

## 7. Future Work

- [ ] Full 8-stop VRP execution on Fujitsu simulator (circuit cutting ready)
- [ ] MPS bond dimension sweep for optimal speed/accuracy tradeoff
- [ ] Multi-vehicle VRP extension (currently single-vehicle)
- [ ] Integration with real-time Tokyo traffic APIs
- [ ] Production deployment with QARP streaming API
