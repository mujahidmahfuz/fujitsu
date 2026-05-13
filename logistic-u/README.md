# Quantum-Enhanced Vehicle Routing for Tokyo SME Delivery
## Fujitsu Quantum Simulator Challenge 2025-26

Hybrid classical-quantum solution for the Vehicle Routing Problem (VRP) targeting Tokyo's SME delivery fleets. Uses QAOA and Grover Adaptive Search on QUBO formulation to optimize routes for 20-50 daily stops with realistic constraints.

### Key Features
- **Dual-Algorithm Approach**: QAOA for fast warm-start + Grover Adaptive Search for exact refinement
- **Real Tokyo Data**: Shibuya Ward street network via OpenStreetMap
- **Full VRP Constraints**: Capacity (15 parcels), time windows (9AM-8PM), dynamic traffic
- **Real-Time Re-routing**: 90-second response to traffic disruptions
- **Fujitsu-Ready**: ARM64 compatible, designed for QARP SDK migration

### Project Structure
```
src/
├── qubo/          # QUBO formulation (encodings, penalties, builder)
├── solvers/       # QAOA, Grover, Hybrid, OR-Tools baseline
├── data/          # Tokyo dataset generation (OSMnx)
├── routing/       # Dynamic re-routing engine
└── visualization/ # Maps, convergence plots, dashboards
```

### Quick Start
```bash
pip install -r requirements.txt
python -m pytest tests/ -v
```

### Tech Stack
| Layer | Tool | Purpose |
|-------|------|---------|
| Quantum SDK | Qiskit 1.2+ → Fujitsu QARP | QAOA, Grover, circuit design |
| Classical Opt | OR-Tools 9.8+ | VRP baselines |
| Graph Data | NetworkX, OSMnx | Tokyo street networks |
| Visualization | Folium, Matplotlib, Plotly | Route maps, analysis |

### Competition Target Metrics
- 39-40 qubit utilization on Fujitsu simulator
- Benchmark vs OR-Tools on 3-8 stop instances
- Real-time re-routing demo (90 seconds)
