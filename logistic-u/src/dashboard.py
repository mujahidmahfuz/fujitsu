"""
Quantum VRP Dashboard — Streamlit App.

Interactive visualization for the Tokyo SME Delivery VRP
quantum optimization solution.

Sections:
1. Problem Setup — load/generate Tokyo delivery instances
2. Solver Comparison — run & compare classical vs quantum
3. Traffic Dynamics — Lyapunov stability visualization
4. Re-routing Demo — simulate disruption & re-optimization
5. Benchmark Results — performance tables & charts

Run: streamlit run src/dashboard.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import time
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.tokyo_generator import TokyoDatasetGenerator, ShibuyaNetworkGenerator
from src.qubo.vrp_qubo import VRPQuboBuilder, VRPInstance
from src.solvers.classical_baseline import ClassicalBaseline
from src.solvers.grover_solver import GroverAdaptiveSearch, QAOAGroverHybrid
from src.solvers.hybrid_solver import HybridSolver
from src.solvers.circuit_cutting import QUBOPartitioner, CircuitCuttingExecutor
from src.routing.traffic_dynamics import (
    TrafficVelocityField, LyapunovEstimator, StabilityWeightedGraph
)
from src.routing.traffic_sim import TrafficSimulator
from src.routing.rerouter import RerouteEngine, RerouteRequest

# --- Page Config ---
st.set_page_config(
    page_title="Quantum VRP — Tokyo Delivery",
    page_icon="🗼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    .metric-card {
        background: rgba(255,255,255,0.05);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255,255,255,0.1);
    }
    h1 { color: #4ecdc4 !important; }
    h2 { color: #ff6b6b !important; }
    .stMetric label { color: #aaa !important; }
</style>
""", unsafe_allow_html=True)


# --- Sidebar ---
st.sidebar.title("🗼 Settings")

n_stops = st.sidebar.slider("Number of Stops", 2, 8, 3)
seed = st.sidebar.number_input("Random Seed", value=42, step=1)
encoding = st.sidebar.selectbox("QUBO Encoding", ["position", "route"])
risk_aversion = st.sidebar.slider("Risk Aversion (α)", 0.0, 2.0, 0.5, 0.1)
time_of_day = st.sidebar.slider("Time of Day (hour)", 0, 23, 8)


# --- Header ---
st.title("🗼 Quantum VRP — Tokyo SME Delivery Optimization")
st.markdown("""
> **Fujitsu Quantum Simulator Challenge 2025-26** — Using QAOA + Grover Adaptive Search
> to optimize delivery routes in Shibuya Ward with traffic dynamics analysis.
""")


# --- Generate Instance ---
@st.cache_data
def generate_data(n, s):
    gen = TokyoDatasetGenerator(seed=s)
    dataset = gen.generate_dataset(n)
    instance = gen.to_vrp_instance(dataset)
    return dataset, instance, gen


dataset, instance, gen = generate_data(n_stops, seed)


# === SECTION 1: Problem Overview ===
st.header("1. Problem Instance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Stops", dataset.n_stops)
col2.metric("Total Demand", f"{dataset.total_demand} parcels")
col3.metric("Vehicle Capacity", dataset.vehicle_capacity)
col4.metric("Depot", "Shibuya Station")

# Stop details
stops_df = pd.DataFrame(dataset.stops)
if 'name' in stops_df.columns:
    display_cols = ['name', 'type', 'demand', 'address']
    display_cols = [c for c in display_cols if c in stops_df.columns]
    st.dataframe(stops_df[display_cols], use_container_width=True)

# Distance matrix heatmap
st.subheader("Distance Matrix")
dm = np.array(dataset.distance_matrix)
labels = ["Depot"] + [f"S{i+1}" for i in range(n_stops)]

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')
im = ax.imshow(dm, cmap='magma')
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, color='white', fontsize=10)
ax.set_yticklabels(labels, color='white', fontsize=10)
ax.set_title("Distance Matrix (meters)", color='white', fontsize=14, fontweight='bold')
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f'{dm[i,j]:.0f}', ha='center', va='center',
                fontsize=8, color='white' if dm[i,j] < dm.max()/2 else 'black')
cbar = fig.colorbar(im, ax=ax)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
st.pyplot(fig)
plt.close()


# === SECTION 2: Solver Comparison ===
st.header("2. Solver Comparison")

if st.button("🚀 Run All Solvers", type="primary"):
    results = {}

    # Classical brute-force
    with st.spinner("Running classical brute-force..."):
        t0 = time.time()
        bf = ClassicalBaseline(instance).brute_force_tsp()
        results["Brute Force"] = {
            "cost": bf.total_cost, "time_ms": (time.time()-t0)*1000,
            "route": bf.routes[0], "type": "Classical"
        }

    # Grover
    with st.spinner("Running Grover Adaptive Search..."):
        builder = VRPQuboBuilder(instance, encoding=encoding)
        qubo = builder.build()
        t0 = time.time()
        gas = GroverAdaptiveSearch(max_gas_iterations=10, seed=seed)
        gas_r = gas.solve(qubo)
        gas_eval = builder.evaluate_solution(gas_r.optimal_bitstring)
        results["Grover GAS"] = {
            "cost": gas_eval["cost"] if gas_eval["feasible"] else gas_r.optimal_cost,
            "time_ms": (time.time()-t0)*1000,
            "feasible": gas_eval["feasible"],
            "type": "Quantum",
            "qubits": qubo.n_qubits,
            "iterations": gas_r.n_iterations,
        }

    # QAOA → GAS
    with st.spinner("Running QAOA→GAS hybrid..."):
        t0 = time.time()
        qg = QAOAGroverHybrid(max_gas_iterations=10, seed=seed)
        qg_r = qg.solve(qubo)
        qg_eval = builder.evaluate_solution(qg_r["optimal_bitstring"])
        results["QAOA→GAS"] = {
            "cost": qg_eval["cost"] if qg_eval["feasible"] else qg_r["optimal_cost"],
            "time_ms": (time.time()-t0)*1000,
            "type": "Quantum Hybrid",
        }

    # Hybrid decomposition
    if n_stops >= 4:
        with st.spinner("Running hybrid decomposition..."):
            t0 = time.time()
            hs = HybridSolver(max_stops_per_quantum=3, seed=seed)
            hs_r = hs.solve(instance, use_quantum=True)
            results["Hybrid Decomp"] = {
                "cost": hs_r.total_cost,
                "time_ms": (time.time()-t0)*1000,
                "routes": hs_r.routes,
                "type": "Hybrid Q-C",
            }

    # Display results
    st.subheader("Results")
    res_df = pd.DataFrame([
        {
            "Solver": name,
            "Cost (m)": f"{r['cost']:.1f}",
            "Time (ms)": f"{r['time_ms']:.1f}",
            "Type": r.get("type", ""),
            "Gap %": f"{(r['cost'] - results['Brute Force']['cost']) / results['Brute Force']['cost'] * 100:.1f}%"
        }
        for name, r in results.items()
    ])
    st.dataframe(res_df, use_container_width=True)

    # Bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#1a1a2e')
    for ax in [ax1, ax2]:
        ax.set_facecolor('#1a1a2e')

    names = list(results.keys())
    costs = [results[n]["cost"] for n in names]
    times = [results[n]["time_ms"] for n in names]
    colors = ['#4ecdc4', '#ff6b6b', '#ffd93d', '#a29bfe'][:len(names)]

    ax1.barh(names, costs, color=colors)
    ax1.set_xlabel("Cost (meters)", color='white')
    ax1.set_title("Solution Quality", color='white', fontweight='bold')
    ax1.tick_params(colors='white')

    ax2.barh(names, times, color=colors)
    ax2.set_xlabel("Runtime (ms)", color='white')
    ax2.set_title("Solver Speed", color='white', fontweight='bold')
    ax2.tick_params(colors='white')

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()


# === SECTION 3: Traffic Dynamics ===
st.header("3. Traffic Dynamics — Lyapunov Analysis 🔥")

@st.cache_data
def compute_traffic_profile(_seed):
    gen = ShibuyaNetworkGenerator(grid_rows=4, grid_cols=4, seed=_seed)
    G = gen.generate()
    tvf = TrafficVelocityField(G, seed=_seed)
    swg = StabilityWeightedGraph(G, tvf, risk_aversion=0.5)

    times_arr = np.arange(0, 24*60, 30)
    mean_lya = []
    pct_chaotic = []
    mean_stability = []

    for t in times_arr:
        s = swg.get_stability_summary(int(t))
        mean_lya.append(s['mean_lyapunov'])
        pct_chaotic.append(s['pct_chaotic'])
        mean_stability.append(s['mean_stability'])

    return times_arr, mean_lya, pct_chaotic, mean_stability

times_arr, mean_lya, pct_chaotic, mean_stability = compute_traffic_profile(seed)

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    hours = times_arr / 60
    ax.plot(hours, mean_lya, color='#ff6b6b', linewidth=2)
    ax.axhline(y=0, color='white', linestyle='--', alpha=0.3)
    ax.fill_between(hours, mean_lya, 0,
                    where=np.array(mean_lya) > 0,
                    color='#ff6b6b', alpha=0.3, label='Chaotic (λ>0)')
    ax.fill_between(hours, mean_lya, 0,
                    where=np.array(mean_lya) <= 0,
                    color='#4ecdc4', alpha=0.3, label='Stable (λ≤0)')
    ax.axvspan(7, 9, color='red', alpha=0.1)
    ax.axvspan(17, 19, color='red', alpha=0.1)
    ax.set_xlabel('Hour', color='white')
    ax.set_ylabel('Mean Lyapunov λ', color='white')
    ax.set_title('Traffic Stability Profile', color='white', fontweight='bold')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    ax.set_xlim(0, 24)
    st.pyplot(fig)
    plt.close()

with col2:
    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#1a1a2e')

    ax.plot(hours, pct_chaotic, color='#ffd93d', linewidth=2)
    ax.axvspan(7, 9, color='red', alpha=0.1, label='Rush hour')
    ax.axvspan(17, 19, color='red', alpha=0.1)
    ax.set_xlabel('Hour', color='white')
    ax.set_ylabel('% Chaotic Edges', color='white')
    ax.set_title('Traffic Chaos Percentage', color='white', fontweight='bold')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#1a1a2e', edgecolor='white', labelcolor='white')
    ax.set_xlim(0, 24)
    st.pyplot(fig)
    plt.close()

st.info(f"📊 At {time_of_day}:00 — Mean λ = {mean_lya[time_of_day*2]:.3f}, "
        f"Chaotic edges: {pct_chaotic[time_of_day*2]:.0f}%, "
        f"Mean stability: {mean_stability[time_of_day*2]:.3f}")


# === SECTION 4: Re-routing Demo ===
st.header("4. Real-Time Re-routing Demo")

if n_stops >= 3 and st.button("🔄 Simulate Disruption & Re-route"):
    with st.spinner("Solving original route..."):
        bf = ClassicalBaseline(instance).brute_force_tsp()
        original_route = bf.routes[0]

    st.write(f"**Original route:** {original_route}")
    st.write(f"**Original cost:** {bf.total_cost:.0f} meters")

    with st.spinner("Generating disruption & re-routing..."):
        # Create disrupted distance matrix
        disrupted_dm = instance.distance_matrix.copy()
        # Block a random edge
        blocked_i = original_route[1]
        blocked_j = original_route[2] if len(original_route) > 2 else 0
        disrupted_dm[blocked_i][blocked_j] *= 5.0
        disrupted_dm[blocked_j][blocked_i] *= 5.0

        # Re-route
        engine = RerouteEngine(max_stops_per_quantum=3, seed=seed)
        request = RerouteRequest(
            current_position=original_route[1],
            remaining_stops=[s for s in original_route[2:-1] if s != 0],
            completed_stops=[original_route[1]],
            disrupted_edges=[(blocked_i, blocked_j)],
            original_route=original_route,
            original_cost=bf.total_cost,
            current_time_minutes=time_of_day * 60,
        )

        result = engine.reroute(request, instance, disrupted_dm)

    col1, col2, col3 = st.columns(3)
    col1.metric("New Route Cost", f"{result.new_cost:.0f}m")
    col2.metric("Cost Saved", f"{result.cost_saving:.0f}m",
               delta=f"{result.improvement_pct:.1f}%")
    col3.metric("Re-route Time", f"{result.reroute_time_ms:.1f}ms")

    st.write(f"**New route:** {result.new_route}")
    st.write(f"**Method:** {result.method}")
    st.success(f"✅ {result.disruption_summary}")


# === SECTION 5: QUBO Details ===
st.header("5. QUBO Matrix Visualization")

builder = VRPQuboBuilder(instance, encoding=encoding)
qubo = builder.build()

col1, col2, col3 = st.columns(3)
col1.metric("Qubits", qubo.n_qubits)
col2.metric("Search Space", f"2^{qubo.n_qubits} = {2**qubo.n_qubits:,}")
col3.metric("Encoding", encoding.title())

fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#1a1a2e')
ax.set_facecolor('#1a1a2e')
Q_sym = (qubo.Q + qubo.Q.T) / 2
im = ax.imshow(Q_sym, cmap='RdBu_r', aspect='auto')
ax.set_title(f'QUBO Matrix ({qubo.n_qubits} qubits)', color='white',
             fontsize=14, fontweight='bold')
ax.set_xlabel('Qubit', color='white')
ax.set_ylabel('Qubit', color='white')
ax.tick_params(colors='white')
cbar = fig.colorbar(im, ax=ax)
cbar.ax.yaxis.set_tick_params(color='white')
plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='white')
st.pyplot(fig)
plt.close()


# --- Footer ---
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Quantum VRP Solution for Fujitsu Quantum Simulator Challenge 2025-26</p>
    <p>QAOA + Grover Adaptive Search + Lyapunov Traffic Dynamics</p>
</div>
""", unsafe_allow_html=True)
