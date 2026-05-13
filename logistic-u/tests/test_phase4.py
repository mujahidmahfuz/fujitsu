"""
Tests for Phase 4-5 modules: re-routing, circuit cutting, QARP, benchmarks.
"""
import numpy as np
import pytest
from src.qubo.vrp_qubo import VRPQuboBuilder, VRPInstance
from src.routing.rerouter import RerouteEngine, RerouteRequest
from src.solvers.circuit_cutting import QUBOPartitioner, CircuitCuttingExecutor
from src.solvers.qarp_integration import QARPInterface, QARPConfig
from src.benchmark import VRPBenchmark


def _make_instance(n_stops=3):
    if n_stops == 3:
        dist = np.array([
            [0, 10, 15, 20],
            [10, 0, 12, 18],
            [15, 12, 0, 14],
            [20, 18, 14, 0],
        ], dtype=float)
        demands = np.array([1, 1, 1])
    elif n_stops == 4:
        dist = np.array([
            [0, 10, 15, 20, 25],
            [10, 0, 12, 18, 22],
            [15, 12, 0, 14, 16],
            [20, 18, 14, 0, 11],
            [25, 22, 16, 11, 0],
        ], dtype=float)
        demands = np.array([1, 1, 1, 1])
    else:
        n = n_stops + 1
        rng = np.random.RandomState(42)
        dist = rng.uniform(5, 30, (n, n))
        dist = (dist + dist.T) / 2
        np.fill_diagonal(dist, 0)
        demands = rng.randint(1, 3, n_stops)
    return VRPInstance(
        n_stops=n_stops,
        distance_matrix=dist,
        demands=demands,
        capacity=max(15, n_stops * 3),
    )


# === Re-routing Tests ===

class TestRerouteEngine:
    def test_basic_reroute(self):
        inst = _make_instance(3)
        engine = RerouteEngine(max_stops_per_quantum=3, seed=42)

        # Simulate: at stop 1, remaining stops [2, 3]
        disrupted_dm = inst.distance_matrix.copy()
        disrupted_dm[1][2] *= 5  # Block edge 1→2

        request = RerouteRequest(
            current_position=1,
            remaining_stops=[2, 3],
            completed_stops=[1],
            disrupted_edges=[(1, 2)],
            original_route=[0, 1, 2, 3, 0],
            original_cost=100,
            current_time_minutes=600,
        )

        result = engine.reroute(request, inst, disrupted_dm)
        assert result.feasible
        assert result.new_cost > 0
        assert result.new_route[-1] == 0  # Ends at depot
        assert result.reroute_time_ms > 0

    def test_empty_remaining(self):
        inst = _make_instance(3)
        engine = RerouteEngine(seed=42)

        request = RerouteRequest(
            current_position=3,
            remaining_stops=[],
            completed_stops=[1, 2, 3],
            disrupted_edges=[],
            original_route=[0, 1, 2, 3, 0],
            original_cost=100,
            current_time_minutes=600,
        )

        result = engine.reroute(request, inst, inst.distance_matrix)
        assert result.method == "trivial"
        assert result.feasible

    def test_reroute_stats(self):
        inst = _make_instance(3)
        engine = RerouteEngine(seed=42)

        # Do two reroutes
        for pos in [1, 2]:
            request = RerouteRequest(
                current_position=pos,
                remaining_stops=[s for s in [1, 2, 3] if s != pos],
                completed_stops=[pos],
                disrupted_edges=[(0, 1)],
                original_route=[0, 1, 2, 3, 0],
                original_cost=100,
                current_time_minutes=600,
            )
            engine.reroute(request, inst, inst.distance_matrix)

        stats = engine.get_reroute_stats()
        assert stats["n_reroutes"] == 2
        assert stats["avg_reroute_ms"] > 0


# === Circuit Cutting Tests ===

class TestCircuitCutting:
    def test_no_cut_needed(self):
        """Small QUBO shouldn't need cutting."""
        inst = _make_instance(2)
        builder = VRPQuboBuilder(inst, encoding='position')
        qubo = builder.build()

        part = QUBOPartitioner(max_fragment_qubits=20)
        result = part.partition(qubo.Q)

        assert len(result.fragments) == 1
        assert result.n_cuts == 0
        assert result.overhead_factor == 1

    def test_partition_large_qubo(self):
        """Larger QUBO should be partitioned."""
        # Create a 16-qubit QUBO
        Q = np.random.RandomState(42).randn(16, 16)
        Q = (Q + Q.T) / 2

        part = QUBOPartitioner(max_fragment_qubits=10)
        result = part.partition(Q, n_fragments=2)

        assert len(result.fragments) == 2
        assert result.max_fragment_qubits <= 10
        assert result.n_cuts >= 0
        # All qubits should be covered
        all_qubits = set()
        for f in result.fragments:
            all_qubits.update(f.qubit_indices)
        assert all_qubits == set(range(16))

    def test_fragment_execution(self):
        Q = np.random.RandomState(42).randn(8, 8)
        Q = (Q + Q.T) / 2

        part = QUBOPartitioner(max_fragment_qubits=5)
        cut = part.partition(Q, n_fragments=2)

        executor = CircuitCuttingExecutor(seed=42)
        result = executor.solve_fragments(cut)

        assert len(result["bitstring"]) == 8
        assert result["n_fragments"] == len(cut.fragments)

    def test_group_based_partition(self):
        """Test partitioning with qubit groups."""
        Q = np.random.RandomState(42).randn(12, 12)
        Q = (Q + Q.T) / 2

        groups = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]]
        part = QUBOPartitioner(max_fragment_qubits=8)
        result = part.partition(Q, n_fragments=2, qubit_groups=groups)

        assert len(result.fragments) >= 2
        all_q = set()
        for f in result.fragments:
            all_q.update(f.qubit_indices)
        assert all_q == set(range(12))


# === QARP Integration Tests ===

class TestQARPInterface:
    def test_qarp_format_conversion(self):
        inst = _make_instance(2)
        builder = VRPQuboBuilder(inst, encoding='position')
        qubo = builder.build()

        qarp = QARPInterface(QARPConfig(n_qubits=qubo.n_qubits))
        fmt = qarp.to_qarp_format(qubo)

        assert fmt["format"] == "qarp_ising_v1"
        assert fmt["n_qubits"] == qubo.n_qubits
        assert "h" in fmt["ising"]
        assert "J" in fmt["ising"]
        assert len(fmt["ising"]["h"]) == qubo.n_qubits

    def test_export_problem(self, tmp_path):
        inst = _make_instance(2)
        builder = VRPQuboBuilder(inst, encoding='position')
        qubo = builder.build()

        qarp = QARPInterface()
        filepath = str(tmp_path / "test_problem.json")
        qarp.export_problem(qubo, filepath)

        import json
        with open(filepath) as f:
            data = json.load(f)
        assert data["n_qubits"] == qubo.n_qubits

    def test_local_grover_simulation(self):
        inst = _make_instance(2)
        builder = VRPQuboBuilder(inst, encoding='position')
        qubo = builder.build()

        qarp = QARPInterface(QARPConfig(n_qubits=qubo.n_qubits, backend="local"))
        result = qarp.submit_grover(qubo, threshold=0)

        assert result.best_bitstring is not None
        assert result.simulator_type == "local_grover"


# === Benchmark Tests ===

class TestBenchmark:
    def test_generate_instance(self):
        bm = VRPBenchmark(seed=42)
        inst = bm.generate_instance(3)
        assert inst.n_stops == 3
        assert inst.distance_matrix.shape == (4, 4)

    def test_run_small_benchmark(self):
        bm = VRPBenchmark(seed=42)
        report = bm.run_benchmark(
            sizes=[2],
            solvers=["brute_force", "grover_gas"],
            n_trials=1,
        )
        assert len(report.results) >= 2
        assert report.timestamp is not None

    def test_format_table(self):
        bm = VRPBenchmark(seed=42)
        report = bm.run_benchmark(
            sizes=[2],
            solvers=["brute_force"],
            n_trials=1,
        )
        table = bm.format_table(report)
        assert "brute_force" in table
        assert "Solver" in table

    def test_save_report(self, tmp_path):
        bm = VRPBenchmark(seed=42)
        report = bm.run_benchmark(sizes=[2], solvers=["brute_force"])
        filepath = str(tmp_path / "report.json")
        bm.save_report(report, filepath)

        import json
        with open(filepath) as f:
            data = json.load(f)
        assert "results" in data
