"""
Tests for Grover Adaptive Search solver.
"""
import numpy as np
import pytest
from src.qubo.vrp_qubo import VRPQuboBuilder, VRPInstance
from src.solvers.grover_solver import (
    GroverOracle, GroverDiffusion, GroverAdaptiveSearch, QAOAGroverHybrid
)


def _make_instance(n_stops=2):
    """Create a small VRP instance for testing."""
    if n_stops == 2:
        dist = np.array([
            [0, 10, 15],
            [10, 0, 12],
            [15, 12, 0],
        ], dtype=float)
        demands = np.array([1, 1])
    elif n_stops == 3:
        dist = np.array([
            [0, 10, 15, 20],
            [10, 0, 12, 18],
            [15, 12, 0, 14],
            [20, 18, 14, 0],
        ], dtype=float)
        demands = np.array([1, 1, 1])
    return VRPInstance(
        n_stops=n_stops,
        distance_matrix=dist,
        demands=demands,
        capacity=15,
    )


class TestGroverOracle:
    def test_oracle_marks_correct_states(self):
        costs = np.array([10.0, 5.0, 8.0, 15.0])
        oracle = GroverOracle(costs, threshold=9.0)
        assert oracle.n_marked == 2  # costs 5.0 and 8.0 are below 9.0

    def test_oracle_phase_flip(self):
        costs = np.array([10.0, 5.0, 8.0, 15.0])
        oracle = GroverOracle(costs, threshold=9.0)
        state = np.array([1.0, 1.0, 1.0, 1.0], dtype=complex)
        result = oracle.apply(state)
        # Marked states should be flipped
        assert result[0] == 1.0   # cost=10 >= 9, not flipped
        assert result[1] == -1.0  # cost=5 < 9, flipped
        assert result[2] == -1.0  # cost=8 < 9, flipped
        assert result[3] == 1.0   # cost=15 >= 9, not flipped

    def test_oracle_update_threshold(self):
        costs = np.array([10.0, 5.0, 8.0, 15.0])
        oracle = GroverOracle(costs, threshold=9.0)
        assert oracle.n_marked == 2
        oracle.update_threshold(6.0)
        assert oracle.n_marked == 1  # only cost=5.0 below 6.0


class TestGroverDiffusion:
    def test_diffusion_on_uniform(self):
        """Diffusion on uniform superposition should return the same state."""
        n = 3
        N = 2**n
        diff = GroverDiffusion(n)
        state = np.full(N, 1.0/np.sqrt(N), dtype=complex)
        result = diff.apply(state)
        np.testing.assert_allclose(np.abs(result), np.abs(state), atol=1e-10)


class TestGroverAdaptiveSearch:
    def test_gas_finds_minimum(self):
        """GAS should find the global QUBO minimum."""
        inst = _make_instance(2)
        builder = VRPQuboBuilder(inst, encoding='position')
        qubo = builder.build()

        gas = GroverAdaptiveSearch(max_gas_iterations=10, seed=42)
        result = gas.solve(qubo)

        # Compare with brute-force
        bf_bits, bf_energy = builder.brute_force_solve()

        assert result.optimal_cost <= bf_energy + 1e-6

    def test_gas_with_warm_start(self):
        """GAS with warm start should converge faster."""
        inst = _make_instance(2)
        builder = VRPQuboBuilder(inst, encoding='position')
        qubo = builder.build()

        bf_bits, bf_energy = builder.brute_force_solve()

        # Warm start with the true optimum
        gas = GroverAdaptiveSearch(max_gas_iterations=10, seed=42)
        result = gas.solve(qubo, warm_start_bitstring=bf_bits)

        assert result.optimal_cost <= bf_energy + 1e-6
        assert result.n_iterations <= 5  # Should converge quickly

    def test_gas_result_structure(self):
        inst = _make_instance(2)
        builder = VRPQuboBuilder(inst, encoding='position')
        qubo = builder.build()

        gas = GroverAdaptiveSearch(seed=42)
        result = gas.solve(qubo)

        assert result.n_qubits == qubo.n_qubits
        assert result.runtime_seconds > 0
        assert len(result.threshold_history) >= 1
        assert result.method == "grover_adaptive_search"


class TestQAOAGroverHybrid:
    def test_hybrid_pipeline(self):
        inst = _make_instance(2)
        builder = VRPQuboBuilder(inst, encoding='position')
        qubo = builder.build()

        # Simulate QAOA warm-start
        bf_bits, bf_energy = builder.brute_force_solve()
        qaoa_result = {"bitstring": bf_bits, "cost": bf_energy}

        hybrid = QAOAGroverHybrid(max_gas_iterations=5, seed=42)
        result = hybrid.solve(qubo, qaoa_result=qaoa_result)

        assert result["method"] == "qaoa_grover_hybrid"
        assert result["optimal_cost"] <= bf_energy + 1e-6
        assert result["runtime_seconds"] > 0

    def test_hybrid_no_warmstart(self):
        inst = _make_instance(2)
        builder = VRPQuboBuilder(inst, encoding='position')
        qubo = builder.build()

        hybrid = QAOAGroverHybrid(seed=42)
        result = hybrid.solve(qubo)

        assert result["optimal_cost"] is not None
        assert result["n_qubits"] == qubo.n_qubits


class TestGASOn3Stops:
    def test_3stop_finds_optimum(self):
        """Test on a larger (3-stop) instance."""
        inst = _make_instance(3)
        builder = VRPQuboBuilder(inst, encoding='position')
        qubo = builder.build()

        bf_bits, bf_energy = builder.brute_force_solve()

        gas = GroverAdaptiveSearch(max_gas_iterations=15, seed=42)
        result = gas.solve(qubo)

        # GAS should find same or better energy
        assert result.optimal_cost <= bf_energy + 1e-6
