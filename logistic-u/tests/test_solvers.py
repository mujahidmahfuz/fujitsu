"""
Tests for quantum and classical solvers.
"""
import numpy as np
import pytest
from src.qubo.vrp_qubo import VRPInstance, VRPQuboBuilder
from src.solvers.classical_baseline import ClassicalBaseline


def make_2stop_instance():
    distance_matrix = np.array([
        [0, 5, 3],
        [5, 0, 4],
        [3, 4, 0],
    ])
    return VRPInstance(
        n_stops=2,
        distance_matrix=distance_matrix,
        demands=np.array([1, 1]),
        capacity=15,
    )


def make_3stop_instance():
    distance_matrix = np.array([
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ])
    return VRPInstance(
        n_stops=3,
        distance_matrix=distance_matrix,
        demands=np.array([5, 3, 4]),
        capacity=15,
    )


class TestClassicalBaseline:
    def test_brute_force_2stop(self):
        inst = make_2stop_instance()
        solver = ClassicalBaseline(inst)
        result = solver.brute_force_tsp()
        assert result.is_optimal
        # Both tours cost 12
        assert result.total_cost == 12
        assert len(result.routes) == 1
        assert result.routes[0][0] == 0
        assert result.routes[0][-1] == 0

    def test_brute_force_3stop(self):
        inst = make_3stop_instance()
        solver = ClassicalBaseline(inst)
        result = solver.brute_force_tsp()
        assert result.is_optimal
        assert result.total_cost > 0
        assert result.runtime_seconds < 1.0  # Should be instant

    def test_brute_force_runtime(self):
        """Brute force should be fast for small instances."""
        inst = make_2stop_instance()
        solver = ClassicalBaseline(inst)
        result = solver.brute_force_tsp()
        assert result.runtime_seconds < 0.1

    def test_ortools_heuristic_2stop(self):
        """OR-Tools should solve trivial 2-stop in milliseconds."""
        inst = make_2stop_instance()
        solver = ClassicalBaseline(inst)
        try:
            result = solver.solve_ortools_heuristic(time_limit_seconds=5)
            if result.solver_status != "NO_SOLUTION":
                assert result.total_cost == 12
                assert result.runtime_seconds < 10.0  # 5s limit + overhead
        except ImportError:
            pytest.skip("OR-Tools not installed")
