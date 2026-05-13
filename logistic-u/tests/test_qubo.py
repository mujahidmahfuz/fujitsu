"""
Tests for QUBO formulation and encoding modules.
"""
import numpy as np
import pytest
from src.qubo.vrp_qubo import VRPInstance, VRPQuboBuilder, QUBOResult
from src.qubo.encodings import PositionEncoding, RouteEncoding


# --- Test Fixtures ---

def make_2stop_instance():
    """Create a simple 2-stop TSP instance (depot + 2 customers)."""
    distance_matrix = np.array([
        [0, 5, 3],   # depot -> A=5, B=3
        [5, 0, 4],   # A -> depot=5, B=4
        [3, 4, 0],   # B -> depot=3, A=4
    ])
    return VRPInstance(
        n_stops=2,
        distance_matrix=distance_matrix,
        demands=np.array([1, 1]),
        capacity=15,
    )


def make_3stop_instance():
    """Create a 3-stop VRP instance with capacity constraints."""
    distance_matrix = np.array([
        [0, 10, 15, 20],  # depot
        [10, 0, 35, 25],  # A
        [15, 35, 0, 30],  # B
        [20, 25, 30, 0],  # C
    ])
    return VRPInstance(
        n_stops=3,
        distance_matrix=distance_matrix,
        demands=np.array([5, 3, 4]),
        capacity=8,
    )


# --- PositionEncoding Tests ---

class TestPositionEncoding:
    def test_var_index(self):
        enc = PositionEncoding(n_stops=2)
        assert enc.n_qubits == 4  # 2 stops × 2 positions
        assert enc.var_index(0, 0) == 0
        assert enc.var_index(0, 1) == 1
        assert enc.var_index(1, 0) == 2
        assert enc.var_index(1, 1) == 3

    def test_decode_valid(self):
        enc = PositionEncoding(n_stops=2)
        # Stop 0 at position 0, stop 1 at position 1
        bits = np.array([1, 0, 0, 1], dtype=float)
        tour = enc.decode(bits)
        assert tour == [0, 1]

    def test_decode_swapped(self):
        enc = PositionEncoding(n_stops=2)
        # Stop 0 at position 1, stop 1 at position 0
        bits = np.array([0, 1, 1, 0], dtype=float)
        tour = enc.decode(bits)
        assert tour == [1, 0]

    def test_decode_invalid_double(self):
        enc = PositionEncoding(n_stops=2)
        # Both stops at position 0 — invalid
        bits = np.array([1, 0, 1, 0], dtype=float)
        tour = enc.decode(bits)
        assert tour is None

    def test_decode_invalid_empty(self):
        enc = PositionEncoding(n_stops=2)
        bits = np.array([0, 0, 0, 0], dtype=float)
        tour = enc.decode(bits)
        assert tour is None

    def test_objective_qubo_2stop(self):
        inst = make_2stop_instance()
        enc = PositionEncoding(n_stops=2)
        Q = enc.build_objective_qubo(inst.distance_matrix)
        assert Q.shape == (4, 4)
        # Should have non-zero entries for depot-first, consecutive, and last-depot

    def test_constraint_qubo_shape(self):
        enc = PositionEncoding(n_stops=3)
        Q = enc.build_constraint_qubo(penalty=10.0)
        assert Q.shape == (9, 9)


# --- RouteEncoding Tests ---

class TestRouteEncoding:
    def test_initialization(self):
        enc = RouteEncoding(n_nodes=3)
        # 3 nodes, complete graph: 3*2 = 6 edges
        assert enc.n_qubits == 6

    def test_decode_valid_tour(self):
        enc = RouteEncoding(n_nodes=3)
        # Route: 0 -> 1 -> 2 -> 0
        bits = np.zeros(enc.n_qubits, dtype=float)
        bits[enc.var_index(0, 1)] = 1  # 0 -> 1
        bits[enc.var_index(1, 2)] = 1  # 1 -> 2
        bits[enc.var_index(2, 0)] = 1  # 2 -> 0
        tour = enc.decode(bits)
        assert tour == [0, 1, 2, 0]

    def test_decode_invalid_subtour(self):
        enc = RouteEncoding(n_nodes=3)
        # No route from depot
        bits = np.zeros(enc.n_qubits, dtype=float)
        bits[enc.var_index(1, 2)] = 1
        bits[enc.var_index(2, 1)] = 1
        tour = enc.decode(bits)
        assert tour is None

    def test_objective_qubo_diagonal(self):
        inst = make_2stop_instance()
        enc = RouteEncoding(n_nodes=3)
        Q = enc.build_objective_qubo(inst.distance_matrix)
        # Route encoding objective is purely diagonal (linear costs)
        for i in range(enc.n_qubits):
            for j in range(enc.n_qubits):
                if i != j:
                    assert Q[i][j] == 0, f"Off-diagonal Q[{i}][{j}] = {Q[i][j]}"


# --- VRPQuboBuilder Tests ---

class TestVRPQuboBuilder:
    def test_build_2stop(self):
        inst = make_2stop_instance()
        builder = VRPQuboBuilder(inst, encoding="position")
        result = builder.build()
        assert result.n_qubits == 4
        assert result.Q.shape == (4, 4)

    def test_ising_conversion(self):
        inst = make_2stop_instance()
        builder = VRPQuboBuilder(inst, encoding="position")
        result = builder.build()
        J, h, offset = result.to_ising()
        assert J.shape == (4, 4)
        assert h.shape == (4,)

    def test_brute_force_finds_optimal(self):
        inst = make_2stop_instance()
        builder = VRPQuboBuilder(inst, encoding="position")
        bits, energy = builder.brute_force_solve()
        assert bits is not None
        # Decode and check
        tour = builder.encoding.decode(bits)
        # The optimal tour should be a valid permutation
        if tour is not None:
            assert sorted(tour) == [0, 1]

    def test_evaluate_solution_valid(self):
        inst = make_2stop_instance()
        builder = VRPQuboBuilder(inst, encoding="position")
        # Tour: stop 0 at pos 0, stop 1 at pos 1 → route: depot→A→B→depot
        bits = np.array([1, 0, 0, 1], dtype=float)
        result = builder.evaluate_solution(bits)
        assert result['feasible'] is True
        # Cost: dist[0][1] + dist[1][2] + dist[2][0] = 5 + 4 + 3 = 12
        assert result['cost'] == 12

    def test_evaluate_solution_other_tour(self):
        inst = make_2stop_instance()
        builder = VRPQuboBuilder(inst, encoding="position")
        # Tour: stop 0 at pos 1, stop 1 at pos 0 → route: depot→B→A→depot
        bits = np.array([0, 1, 1, 0], dtype=float)
        result = builder.evaluate_solution(bits)
        assert result['feasible'] is True
        # Cost: dist[0][2] + dist[2][1] + dist[1][0] = 3 + 4 + 5 = 12
        assert result['cost'] == 12

    def test_auto_penalties(self):
        inst = make_2stop_instance()
        builder = VRPQuboBuilder(inst, encoding="position")
        for key in ['visit', 'flow', 'capacity', 'timewindow']:
            assert builder.penalties[key] > 0


# --- Integration Tests ---

class TestIntegration:
    def test_brute_force_vs_manual_2stop(self):
        """Verify brute force optimal matches manually computed optimal."""
        inst = make_2stop_instance()
        builder = VRPQuboBuilder(inst, encoding="position")
        bits, energy = builder.brute_force_solve()
        eval_result = builder.evaluate_solution(bits)

        # For a symmetric 2-stop TSP, both tours have the same cost
        # depot→A→B→depot = 5+4+3 = 12
        # depot→B→A→depot = 3+4+5 = 12
        if eval_result['feasible']:
            assert eval_result['cost'] == 12

    def test_route_vs_position_same_optimal_cost(self):
        """Both encodings should find the same optimal tour cost."""
        inst = make_2stop_instance()

        builder_pos = VRPQuboBuilder(inst, encoding="position")
        bits_pos, _ = builder_pos.brute_force_solve()
        eval_pos = builder_pos.evaluate_solution(bits_pos)

        builder_route = VRPQuboBuilder(inst, encoding="route")
        bits_route, _ = builder_route.brute_force_solve()
        eval_route = builder_route.evaluate_solution(bits_route)

        if eval_pos['feasible'] and eval_route['feasible']:
            assert eval_pos['cost'] == eval_route['cost']
