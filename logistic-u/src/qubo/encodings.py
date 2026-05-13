"""
VRP QUBO Encoding Strategies.

Two encoding approaches for mapping VRP to QUBO:
1. PositionEncoding: x[i][t] = 1 if stop i is visited at position t in the tour.
   - Qubits: O(n^2), simple, standard in literature.
2. RouteEncoding: x[i][j] = 1 if edge (i->j) is in the route.
   - Qubits: O(n * log(n)) for sparse graphs, better for real road networks.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class PositionEncoding:
    """
    Position-based encoding: x[i][t] = 1 if node i is at position t in the tour.

    For n stops (excluding depot), we need n positions.
    Total qubits: n * n = n^2 (each node can be at each position).

    Constraints:
        - Each position filled by exactly one node: sum_i x[i][t] = 1 for all t
        - Each node appears at exactly one position: sum_t x[i][t] = 1 for all i
    """

    def __init__(self, n_stops: int, include_depot: bool = True):
        """
        Args:
            n_stops: Number of customer stops (excluding depot).
            include_depot: If True, depot transitions are included in the cost
                          but depot position is fixed (position 0 and n+1).
        """
        self.n_stops = n_stops
        self.include_depot = include_depot
        # Number of positions in the tour (just the customer stops)
        self.n_positions = n_stops
        # Total qubits: one binary variable per (stop, position) pair
        self.n_qubits = n_stops * n_stops

    def var_index(self, stop: int, position: int) -> int:
        """Map (stop, position) to qubit index.

        Args:
            stop: Customer stop index (0 to n_stops-1, NOT the depot).
            position: Position in tour (0 to n_positions-1).

        Returns:
            Qubit index in the QUBO matrix.
        """
        assert 0 <= stop < self.n_stops, f"Stop {stop} out of range [0, {self.n_stops})"
        assert 0 <= position < self.n_positions, f"Position {position} out of range [0, {self.n_positions})"
        return stop * self.n_positions + position

    def decode(self, bitstring: np.ndarray) -> List[int]:
        """Decode a bitstring into a tour (list of stop indices).

        Args:
            bitstring: Binary array of length n_qubits.

        Returns:
            Tour as ordered list of stop indices.
            Returns None if the bitstring is invalid (constraint violation).
        """
        assert len(bitstring) == self.n_qubits
        tour = [None] * self.n_positions
        for stop in range(self.n_stops):
            for pos in range(self.n_positions):
                idx = self.var_index(stop, pos)
                if bitstring[idx] == 1:
                    if tour[pos] is not None:
                        return None  # Two stops at same position
                    tour[pos] = stop
        if None in tour:
            return None  # Some position unfilled
        return tour

    def build_objective_qubo(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Build the objective (cost) part of the QUBO matrix.

        The cost is: sum over consecutive positions t, t+1:
            dist[tour[t]][tour[t+1]] = sum_{i,j} dist[i][j] * x[i][t] * x[j][t+1]

        For depot transitions (first and last):
            dist[depot][tour[0]] + dist[tour[-1]][depot]

        Args:
            distance_matrix: Full distance matrix including depot at index 0.
                            Shape: (n_stops+1, n_stops+1) where row/col 0 is depot.

        Returns:
            QUBO matrix of shape (n_qubits, n_qubits) for the objective.
        """
        n = self.n_stops
        Q = np.zeros((self.n_qubits, self.n_qubits))

        # Cost between consecutive positions in the tour
        for t in range(n - 1):
            for i in range(n):
                for j in range(n):
                    # x[i][t] * x[j][t+1] contributes dist[i+1][j+1]
                    # (+1 because depot is index 0 in distance_matrix, stops are 1..n)
                    idx_it = self.var_index(i, t)
                    idx_jt1 = self.var_index(j, t + 1)
                    cost = distance_matrix[i + 1][j + 1]
                    if idx_it == idx_jt1:
                        Q[idx_it][idx_it] += cost
                    elif idx_it < idx_jt1:
                        Q[idx_it][idx_jt1] += cost
                    else:
                        Q[idx_jt1][idx_it] += cost

        if self.include_depot:
            # Depot -> first position: dist[0][tour[0]]
            for i in range(n):
                idx = self.var_index(i, 0)
                Q[idx][idx] += distance_matrix[0][i + 1]

            # Last position -> depot: dist[tour[-1]][0]
            for i in range(n):
                idx = self.var_index(i, n - 1)
                Q[idx][idx] += distance_matrix[i + 1][0]

        return Q

    def build_constraint_qubo(self, penalty: float) -> np.ndarray:
        """Build constraint penalty QUBO.

        Constraints:
            1. Each stop visited exactly once: sum_t x[i][t] = 1 for all i
            2. Each position has exactly one stop: sum_i x[i][t] = 1 for all t

        Penalty form: P * (sum - 1)^2 = P * (sum^2 - 2*sum + 1)

        Args:
            penalty: Penalty weight P.

        Returns:
            QUBO matrix of shape (n_qubits, n_qubits) for constraints.
        """
        n = self.n_stops
        Q = np.zeros((self.n_qubits, self.n_qubits))

        # Constraint 1: Each stop at exactly one position
        for i in range(n):
            indices = [self.var_index(i, t) for t in range(n)]
            # (sum x - 1)^2 = sum x_a x_b - 2 sum x_a + 1
            for a in indices:
                Q[a][a] += penalty * (-2 + 1)  # linear: -2x + x^2 (x^2 = x for binary)
                # Actually: (sum - 1)^2 = sum_a sum_b x_a x_b - 2 sum_a x_a + 1
                # Diagonal (a == b): x_a^2 = x_a, so coefficient is (1 - 2) = -1
                # Off-diagonal: 2 * x_a * x_b, coefficient is 2
            Q_diag_correction = penalty * (-1)  # -2 + 1 for the x^2=x identity
            for a in indices:
                Q[a][a] = Q[a][a] - penalty * (-2 + 1) + Q_diag_correction  # fix
            # Let me redo this cleanly:
            pass

        # Reset and do it properly
        Q = np.zeros((self.n_qubits, self.n_qubits))

        # Constraint 1: Each stop visits exactly one position
        # Penalty: P * (sum_t x[i][t] - 1)^2
        for i in range(n):
            indices = [self.var_index(i, t) for t in range(n)]
            for a_pos, a_idx in enumerate(indices):
                # Diagonal: from x_a^2 = x_a in binary, and -2*x_a
                # (sum - 1)^2 = sum_a x_a^2 + 2*sum_{a<b} x_a*x_b - 2*sum_a x_a + 1
                #             = sum_a x_a + 2*sum_{a<b} x_a*x_b - 2*sum_a x_a + 1
                #             = -sum_a x_a + 2*sum_{a<b} x_a*x_b + 1
                Q[a_idx][a_idx] += penalty * (-1)  # from x_a - 2*x_a = -x_a
                for b_pos in range(a_pos + 1, len(indices)):
                    b_idx = indices[b_pos]
                    row, col = min(a_idx, b_idx), max(a_idx, b_idx)
                    Q[row][col] += penalty * 2  # from 2*x_a*x_b

        # Constraint 2: Each position has exactly one stop
        # Penalty: P * (sum_i x[i][t] - 1)^2
        for t in range(n):
            indices = [self.var_index(i, t) for i in range(n)]
            for a_pos, a_idx in enumerate(indices):
                Q[a_idx][a_idx] += penalty * (-1)
                for b_pos in range(a_pos + 1, len(indices)):
                    b_idx = indices[b_pos]
                    row, col = min(a_idx, b_idx), max(a_idx, b_idx)
                    Q[row][col] += penalty * 2

        return Q


class RouteEncoding:
    """
    Route/Edge-based encoding: x[i][j] = 1 if edge (i->j) is in the route.

    For a graph with n+1 nodes (depot + n stops), edges are (i, j) for i != j.
    Total qubits: (n+1)*n for complete graph, fewer for sparse (road) networks.

    For sparse Tokyo street networks, this is more qubit-efficient than PositionEncoding.

    Constraints:
        - Each stop has exactly one incoming edge: sum_j x[j][i] = 1
        - Each stop has exactly one outgoing edge: sum_j x[i][j] = 1
        - Subtour elimination (MTZ or similar)
    """

    def __init__(self, n_nodes: int, edges: Optional[List[Tuple[int, int]]] = None):
        """
        Args:
            n_nodes: Total number of nodes including depot (depot = node 0).
            edges: List of valid edges (i, j). If None, complete graph is assumed.
        """
        self.n_nodes = n_nodes
        self.n_stops = n_nodes - 1  # Exclude depot

        if edges is None:
            # Complete directed graph (all possible edges excluding self-loops)
            self.edges = [(i, j) for i in range(n_nodes)
                         for j in range(n_nodes) if i != j]
        else:
            self.edges = edges

        # Map edge -> qubit index
        self.edge_to_idx: Dict[Tuple[int, int], int] = {}
        for idx, edge in enumerate(self.edges):
            self.edge_to_idx[edge] = idx

        self.n_qubits = len(self.edges)

    def var_index(self, i: int, j: int) -> int:
        """Map edge (i, j) to qubit index."""
        return self.edge_to_idx[(i, j)]

    def decode(self, bitstring: np.ndarray) -> Optional[List[int]]:
        """Decode bitstring into a route.

        Args:
            bitstring: Binary array of length n_qubits.

        Returns:
            Route as list of node indices starting and ending at depot (0),
            or None if invalid.
        """
        assert len(bitstring) == self.n_qubits
        active_edges = []
        for idx, edge in enumerate(self.edges):
            if bitstring[idx] == 1:
                active_edges.append(edge)

        # Build adjacency from active edges
        adj = {}
        for i, j in active_edges:
            if i in adj:
                return None  # Multiple outgoing edges from same node
            adj[i] = j

        # Trace route from depot
        route = [0]
        current = 0
        visited = {0}
        while True:
            if current not in adj:
                return None
            nxt = adj[current]
            if nxt == 0:
                route.append(0)
                break
            if nxt in visited:
                return None  # Cycle not through depot
            visited.add(nxt)
            route.append(nxt)
            current = nxt
            if len(route) > self.n_nodes + 1:
                return None  # Infinite loop protection

        # Check all stops visited
        if len(visited) != self.n_nodes:
            return None

        return route

    def build_objective_qubo(self, distance_matrix: np.ndarray) -> np.ndarray:
        """Build objective QUBO: minimize total route distance.

        Cost = sum_{(i,j) in edges} dist[i][j] * x[i][j]

        This is purely linear in the edge variables, so only diagonal entries.

        Args:
            distance_matrix: Distance matrix of shape (n_nodes, n_nodes).

        Returns:
            QUBO matrix of shape (n_qubits, n_qubits).
        """
        Q = np.zeros((self.n_qubits, self.n_qubits))
        for (i, j), idx in self.edge_to_idx.items():
            Q[idx][idx] = distance_matrix[i][j]
        return Q

    def build_constraint_qubo(self, penalty: float) -> np.ndarray:
        """Build constraint QUBO for flow conservation.

        Constraints:
            1. Each node has exactly one outgoing edge: sum_j x[i][j] = 1
            2. Each node has exactly one incoming edge: sum_i x[i][j] = 1

        Args:
            penalty: Penalty weight.

        Returns:
            QUBO matrix of shape (n_qubits, n_qubits).
        """
        Q = np.zeros((self.n_qubits, self.n_qubits))

        # Constraint 1: Exactly one outgoing edge per node
        for node in range(self.n_nodes):
            outgoing = [self.edge_to_idx[(node, j)]
                       for j in range(self.n_nodes)
                       if (node, j) in self.edge_to_idx]
            if not outgoing:
                continue
            for a_pos, a_idx in enumerate(outgoing):
                Q[a_idx][a_idx] += penalty * (-1)
                for b_pos in range(a_pos + 1, len(outgoing)):
                    b_idx = outgoing[b_pos]
                    row, col = min(a_idx, b_idx), max(a_idx, b_idx)
                    Q[row][col] += penalty * 2

        # Constraint 2: Exactly one incoming edge per node
        for node in range(self.n_nodes):
            incoming = [self.edge_to_idx[(i, node)]
                       for i in range(self.n_nodes)
                       if (i, node) in self.edge_to_idx]
            if not incoming:
                continue
            for a_pos, a_idx in enumerate(incoming):
                Q[a_idx][a_idx] += penalty * (-1)
                for b_pos in range(a_pos + 1, len(incoming)):
                    b_idx = incoming[b_pos]
                    row, col = min(a_idx, b_idx), max(a_idx, b_idx)
                    Q[row][col] += penalty * 2

        return Q
