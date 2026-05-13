"""
VRP QUBO Builder — Core module for constructing QUBO matrices from VRP instances.

Supports:
- Capacity constraints (truck load limits)
- Time window constraints (delivery time ranges)
- Multiple encoding strategies (position-based, route-based)
- Configurable penalty weights with automatic calibration

The QUBO is constructed as:
    Q_total = Q_objective + P1*Q_visit_once + P2*Q_flow + P3*Q_capacity + P4*Q_timewindow

References:
    - Lucas (2014), "Ising formulations of many NP problems", arXiv:1302.5843
    - Borowski et al. (2020), "New Hybrid Quantum Annealing Algorithms for VRP"
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from src.qubo.encodings import PositionEncoding, RouteEncoding


@dataclass
class VRPInstance:
    """A complete VRP problem instance.

    Attributes:
        n_stops: Number of customer stops (excluding depot).
        distance_matrix: Distance matrix of shape (n_stops+1, n_stops+1).
                        Index 0 is the depot.
        demands: Demand (parcel count or weight) for each stop.
                Shape: (n_stops,). Indexed 0..n_stops-1 for customers.
        capacity: Maximum truck capacity.
        time_windows: Optional list of (earliest, latest) tuples for each stop.
                     Time in minutes from start of day.
        service_times: Optional service time at each stop (minutes).
        travel_times: Optional travel time matrix (minutes). If None, derived
                     from distance_matrix with assumed speed.
        stop_names: Optional human-readable names for stops.
        stop_coords: Optional (lat, lon) coordinates for each stop.
    """
    n_stops: int
    distance_matrix: np.ndarray
    demands: np.ndarray
    capacity: float
    time_windows: Optional[List[Tuple[float, float]]] = None
    service_times: Optional[np.ndarray] = None
    travel_times: Optional[np.ndarray] = None
    stop_names: Optional[List[str]] = None
    stop_coords: Optional[List[Tuple[float, float]]] = None

    def __post_init__(self):
        n = self.n_stops + 1  # Including depot
        assert self.distance_matrix.shape == (n, n), \
            f"Distance matrix shape {self.distance_matrix.shape} != ({n}, {n})"
        assert len(self.demands) == self.n_stops, \
            f"Demands length {len(self.demands)} != {self.n_stops}"
        if self.time_windows is not None:
            assert len(self.time_windows) == self.n_stops, \
                f"Time windows length {len(self.time_windows)} != {self.n_stops}"

    def total_demand(self) -> float:
        return float(np.sum(self.demands))

    def is_feasible(self) -> bool:
        """Check if the instance is feasible (total demand <= capacity for single vehicle)."""
        return self.total_demand() <= self.capacity


@dataclass
class QUBOResult:
    """Result of QUBO construction.

    Attributes:
        Q: The QUBO matrix (upper triangular).
        n_qubits: Number of qubits (size of Q).
        encoding: The encoding used.
        penalties: Dictionary of penalty weights used.
        offset: Constant energy offset from constraint penalties.
    """
    Q: np.ndarray
    n_qubits: int
    encoding: Any
    penalties: Dict[str, float]
    offset: float = 0.0

    def to_ising(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Convert QUBO to Ising Hamiltonian.

        QUBO: E = x^T Q x  (x ∈ {0,1}^n)
        Ising: E = s^T J s + h^T s + c  (s ∈ {-1,+1}^n)

        Substitution: x = (1 - s) / 2  →  x = (1 + s) / 2 for standard convention.

        Returns:
            J: Coupling matrix (n x n), upper triangular.
            h: Local field vector (n,).
            offset: Constant energy offset.
        """
        n = self.n_qubits
        Q = self.Q

        # Make Q symmetric for conversion
        Q_sym = (Q + Q.T) / 2.0

        # Ising conversion: x_i = (1 + s_i) / 2
        # E = sum_{i<j} Q_ij x_i x_j + sum_i Q_ii x_i
        # Substituting x = (1+s)/2:
        # x_i x_j = (1 + s_i)(1 + s_j)/4 = (1 + s_i + s_j + s_i*s_j)/4
        # x_i = (1 + s_i)/2

        J = np.zeros((n, n))
        h = np.zeros(n)
        offset = self.offset

        for i in range(n):
            # Diagonal term: Q_ii * x_i = Q_ii * (1 + s_i) / 2
            h[i] += Q_sym[i][i] / 2.0
            offset += Q_sym[i][i] / 2.0

            for j in range(i + 1, n):
                # Off-diagonal: Q_ij * x_i * x_j
                # = Q_ij * (1 + s_i + s_j + s_i*s_j) / 4
                q = Q_sym[i][j]
                J[i][j] += q / 4.0
                h[i] += q / 4.0
                h[j] += q / 4.0
                offset += q / 4.0

        return J, h, offset


class VRPQuboBuilder:
    """Builds QUBO matrices for Vehicle Routing Problems.

    Supports both position-based and route-based encodings with
    capacity and time window constraints.
    """

    def __init__(
        self,
        instance: VRPInstance,
        encoding: str = "position",
        penalties: Optional[Dict[str, float]] = None,
    ):
        """
        Args:
            instance: The VRP problem instance.
            encoding: Either "position" or "route".
            penalties: Dictionary with keys 'visit', 'flow', 'capacity', 'timewindow'.
                      If None, uses automatic calibration.
        """
        self.instance = instance
        self.encoding_type = encoding

        # Create encoding
        if encoding == "position":
            self.encoding = PositionEncoding(instance.n_stops)
        elif encoding == "route":
            self.encoding = RouteEncoding(instance.n_stops + 1)
        else:
            raise ValueError(f"Unknown encoding: {encoding}. Use 'position' or 'route'.")

        # Set penalty weights
        if penalties is None:
            self.penalties = self._auto_calibrate_penalties()
        else:
            self.penalties = penalties

    def _auto_calibrate_penalties(self) -> Dict[str, float]:
        """Automatically calibrate penalty weights based on problem scaling.

        Strategy: Set penalties proportional to max distance to ensure
        constraint violations always cost more than the worst valid route.
        """
        max_dist = np.max(self.instance.distance_matrix)
        n = self.instance.n_stops

        # Base penalty: should dominate the maximum possible objective value
        base = max_dist * n * 1.5

        return {
            'visit': base,        # Each-stop-visited-once constraint
            'flow': base,         # Flow conservation (route encoding)
            'capacity': base * 2, # Capacity is harder to satisfy
            'timewindow': base,   # Time window constraint
        }

    def build(self) -> QUBOResult:
        """Build the complete QUBO matrix.

        Returns:
            QUBOResult with the QUBO matrix and metadata.
        """
        # Objective: minimize total distance
        Q_obj = self.encoding.build_objective_qubo(self.instance.distance_matrix)

        # Structural constraints (visit-once, flow conservation)
        Q_struct = self.encoding.build_constraint_qubo(self.penalties['visit'])

        # Start with objective + structural constraints
        Q_total = Q_obj + Q_struct

        # Add capacity constraint if demands are non-trivial
        if np.any(self.instance.demands > 0):
            Q_cap = self._build_capacity_qubo()
            Q_total += Q_cap

        # Add time window constraints if specified
        if self.instance.time_windows is not None:
            Q_tw = self._build_timewindow_qubo()
            Q_total += Q_tw

        # Make upper triangular (standard QUBO form)
        Q_upper = np.triu(Q_total + Q_total.T) - np.diag(np.diag(Q_total))

        return QUBOResult(
            Q=Q_upper,
            n_qubits=self.encoding.n_qubits,
            encoding=self.encoding,
            penalties=self.penalties,
        )

    def _build_capacity_qubo(self) -> np.ndarray:
        """Build capacity constraint QUBO.

        For position encoding:
            At each position t, the cumulative load must not exceed capacity.
            We use slack variables to encode: sum_{t'<=t} demand[tour[t']] + slack = C

        For simplicity in early phases, we use a soft penalty:
            P * max(0, total_demand_on_route - C)^2
            where total_demand = sum_i demand[i] * (sum_t x[i][t])

        Since we have a single vehicle visiting all stops, total demand is fixed.
        The capacity constraint matters when we have MULTIPLE vehicles or
        when partition-based encoding is used.

        For single-vehicle: if total_demand > capacity, the instance is infeasible.
        We still add a soft penalty to penalize partial assignments that exceed capacity.
        """
        n = self.encoding.n_qubits
        Q = np.zeros((n, n))

        if self.encoding_type == "position":
            # For single vehicle: penalize if the tour includes too many heavy stops
            # This is relevant when we extend to multi-vehicle later.
            # For now, we add pairwise penalties for stops whose combined demand > capacity
            P = self.penalties['capacity']
            demands = self.instance.demands
            cap = self.instance.capacity

            for i in range(self.instance.n_stops):
                for j in range(i + 1, self.instance.n_stops):
                    if demands[i] + demands[j] > cap:
                        # These two stops cannot be on the same vehicle
                        # Penalize all pairs (x[i][t], x[j][t']) where both are 1
                        for ti in range(self.encoding.n_positions):
                            for tj in range(self.encoding.n_positions):
                                idx_i = self.encoding.var_index(i, ti)
                                idx_j = self.encoding.var_index(j, tj)
                                row, col = min(idx_i, idx_j), max(idx_i, idx_j)
                                if row == col:
                                    Q[row][col] += P
                                else:
                                    Q[row][col] += P

        elif self.encoding_type == "route":
            # For route encoding: similar pairwise penalty on edges
            # that would connect stops exceeding capacity
            pass  # Extended in Phase 2

        return Q

    def _build_timewindow_qubo(self) -> np.ndarray:
        """Build time window constraint QUBO.

        For position encoding:
            If stop i is at position t, the arrival time at position t
            must be within [earliest_i, latest_i].

            We discretize time into slots and add penalties for invalid assignments.

        For now: soft penalty that penalizes stop i being at position t
        if the estimated arrival at position t falls outside [earliest, latest].
        """
        n = self.encoding.n_qubits
        Q = np.zeros((n, n))

        if self.instance.time_windows is None:
            return Q

        if self.encoding_type != "position":
            return Q  # Time window encoding for route-based: Phase 2

        P = self.penalties['timewindow']
        tw = self.instance.time_windows
        dist = self.instance.distance_matrix

        # Estimate average travel time between consecutive stops
        avg_dist = np.mean(dist[dist > 0])
        # Assume speed: 30 km/h in Tokyo traffic → time per unit distance
        # This is a simplification; in Phase 2 we use actual travel times
        time_per_dist = 2.0  # minutes per distance unit (calibrate later)

        for i in range(self.instance.n_stops):
            earliest, latest = tw[i]
            for t in range(self.encoding.n_positions):
                # Estimated arrival time at position t: t * avg_travel_time
                est_arrival = t * avg_dist * time_per_dist
                if est_arrival < earliest or est_arrival > latest:
                    # Penalize this assignment
                    idx = self.encoding.var_index(i, t)
                    Q[idx][idx] += P

        return Q

    def evaluate_solution(self, bitstring: np.ndarray) -> Dict[str, Any]:
        """Evaluate a candidate solution.

        Args:
            bitstring: Binary solution vector.

        Returns:
            Dictionary with:
                - 'tour': Decoded tour (or None if invalid)
                - 'cost': Total distance (or inf if invalid)
                - 'qubo_energy': Energy of the QUBO
                - 'feasible': Whether all constraints are satisfied
                - 'violations': List of constraint violations
        """
        tour = self.encoding.decode(bitstring)

        result = {
            'tour': tour,
            'cost': float('inf'),
            'qubo_energy': float(bitstring @ self.build().Q @ bitstring),
            'feasible': tour is not None,
            'violations': [],
        }

        if tour is None:
            result['violations'].append('INVALID_ENCODING')
            return result

        # Calculate actual route cost
        if self.encoding_type == "position":
            # Tour is list of stop indices. Add depot at start and end.
            full_tour = [0] + [s + 1 for s in tour] + [0]
        else:
            full_tour = tour  # Already includes depot

        cost = sum(
            self.instance.distance_matrix[full_tour[i]][full_tour[i + 1]]
            for i in range(len(full_tour) - 1)
        )
        result['cost'] = cost

        # Check capacity
        if self.encoding_type == "position":
            total_load = sum(self.instance.demands[s] for s in tour)
            if total_load > self.instance.capacity:
                result['feasible'] = False
                result['violations'].append(
                    f'CAPACITY: {total_load} > {self.instance.capacity}'
                )

        # Check time windows
        if self.instance.time_windows is not None and self.encoding_type == "position":
            # Simplified check: would need actual travel times for accurate check
            pass  # Extended in Phase 2

        return result

    def brute_force_solve(self) -> Tuple[np.ndarray, float]:
        """Solve the QUBO by brute force (for small instances only).

        Returns:
            Tuple of (optimal_bitstring, optimal_energy).

        Warning: Exponential in n_qubits. Only use for n_qubits <= 20.
        """
        qubo_result = self.build()
        Q = qubo_result.Q
        n = qubo_result.n_qubits

        if n > 20:
            raise ValueError(f"Brute force not feasible for {n} qubits (max 20)")

        best_energy = float('inf')
        best_bits = None

        for i in range(2 ** n):
            bits = np.array([(i >> b) & 1 for b in range(n)], dtype=float)
            energy = bits @ Q @ bits
            if energy < best_energy:
                best_energy = energy
                best_bits = bits.copy()

        return best_bits, best_energy
