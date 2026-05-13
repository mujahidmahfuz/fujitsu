"""
QUBO Encoder for VRP Repair Subproblem.

Encodes the VRP repair subproblem as a Quadratic Unconstrained Binary Optimization (QUBO)
problem suitable for quantum annealing / Ising machines.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ..data.synthetic_generator import Customer, VRPInstance

logger = logging.getLogger(__name__)


@dataclass
class QUBOProblem:
    """Represents a QUBO problem."""

    # QUBO matrix Q such that cost = x^T Q x
    Q: npt.NDArray[np.float64]

    # Variable mapping: qubit_index -> (customer_id, route_idx, position)
    variable_mapping: list[tuple[int, int, int]]

    # Number of variables (qubits)
    num_variables: int

    # Constraint penalties used
    penalties: dict[str, float]

    # Original problem info
    num_customers: int
    num_routes: int
    removed_customers: list[int]

    # Offset to add to objective (constant term)
    offset: float = 0.0

    metadata: dict = field(default_factory=dict)

    def get_linear_terms(self) -> npt.NDArray[np.float64]:
        """Get diagonal (linear) terms of QUBO."""
        return np.diag(self.Q)

    def get_quadratic_terms(self) -> list[tuple[int, int, float]]:
        """Get off-diagonal (quadratic) terms."""
        terms = []
        for i in range(self.num_variables):
            for j in range(i + 1, self.num_variables):
                if self.Q[i, j] != 0:
                    terms.append((i, j, self.Q[i, j]))
        return terms


@dataclass
class EncodingConfig:
    """Configuration for QUBO encoding."""

    # Maximum qubits (Fujitsu constraint: 40)
    max_qubits: int = 40

    # Penalty coefficients
    assignment_penalty: float = 1000.0  # Each customer assigned once
    capacity_penalty: float = 100.0  # Capacity constraint
    time_window_penalty: float = 50.0  # Time window soft constraint

    # Encoding type
    encoding_type: str = "slot"  # "slot", "edge", "position"

    # Approximation settings
    approximate_constraints: bool = True
    constraint_strength_factor: float = 2.0


class VRP_QUBOEncoder:
    """Encodes VRP repair subproblem as QUBO.

    The repair subproblem: Given a partial solution with removed customers,
    find the optimal insertion positions for these customers.

    Variables:
        x_{i,j,k} = 1 if customer i is inserted at position k in route j

    Objective:
        Minimize insertion cost + penalty for constraint violations
    """

    def __init__(self, config: EncodingConfig | None = None) -> None:
        """Initialize encoder with configuration."""
        self.config = config or EncodingConfig()

    def encode_repair_subproblem(
        self,
        instance: VRPInstance,
        partial_routes: list[list[int]],
        removed_customers: list[int],
        current_route_times: dict[int, float] | None = None,
    ) -> QUBOProblem:
        """Encode the repair subproblem as QUBO.

        Args:
            instance: VRP instance
            partial_routes: Current partial routes (customers removed)
            removed_customers: Customers to be reinserted
            current_route_times: Current arrival times at route positions

        Returns:
            QUBOProblem ready for quantum solving
        """
        # Determine encoding based on problem size
        num_customers_to_insert = len(removed_customers)

        # Estimate qubits needed for each encoding
        qubits_slot = self._estimate_slot_encoding_qubits(
            partial_routes, num_customers_to_insert
        )
        qubits_position = num_customers_to_insert * len(partial_routes)

        # Choose encoding
        if qubits_slot <= self.config.max_qubits:
            return self._encode_slot_based(
                instance, partial_routes, removed_customers, current_route_times
            )
        else:
            # Use position encoding (more compact but less precise)
            return self._encode_position_based(
                instance, partial_routes, removed_customers, current_route_times
            )

    def _estimate_slot_encoding_qubits(
        self,
        partial_routes: list[list[int]],
        num_customers: int,
    ) -> int:
        """Estimate number of qubits for slot encoding."""
        total_slots = sum(len(r) - 1 for r in partial_routes) + num_customers
        return num_customers * total_slots

    def _encode_slot_based(
        self,
        instance: VRPInstance,
        partial_routes: list[list[int]],
        removed_customers: list[int],
        current_route_times: dict[int, float] | None = None,
    ) -> QUBOProblem:
        """Encode using slot-based formulation.

        Variables: x_{i,s} = 1 if customer i assigned to slot s
        Slots: Available insertion positions across all routes
        """
        # Build slot list
        slots = []  # (route_idx, position)
        for route_idx, route in enumerate(partial_routes):
            for pos in range(1, len(route)):  # Insert after position pos-1
                slots.append((route_idx, pos))

        num_customers = len(removed_customers)
        num_slots = len(slots)
        num_vars = num_customers * num_slots

        if num_vars > self.config.max_qubits:
            # Truncate to fit within qubit limit
            num_slots = self.config.max_qubits // num_customers
            slots = slots[:num_slots]
            num_vars = num_customers * num_slots
            logger.warning(
                f"Truncated slots to {num_slots} to fit {self.config.max_qubits} qubit limit"
            )

        # Initialize QUBO matrix
        Q = np.zeros((num_vars, num_vars))
        offset = 0.0

        # Variable mapping: qubit_index -> (customer_id, route_idx, position)
        variable_mapping = []
        for i, customer in enumerate(removed_customers):
            for s, (route_idx, pos) in enumerate(slots):
                var_idx = i * num_slots + s
                variable_mapping.append((customer, route_idx, pos))

        # --- Objective: Minimize insertion cost ---
        for i, customer in enumerate(removed_customers):
            for s, (route_idx, pos) in enumerate(slots):
                var_idx = i * num_slots + s

                # Insertion cost
                route = partial_routes[route_idx]
                if pos <= len(route) - 1:
                    prev_node = route[pos - 1]
                    next_node = route[pos]
                else:
                    prev_node = route[-1]
                    next_node = 0  # Depot

                cost = (
                    instance.distance_matrix[prev_node, customer]
                    + instance.distance_matrix[customer, next_node]
                    - instance.distance_matrix[prev_node, next_node]
                )

                Q[var_idx, var_idx] = cost

        # --- Constraint: Each customer assigned exactly once ---
        P_assign = self.config.assignment_penalty
        for i in range(num_customers):
            # Sum over slots for customer i should be 1
            # Penalty: (sum - 1)^2 = sum^2 - 2*sum + 1
            for s1 in range(num_slots):
                idx1 = i * num_slots + s1
                for s2 in range(num_slots):
                    idx2 = i * num_slots + s2
                    if s1 == s2:
                        Q[idx1, idx1] += P_assign  # x^2 = x
                    else:
                        Q[idx1, idx2] += P_assign  # 2 * x1 * x2
                Q[idx1, idx1] -= 2 * P_assign  # -2x
            offset += P_assign  # +1

        # --- Constraint: Each slot assigned at most once ---
        for s in range(num_slots):
            for i1 in range(num_customers):
                idx1 = i1 * num_slots + s
                for i2 in range(i1 + 1, num_customers):
                    idx2 = i2 * num_slots + s
                    Q[idx1, idx2] += P_assign * 0.5  # Penalty for double assignment

        # --- Soft constraint: Capacity ---
        P_cap = self.config.capacity_penalty
        for route_idx, route in enumerate(partial_routes):
            # Current demand in route
            current_demand = sum(
                instance.customers[n - 1].demand for n in route[1:-1] if n > 0
            )
            capacity = instance.vehicles[route_idx].capacity

            # For each slot in this route
            route_slots = [
                s for s, (r, _) in enumerate(slots) if r == route_idx
            ]

            # Soft penalty for exceeding capacity
            for s in route_slots:
                for i, customer in enumerate(removed_customers):
                    var_idx = i * num_slots + s
                    cust_demand = instance.customers[customer - 1].demand

                    # Linear penalty for demand contribution
                    if current_demand + cust_demand > capacity:
                        Q[var_idx, var_idx] += P_cap * (current_demand + cust_demand - capacity)

        # --- Soft constraint: Time windows ---
        P_tw = self.config.time_window_penalty
        for i, customer in enumerate(removed_customers):
            cust = instance.customers[customer - 1]

            for s, (route_idx, pos) in enumerate(slots):
                var_idx = i * num_slots + s

                # Estimate arrival time (simplified)
                # This is a soft penalty based on time window tightness
                tw_slack = cust.time_window_end - cust.time_window_start
                time_penalty = P_tw * (1.0 / (tw_slack + 1.0))  # Tighter TW = higher penalty

                Q[var_idx, var_idx] += time_penalty * 0.1

        # Create QUBO problem
        return QUBOProblem(
            Q=Q,
            variable_mapping=variable_mapping,
            num_variables=num_vars,
            penalties={
                "assignment": P_assign,
                "capacity": P_cap,
                "time_window": P_tw,
            },
            num_customers=num_customers,
            num_routes=len(partial_routes),
            removed_customers=removed_customers,
            offset=offset,
            metadata={
                "encoding": "slot",
                "num_slots": num_slots,
                "slots": slots,
            },
        )

    def _encode_position_based(
        self,
        instance: VRPInstance,
        partial_routes: list[list[int]],
        removed_customers: list[int],
        current_route_times: dict[int, float] | None = None,
    ) -> QUBOProblem:
        """Encode using position-based formulation.

        Variables: x_{i,j} = 1 if customer i assigned to route j
        Simplified encoding: each customer must go to some route
        """
        num_customers = len(removed_customers)
        num_routes = len(partial_routes)
        num_vars = num_customers * num_routes

        Q = np.zeros((num_vars, num_vars))
        offset = 0.0

        variable_mapping = []
        for i, customer in enumerate(removed_customers):
            for j in range(num_routes):
                var_idx = i * num_routes + j
                variable_mapping.append((customer, j, 0))  # Position determined later

        # Objective: Minimize approximate insertion cost
        for i, customer in enumerate(removed_customers):
            cust = instance.customers[customer - 1]

            for j, route in enumerate(partial_routes):
                var_idx = i * num_routes + j

                # Approximate cost: distance from depot to customer
                # (Simplified: actual position determined in repair)
                depot = instance.depot
                cost = instance.distance_matrix[depot.id, customer]

                # Add distance from route centroid
                if len(route) > 2:
                    centroid_x = np.mean(
                        [instance.customers[n - 1].x for n in route[1:-1] if n > 0]
                        or [depot.x]
                    )
                    centroid_y = np.mean(
                        [instance.customers[n - 1].y for n in route[1:-1] if n > 0]
                        or [depot.y]
                    )
                    dist_to_centroid = np.sqrt(
                        (cust.x - centroid_x) ** 2 + (cust.y - centroid_y) ** 2
                    )
                    cost += dist_to_centroid

                Q[var_idx, var_idx] = cost

        # Constraint: Each customer assigned to exactly one route
        P = self.config.assignment_penalty
        for i in range(num_customers):
            for j1 in range(num_routes):
                idx1 = i * num_routes + j1
                for j2 in range(num_routes):
                    idx2 = i * num_routes + j2
                    if j1 == j2:
                        Q[idx1, idx1] += P
                    else:
                        Q[idx1, idx2] += P
                Q[idx1, idx1] -= 2 * P
            offset += P

        # Capacity soft constraint
        P_cap = self.config.capacity_penalty
        for j, route in enumerate(partial_routes):
            current_demand = sum(
                instance.customers[n - 1].demand for n in route[1:-1] if n > 0
            )
            capacity = instance.vehicles[j].capacity
            slack = capacity - current_demand

            for i, customer in enumerate(removed_customers):
                var_idx = i * num_routes + j
                cust_demand = instance.customers[customer - 1].demand

                # Penalty if would exceed capacity
                if cust_demand > slack:
                    Q[var_idx, var_idx] += P_cap * (cust_demand - slack)

        return QUBOProblem(
            Q=Q,
            variable_mapping=variable_mapping,
            num_variables=num_vars,
            penalties={
                "assignment": P,
                "capacity": P_cap,
            },
            num_customers=num_customers,
            num_routes=num_routes,
            removed_customers=removed_customers,
            offset=offset,
            metadata={"encoding": "position"},
        )

    def decode_solution(
        self,
        qubo: QUBOProblem,
        solution_bits: npt.NDArray[np.int32],
        partial_routes: list[list[int]],
        instance: VRPInstance,
    ) -> list[list[int]]:
        """Decode QUBO solution back to routes.

        Args:
            qubo: Original QUBO problem
            solution_bits: Binary solution vector
            partial_routes: Original partial routes
            instance: VRP instance

        Returns:
            Complete routes with customers inserted
        """
        routes = [list(r) for r in partial_routes]

        # Group assignments by customer
        customer_assignments = {}
        for i, (customer, route_idx, pos) in enumerate(qubo.variable_mapping):
            if solution_bits[i] == 1:
                if customer not in customer_assignments:
                    customer_assignments[customer] = []
                customer_assignments[customer].append((route_idx, pos))

        # Insert customers at their assigned positions
        # Sort by position to avoid index shifting issues
        all_insertions = []
        for customer, assignments in customer_assignments.items():
            if assignments:
                # Take first valid assignment
                route_idx, pos = assignments[0]
                all_insertions.append((route_idx, pos, customer))

        # Sort by route and reverse position to insert from end
        all_insertions.sort(key=lambda x: (x[0], -x[1]))

        for route_idx, pos, customer in all_insertions:
            if route_idx < len(routes):
                # Insert at position
                if pos <= len(routes[route_idx]):
                    routes[route_idx].insert(pos, customer)
                else:
                    routes[route_idx].insert(len(routes[route_idx]) - 1, customer)

        # Handle unassigned customers (greedy insertion)
        assigned = set(customer_assignments.keys())
        unassigned = [c for c in qubo.removed_customers if c not in assigned]

        if unassigned:
            logger.warning(f"{len(unassigned)} customers not assigned by QUBO, using greedy insertion")
            greedy = GreedyRepair()
            routes = greedy.repair(unassigned, routes, instance, None)

        return routes


class GreedyRepair:
    """Simple greedy repair for unassigned customers."""

    def find_best_insertion(
        self,
        customer: int,
        routes: list[list[int]],
        instance: VRPInstance,
    ) -> tuple[int, int]:
        """Find best insertion position."""
        best_route = 0
        best_pos = 1
        best_cost = float("inf")

        for route_idx, route in enumerate(routes):
            for pos in range(1, len(route)):
                prev_node = route[pos - 1]
                next_node = route[pos]

                cost = (
                    instance.distance_matrix[prev_node, customer]
                    + instance.distance_matrix[customer, next_node]
                    - instance.distance_matrix[prev_node, next_node]
                )

                if cost < best_cost:
                    best_cost = cost
                    best_route = route_idx
                    best_pos = pos

        return best_route, best_pos

    def repair(
        self,
        removed_customers: list[int],
        partial_solution: list[list[int]],
        instance: VRPInstance,
        rng,  # Unused
    ) -> list[list[int]]:
        """Greedy repair."""
        import random

        routes = [list(r) for r in partial_solution]

        for customer in removed_customers:
            route_idx, pos = self.find_best_insertion(customer, routes, instance)
            routes[route_idx].insert(pos, customer)

        return routes