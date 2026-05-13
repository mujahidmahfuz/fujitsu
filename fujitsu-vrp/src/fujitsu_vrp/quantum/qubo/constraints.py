"""
Constraint Hamiltonians for VRP QUBO Formulation.

Implements penalty terms for VRP constraints as Ising Hamiltonians.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ...data.synthetic_generator import VRPInstance


@dataclass
class ConstraintPenalties:
    """Container for constraint penalty strengths."""

    assignment: float = 1000.0  # Each customer assigned once
    capacity: float = 100.0  # Capacity constraint
    time_window: float = 50.0  # Time window constraint
    route_continuity: float = 10.0  # Route structure constraint


def assignment_constraint_hamiltonian(
    num_customers: int,
    num_slots: int,
    penalty: float,
) -> npt.NDArray[np.float64]:
    """Construct Hamiltonian for assignment constraint.

    Each customer must be assigned to exactly one slot.

    Constraint: Sum_s x_{i,s} = 1 for all customers i

    Penalty: (Sum_s x_{i,s} - 1)^2

    Args:
        num_customers: Number of customers to assign
        num_slots: Number of available slots
        penalty: Penalty strength

    Returns:
        Hamiltonian matrix (n x n where n = num_customers * num_slots)
    """
    num_vars = num_customers * num_slots
    H = np.zeros((num_vars, num_vars))

    for i in range(num_customers):
        for s1 in range(num_slots):
            idx1 = i * num_slots + s1

            # Linear term: (sum - 1)^2 = -2x + ...
            H[idx1, idx1] += penalty * (-2.0 + 1.0)  # x contributes 1 (x^2 = x for binary)

            for s2 in range(num_slots):
                idx2 = i * num_slots + s2
                # Quadratic term: x_{i,s1} * x_{i,s2}
                if s1 != s2:
                    H[idx1, idx2] += penalty * 1.0

    # Add constant offset (not in Hamiltonian but in total energy)
    # We handle this separately in the solver
    return H


def slot_uniqueness_constraint(
    num_customers: int,
    num_slots: int,
    penalty: float,
) -> npt.NDArray[np.float64]:
    """Construct Hamiltonian for slot uniqueness.

    Each slot can have at most one customer assigned.

    Constraint: Sum_i x_{i,s} <= 1 for all slots s

    Penalty: (Sum_i x_{i,s}) * (Sum_i x_{i,s} - 1)
           = Sum_i x_{i,s} + Sum_{i!=j} x_{i,s} x_{j,s}

    Args:
        num_customers: Number of customers
        num_slots: Number of slots
        penalty: Penalty strength

    Returns:
        Hamiltonian matrix
    """
    num_vars = num_customers * num_slots
    H = np.zeros((num_vars, num_vars))

    for s in range(num_slots):
        for i1 in range(num_customers):
            idx1 = i1 * num_slots + s

            # Linear penalty (from i1 * (i1 - 1) term when only one assigned)
            H[idx1, idx1] += penalty * (-1.0)  # Penalty reduced if only this customer

            for i2 in range(i1 + 1, num_customers):
                idx2 = i2 * num_slots + s
                # Quadratic penalty for double assignment
                H[idx1, idx2] += penalty * 2.0
                H[idx2, idx1] += penalty * 2.0

    return H


def capacity_constraint_hamiltonian(
    instance: VRPInstance,
    removed_customers: list[int],
    partial_routes: list[list[int]],
    slots: list[tuple[int, int]],
    penalty: float,
) -> npt.NDArray[np.float64]:
    """Construct Hamiltonian for capacity constraints.

    Each route must not exceed vehicle capacity.

    Args:
        instance: VRP instance
        removed_customers: Customers to be inserted
        partial_routes: Current partial routes
        slots: List of (route_idx, position) slots
        penalty: Penalty strength

    Returns:
        Hamiltonian matrix
    """
    num_customers = len(removed_customers)
    num_slots = len(slots)
    num_vars = num_customers * num_slots

    H = np.zeros((num_vars, num_vars))

    for route_idx, route in enumerate(partial_routes):
        # Current demand in route
        current_demand = sum(
            instance.customers[n - 1].demand if n > 0 else 0 for n in route[1:-1]
        )
        capacity = instance.vehicles[route_idx].capacity
        slack = capacity - current_demand

        # Get slots for this route
        route_slots = [s for s, (r, _) in enumerate(slots) if r == route_idx]

        # Penalty for exceeding capacity
        # Soft constraint: penalize demand exceeding slack
        for s in route_slots:
            for i, customer in enumerate(removed_customers):
                var_idx = i * num_slots + s
                cust_demand = instance.customers[customer - 1].demand

                # Linear penalty for demand contribution
                if cust_demand > slack:
                    H[var_idx, var_idx] += penalty * (cust_demand - slack)

        # Quadratic penalty for combinations that exceed capacity
        for s1 in route_slots:
            for i1, c1 in enumerate(removed_customers):
                idx1 = i1 * num_slots + s1
                d1 = instance.customers[c1 - 1].demand

                for s2 in route_slots:
                    if s2 < s1:
                        continue
                    for i2, c2 in enumerate(removed_customers):
                        if i2 <= i1:
                            continue
                        idx2 = i2 * num_slots + s2
                        d2 = instance.customers[c2 - 1].demand

                        # Penalty for combined demand exceeding slack
                        if d1 + d2 > slack:
                            H[idx1, idx2] += penalty * 0.5 * (d1 + d2 - slack)
                            H[idx2, idx1] += penalty * 0.5 * (d1 + d2 - slack)

    return H


def time_window_constraint_hamiltonian(
    instance: VRPInstance,
    removed_customers: list[int],
    partial_routes: list[list[int]],
    slots: list[tuple[int, int]],
    penalty: float,
) -> npt.NDArray[np.float64]:
    """Construct Hamiltonian for time window soft constraints.

    This is a simplified soft penalty based on time window tightness.
    A full formulation would require auxiliary variables for arrival times.

    Args:
        instance: VRP instance
        removed_customers: Customers to be inserted
        partial_routes: Current partial routes
        slots: List of (route_idx, position) slots
        penalty: Penalty strength

    Returns:
        Hamiltonian matrix
    """
    num_customers = len(removed_customers)
    num_slots = len(slots)
    num_vars = num_customers * num_slots

    H = np.zeros((num_vars, num_vars))

    for i, customer in enumerate(removed_customers):
        cust = instance.customers[customer - 1]

        # Time window tightness
        tw_width = cust.time_window_end - cust.time_window_start

        # Penalize tight time windows
        tw_penalty = penalty * (1.0 / (tw_width + 30.0))  # Normalize

        for s, (route_idx, pos) in enumerate(slots):
            var_idx = i * num_slots + s

            # Add time window penalty
            H[var_idx, var_idx] += tw_penalty

            # Additional penalty for late positions in route
            route = partial_routes[route_idx]
            if pos > len(route) * 0.7:  # Late position
                H[var_idx, var_idx] += penalty * 0.1

    return H


def objective_hamiltonian(
    instance: VRPInstance,
    removed_customers: list[int],
    partial_routes: list[list[int]],
    slots: list[tuple[int, int]],
    cost_weight: float = 1.0,
) -> tuple[npt.NDArray[np.float64], float]:
    """Construct Hamiltonian for the cost objective.

    Minimizes total insertion distance cost.

    Args:
        instance: VRP instance
        removed_customers: Customers to be inserted
        partial_routes: Current partial routes
        slots: List of (route_idx, position) slots
        cost_weight: Weight for cost objective

    Returns:
        (Hamiltonian matrix, constant offset)
    """
    num_customers = len(removed_customers)
    num_slots = len(slots)
    num_vars = num_customers * num_slots

    H = np.zeros((num_vars, num_vars))
    offset = 0.0

    for i, customer in enumerate(removed_customers):
        for s, (route_idx, pos) in enumerate(slots):
            var_idx = i * num_slots + s

            route = partial_routes[route_idx]

            # Find previous and next nodes for insertion
            if pos <= len(route) - 1:
                prev_node = route[pos - 1] if pos > 0 else 0
                next_node = route[pos]
            else:
                prev_node = route[-1]
                next_node = 0  # Return to depot

            # Insertion cost
            cost_before = instance.distance_matrix[prev_node, next_node]
            cost_after = (
                instance.distance_matrix[prev_node, customer]
                + instance.distance_matrix[customer, next_node]
            )
            insertion_cost = cost_after - cost_before

            H[var_idx, var_idx] += cost_weight * insertion_cost

    return H, offset


def build_full_hamiltonian(
    instance: VRPInstance,
    removed_customers: list[int],
    partial_routes: list[list[int]],
    slots: list[tuple[int, int]],
    penalties: ConstraintPenalties,
) -> tuple[npt.NDArray[np.float64], float]:
    """Build the full QUBO Hamiltonian.

    Combines objective and all constraint Hamiltonians.

    Args:
        instance: VRP instance
        removed_customers: Customers to insert
        partial_routes: Current routes
        slots: Available insertion slots
        penalties: Constraint penalty strengths

    Returns:
        (Full Hamiltonian, constant offset)
    """
    num_customers = len(removed_customers)
    num_slots = len(slots)

    # Objective
    H_obj, offset = objective_hamiltonian(
        instance, removed_customers, partial_routes, slots
    )

    # Assignment constraint
    H_assign = assignment_constraint_hamiltonian(
        num_customers, num_slots, penalties.assignment
    )

    # Slot uniqueness
    H_slot = slot_uniqueness_constraint(
        num_customers, num_slots, penalties.assignment * 0.5
    )

    # Capacity constraint
    H_cap = capacity_constraint_hamiltonian(
        instance, removed_customers, partial_routes, slots, penalties.capacity
    )

    # Time window constraint
    H_tw = time_window_constraint_hamiltonian(
        instance, removed_customers, partial_routes, slots, penalties.time_window
    )

    # Combine
    H_total = H_obj + H_assign + H_slot + H_cap + H_tw

    return H_total, offset