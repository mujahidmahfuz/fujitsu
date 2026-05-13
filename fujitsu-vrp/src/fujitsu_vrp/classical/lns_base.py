"""
Large Neighborhood Search (LNS) Base Implementation.

Provides the foundation for classical LNS and quantum-enhanced QLNRS algorithms.
"""

from __future__ import annotations

import abc
import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from ..data.problem_builder import Solution
    from ..data.synthetic_generator import VRPInstance

logger = logging.getLogger(__name__)


@dataclass
class LNSConfig:
    """Configuration for Large Neighborhood Search."""

    # Iteration parameters
    max_iterations: int = 1000
    time_limit_seconds: float = 300.0
    no_improvement_limit: int = 100  # Stop after N iterations without improvement

    # Destroy parameters
    destroy_fraction: float = 0.3  # Fraction of customers to remove
    min_destroy: int = 2  # Minimum customers to destroy
    max_destroy: int = 10  # Maximum customers to destroy

    # Repair parameters
    repair_method: str = "greedy"  # greedy, regret_2, regret_3

    # Acceptance parameters (Simulated Annealing)
    initial_temperature: float = 100.0
    final_temperature: float = 0.1
    cooling_rate: float = 0.995

    # Seed
    seed: int | None = None


@dataclass
class DestroyOperator:
    """Abstract base class for destroy operators."""

    name: str
    weight: float = 1.0

    @abc.abstractmethod
    def destroy(
        self,
        solution: Solution,
        num_remove: int,
        instance: VRPInstance,
        rng: random.Random,
    ) -> tuple[list[int], list[list[int]]]:
        """Destroy part of the solution.

        Args:
            solution: Current solution
            num_remove: Number of customers to remove
            instance: VRP instance
            rng: Random number generator

        Returns:
            Tuple of (removed_customers, modified_routes)
        """
        pass


class RandomDestroy(DestroyOperator):
    """Randomly removes customers from routes."""

    def __init__(self) -> None:
        super().__init__(name="random")

    def destroy(
        self,
        solution: Solution,
        num_remove: int,
        instance: VRPInstance,
        rng: random.Random,
    ) -> tuple[list[int], list[list[int]]]:
        """Randomly remove customers."""
        # Collect all customers in routes
        all_customers = []
        for route_idx, route in enumerate(solution.routes):
            for pos in range(1, len(route) - 1):  # Skip depot at start/end
                all_customers.append((route_idx, pos, route[pos]))

        # Randomly select customers to remove
        num_remove = min(num_remove, len(all_customers))
        to_remove = rng.sample(all_customers, num_remove)
        removed = [c[2] for c in to_remove]

        # Create modified routes
        modified_routes = []
        for route in solution.routes:
            new_route = [n for n in route if n not in removed]
            modified_routes.append(new_route)

        return removed, modified_routes


class WorstDestroy(DestroyOperator):
    """Removes customers with highest insertion cost."""

    def __init__(self) -> None:
        super().__init__(name="worst")

    def destroy(
        self,
        solution: Solution,
        num_remove: int,
        instance: VRPInstance,
        rng: random.Random,
    ) -> tuple[list[int], list[list[int]]]:
        """Remove customers with highest marginal cost."""
        # Calculate removal savings for each customer
        removal_costs = []
        for route_idx, route in enumerate(solution.routes):
            for pos in range(1, len(route) - 1):
                customer = route[pos]
                # Cost reduction if customer removed
                prev_node = route[pos - 1]
                next_node = route[pos + 1]
                original_cost = (
                    instance.distance_matrix[prev_node, customer]
                    + instance.distance_matrix[customer, next_node]
                )
                new_cost = instance.distance_matrix[prev_node, next_node]
                saving = original_cost - new_cost  # Negative saving = high cost
                removal_costs.append((route_idx, pos, customer, -saving))

        # Sort by cost (highest first)
        removal_costs.sort(key=lambda x: x[3], reverse=True)

        # Select top num_remove with some randomness
        num_remove = min(num_remove, len(removal_costs))
        # Use stochastic selection: higher cost customers more likely to be selected
        selected_indices = []
        remaining = list(range(len(removal_costs)))
        while len(selected_indices) < num_remove and remaining:
            # Weight by position (earlier = higher cost)
            weights = [1.0 / (i + 1) for i in range(len(remaining))]
            idx = rng.choices(remaining, weights=weights)[0]
            selected_indices.append(idx)
            remaining.remove(idx)

        removed = [removal_costs[i][2] for i in selected_indices]

        # Create modified routes
        modified_routes = []
        for route in solution.routes:
            new_route = [n for n in route if n not in removed]
            modified_routes.append(new_route)

        return removed, modified_routes


class RelatedDestroy(DestroyOperator):
    """Removes customers that are close to each other."""

    def __init__(self) -> None:
        super().__init__(name="related")

    def destroy(
        self,
        solution: Solution,
        num_remove: int,
        instance: VRPInstance,
        rng: random.Random,
    ) -> tuple[list[int], list[list[int]]]:
        """Remove customers that are related (geographically close)."""
        # Collect all customers
        all_customers = []
        for route_idx, route in enumerate(solution.routes):
            for pos in range(1, len(route) - 1):
                all_customers.append((route_idx, pos, route[pos]))

        if not all_customers:
            return [], solution.routes

        # Start with random customer
        removed = []
        remaining = list(all_customers)
        first = rng.choice(remaining)
        removed.append(first[2])
        remaining.remove(first)

        # Select related customers
        while len(removed) < num_remove and remaining:
            # Find closest customer to any removed customer
            best_customer = None
            best_distance = float("inf")

            for cand in remaining:
                for rem_customer in removed:
                    dist = instance.distance_matrix[cand[2], rem_customer]
                    if dist < best_distance:
                        best_distance = dist
                        best_customer = cand

            if best_customer is None:
                break

            removed.append(best_customer[2])
            remaining.remove(best_customer)

        # Create modified routes
        modified_routes = []
        for route in solution.routes:
            new_route = [n for n in route if n not in removed]
            modified_routes.append(new_route)

        return removed, modified_routes


class RiskWeightedDestroy(DestroyOperator):
    """Removes customers based on risk contribution."""

    def __init__(self, alpha: float = 0.3, beta: float = 0.3, gamma: float = 0.4):
        """
        Args:
            alpha: Time window risk weight
            beta: Capacity risk weight
            gamma: Operational risk weight
        """
        super().__init__(name="risk_weighted")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_customer_risk(
        self,
        customer_id: int,
        route: list[int],
        pos: int,
        instance: VRPInstance,
    ) -> float:
        """Compute risk score for a customer in its current position."""
        if customer_id == 0:  # Depot
            return 0.0

        customer = instance.customers[customer_id - 1]

        # Time window risk
        tw_risk = self._compute_tw_risk(customer, route, pos, instance)

        # Capacity risk
        cap_risk = self._compute_capacity_risk(customer, route, instance)

        # Operational risk (based on distance from route centroid)
        op_risk = self._compute_operational_risk(customer, route, instance)

        return self.alpha * tw_risk + self.beta * cap_risk + self.gamma * op_risk

    def _compute_tw_risk(
        self,
        customer,
        route: list[int],
        pos: int,
        instance: VRPInstance,
    ) -> float:
        """Compute time window violation risk."""
        # Estimate arrival time at customer
        arrival_time = 0.0
        for i in range(pos):
            arrival_time += instance.time_matrix[route[i], route[i + 1]]
            if i > 0:
                arr_c = instance.customers[route[i] - 1]
                arrival_time += arr_c.service_time

        # Time window slack
        tw_slack = customer.time_window_end - arrival_time

        # Risk increases as slack decreases
        if tw_slack < 0:
            return 1.0 + abs(tw_slack) / 100.0  # Already violated
        elif tw_slack < 30:
            return 0.5 + (30 - tw_slack) / 60.0  # Tight
        else:
            return 0.1  # Safe

    def _compute_capacity_risk(self, customer, route: list[int], instance: VRPInstance) -> float:
        """Compute capacity utilization risk."""
        # Calculate route demand
        route_demand = sum(
            instance.customers[n - 1].demand for n in route[1:-1] if n > 0
        )
        avg_capacity = np.mean([v.capacity for v in instance.vehicles])

        utilization = route_demand / avg_capacity

        # Risk increases with utilization
        if utilization > 0.9:
            return 1.0
        elif utilization > 0.7:
            return 0.5
        else:
            return 0.1

    def _compute_operational_risk(
        self,
        customer,
        route: list[int],
        instance: VRPInstance,
    ) -> float:
        """Compute operational risk based on geographic isolation."""
        if len(route) <= 2:
            return 0.1

        # Calculate distance from customer to route centroid
        customer_lon = customer.x
        customer_lat = customer.y

        route_customers = [
            instance.customers[n - 1] for n in route[1:-1] if n > 0
        ]
        if not route_customers:
            return 0.1

        centroid_lon = np.mean([c.x for c in route_customers])
        centroid_lat = np.mean([c.y for c in route_customers])

        # Distance to centroid
        dist = np.sqrt(
            (customer_lon - centroid_lon) ** 2 + (customer_lat - centroid_lat) ** 2
        )

        # Normalize by typical route spread
        spreads = []
        for c in route_customers:
            for c2 in route_customers:
                spread = np.sqrt((c.x - c2.x) ** 2 + (c.y - c2.y) ** 2)
                spreads.append(spread)

        avg_spread = np.mean(spreads) if spreads else 1.0
        normalized_dist = dist / (avg_spread + 0.01)

        return min(1.0, normalized_dist)

    def destroy(
        self,
        solution: Solution,
        num_remove: int,
        instance: VRPInstance,
        rng: random.Random,
    ) -> tuple[list[int], list[list[int]]]:
        """Remove customers with highest risk contribution."""
        # Compute risk for each customer
        customer_risks = []
        for route_idx, route in enumerate(solution.routes):
            for pos in range(1, len(route) - 1):
                customer_id = route[pos]
                risk = self.compute_customer_risk(customer_id, route, pos, instance)
                customer_risks.append((route_idx, pos, customer_id, risk))

        # Sort by risk (highest first)
        customer_risks.sort(key=lambda x: x[3], reverse=True)

        # Select customers with probability proportional to risk
        num_remove = min(num_remove, len(customer_risks))
        removed = []

        remaining = list(range(len(customer_risks)))
        while len(removed) < num_remove and remaining:
            # Weight selection by risk
            weights = [customer_risks[i][3] + 0.01 for i in remaining]  # Add small constant
            total = sum(weights)
            probs = [w / total for w in weights]

            idx = rng.choices(remaining, weights=probs)[0]
            removed.append(customer_risks[idx][2])
            remaining.remove(idx)

        # Create modified routes
        modified_routes = []
        for route in solution.routes:
            new_route = [n for n in route if n not in removed]
            modified_routes.append(new_route)

        return removed, modified_routes


@dataclass
class RepairOperator:
    """Abstract base class for repair operators."""

    name: str

    @abc.abstractmethod
    def repair(
        self,
        removed_customers: list[int],
        partial_solution: list[list[int]],
        instance: VRPInstance,
        rng: random.Random,
    ) -> list[list[int]]:
        """Repair solution by inserting removed customers.

        Args:
            removed_customers: Customers to insert
            partial_solution: Current partial routes
            instance: VRP instance
            rng: Random number generator

        Returns:
            Complete solution with all customers inserted
        """
        pass


class GreedyRepair(RepairOperator):
    """Greedy insertion - insert at best position."""

    def __init__(self) -> None:
        super().__init__(name="greedy")

    def find_best_insertion(
        self,
        customer: int,
        routes: list[list[int]],
        instance: VRPInstance,
    ) -> tuple[int, int, float]:
        """Find best insertion position for a customer.

        Returns:
            (route_index, position, insertion_cost)
        """
        best_route = -1
        best_pos = -1
        best_cost = float("inf")

        for route_idx, route in enumerate(routes):
            for pos in range(1, len(route)):  # Can insert before any position
                prev_node = route[pos - 1]
                next_node = route[pos]

                # Insertion cost
                cost = (
                    instance.distance_matrix[prev_node, customer]
                    + instance.distance_matrix[customer, next_node]
                    - instance.distance_matrix[prev_node, next_node]
                )

                if cost < best_cost:
                    best_cost = cost
                    best_route = route_idx
                    best_pos = pos

        return best_route, best_pos, best_cost

    def repair(
        self,
        removed_customers: list[int],
        partial_solution: list[list[int]],
        instance: VRPInstance,
        rng: random.Random,
    ) -> list[list[int]]:
        """Greedy repair - insert each customer at best position."""
        routes = [list(r) for r in partial_solution]  # Deep copy

        # Sort customers by demand (largest first)
        sorted_customers = sorted(
            removed_customers,
            key=lambda c: instance.customers[c - 1].demand if c > 0 else 0,
            reverse=True,
        )

        for customer in sorted_customers:
            best_route, best_pos, _ = self.find_best_insertion(
                customer, routes, instance
            )

            if best_route >= 0:
                routes[best_route].insert(best_pos, customer)
            else:
                # Create new route if possible
                if len(routes) < instance.num_vehicles:
                    routes.append([0, customer, 0])
                else:
                    # Force insert into first route
                    routes[0].insert(1, customer)

        return routes


class RegretRepair(RepairOperator):
    """Regret-k insertion - minimize regret of not choosing best."""

    def __init__(self, k: int = 2) -> None:
        """
        Args:
            k: Number of alternatives to consider (regret-2, regret-3, etc.)
        """
        super().__init__(name=f"regret_{k}")
        self.k = k
        self.greedy = GreedyRepair()

    def repair(
        self,
        removed_customers: list[int],
        partial_solution: list[list[int]],
        instance: VRPInstance,
        rng: random.Random,
    ) -> list[list[int]]:
        """Regret-k repair."""
        routes = [list(r) for r in partial_solution]
        remaining = list(removed_customers)

        while remaining:
            best_customer = None
            best_regret = -1
            best_insertions = {}

            for customer in remaining:
                # Find best k insertion costs across routes
                insertion_costs = []
                for route_idx in range(len(routes)):
                    route = routes[route_idx]
                    for pos in range(1, len(route)):
                        prev_node = route[pos - 1]
                        next_node = route[pos] if pos < len(route) else 0

                        cost = (
                            instance.distance_matrix[prev_node, customer]
                            + instance.distance_matrix[customer, next_node]
                            - instance.distance_matrix[prev_node, next_node]
                        )
                        insertion_costs.append((route_idx, pos, cost))

                # Sort by cost
                insertion_costs.sort(key=lambda x: x[2])

                # Compute regret (difference between best and k-th best)
                if len(insertion_costs) >= self.k:
                    regret = insertion_costs[self.k - 1][2] - insertion_costs[0][2]
                elif len(insertion_costs) >= 1:
                    regret = insertion_costs[-1][2] - insertion_costs[0][2] + 1000  # Penalty
                else:
                    regret = 10000  # No valid insertion

                if regret > best_regret:
                    best_regret = regret
                    best_customer = customer
                    best_insertions[customer] = insertion_costs

            if best_customer is None:
                break

            # Insert at best position
            insertions = best_insertions.get(best_customer, [])
            if insertions:
                route_idx, pos, _ = insertions[0]
                routes[route_idx].insert(pos, best_customer)
            else:
                # Create new route if possible
                if len(routes) < instance.num_vehicles:
                    routes.append([0, best_customer, 0])

            remaining.remove(best_customer)

        return routes


@dataclass
class LNSState:
    """State of the LNS algorithm."""

    current_solution: Solution
    best_solution: Solution
    current_cost: float
    best_cost: float
    iteration: int = 0
    temperature: float = 100.0
    no_improvement_count: int = 0

    # Statistics
    destroy_counts: dict[str, int] = field(default_factory=dict)
    repair_counts: dict[str, int] = field(default_factory=dict)
    accepted_count: int = 0
    improved_count: int = 0


class LNSBase:
    """Base class for Large Neighborhood Search."""

    def __init__(
        self,
        config: LNSConfig | None = None,
        destroy_operators: list[DestroyOperator] | None = None,
        repair_operators: list[RepairOperator] | None = None,
    ) -> None:
        """Initialize LNS solver.

        Args:
            config: LNS configuration
            destroy_operators: List of destroy operators (uses defaults if None)
            repair_operators: List of repair operators (uses defaults if None)
        """
        self.config = config or LNSConfig()
        self.rng = random.Random(self.config.seed)

        # Default operators
        self.destroy_operators = destroy_operators or [
            RandomDestroy(),
            WorstDestroy(),
            RelatedDestroy(),
            RiskWeightedDestroy(),
        ]

        self.repair_operators = repair_operators or [
            GreedyRepair(),
            RegretRepair(k=2),
            RegretRepair(k=3),
        ]

        # Adaptive weights
        self.destroy_weights = {op.name: 1.0 for op in self.destroy_operators}
        self.repair_weights = {op.name: 1.0 for op in self.repair_operators}

    def compute_cost(self, solution: Solution, instance: VRPInstance) -> float:
        """Compute total solution cost."""
        from ..data.geospatial import compute_route_statistics

        stats = compute_route_statistics(instance, solution.routes)

        # Base cost: total distance
        cost = stats["total_distance"]

        # Penalty for violations
        cost += stats["total_tw_violation"] * 10.0

        # Penalty for unassigned customers
        total_demand = sum(c.demand for c in instance.customers)
        unassigned = total_demand - stats["total_demand_served"]
        cost += unassigned * 1000.0

        return cost

    def select_destroy_operator(self) -> DestroyOperator:
        """Select destroy operator based on adaptive weights."""
        names = list(self.destroy_weights.keys())
        weights = [self.destroy_weights[n] for n in names]
        total = sum(weights)
        probs = [w / total for w in weights]

        selected_name = self.rng.choices(names, weights=probs)[0]
        for op in self.destroy_operators:
            if op.name == selected_name:
                return op

        return self.destroy_operators[0]

    def select_repair_operator(self) -> RepairOperator:
        """Select repair operator based on adaptive weights."""
        names = list(self.repair_weights.keys())
        weights = [self.repair_weights[n] for n in names]
        total = sum(weights)
        probs = [w / total for w in weights]

        selected_name = self.rng.choices(names, weights=probs)[0]
        for op in self.repair_operators:
            if op.name == selected_name:
                return op

        return self.repair_operators[0]

    def accept_solution(
        self,
        current_cost: float,
        new_cost: float,
        temperature: float,
    ) -> bool:
        """Decide whether to accept a new solution (Simulated Annealing)."""
        if new_cost < current_cost:
            return True

        # Accept worse solution with probability
        delta = new_cost - current_cost
        probability = np.exp(-delta / temperature)

        return self.rng.random() < probability

    def cool_temperature(self, temperature: float) -> float:
        """Apply cooling schedule."""
        return temperature * self.config.cooling_rate

    def get_num_remove(self, instance: VRPInstance) -> int:
        """Determine number of customers to remove."""
        total_customers = instance.num_customers

        # Use configured fraction
        num = int(total_customers * self.config.destroy_fraction)

        # Apply bounds
        num = max(self.config.min_destroy, min(self.config.max_destroy, num))

        return num