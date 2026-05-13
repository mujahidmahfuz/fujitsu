"""
Risk Metrics for VRP Solutions.

Computes multi-component risk metrics for robust VRP solutions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..data.problem_builder import Solution
    from ..data.synthetic_generator import VRPInstance


@dataclass
class RiskConfig:
    """Configuration for risk evaluation."""

    # Risk weights
    time_window_weight: float = 0.3
    capacity_weight: float = 0.3
    operational_weight: float = 0.4

    # Time window risk parameters
    tw_early_penalty: float = 1.0
    tw_late_penalty: float = 2.0

    # Capacity risk parameters
    capacity_threshold: float = 0.8  # Risk increases above this utilization

    # Operational risk parameters
    distance_weight: float = 1.0
    traffic_uncertainty: float = 0.2  # 20% uncertainty in travel time


@dataclass
class RiskMetrics:
    """Risk metrics for a solution."""

    total_risk: float
    time_window_risk: float
    capacity_risk: float
    operational_risk: float

    # Breakdown by route
    route_risks: dict[int, float] = field(default_factory=dict)

    # Details
    details: dict[str, Any] = field(default_factory=dict)


class RiskEvaluator:
    """Evaluates risk for VRP solutions.

    Risk(S) = α·TW_risk(S) + β·Cap_risk(S) + γ·Op_risk(S)

    Components:
    - TW_risk: Time window violation probability
    - Cap_risk: Capacity utilization stress
    - Op_risk: Operational uncertainty (traffic, service time)
    """

    def __init__(self, config: RiskConfig | None = None) -> None:
        """Initialize risk evaluator."""
        self.config = config or RiskConfig()

    def evaluate(self, solution: Solution, instance: VRPInstance) -> RiskMetrics:
        """Compute risk metrics for a solution.

        Args:
            solution: VRP solution
            instance: VRP instance

        Returns:
            RiskMetrics with all components
        """
        # Compute individual risk components
        tw_risk = self._compute_time_window_risk(solution, instance)
        cap_risk = self._compute_capacity_risk(solution, instance)
        op_risk = self._compute_operational_risk(solution, instance)

        # Weighted sum
        total = (
            self.config.time_window_weight * tw_risk
            + self.config.capacity_weight * cap_risk
            + self.config.operational_weight * op_risk
        )

        # Route-level risk
        route_risks = {}
        for i, route in enumerate(solution.routes):
            if len(route) > 2:
                route_risks[i] = self._compute_route_risk(
                    route, instance, i
                )

        return RiskMetrics(
            total_risk=total,
            time_window_risk=tw_risk,
            capacity_risk=cap_risk,
            operational_risk=op_risk,
            route_risks=route_risks,
            details={
                "tw_weight": self.config.time_window_weight,
                "cap_weight": self.config.capacity_weight,
                "op_weight": self.config.operational_weight,
            },
        )

    def _compute_time_window_risk(
        self, solution: Solution, instance: VRPInstance
    ) -> float:
        """Compute time window risk.

        Risk increases with:
        - Tight time windows
        - Late arrivals
        - Routes with many time-critical customers
        """
        total_risk = 0.0
        all_nodes = [instance.depot] + instance.customers

        for route in solution.routes:
            if len(route) <= 2:
                continue

            # Track arrival times
            current_time = 0.0

            for i in range(len(route) - 1):
                # Travel time
                current_time += instance.time_matrix[route[i], route[i + 1]]

                if route[i + 1] == 0:  # Back to depot
                    continue

                customer = instance.customers[route[i + 1] - 1]
                tw_start = customer.time_window_start
                tw_end = customer.time_window_end
                tw_width = tw_end - tw_start

                # Risk from tight windows
                tightness_risk = 1.0 / (tw_width + 30.0)  # Normalize by ~30 min

                # Risk from lateness
                if current_time > tw_end:
                    lateness = current_time - tw_end
                    lateness_risk = self.config.tw_late_penalty * lateness / 60.0
                else:
                    lateness_risk = 0.0

                # Risk from earliness (waiting)
                if current_time < tw_start:
                    wait = tw_start - current_time
                    earliness_risk = self.config.tw_early_penalty * wait / 60.0
                else:
                    earliness_risk = 0.0

                route_risk = tightness_risk + lateness_risk + earliness_risk
                total_risk += route_risk

                # Add service time
                current_time += customer.service_time

        return total_risk

    def _compute_capacity_risk(
        self, solution: Solution, instance: VRPInstance
    ) -> float:
        """Compute capacity utilization risk.

        Risk increases with:
        - High utilization (above threshold)
        - Uneven load distribution
        """
        total_risk = 0.0

        for i, route in enumerate(solution.routes):
            if len(route) <= 2:
                continue

            # Calculate route demand
            route_demand = sum(
                instance.customers[n - 1].demand
                for n in route[1:-1]
                if n > 0
            )

            capacity = instance.vehicles[i].capacity if i < len(instance.vehicles) else 100.0
            utilization = route_demand / capacity

            # Risk increases above threshold
            if utilization > self.config.capacity_threshold:
                excess = utilization - self.config.capacity_threshold
                risk = excess ** 2 * 10.0  # Quadratic penalty
            else:
                risk = utilization * 0.1  # Small base risk

            total_risk += risk

        return total_risk

    def _compute_operational_risk(
        self, solution: Solution, instance: VRPInstance
    ) -> float:
        """Compute operational risk.

        Includes:
        - Distance uncertainty
        - Traffic uncertainty
        - Service time variability
        """
        total_risk = 0.0

        for route in solution.routes:
            if len(route) <= 2:
                continue

            # Route distance risk
            route_distance = sum(
                instance.distance_matrix[route[i], route[i + 1]]
                for i in range(len(route) - 1)
            )

            # Traffic uncertainty increases risk
            distance_risk = route_distance * self.config.distance_weight

            # Traffic uncertainty
            traffic_risk = route_distance * self.config.traffic_uncertainty * 0.5

            # Service time variability (more customers = more risk)
            num_customers = len(route) - 2
            service_risk = num_customers * 0.1  # Each customer adds variability

            total_risk += distance_risk + traffic_risk + service_risk

        return total_risk

    def _compute_route_risk(
        self, route: list[int], instance: VRPInstance, route_idx: int
    ) -> float:
        """Compute risk for a single route."""
        # Simplified route risk
        tw_risk = 0.0
        cap_risk = 0.0
        op_risk = 0.0

        # Time window risk (simplified)
        for n in route[1:-1]:
            if n > 0:
                customer = instance.customers[n - 1]
                tw_width = customer.time_window_end - customer.time_window_start
                tw_risk += 1.0 / (tw_width + 30.0)

        # Capacity risk
        route_demand = sum(
            instance.customers[n - 1].demand
            for n in route[1:-1]
            if n > 0
        )
        capacity = instance.vehicles[route_idx].capacity if route_idx < len(instance.vehicles) else 100.0
        utilization = route_demand / capacity
        cap_risk = max(0, utilization - self.config.capacity_threshold) ** 2

        # Operational risk
        route_distance = sum(
            instance.distance_matrix[route[i], route[i + 1]]
            for i in range(len(route) - 1)
        )
        op_risk = route_distance * 0.1

        return (
            self.config.time_window_weight * tw_risk
            + self.config.capacity_weight * cap_risk
            + self.config.operational_weight * op_risk
        )


def compute_solution_risk(
    solution: Solution, instance: VRPInstance
) -> RiskMetrics:
    """Convenience function to compute solution risk."""
    evaluator = RiskEvaluator()
    return evaluator.evaluate(solution, instance)