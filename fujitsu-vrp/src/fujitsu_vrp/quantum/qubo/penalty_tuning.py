"""
Penalty Coefficient Optimization for QUBO.

Implements methods to find appropriate penalty coefficients that balance
objective minimization with constraint satisfaction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...data.synthetic_generator import VRPInstance


@dataclass
class PenaltyTuningConfig:
    """Configuration for penalty coefficient tuning."""

    # Base penalties
    base_assignment_penalty: float = 1000.0
    base_capacity_penalty: float = 100.0
    base_time_window_penalty: float = 50.0

    # Tuning parameters
    scale_with_problem_size: bool = True
    relative_gap_factor: float = 2.0  # Penalty = gap_factor * max_cost_gap

    # Adaptive tuning
    use_adaptive: bool = True
    violation_tolerance: float = 0.01
    max_iterations: int = 10


class PenaltyTuner:
    """Tunes penalty coefficients for QUBO formulation.

    The key insight is that penalties should be:
    1. Large enough to discourage constraint violations
    2. Not too large to overwhelm the objective function

    Common approaches:
    - Scale with problem size
    - Use gap-based scaling
    - Adaptive tuning based on violation frequency
    """

    def __init__(self, config: PenaltyTuningConfig | None = None) -> None:
        """Initialize penalty tuner."""
        self.config = config or PenaltyTuningConfig()

    def compute_penalties(
        self,
        instance: VRPInstance,
        partial_routes: list[list[int]],
        removed_customers: list[int],
    ) -> dict[str, float]:
        """Compute appropriate penalty coefficients.

        Args:
            instance: VRP instance
            partial_routes: Current partial routes
            removed_customers: Customers to be inserted

        Returns:
            Dictionary of penalty coefficients
        """
        if self.config.scale_with_problem_size:
            return self._compute_scaled_penalties(instance, removed_customers)
        else:
            return {
                "assignment": self.config.base_assignment_penalty,
                "capacity": self.config.base_capacity_penalty,
                "time_window": self.config.base_time_window_penalty,
            }

    def _compute_scaled_penalties(
        self,
        instance: VRPInstance,
        removed_customers: list[int],
    ) -> dict[str, float]:
        """Compute penalties scaled to problem size and objective range."""
        # Estimate objective magnitude
        distances = []
        for i in removed_customers:
            for j in removed_customers:
                if i != j:
                    distances.append(instance.distance_matrix[i, j])
            distances.append(instance.distance_matrix[0, i])  # Depot distance

        if distances:
            max_distance = max(distances)
            avg_distance = np.mean(distances)
        else:
            max_distance = 10.0
            avg_distance = 5.0

        # Assignment penalty: should be larger than potential cost difference
        # from violating the constraint
        assignment_penalty = max(
            self.config.base_assignment_penalty,
            self.config.relative_gap_factor * max_distance * len(removed_customers),
        )

        # Capacity penalty: scale with demands
        if removed_customers:
            demands = [instance.customers[c - 1].demand for c in removed_customers]
            max_demand = max(demands)
            avg_capacity = np.mean([v.capacity for v in instance.vehicles])
            capacity_penalty = max(
                self.config.base_capacity_penalty,
                avg_distance * max_demand / avg_capacity * 10,
            )
        else:
            capacity_penalty = self.config.base_capacity_penalty

        # Time window penalty: scale with time window widths
        if removed_customers:
            tw_widths = [
                instance.customers[c - 1].time_window_end
                - instance.customers[c - 1].time_window_start
                for c in removed_customers
            ]
            avg_tw_width = np.mean(tw_widths)
            # Tighter windows need higher penalty
            tw_penalty = max(
                self.config.base_time_window_penalty,
                avg_distance / (avg_tw_width + 1) * 100,
            )
        else:
            tw_penalty = self.config.base_time_window_penalty

        return {
            "assignment": assignment_penalty,
            "capacity": capacity_penalty,
            "time_window": tw_penalty,
        }

    def adaptive_tune(
        self,
        current_penalties: dict[str, float],
        violations: dict[str, float],
        iteration: int,
    ) -> dict[str, float]:
        """Adaptively adjust penalties based on violations.

        Args:
            current_penalties: Current penalty values
            violations: Current constraint violation rates
            iteration: Current iteration number

        Returns:
            Updated penalty values
        """
        if not self.config.use_adaptive:
            return current_penalties

        new_penalties = {}
        for key, penalty in current_penalties.items():
            violation_rate = violations.get(key, 0.0)

            if violation_rate > self.config.violation_tolerance:
                # Increase penalty
                factor = 1.5 + 0.5 * violation_rate
                new_penalty = penalty * factor
            else:
                # Slight decrease to avoid over-penalizing
                factor = 0.95
                new_penalty = penalty * factor

            new_penalties[key] = new_penalty

        return new_penalties


def estimate_penalty_range(
    instance: VRPInstance,
    removed_customers: list[int],
) -> tuple[float, float, float]:
    """Estimate reasonable penalty ranges.

    Returns:
        (min_penalty, recommended_penalty, max_penalty)
    """
    if not removed_customers:
        return 100.0, 1000.0, 10000.0

    # Compute objective scale
    distances = []
    for c in removed_customers:
        distances.append(instance.distance_matrix[0, c])
        for c2 in removed_customers:
            if c != c2:
                distances.append(instance.distance_matrix[c, c2])

    max_dist = max(distances) if distances else 10.0
    num_customers = len(removed_customers)

    # Penalties should be larger than potential objective improvement
    # from violating constraints
    min_penalty = max_dist
    recommended = max_dist * num_customers
    max_penalty = max_dist * num_customers * 10

    return min_penalty, recommended, max_penalty


def validate_penalty_strength(
    Q: np.ndarray,
    penalty_key: str,
    penalty_value: float,
    objective_scale: float,
) -> float:
    """Validate that penalty strength is appropriate.

    Checks that penalty doesn't overwhelm objective.

    Args:
        Q: QUBO matrix
        penalty_key: Which constraint this penalty is for
        penalty_value: Current penalty value
        objective_scale: Scale of objective function

    Returns:
        Recommended penalty value (may adjust if too strong/weak)
    """
    ratio = penalty_value / (objective_scale + 1e-10)

    # Penalty should be 2-10x objective scale
    if ratio < 1.0:
        return objective_scale * 2.0
    elif ratio > 1000.0:
        return objective_scale * 100.0
    else:
        return penalty_value