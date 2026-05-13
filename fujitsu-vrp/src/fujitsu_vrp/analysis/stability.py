"""
Solution Stability Analysis.

Analyzes solution stability and convergence properties.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..data.problem_builder import Solution


@dataclass
class StabilityConfig:
    """Configuration for stability analysis."""

    window_size: int = 50
    convergence_threshold: float = 0.01
    oscillation_threshold: float = 0.1


@dataclass
class SolutionStability:
    """Stability metrics for a solution trajectory."""

    # Convergence
    is_converged: bool
    convergence_rate: float
    convergence_iteration: int | None

    # Stability
    stability_score: float  # 0-1, higher = more stable
    oscillation_score: float  # 0-1, higher = more oscillation

    # Variability
    cost_variance: float
    cost_range: tuple[float, float]

    # Trajectory
    improvement_rate: float
    final_improvement: float

    details: dict[str, Any] = field(default_factory=dict)


class StabilityAnalyzer:
    """Analyzes stability of solution trajectory."""

    def __init__(self, config: StabilityConfig | None = None) -> None:
        """Initialize stability analyzer."""
        self.config = config or StabilityConfig()
        self.cost_history: list[float] = []
        self.best_history: list[float] = []

    def add_point(self, cost: float, best_cost: float) -> None:
        """Add a point to the trajectory."""
        self.cost_history.append(cost)
        self.best_history.append(best_cost)

    def analyze(self) -> SolutionStability:
        """Analyze stability of trajectory."""
        if len(self.cost_history) < 10:
            return SolutionStability(
                is_converged=False,
                convergence_rate=0.0,
                convergence_iteration=None,
                stability_score=0.0,
                oscillation_score=0.0,
                cost_variance=0.0,
                cost_range=(0.0, 0.0),
                improvement_rate=0.0,
                final_improvement=0.0,
            )

        costs = np.array(self.cost_history)
        best = np.array(self.best_history)

        # Convergence analysis
        is_converged, convergence_rate, convergence_iter = self._analyze_convergence(
            best
        )

        # Stability score
        stability = self._compute_stability(costs)

        # Oscillation score
        oscillation = self._compute_oscillation(costs)

        # Variability
        variance = float(np.var(costs[-self.config.window_size :]))
        cost_range = (float(np.min(costs)), float(np.max(costs)))

        # Improvement
        improvement_rate = (best[0] - best[-1]) / (best[0] + 1e-10)
        final_improvement = (best[0] - best[-1]) / (best[0] + 1e-10)

        return SolutionStability(
            is_converged=is_converged,
            convergence_rate=convergence_rate,
            convergence_iteration=convergence_iter,
            stability_score=stability,
            oscillation_score=oscillation,
            cost_variance=variance,
            cost_range=cost_range,
            improvement_rate=improvement_rate,
            final_improvement=final_improvement,
        )

    def _analyze_convergence(
        self, best: np.ndarray
    ) -> tuple[bool, float, int | None]:
        """Analyze convergence of best solution."""
        if len(best) < self.config.window_size:
            return False, 0.0, None

        # Check if improvement rate is below threshold
        window = best[-self.config.window_size :]
        improvement = (window[0] - window[-1]) / (window[0] + 1e-10)

        if improvement < self.config.convergence_threshold:
            # Find convergence iteration
            for i in range(len(best) - self.config.window_size):
                segment = best[i : i + self.config.window_size]
                seg_improvement = (segment[0] - segment[-1]) / (
                    segment[0] + 1e-10
                )
                if seg_improvement < self.config.convergence_threshold:
                    return True, float(improvement), i

            return True, float(improvement), len(best) - self.config.window_size

        return False, float(improvement), None

    def _compute_stability(self, costs: np.ndarray) -> float:
        """Compute stability score."""
        if len(costs) < 2:
            return 0.0

        # Variance of recent costs
        recent = costs[-self.config.window_size :]
        variance = np.var(recent)

        # Normalize by mean
        mean_cost = np.mean(recent) + 1e-10
        normalized_variance = variance / (mean_cost**2)

        # Convert to stability (lower variance = higher stability)
        stability = 1.0 / (1.0 + normalized_variance * 100)

        return float(np.clip(stability, 0, 1))

    def _compute_oscillation(self, costs: np.ndarray) -> float:
        """Compute oscillation score."""
        if len(costs) < 3:
            return 0.0

        # Count direction changes
        differences = np.diff(costs)
        signs = np.sign(differences)
        sign_changes = np.sum(np.abs(np.diff(signs))) / 2

        # Normalize by number of iterations
        oscillation = sign_changes / (len(costs) - 1)

        return float(np.clip(oscillation, 0, 1))

    def reset(self) -> None:
        """Reset analyzer."""
        self.cost_history = []
        self.best_history = []