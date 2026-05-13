"""
Lyapunov Exponent Analysis for QLNRS.

Computes Lyapunov exponents from solution trajectory to guide
adaptive parameter control in the search algorithm.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ...data.problem_builder import Solution

logger = logging.getLogger(__name__)


@dataclass
class LyapunovConfig:
    """Configuration for Lyapunov exponent computation."""

    # Window size for computation
    window_size: int = 50

    # Minimum points needed
    min_points: int = 10

    # Embedding dimension for trajectory
    embedding_dimension: int = 3

    # Time delay for embedding
    time_delay: int = 1

    # Smoothing factor
    smoothing: float = 0.1


@dataclass
class TrajectoryPoint:
    """A point in solution trajectory space."""

    iteration: int
    cost: float
    distance: float
    time_violation: float
    capacity_violation: float

    # Optional features
    num_routes: int = 0
    features: dict[str, float] = field(default_factory=dict)

    def to_vector(self) -> npt.NDArray[np.float64]:
        """Convert to feature vector."""
        base = np.array([
            self.cost,
            self.distance,
            self.time_violation,
            self.capacity_violation,
            self.num_routes,
        ])
        if self.features:
            extra = np.array(list(self.features.values()))
            return np.concatenate([base, extra])
        return base


@dataclass
class LyapunovResult:
    """Result of Lyapunov analysis."""

    # Computed Lyapunov exponent
    exponent: float

    # Confidence/validity of the estimate
    confidence: float

    # Trajectory stability classification
    stability: str  # "stable", "periodic", "chaotic"

    # Recommendations
    exploration_recommendation: float  # 0-1, higher = more exploration
    parameter_adjustment: dict[str, float]

    # Diagnostics
    divergence_history: list[float] = field(default_factory=list)
    trajectory_length: int = 0


class LyapunovAnalyzer:
    """Analyzes solution trajectory stability using Lyapunov exponents.

    The Lyapunov exponent measures the rate of separation of infinitesimally
    close trajectories in phase space. For LNS search:

    - λ < 0.1: Converging/stable search → May be stuck, need more exploration
    - 0.1 ≤ λ ≤ 0.5: Balanced search → Maintain current parameters
    - λ > 0.5: Chaotic search → May miss optima, need more exploitation

    The exponent is computed from the solution trajectory using time-series
    methods adapted for discrete optimization.
    """

    def __init__(self, config: LyapunovConfig | None = None) -> None:
        """Initialize analyzer with configuration."""
        self.config = config or LyapunovConfig()
        self.trajectory: list[TrajectoryPoint] = []

    def add_point(self, point: TrajectoryPoint) -> None:
        """Add a point to the trajectory."""
        self.trajectory.append(point)

    def add_solution(
        self,
        solution: Solution,
        iteration: int,
        cost: float,
    ) -> None:
        """Add solution state to trajectory."""
        point = TrajectoryPoint(
            iteration=iteration,
            cost=cost,
            distance=solution.total_distance,
            time_violation=solution.time_window_violations,
            capacity_violation=solution.capacity_violations,
            num_routes=solution.num_routes,
        )
        self.add_point(point)

    def compute_lyapunov(self) -> LyapunovResult:
        """Compute Lyapunov exponent from trajectory.

        Uses the Rosenstein algorithm for estimating the largest
        Lyapunov exponent from a time series.

        Returns:
            LyapunovResult with exponent and recommendations
        """
        if len(self.trajectory) < self.config.min_points:
            return LyapunovResult(
                exponent=0.0,
                confidence=0.0,
                stability="unknown",
                exploration_recommendation=0.5,
                parameter_adjustment={},
                trajectory_length=len(self.trajectory),
            )

        # Extract trajectory vectors
        vectors = np.array([p.to_vector() for p in self.trajectory])

        # Normalize features
        vectors = self._normalize_vectors(vectors)

        # Compute divergence rates
        divergences = self._compute_divergences(vectors)

        if not divergences:
            return LyapunovResult(
                exponent=0.0,
                confidence=0.0,
                stability="unknown",
                exploration_recommendation=0.5,
                parameter_adjustment={},
                trajectory_length=len(self.trajectory),
            )

        # Estimate Lyapunov exponent from divergence slope
        lyapunov = self._estimate_from_divergences(divergences)

        # Classify stability
        stability = self._classify_stability(lyapunov)

        # Generate recommendations
        exploration_rec = self._compute_exploration_recommendation(lyapunov)
        param_adjust = self._compute_parameter_adjustments(lyapunov)

        # Compute confidence
        confidence = self._compute_confidence(divergences)

        return LyapunovResult(
            exponent=lyapunov,
            confidence=confidence,
            stability=stability,
            exploration_recommendation=exploration_rec,
            parameter_adjustment=param_adjust,
            divergence_history=divergences,
            trajectory_length=len(self.trajectory),
        )

    def _normalize_vectors(
        self, vectors: npt.NDArray[np.float64]
    ) -> npt.NDArray[np.float64]:
        """Normalize trajectory vectors to zero mean and unit variance."""
        mean = np.mean(vectors, axis=0)
        std = np.std(vectors, axis=0) + 1e-10
        return (vectors - mean) / std

    def _compute_divergences(
        self, vectors: npt.NDArray[np.float64]
    ) -> list[float]:
        """Compute divergence rates between trajectory points.

        Uses nearest-neighbor distances in embedded space.
        """
        n = len(vectors)
        window = min(self.config.window_size, n // 2)

        divergences = []

        for i in range(window, n - window):
            # Find nearest neighbor in the past
            past_vectors = vectors[max(0, i - window) : i]
            current = vectors[i]

            if len(past_vectors) == 0:
                continue

            # Distance to all past points
            distances = np.sqrt(np.sum((past_vectors - current) ** 2, axis=1))

            if len(distances) == 0:
                continue

            # Find nearest neighbor
            nearest_idx = np.argmin(distances)
            min_distance = distances[nearest_idx]

            # Track divergence over time
            if min_distance < 1e-10:
                # Points are identical, maximum stability
                continue

            # Compute divergence rate
            # Look at distance after k steps
            k_steps = min(window, n - i - 1)
            if k_steps <= 0:
                continue

            future_current = vectors[i + k_steps]
            future_nearest = vectors[max(0, i - window + nearest_idx) + k_steps]
            future_distance = np.sqrt(
                np.sum((future_current - future_nearest) ** 2)
            )

            # Divergence rate = ln(d_future / d_initial) / k
            if min_distance > 1e-10 and future_distance > 1e-10:
                divergence = np.log(future_distance / min_distance) / k_steps
                divergences.append(divergence)

        return divergences

    def _estimate_from_divergences(
        self, divergences: list[float]
    ) -> float:
        """Estimate Lyapunov exponent from divergence rates."""
        if not divergences:
            return 0.0

        # Use median for robustness
        return float(np.median(divergences))

    def _classify_stability(self, lyapunov: float) -> str:
        """Classify trajectory stability from Lyapunov exponent."""
        if lyapunov < 0.1:
            return "stable"
        elif lyapunov < 0.5:
            return "periodic"
        else:
            return "chaotic"

    def _compute_exploration_recommendation(self, lyapunov: float) -> float:
        """Compute exploration recommendation from Lyapunov exponent.

        Returns:
            Value in [0, 1] where higher = more exploration recommended
        """
        # Stable (low λ): need more exploration
        # Chaotic (high λ): need more exploitation
        if lyapunov < 0.1:
            return 1.0  # Maximum exploration
        elif lyapunov > 0.5:
            return 0.0  # Maximum exploitation
        else:
            # Linear interpolation
            return 1.0 - (lyapunov - 0.1) / 0.4

    def _compute_parameter_adjustments(
        self, lyapunov: float
    ) -> dict[str, float]:
        """Compute parameter adjustment recommendations."""
        adjustments = {}

        if lyapunov < 0.1:
            # Too stable - need more chaos
            adjustments["temperature_increase"] = 0.2
            adjustments["destroy_fraction_increase"] = 0.1
            adjustments["chaotic_map_mle_increase"] = 0.1

        elif lyapunov > 0.5:
            # Too chaotic - need more stability
            adjustments["temperature_decrease"] = 0.1
            adjustments["destroy_fraction_decrease"] = 0.05
            adjustments["chaotic_map_mle_decrease"] = 0.1

        else:
            # Balanced - minor adjustments
            adjustments["maintain_current"] = 1.0

        return adjustments

    def _compute_confidence(self, divergences: list[float]) -> float:
        """Compute confidence in Lyapunov estimate."""
        if len(divergences) < 5:
            return 0.0

        # Higher confidence with more data and less variance
        variance = np.var(divergences)
        count = len(divergences)

        # Confidence increases with more data, decreases with variance
        confidence = min(1.0, count / 50.0) * (1.0 / (1.0 + variance))

        return float(confidence)

    def reset(self) -> None:
        """Reset trajectory."""
        self.trajectory = []

    def get_trajectory_summary(self) -> dict[str, Any]:
        """Get summary statistics of trajectory."""
        if not self.trajectory:
            return {"length": 0}

        costs = [p.cost for p in self.trajectory]
        distances = [p.distance for p in self.trajectory]

        return {
            "length": len(self.trajectory),
            "cost_mean": float(np.mean(costs)),
            "cost_std": float(np.std(costs)),
            "cost_min": float(min(costs)),
            "cost_max": float(max(costs)),
            "distance_mean": float(np.mean(distances)),
            "distance_std": float(np.std(distances)),
            "improvement_rate": float(
                (costs[0] - min(costs)) / (costs[0] + 1e-10)
            ),
        }


class AdaptiveLyapunovController:
    """Controls algorithm parameters based on Lyapunov analysis.

    This is the key novelty of QLNRS: using Lyapunov stability
    analysis to adaptively control the search dynamics.
    """

    def __init__(
        self,
        initial_temperature: float = 100.0,
        initial_destroy_fraction: float = 0.3,
        target_lyapunov: float = 0.25,
        adaptation_rate: float = 0.1,
    ) -> None:
        """Initialize adaptive controller.

        Args:
            initial_temperature: Initial SA temperature
            initial_destroy_fraction: Initial destroy fraction
            target_lyapunov: Target Lyapunov exponent (balanced)
            adaptation_rate: Rate of parameter adaptation
        """
        self.temperature = initial_temperature
        self.destroy_fraction = initial_destroy_fraction
        self.target_lyapunov = target_lyapunov
        self.adaptation_rate = adaptation_rate

        self.analyzer = LyapunovAnalyzer()

        # Bounds
        self.min_temperature = 0.1
        self.max_temperature = 1000.0
        self.min_destroy_fraction = 0.05
        self.max_destroy_fraction = 0.5

    def update(
        self,
        solution: Solution,
        iteration: int,
        cost: float,
    ) -> dict[str, float]:
        """Update parameters based on current solution.

        Args:
            solution: Current solution
            iteration: Current iteration
            cost: Current solution cost

        Returns:
            Dictionary of updated parameters
        """
        # Add to trajectory
        self.analyzer.add_solution(solution, iteration, cost)

        # Compute Lyapunov
        result = self.analyzer.compute_lyapunov()

        # Adapt parameters
        if result.confidence > 0.3:  # Only adapt with reasonable confidence
            self._adapt_parameters(result.exponent)

        return {
            "temperature": self.temperature,
            "destroy_fraction": self.destroy_fraction,
            "lyapunov": result.exponent,
            "stability": result.stability,
            "confidence": result.confidence,
        }

    def _adapt_parameters(self, lyapunov: float) -> None:
        """Adapt parameters based on Lyapunov exponent."""
        error = lyapunov - self.target_lyapunov
        adjustment = self.adaptation_rate * error

        # If λ is too high (chaotic), reduce exploration
        # If λ is too low (stable), increase exploration

        # Temperature adjustment
        self.temperature = np.clip(
            self.temperature * (1.0 - adjustment),
            self.min_temperature,
            self.max_temperature,
        )

        # Destroy fraction adjustment
        self.destroy_fraction = np.clip(
            self.destroy_fraction + adjustment * 0.1,
            self.min_destroy_fraction,
            self.max_destroy_fraction,
        )

    def get_chaotic_map_mle(self) -> float:
        """Get target MLE for chaotic map based on current state."""
        # Inverse relationship: lower temperature = more chaotic map needed
        # Higher temperature = less chaotic map needed
        temp_factor = np.log(self.temperature + 1) / np.log(1001)
        target_mle = 0.7 - 0.3 * temp_factor
        return float(np.clip(target_mle, 0.1, 0.7))

    def reset(self) -> None:
        """Reset controller state."""
        self.analyzer.reset()