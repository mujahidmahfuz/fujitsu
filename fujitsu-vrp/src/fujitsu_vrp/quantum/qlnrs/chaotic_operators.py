"""
Chaotic Operators for Operator Selection in QLNRS.

Implements chaotic maps for adaptive operator selection based on
Lyapunov exponent analysis.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...classical.lns_base import DestroyOperator

logger = logging.getLogger(__name__)


@dataclass
class ChaoticState:
    """State of a chaotic map."""

    x: float  # Current state value
    history: list[float]  # State history for Lyapunov computation

    def __post_init__(self):
        if self.history is None:
            self.history = [self.x]


class ChaoticMap(ABC):
    """Abstract base class for chaotic maps.

    Chaotic maps provide deterministic chaos for operator selection,
    with controllable Maximum Lyapunov Exponent (MLE) for balancing
    exploration vs exploitation.
    """

    def __init__(self, initial_state: float | None = None, seed: int | None = None):
        """Initialize chaotic map."""
        self.rng = np.random.default_rng(seed)
        self.state = ChaoticState(
            x=initial_state if initial_state else self.rng.random(),
            history=[],
        )

    @property
    @abstractmethod
    def mle(self) -> float:
        """Maximum Lyapunov Exponent for this map.

        MLE > 0 indicates chaos.
        Higher MLE = more chaotic = more exploration.
        Lower MLE = less chaotic = more exploitation.
        """
        pass

    @abstractmethod
    def iterate(self) -> float:
        """Perform one iteration of the chaotic map.

        Returns:
            Next state value
        """
        pass

    @property
    def name(self) -> str:
        """Name of this chaotic map."""
        return self.__class__.__name__

    def reset(self, initial_state: float | None = None) -> None:
        """Reset to initial state."""
        self.state.x = initial_state if initial_state else self.rng.random()
        self.state.history = [self.state.x]

    def get_sequence(self, n: int) -> list[float]:
        """Generate n values from the chaotic map."""
        sequence = []
        for _ in range(n):
            sequence.append(self.iterate())
        return sequence


class LogisticMap(ChaoticMap):
    """Logistic map: x_{n+1} = r * x_n * (1 - x_n).

    The logistic map exhibits chaos for r > 3.57.
    At r = 4.0, MLE = ln(2) ≈ 0.693 (fully chaotic).

    MLE varies with r:
    - r ≈ 3.57: MLE → 0 (onset of chaos)
    - r = 4.0: MLE = ln(2) ≈ 0.693
    """

    def __init__(
        self,
        r: float = 4.0,
        initial_state: float | None = None,
        seed: int | None = None,
    ):
        """Initialize logistic map.

        Args:
            r: Control parameter (chaotic for r > 3.57)
            initial_state: Initial x value in (0, 1)
            seed: Random seed
        """
        super().__init__(initial_state, seed)
        self.r = r

    @property
    def mle(self) -> float:
        """Maximum Lyapunov Exponent.

        For logistic map: MLE ≈ ln(r) for r close to 4.
        At r = 4: MLE = ln(2) ≈ 0.693
        """
        if self.r <= 3.57:
            return 0.0
        elif self.r >= 4.0:
            return np.log(2)
        else:
            # Approximate MLE for intermediate r
            return max(0.0, np.log(self.r) - np.log(3.57))

    def iterate(self) -> float:
        """Perform one iteration: x_{n+1} = r * x_n * (1 - x_n)."""
        x = self.state.x
        # Avoid fixed points
        if x <= 0.001 or x >= 0.999:
            x = 0.5

        x_new = self.r * x * (1 - x)

        # Handle numerical issues
        if x_new <= 0 or x_new >= 1:
            x_new = 0.5

        self.state.x = x_new
        self.state.history.append(x_new)

        return x_new


class TentMap(ChaoticMap):
    """Tent map: x_{n+1} = 1 - 2|x_n - 0.5|.

    The tent map has MLE = ln(2) ≈ 0.693 for full parameter range.
    Fully chaotic when the slope magnitude is 2.
    """

    def __init__(
        self,
        initial_state: float | None = None,
        seed: int | None = None,
    ):
        """Initialize tent map."""
        super().__init__(initial_state, seed)

    @property
    def mle(self) -> float:
        """MLE = ln(2) for tent map."""
        return np.log(2)

    def iterate(self) -> float:
        """Perform one iteration: x_{n+1} = 1 - 2|x_n - 0.5|."""
        x = self.state.x

        # Avoid edge cases
        if x <= 0.001 or x >= 0.999:
            x = 0.5

        if x < 0.5:
            x_new = 2 * x
        else:
            x_new = 2 * (1 - x)

        # Handle numerical issues
        if x_new <= 0 or x_new >= 1:
            x_new = 0.5

        self.state.x = x_new
        self.state.history.append(x_new)

        return x_new


class ChebyshevMap(ChaoticMap):
    """Chebyshev map: x_{n+1} = cos(k * arccos(x_n)).

    The Chebyshev map has MLE = ln(k) for degree k.
    Common values:
    - k = 2: MLE = ln(2) ≈ 0.693
    - k = 3: MLE = ln(3) ≈ 1.099
    - k = 4: MLE = ln(4) ≈ 1.386
    """

    def __init__(
        self,
        k: int = 3,
        initial_state: float | None = None,
        seed: int | None = None,
    ):
        """Initialize Chebyshev map.

        Args:
            k: Degree of Chebyshev polynomial
            initial_state: Initial x value in [-1, 1]
            seed: Random seed
        """
        super().__init__(initial_state, seed)
        self.k = k

        # Transform initial state to [-1, 1]
        if initial_state is None:
            self.state.x = self.rng.random() * 2 - 1

    @property
    def mle(self) -> float:
        """MLE = ln(k) for Chebyshev map."""
        return np.log(self.k)

    def iterate(self) -> float:
        """Perform one iteration: x_{n+1} = cos(k * arccos(x_n))."""
        x = self.state.x

        # Avoid edge cases
        if abs(x) >= 0.999:
            x = 0.5

        # Chebyshev iteration
        # cos(k * arccos(x)) = T_k(x) where T_k is Chebyshev polynomial
        x_new = np.cos(self.k * np.arccos(np.clip(x, -1, 1)))

        # Transform to [0, 1] for operator selection
        normalized = (x_new + 1) / 2

        self.state.x = x_new
        self.state.history.append(normalized)

        return normalized


class BernoulliMap(ChaoticMap):
    """Bernoulli map: x_{n+1} = 2 * x_n mod 1.

    Simple chaotic map with MLE = ln(2) ≈ 0.693.
    """

    def __init__(
        self,
        initial_state: float | None = None,
        seed: int | None = None,
    ):
        """Initialize Bernoulli map."""
        super().__init__(initial_state, seed)

    @property
    def mle(self) -> float:
        """MLE = ln(2) for Bernoulli map."""
        return np.log(2)

    def iterate(self) -> float:
        """Perform one iteration: x_{n+1} = 2 * x_n mod 1."""
        x = self.state.x

        # Avoid edge cases
        if x <= 0.001 or x >= 0.999:
            x = 0.5

        x_new = (2 * x) % 1.0

        # Handle numerical issues
        if x_new <= 0 or x_new >= 1:
            x_new = 0.5

        self.state.x = x_new
        self.state.history.append(x_new)

        return x_new


class ChaoticOperatorSelector:
    """Selects destroy operators using chaotic maps.

    Uses Lyapunov-adaptive control to balance exploration
    and exploitation in operator selection.

    The key insight:
    - High MLE (chaotic) → More exploration → Select diverse operators
    - Low MLE (stable) → More exploitation → Select best-performing operators
    """

    def __init__(
        self,
        operators: list[DestroyOperator],
        chaotic_map: ChaoticMap | None = None,
        adaptation_mode: str = "lyapunov",
        lyapunov_min: float = 0.1,
        lyapunov_max: float = 0.5,
        seed: int | None = None,
    ) -> None:
        """Initialize chaotic operator selector.

        Args:
            operators: List of destroy operators
            chaotic_map: Chaotic map for selection (default: LogisticMap)
            adaptation_mode: "lyapunov", "static", or "performance"
            lyapunov_min: Minimum Lyapunov exponent threshold
            lyapunov_max: Maximum Lyapunov exponent threshold
            seed: Random seed
        """
        self.operators = operators
        self.chaotic_map = chaotic_map or LogisticMap(seed=seed)
        self.adaptation_mode = adaptation_mode
        self.lyapunov_min = lyapunov_min
        self.lyapunov_max = lyapunov_max

        self.rng = np.random.default_rng(seed)

        # Performance tracking
        self.operator_scores = {op.name: 0.0 for op in operators}
        self.operator_counts = {op.name: 0 for op in operators}
        self.operator_improvements = {op.name: [] for op in operators}

        # Chaotic state for each operator
        self.operator_chaotic_states = {
            op.name: self.chaotic_map.state.x for op in operators
        }

        # Lyapunov history
        self.lyapunov_history: list[float] = []

    def select(
        self,
        lyapunov_exponent: float | None = None,
    ) -> DestroyOperator:
        """Select a destroy operator using chaotic dynamics.

        Args:
            lyapunov_exponent: Current Lyapunov exponent of search trajectory

        Returns:
            Selected destroy operator
        """
        # Iterate chaotic map
        chaotic_value = self.chaotic_map.iterate()

        # Map chaotic value to operator
        if self.adaptation_mode == "lyapunov" and lyapunov_exponent is not None:
            operator = self._select_lyapunov_adaptive(
                chaotic_value, lyapunov_exponent
            )
        elif self.adaptation_mode == "performance":
            operator = self._select_performance_adaptive(chaotic_value)
        else:
            operator = self._select_static(chaotic_value)

        return operator

    def _select_static(self, chaotic_value: float) -> DestroyOperator:
        """Select operator based purely on chaotic value."""
        # Map chaotic value to operator index
        index = int(chaotic_value * len(self.operators)) % len(self.operators)
        return self.operators[index]

    def _select_lyapunov_adaptive(
        self,
        chaotic_value: float,
        lyapunov: float,
    ) -> DestroyOperator:
        """Select operator adaptively based on Lyapunov exponent.

        - Low Lyapunov (converging): Need more exploration → use chaotic selection
        - High Lyapunov (chaotic): Need more exploitation → use performance selection
        """
        self.lyapunov_history.append(lyapunov)

        if lyapunov < self.lyapunov_min:
            # Converging: increase exploration
            # Use chaotic selection with higher-MLE map
            return self._select_static(chaotic_value)

        elif lyapunov > self.lyapunov_max:
            # Chaotic: increase exploitation
            # Weight by performance
            return self._select_performance_adaptive(chaotic_value)

        else:
            # Balanced: mix chaotic and performance
            if self.rng.random() < 0.5:
                return self._select_static(chaotic_value)
            else:
                return self._select_performance_adaptive(chaotic_value)

    def _select_performance_adaptive(
        self, chaotic_value: float
    ) -> DestroyOperator:
        """Select operator based on performance scores."""
        # Compute selection probabilities
        scores = np.array(
            [self.operator_scores[op.name] for op in self.operators]
        )

        # Add small constant to avoid zero probabilities
        scores = scores + 0.1

        # Normalize to probabilities
        probs = scores / scores.sum()

        # Add chaotic perturbation
        probs = probs * (1 + 0.1 * chaotic_value)
        probs = probs / probs.sum()

        # Select operator
        index = self.rng.choice(len(self.operators), p=probs)
        return self.operators[index]

    def update_performance(
        self,
        operator: DestroyOperator,
        improvement: float,
    ) -> None:
        """Update operator performance score.

        Args:
            operator: Operator that was used
            improvement: Solution improvement achieved
        """
        self.operator_counts[operator.name] += 1
        self.operator_improvements[operator.name].append(improvement)

        # Update score using exponential moving average
        alpha = 0.3  # Learning rate
        current_score = self.operator_scores[operator.name]
        self.operator_scores[operator.name] = (
            (1 - alpha) * current_score + alpha * improvement
        )

    def adapt_chaotic_map(self, lyapunov: float) -> None:
        """Adapt chaotic map parameters based on Lyapunov exponent.

        If search is too convergent (low λ), increase chaos.
        If search is too chaotic (high λ), decrease chaos.
        """
        if isinstance(self.chaotic_map, LogisticMap):
            # Adjust r parameter
            if lyapunov < self.lyapunov_min:
                # Need more chaos
                self.chaotic_map.r = min(4.0, self.chaotic_map.r + 0.05)
            elif lyapunov > self.lyapunov_max:
                # Need less chaos
                self.chaotic_map.r = max(3.58, self.chaotic_map.r - 0.05)

        elif isinstance(self.chaotic_map, ChebyshevMap):
            # Adjust k parameter
            if lyapunov < self.lyapunov_min:
                self.chaotic_map.k = min(4, self.chaotic_map.k + 1)
            elif lyapunov > self.lyapunov_max:
                self.chaotic_map.k = max(2, self.chaotic_map.k - 1)

    def get_statistics(self) -> dict:
        """Get operator selection statistics."""
        stats = {
            "operator_counts": dict(self.operator_counts),
            "operator_scores": dict(self.operator_scores),
            "operator_avg_improvements": {
                name: np.mean(imps) if imps else 0.0
                for name, imps in self.operator_improvements.items()
            },
            "chaotic_map": self.chaotic_map.name,
            "chaotic_mle": self.chaotic_map.mle,
            "lyapunov_history": list(self.lyapunov_history),
        }
        return stats


def create_chaotic_map(
    map_type: str,
    mle_target: float = 0.5,
    seed: int | None = None,
) -> ChaoticMap:
    """Factory function to create chaotic maps with target MLE.

    Args:
        map_type: "logistic", "tent", "chebyshev", or "bernoulli"
        mle_target: Target maximum Lyapunov exponent
        seed: Random seed

    Returns:
        ChaoticMap instance
    """
    if map_type == "logistic":
        # For logistic map, MLE ≈ ln(r) for r near 4
        # Target: mle_target = ln(r) - ln(2)
        # => r = exp(mle_target) * 2
        r = min(4.0, max(3.58, np.exp(mle_target) * 2))
        return LogisticMap(r=r, seed=seed)

    elif map_type == "tent":
        # Tent map has fixed MLE = ln(2)
        return TentMap(seed=seed)

    elif map_type == "chebyshev":
        # MLE = ln(k), so k = exp(mle_target)
        k = max(2, min(4, int(np.exp(mle_target))))
        return ChebyshevMap(k=k, seed=seed)

    elif map_type == "bernoulli":
        # Bernoulli has fixed MLE = ln(2)
        return BernoulliMap(seed=seed)

    else:
        raise ValueError(f"Unknown map type: {map_type}")