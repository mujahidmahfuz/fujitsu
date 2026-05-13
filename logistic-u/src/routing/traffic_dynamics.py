"""
Traffic Dynamics Module — Lyapunov Exponent Stability Model.

Models Tokyo traffic as a time-varying velocity field on the road network.
Computes local Lyapunov exponents for each road segment to quantify
traffic predictability/stability. High λ_max = turbulent (avoid),
low λ_max = laminar (prefer).

This is integrated as an edge-weight modifier in the QUBO formulation:
    risk_cost(e, t) = distance(e) × (1 + α · max(0, λ_max(e, t)))

Scientific basis:
    - Traffic flow exhibits chaotic dynamics (positive Lyapunov exponents)
    - The largest Lyapunov exponent (MLE) determines the predictability horizon
    - Routing through stable-flow segments improves delivery reliability

This concept draws inspiration from the 2024 2nd-place winner (TU Ilmenau)
who applied quantum computing to Particle Image Velocimetry (fluid analysis).
We reverse the direction: using fluid dynamics concepts TO ENHANCE quantum VRP.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class TrafficState:
    """Traffic state for a single road segment at a given time."""
    edge: Tuple[int, int]
    speed_mean: float       # km/h average
    speed_std: float        # km/h standard deviation
    lyapunov_exp: float     # maximal Lyapunov exponent
    stability_score: float  # exp(-α * max(0, λ_max)), ∈ (0, 1]
    risk_cost: float        # distance / stability_score


class TrafficVelocityField:
    """Models traffic as a time-varying velocity field on the road network.

    For each road segment (edge) and time window, generates a synthetic
    but realistic speed profile based on:
    - Road type (arterial vs residential)
    - Time of day (rush hours, midday, evening)
    - Stochastic fluctuations (represent incident-driven chaos)

    The speed profiles are generated as time series with controlled
    statistical properties, enabling Lyapunov exponent computation.
    """

    # Tokyo traffic speed profiles (km/h) by road type and time period
    SPEED_PROFILES = {
        "arterial": {
            "early_morning": (35, 5),     # (mean, std) — 5-7 AM, low traffic
            "morning_rush":  (12, 8),     # 7-9 AM, severe congestion
            "midday":        (25, 4),     # 9 AM - 5 PM
            "evening_rush":  (10, 9),     # 5-7 PM, worst congestion
            "evening":       (30, 3),     # 7-10 PM
            "night":         (40, 2),     # 10 PM - 5 AM
        },
        "residential": {
            "early_morning": (25, 3),
            "morning_rush":  (15, 5),
            "midday":        (20, 3),
            "evening_rush":  (12, 6),
            "evening":       (22, 3),
            "night":         (25, 2),
        },
        "diagonal": {
            "early_morning": (30, 4),
            "morning_rush":  (14, 7),
            "midday":        (22, 4),
            "evening_rush":  (11, 8),
            "evening":       (28, 3),
            "night":         (35, 2),
        },
    }

    # Time period boundaries (minutes since midnight)
    TIME_PERIODS = [
        ("early_morning", 5 * 60, 7 * 60),
        ("morning_rush",  7 * 60, 9 * 60),
        ("midday",        9 * 60, 17 * 60),
        ("evening_rush",  17 * 60, 19 * 60),
        ("evening",       19 * 60, 22 * 60),
        ("night",         22 * 60, 29 * 60),  # wraps to 5 AM
    ]

    def __init__(
        self,
        G: nx.DiGraph,
        seed: int = 42,
        time_series_length: int = 120,
        sampling_interval: float = 30.0,  # seconds
    ):
        """
        Args:
            G: Road network graph with edge attributes 'road_type'.
            seed: Random seed.
            time_series_length: Number of samples in each speed time series.
            sampling_interval: Seconds between speed samples.
        """
        self.G = G
        self.rng = np.random.RandomState(seed)
        self.ts_length = time_series_length
        self.dt = sampling_interval

        # Generate speed time series for each edge and time period
        self._speed_series: Dict[Tuple, Dict[str, np.ndarray]] = {}
        self._generate_all_speed_series()

    def _generate_all_speed_series(self):
        """Pre-generate speed time series for all edges and time periods."""
        for u, v, data in self.G.edges(data=True):
            road_type = data.get("road_type", "residential")
            if road_type not in self.SPEED_PROFILES:
                road_type = "residential"

            self._speed_series[(u, v)] = {}
            for period_name, _, _ in self.TIME_PERIODS:
                mean, std = self.SPEED_PROFILES[road_type][period_name]
                # Generate a correlated time series with controlled chaos
                series = self._generate_traffic_series(mean, std, road_type, period_name)
                self._speed_series[(u, v)][period_name] = series

    def _generate_traffic_series(
        self, mean: float, std: float, road_type: str, period: str
    ) -> np.ndarray:
        """Generate a realistic traffic speed time series.

        Uses an AR(1) process with controlled noise injection to produce
        time series that exhibit varying degrees of chaos depending on
        road type and time period.

        Args:
            mean: Mean speed (km/h).
            std: Speed standard deviation.
            road_type: 'arterial', 'residential', or 'diagonal'.
            period: Time period name.

        Returns:
            Speed time series array of length self.ts_length.
        """
        n = self.ts_length

        # AR(1) autoregressive coefficient — higher = more persistent
        # Rush hours have lower persistence (more chaotic transitions)
        if "rush" in period:
            phi = 0.6 + self.rng.uniform(-0.1, 0.1)  # More chaotic
        else:
            phi = 0.85 + self.rng.uniform(-0.05, 0.05)  # More stable

        # Occasional traffic "shocks" (incidents) during rush hours
        shock_prob = 0.05 if "rush" in period else 0.01
        shock_magnitude = 3.0 if "rush" in period else 1.5

        # Generate AR(1) with shocks
        series = np.zeros(n)
        series[0] = mean
        innovation_std = std * np.sqrt(1 - phi**2)  # Stationary variance

        for t in range(1, n):
            innovation = self.rng.normal(0, innovation_std)

            # Random shock (sudden speed drop)
            if self.rng.random() < shock_prob:
                innovation -= shock_magnitude * std

            series[t] = mean + phi * (series[t-1] - mean) + innovation

        # Clip to realistic range
        series = np.clip(series, 2.0, 60.0)

        return series

    def get_speed_series(
        self, edge: Tuple[int, int], time_minutes: int
    ) -> np.ndarray:
        """Get the speed time series for an edge at a given time.

        Args:
            edge: (u, v) edge tuple.
            time_minutes: Time as minutes since midnight.

        Returns:
            Speed time series array.
        """
        period = self._get_time_period(time_minutes)

        if edge in self._speed_series:
            return self._speed_series[edge].get(
                period,
                np.full(self.ts_length, 20.0),  # fallback
            )
        return np.full(self.ts_length, 20.0)

    def _get_time_period(self, time_minutes: int) -> str:
        """Map minutes since midnight to time period name."""
        t = time_minutes % (24 * 60)
        for name, start, end in self.TIME_PERIODS:
            if name == "night":
                if t >= start or t < (end % (24 * 60)):
                    return name
            elif start <= t < end:
                return name
        return "midday"  # fallback


class LyapunovEstimator:
    """Estimates the maximal Lyapunov exponent (MLE) from time series.

    Uses the Rosenstein algorithm: track the divergence of initially
    nearby trajectories in reconstructed phase space.

    For traffic: λ_max > 0 means chaotic (trajectories diverge exponentially)
                 λ_max ≤ 0 means stable (trajectories converge or stay bounded)
    """

    def __init__(self, embedding_dim: int = 3, time_delay: int = 1):
        """
        Args:
            embedding_dim: Dimension for Takens' embedding (m).
            time_delay: Time delay for embedding (τ).
        """
        self.m = embedding_dim
        self.tau = time_delay

    def compute(self, series: np.ndarray) -> float:
        """Compute the maximal Lyapunov exponent for a time series.

        Uses the Rosenstein (1993) algorithm:
        1. Reconstruct attractor via Takens' delay embedding
        2. For each point, find nearest neighbor (not within Theiler window)
        3. Track divergence of nearest-neighbor pairs over time
        4. λ_max ≈ slope of log(divergence) vs time

        Args:
            series: 1D time series (speed measurements).

        Returns:
            Estimated maximal Lyapunov exponent.
            Positive = chaotic, negative = stable, near-zero = marginally stable.
        """
        if len(series) < 20:
            return 0.0

        # Normalize
        series = (series - np.mean(series)) / (np.std(series) + 1e-10)

        # Takens' delay embedding
        N = len(series)
        M = N - (self.m - 1) * self.tau
        if M < 10:
            return 0.0

        # Build embedding matrix
        embed = np.zeros((M, self.m))
        for i in range(self.m):
            embed[:, i] = series[i * self.tau : i * self.tau + M]

        # For each point, find nearest neighbor (Rosenstein method)
        # Use Theiler window to avoid temporal neighbors
        theiler_window = max(self.tau * self.m, 5)
        max_divergence_steps = min(M // 4, 20)

        if max_divergence_steps < 3:
            return 0.0

        divergence = np.zeros(max_divergence_steps)
        count = np.zeros(max_divergence_steps)

        for i in range(M - max_divergence_steps):
            # Find nearest neighbor outside Theiler window
            min_dist = float("inf")
            nn_idx = -1

            for j in range(M - max_divergence_steps):
                if abs(i - j) <= theiler_window:
                    continue
                dist = np.linalg.norm(embed[i] - embed[j])
                if dist < min_dist and dist > 1e-10:
                    min_dist = dist
                    nn_idx = j

            if nn_idx < 0:
                continue

            # Track divergence over future timesteps
            for k in range(max_divergence_steps):
                if i + k < M and nn_idx + k < M:
                    d = np.linalg.norm(embed[i + k] - embed[nn_idx + k])
                    if d > 1e-10:
                        divergence[k] += np.log(d)
                        count[k] += 1

        # Average log-divergence
        valid = count > 0
        if np.sum(valid) < 3:
            return 0.0

        avg_divergence = np.where(valid, divergence / (count + 1e-10), 0)
        time_steps = np.arange(max_divergence_steps)

        # Linear fit: slope = λ_max
        valid_mask = valid & (time_steps > 0)
        if np.sum(valid_mask) < 2:
            return 0.0

        t = time_steps[valid_mask].astype(float)
        d = avg_divergence[valid_mask]

        # Simple least squares
        n_pts = len(t)
        slope = (n_pts * np.sum(t * d) - np.sum(t) * np.sum(d)) / \
                (n_pts * np.sum(t**2) - np.sum(t)**2 + 1e-10)

        return float(slope)


class StabilityWeightedGraph:
    """Produces stability-weighted distance matrices for the VRP.

    Combines static distances with Lyapunov-based traffic stability
    to produce risk-adjusted edge costs for the QUBO formulation.

    This is the key innovation: the quantum optimizer automatically
    avoids chaotic-traffic road segments without any algorithmic changes.
    """

    def __init__(
        self,
        G: nx.DiGraph,
        velocity_field: TrafficVelocityField,
        risk_aversion: float = 0.5,
        seed: int = 42,
    ):
        """
        Args:
            G: Road network graph.
            velocity_field: Traffic velocity field model.
            risk_aversion: α parameter in risk_cost formula. Higher = avoid chaos more.
                0.0 = ignore traffic dynamics (pure distance)
                0.3 = mild traffic awareness
                0.5 = moderate (recommended)
                1.0 = strong chaos avoidance
            seed: Random seed.
        """
        self.G = G
        self.velocity_field = velocity_field
        self.lyapunov = LyapunovEstimator()
        self.alpha = risk_aversion
        self.rng = np.random.RandomState(seed)

        # Cache Lyapunov exponents per (edge, time_period)
        self._lyapunov_cache: Dict[Tuple, Dict[str, float]] = {}

    def compute_edge_stability(
        self, edge: Tuple[int, int], time_minutes: int
    ) -> TrafficState:
        """Compute traffic stability for a single edge at a given time.

        Args:
            edge: (u, v) edge.
            time_minutes: Time as minutes since midnight.

        Returns:
            TrafficState with Lyapunov exponent and risk-adjusted cost.
        """
        # Get speed time series
        series = self.velocity_field.get_speed_series(edge, time_minutes)

        # Compute Lyapunov exponent
        period = self.velocity_field._get_time_period(time_minutes)
        cache_key = (edge, period)

        if cache_key in self._lyapunov_cache:
            lya = self._lyapunov_cache[cache_key]
        else:
            lya = self.lyapunov.compute(series)
            self._lyapunov_cache[cache_key] = lya

        # Speed statistics
        speed_mean = float(np.mean(series))
        speed_std = float(np.std(series))

        # Stability score: exp(-α * max(0, λ_max))
        stability = float(np.exp(-self.alpha * max(0, lya)))

        # Risk-adjusted cost
        edge_data = self.G[edge[0]][edge[1]]
        distance = edge_data.get("distance", 100.0)
        risk_cost = distance / stability

        return TrafficState(
            edge=edge,
            speed_mean=speed_mean,
            speed_std=speed_std,
            lyapunov_exp=lya,
            stability_score=stability,
            risk_cost=risk_cost,
        )

    def compute_all_stabilities(
        self, time_minutes: int = 9 * 60
    ) -> Dict[Tuple[int, int], TrafficState]:
        """Compute traffic stability for ALL edges at a given time.

        Args:
            time_minutes: Time (default 9:00 AM delivery start).

        Returns:
            Dict mapping edge -> TrafficState.
        """
        states = {}
        for u, v in self.G.edges():
            states[(u, v)] = self.compute_edge_stability((u, v), time_minutes)
        return states

    def get_stability_weighted_distance_matrix(
        self,
        depot_node: int,
        stop_nodes: List[int],
        time_minutes: int = 9 * 60,
    ) -> np.ndarray:
        """Compute stability-weighted shortest-path distance matrix.

        This is the primary output: a distance matrix where each edge cost
        includes a Lyapunov stability penalty, so the QUBO solver naturally
        avoids chaotic-traffic road segments.

        Args:
            depot_node: Depot node ID.
            stop_nodes: List of stop node IDs.
            time_minutes: Delivery start time.

        Returns:
            Stability-weighted distance matrix of shape (n+1, n+1).
        """
        # First, update graph weights with stability-adjusted costs
        G_weighted = self.G.copy()
        stabilities = self.compute_all_stabilities(time_minutes)

        for (u, v), state in stabilities.items():
            G_weighted[u][v]["weight"] = state.risk_cost

        # Compute shortest paths on stability-weighted graph
        all_nodes = [depot_node] + stop_nodes
        n = len(all_nodes)
        dist_matrix = np.zeros((n, n))

        for i, src in enumerate(all_nodes):
            try:
                lengths = nx.single_source_dijkstra_path_length(
                    G_weighted, src, weight="weight"
                )
                for j, dst in enumerate(all_nodes):
                    if i != j:
                        dist_matrix[i][j] = lengths.get(dst, float("inf"))
            except nx.NetworkXError:
                for j, dst in enumerate(all_nodes):
                    if i != j:
                        dist_matrix[i][j] = 10000.0

        # Handle infinities
        max_finite = dist_matrix[dist_matrix < float("inf")].max() \
            if np.any(dist_matrix < float("inf")) else 10000
        dist_matrix[dist_matrix == float("inf")] = max_finite * 3

        return dist_matrix

    def get_stability_summary(
        self, time_minutes: int = 9 * 60
    ) -> Dict[str, float]:
        """Get summary statistics of network stability.

        Args:
            time_minutes: Time for analysis.

        Returns:
            Dict with network-wide stability metrics.
        """
        stabilities = self.compute_all_stabilities(time_minutes)

        lya_values = [s.lyapunov_exp for s in stabilities.values()]
        stability_scores = [s.stability_score for s in stabilities.values()]

        return {
            "n_edges": len(lya_values),
            "mean_lyapunov": float(np.mean(lya_values)),
            "max_lyapunov": float(np.max(lya_values)),
            "min_lyapunov": float(np.min(lya_values)),
            "pct_chaotic": float(np.mean(np.array(lya_values) > 0) * 100),
            "mean_stability": float(np.mean(stability_scores)),
            "min_stability": float(np.min(stability_scores)),
            "time_period": self.velocity_field._get_time_period(time_minutes),
        }
