"""
Traffic Simulation Module — Dynamic Events for Re-routing Demos.

Generates realistic traffic disruption scenarios on the Tokyo road network:
- Rush hour congestion (time-varying speed multipliers)
- Random incidents (edge weight → very high cost)
- Weather effects (global speed reduction)
- Construction zones (edge temporarily blocked)

Used to test the quantum solver's ability to re-route in real time.
"""

import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class EventType(Enum):
    """Types of traffic disruption events."""
    ACCIDENT = "accident"
    CONSTRUCTION = "construction"
    WEATHER = "weather"
    RUSH_HOUR = "rush_hour"
    SPECIAL_EVENT = "special_event"  # Festival, parade, etc.


@dataclass
class TrafficEvent:
    """A single traffic disruption event."""
    event_type: EventType
    affected_edges: List[Tuple[int, int]]
    severity: float  # Multiplier: 1.0 = no change, ∞ = blocked
    start_time: int  # Minutes since midnight
    duration: int    # Minutes
    description: str

    @property
    def end_time(self) -> int:
        return self.start_time + self.duration


class TrafficSimulator:
    """Simulates dynamic traffic conditions on the road network.

    Generates a sequence of events that modify edge weights, producing
    "disrupted" distance matrices that trigger re-routing in the
    quantum solver.
    """

    # Rush hour congestion profiles
    RUSH_HOUR_PROFILES = {
        "morning": {
            "start": 7 * 60,      # 7:00 AM
            "peak": 8 * 60,       # 8:00 AM
            "end": 9 * 60 + 30,   # 9:30 AM
            "arterial_factor": 2.5,
            "residential_factor": 1.3,
        },
        "evening": {
            "start": 17 * 60,     # 5:00 PM
            "peak": 18 * 60,      # 6:00 PM
            "end": 19 * 60 + 30,  # 7:30 PM
            "arterial_factor": 3.0,
            "residential_factor": 1.5,
        },
    }

    def __init__(self, G: nx.DiGraph, seed: int = 42):
        """
        Args:
            G: Road network graph with edge attributes.
            seed: Random seed.
        """
        self.G = G
        self.rng = np.random.RandomState(seed)
        self.events: List[TrafficEvent] = []
        self._base_distances = {}

        # Cache base distances
        for u, v, data in G.edges(data=True):
            self._base_distances[(u, v)] = data.get("distance", 100.0)

    def generate_scenario(
        self,
        n_incidents: int = 2,
        weather_severity: float = 0.0,
        include_rush_hour: bool = True,
        time_minutes: int = 9 * 60,
    ) -> List[TrafficEvent]:
        """Generate a realistic traffic scenario.

        Args:
            n_incidents: Number of random incidents (accidents/construction).
            weather_severity: 0.0 = clear, 0.5 = rain, 1.0 = heavy storm.
            include_rush_hour: Include rush hour congestion.
            time_minutes: Current time for scenario.

        Returns:
            List of TrafficEvent objects.
        """
        self.events = []

        # Rush hour
        if include_rush_hour:
            self._add_rush_hour_events(time_minutes)

        # Random incidents
        for _ in range(n_incidents):
            self._add_random_incident(time_minutes)

        # Weather
        if weather_severity > 0:
            self._add_weather_event(weather_severity, time_minutes)

        return self.events

    def _add_rush_hour_events(self, current_time: int):
        """Add rush hour congestion events if within rush hour window."""
        for name, profile in self.RUSH_HOUR_PROFILES.items():
            if profile["start"] <= current_time <= profile["end"]:
                # Compute intensity based on proximity to peak
                peak_dist = abs(current_time - profile["peak"])
                max_dist = max(profile["peak"] - profile["start"],
                             profile["end"] - profile["peak"])
                intensity = 1.0 - (peak_dist / max_dist)

                # Arterial congestion
                arterial_edges = [
                    (u, v) for u, v, d in self.G.edges(data=True)
                    if d.get("road_type") == "arterial"
                ]
                if arterial_edges:
                    factor = 1.0 + (profile["arterial_factor"] - 1.0) * intensity
                    self.events.append(TrafficEvent(
                        event_type=EventType.RUSH_HOUR,
                        affected_edges=arterial_edges,
                        severity=factor,
                        start_time=profile["start"],
                        duration=profile["end"] - profile["start"],
                        description=f"{name.title()} rush: {factor:.1f}x arterial congestion",
                    ))

    def _add_random_incident(self, current_time: int):
        """Add a random accident or construction event."""
        edges = list(self.G.edges())
        if not edges:
            return

        # Pick random edge(s) for incident
        incident_edge = edges[self.rng.randint(len(edges))]
        u, v = incident_edge

        # Also affect adjacent edges (traffic backs up)
        affected = [(u, v)]
        for neighbor in self.G.predecessors(u):
            if (neighbor, u) in self._base_distances:
                affected.append((neighbor, u))
        for neighbor in self.G.successors(v):
            if (v, neighbor) in self._base_distances:
                affected.append((v, neighbor))

        # Random severity and duration
        is_accident = self.rng.random() < 0.7
        if is_accident:
            severity = self.rng.uniform(5.0, 50.0)  # Major slowdown
            duration = self.rng.randint(15, 90)  # 15-90 minutes
            desc = f"Accident at edge ({u},{v}): {severity:.0f}x traffic delay"
            event_type = EventType.ACCIDENT
        else:
            severity = self.rng.uniform(2.0, 10.0)
            duration = self.rng.randint(60, 480)
            desc = f"Construction at edge ({u},{v}): {severity:.0f}x delay"
            event_type = EventType.CONSTRUCTION

        self.events.append(TrafficEvent(
            event_type=event_type,
            affected_edges=affected,
            severity=severity,
            start_time=current_time,
            duration=duration,
            description=desc,
        ))

    def _add_weather_event(self, severity: float, current_time: int):
        """Add weather-based global slowdown."""
        all_edges = list(self.G.edges())
        # Weather affects all roads
        factor = 1.0 + severity * 1.5  # 1.0 (clear) to 2.5 (storm)

        weather_type = "Clear"
        if severity < 0.3:
            weather_type = "Light rain"
        elif severity < 0.7:
            weather_type = "Rain"
        else:
            weather_type = "Heavy storm"

        self.events.append(TrafficEvent(
            event_type=EventType.WEATHER,
            affected_edges=all_edges,
            severity=factor,
            start_time=current_time,
            duration=120,
            description=f"{weather_type}: {factor:.1f}x global slowdown",
        ))

    def get_disrupted_distances(
        self, current_time: int
    ) -> Dict[Tuple[int, int], float]:
        """Get disrupted edge distances at a given time.

        Args:
            current_time: Minutes since midnight.

        Returns:
            Dict mapping edge -> disrupted distance.
        """
        disrupted = dict(self._base_distances)

        for event in self.events:
            if event.start_time <= current_time < event.end_time:
                for edge in event.affected_edges:
                    if edge in disrupted:
                        disrupted[edge] *= event.severity

        return disrupted

    def get_disrupted_distance_matrix(
        self,
        depot_node: int,
        stop_nodes: List[int],
        current_time: int,
    ) -> np.ndarray:
        """Compute disrupted shortest-path distance matrix.

        Args:
            depot_node: Depot node ID.
            stop_nodes: Stop node IDs.
            current_time: Current time in minutes.

        Returns:
            Distance matrix with disruption effects.
        """
        # Create a copy of the graph with disrupted weights
        G_disrupted = self.G.copy()
        disrupted = self.get_disrupted_distances(current_time)

        for (u, v), dist in disrupted.items():
            if G_disrupted.has_edge(u, v):
                G_disrupted[u][v]["distance"] = dist

        # Compute shortest paths
        all_nodes = [depot_node] + stop_nodes
        n = len(all_nodes)
        dist_matrix = np.zeros((n, n))

        for i, src in enumerate(all_nodes):
            try:
                lengths = nx.single_source_dijkstra_path_length(
                    G_disrupted, src, weight="distance"
                )
                for j, dst in enumerate(all_nodes):
                    if i != j:
                        dist_matrix[i][j] = lengths.get(dst, float("inf"))
            except nx.NetworkXError:
                for j, dst in enumerate(all_nodes):
                    if i != j:
                        dist_matrix[i][j] = 50000.0

        max_finite = dist_matrix[dist_matrix < float("inf")].max() \
            if np.any(dist_matrix < float("inf")) else 10000
        dist_matrix[dist_matrix == float("inf")] = max_finite * 3

        return dist_matrix

    def get_active_events(self, current_time: int) -> List[TrafficEvent]:
        """Get events currently active at the given time.

        Args:
            current_time: Minutes since midnight.

        Returns:
            List of active TrafficEvent objects.
        """
        return [
            e for e in self.events
            if e.start_time <= current_time < e.end_time
        ]

    def summary(self, current_time: int) -> str:
        """Human-readable summary of current traffic conditions."""
        active = self.get_active_events(current_time)
        if not active:
            return "Traffic conditions: Normal"

        lines = [f"🚦 Traffic Conditions at {current_time//60:02d}:{current_time%60:02d}:"]
        for e in active:
            icon = {"accident": "🚗💥", "construction": "🚧",
                    "weather": "🌧️", "rush_hour": "🕐",
                    "special_event": "🎪"}.get(e.event_type.value, "⚠️")
            lines.append(f"  {icon} {e.description}")
        return "\n".join(lines)
