"""
Geospatial utilities for Tokyo VRP.

Provides distance calculations, ward information, and visualization utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .synthetic_generator import Customer, VRPInstance

# Earth radius in km
EARTH_RADIUS_KM = 6371.0

# Tokyo approximate center
TOKYO_CENTER = (35.6762, 139.6503)  # (lat, lon)


@dataclass
class BoundingBox:
    """Geographic bounding box."""

    min_lat: float
    max_lat: float
    min_lon: float
    max_lon: float

    def contains(self, lat: float, lon: float) -> bool:
        """Check if a point is inside the bounding box."""
        return (
            self.min_lat <= lat <= self.max_lat
            and self.min_lon <= lon <= self.max_lon
        )


# Tokyo bounding box
TOKYO_BOUNDS = BoundingBox(
    min_lat=35.5,
    max_lat=35.9,
    min_lon=139.5,
    max_lon=140.0,
)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the great circle distance between two points (in km).

    Args:
        lat1, lon1: First point coordinates (degrees)
        lat2, lon2: Second point coordinates (degrees)

    Returns:
        Distance in kilometers
    """
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    lon1_rad = np.radians(lon1)
    lon2_rad = np.radians(lon2)

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return EARTH_RADIUS_KM * c


def euclidean_distance_approx(
    lon1: float, lat1: float, lon2: float, lat2: float
) -> float:
    """Approximate Euclidean distance in km for Tokyo area.

    Uses local projection approximation. Valid for small distances within Tokyo.

    Args:
        lon1, lat1: First point coordinates (degrees)
        lon2, lat2: Second point coordinates (degrees)

    Returns:
        Approximate distance in kilometers
    """
    # Tokyo latitude scaling
    lat_scale = 111.0  # km per degree latitude
    lon_scale = 111.0 * np.cos(np.radians(35.7))  # km per degree longitude

    dx = (lon2 - lon1) * lon_scale
    dy = (lat2 - lat1) * lat_scale

    return np.sqrt(dx**2 + dy**2)


def compute_distance_matrix_haversine(
    locations: list[tuple[float, float]]
) -> np.ndarray:
    """Compute distance matrix using Haversine formula.

    Args:
        locations: List of (lon, lat) tuples

    Returns:
        Distance matrix in km
    """
    n = len(locations)
    distances = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            lon1, lat1 = locations[i]
            lon2, lat2 = locations[j]
            dist = haversine_distance(lat1, lon1, lat2, lon2)
            distances[i, j] = dist
            distances[j, i] = dist

    return distances


def get_ward_center(ward_name: str) -> tuple[float, float]:
    """Get the approximate center coordinates of a Tokyo ward.

    Args:
        ward_name: Name of the ward (in English)

    Returns:
        (lon, lat) tuple
    """
    from .synthetic_generator import TOKYO_WARDS

    ward_data = TOKYO_WARDS.get(ward_name)
    if ward_data is None:
        raise ValueError(f"Unknown ward: {ward_name}")

    return ward_data["lon"], ward_data["lat"]


def compute_route_distance(
    route: list[int],
    distance_matrix: np.ndarray,
) -> float:
    """Compute total distance of a route.

    Args:
        route: List of customer indices (0-indexed, includes depot at start/end)
        distance_matrix: Distance matrix

    Returns:
        Total route distance
    """
    total = 0.0
    for i in range(len(route) - 1):
        total += distance_matrix[route[i], route[i + 1]]
    return total


def compute_route_time(
    route: list[int],
    time_matrix: np.ndarray,
    service_times: list[float],
    include_service: bool = True,
) -> float:
    """Compute total time of a route.

    Args:
        route: List of customer indices (includes depot)
        time_matrix: Travel time matrix (minutes)
        service_times: Service time at each location
        include_service: Whether to include service times

    Returns:
        Total route time in minutes
    """
    total = 0.0
    for i in range(len(route) - 1):
        total += time_matrix[route[i], route[i + 1]]
        if include_service and i < len(route) - 2:
            total += service_times[route[i + 1]]
    return total


def check_time_window_violation(
    route: list[int],
    time_matrix: np.ndarray,
    service_times: list[float],
    time_windows: list[tuple[float, float]],
    arrival_times: list[float] | None = None,
) -> tuple[float, list[float]]:
    """Check time window violations for a route.

    Args:
        route: List of customer indices (includes depot)
        time_matrix: Travel time matrix
        service_times: Service time at each location
        time_windows: List of (start, end) time windows
        arrival_times: Pre-computed arrival times (optional)

    Returns:
        (total_violation, arrival_times)
    """
    if arrival_times is None:
        arrival_times = [0.0]  # Start at depot at time 0

        current_time = 0.0
        for i in range(len(route) - 1):
            # Travel to next location
            current_time += time_matrix[route[i], route[i + 1]]

            # Wait if early
            tw_start, tw_end = time_windows[route[i + 1]]
            if current_time < tw_start:
                current_time = tw_start

            arrival_times.append(current_time)

            # Add service time (except for final depot)
            if i < len(route) - 2:
                current_time += service_times[route[i + 1]]

    total_violation = 0.0
    for idx, node in enumerate(route[1:-1], start=1):  # Skip depot at start/end
        tw_start, tw_end = time_windows[node]
        if arrival_times[idx] > tw_end:
            total_violation += arrival_times[idx] - tw_end

    return total_violation, arrival_times


def compute_route_statistics(
    instance: VRPInstance,
    routes: list[list[int]],
) -> dict:
    """Compute statistics for a solution.

    Args:
        instance: VRP instance
        routes: List of routes, each route is list of customer indices

    Returns:
        Dictionary with statistics
    """
    all_locations = [instance.depot] + instance.customers

    total_distance = 0.0
    total_time = 0.0
    total_demand_served = 0.0
    total_tw_violation = 0.0
    num_routes_used = 0

    route_details = []

    for vehicle_id, route in enumerate(routes):
        if len(route) <= 2:  # Just depot to depot
            continue

        num_routes_used += 1

        # Route distance
        distance = compute_route_distance(route, instance.distance_matrix)
        total_distance += distance

        # Route time
        service_times = [c.service_time for c in all_locations]
        time = compute_route_time(route, instance.time_matrix, service_times)
        total_time += time

        # Demand served
        demand = sum(all_locations[node].demand for node in route[1:-1])
        total_demand_served += demand

        # Time window violations
        time_windows = [
            (c.time_window_start, c.time_window_end) for c in all_locations
        ]
        tw_violation, _ = check_time_window_violation(
            route, instance.time_matrix, service_times, time_windows
        )
        total_tw_violation += tw_violation

        route_details.append(
            {
                "vehicle_id": vehicle_id,
                "distance": distance,
                "time": time,
                "demand": demand,
                "tw_violation": tw_violation,
                "num_customers": len(route) - 2,  # Exclude depot
            }
        )

    return {
        "total_distance": total_distance,
        "total_time": total_time,
        "total_demand_served": total_demand_served,
        "total_tw_violation": total_tw_violation,
        "num_routes_used": num_routes_used,
        "route_details": route_details,
        "avg_route_distance": total_distance / num_routes_used if num_routes_used > 0 else 0,
        "avg_route_time": total_time / num_routes_used if num_routes_used > 0 else 0,
    }