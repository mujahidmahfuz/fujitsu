"""
Tokyo SME Synthetic Data Generator

Generates realistic delivery scenarios for Tokyo small and medium enterprises.
Includes geographic distribution based on Tokyo ward data, time windows,
and capacity constraints.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Customer:
    """Represents a customer/delivery point."""

    id: int
    name: str
    x: float  # Longitude (Tokyo: ~139.5-139.95)
    y: float  # Latitude (Tokyo: ~35.5-35.85)
    demand: float  # Package weight/volume (kg or arbitrary units)
    service_time: float  # Time needed for delivery (minutes)
    time_window_start: float  # Earliest delivery time (minutes from start)
    time_window_end: float  # Latest delivery time (minutes from start)
    priority: int = 0  # Priority level (0=normal, 1=high, 2=urgent)
    ward: str = ""  # Tokyo ward name

    def __repr__(self) -> str:
        return f"Customer({self.id}, {self.name}, demand={self.demand})"


@dataclass
class Vehicle:
    """Represents a delivery vehicle."""

    id: int
    capacity: float  # Maximum load capacity
    start_location: tuple[float, float]  # (x, y) depot location
    end_location: tuple[float, float] | None = None  # If different from start
    max_route_time: float = 480.0  # Maximum route duration (minutes)
    speed: float = 40.0  # Average speed (km/h)

    def __post_init__(self) -> None:
        if self.end_location is None:
            self.end_location = self.start_location


@dataclass
class VRPInstance:
    """Complete VRP problem instance."""

    name: str
    customers: list[Customer]
    vehicles: list[Vehicle]
    depot: Customer  # Depot as a special customer
    distance_matrix: np.ndarray
    time_matrix: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_customers(self) -> int:
        return len(self.customers)

    @property
    def num_vehicles(self) -> int:
        return len(self.vehicles)

    def get_distance(self, i: int, j: int) -> float:
        """Get distance between two customers (indices)."""
        return self.distance_matrix[i, j]

    def get_time(self, i: int, j: int) -> float:
        """Get travel time between two customers (indices)."""
        return self.time_matrix[i, j]


# Tokyo ward data (simplified, with approximate centroids)
TOKYO_WARDS: dict[str, dict[str, Any]] = {
    "Chiyoda": {"lat": 35.694, "lon": 139.754, "population": 54000, "commercial": 0.8},
    "Chuo": {"lat": 35.671, "lon": 139.771, "population": 140000, "commercial": 0.9},
    "Minato": {"lat": 35.658, "lon": 139.737, "population": 250000, "commercial": 0.85},
    "Shinjuku": {"lat": 35.694, "lon": 139.704, "population": 340000, "commercial": 0.9},
    "Bunkyo": {"lat": 35.710, "lon": 139.751, "population": 200000, "commercial": 0.6},
    "Taito": {"lat": 35.713, "lon": 139.782, "population": 190000, "commercial": 0.7},
    "Sumida": {"lat": 35.697, "lon": 139.804, "population": 270000, "commercial": 0.5},
    "Koto": {"lat": 35.673, "lon": 139.815, "population": 510000, "commercial": 0.4},
    "Shinagawa": {"lat": 35.627, "lon": 139.739, "population": 400000, "commercial": 0.7},
    "Meguro": {"lat": 35.640, "lon": 139.710, "population": 280000, "commercial": 0.5},
    "Ota": {"lat": 35.562, "lon": 139.723, "population": 720000, "commercial": 0.3},
    "Setagaya": {"lat": 35.647, "lon": 139.651, "population": 900000, "commercial": 0.4},
    "Shibuya": {"lat": 35.662, "lon": 139.703, "population": 230000, "commercial": 0.95},
    "Nakano": {"lat": 35.708, "lon": 139.663, "population": 330000, "commercial": 0.5},
    "Suginami": {"lat": 35.695, "lon": 139.635, "population": 580000, "commercial": 0.4},
    "Toshima": {"lat": 35.730, "lon": 139.719, "population": 300000, "commercial": 0.7},
    "Kita": {"lat": 35.754, "lon": 139.742, "population": 350000, "commercial": 0.3},
    "Arakawa": {"lat": 35.737, "lon": 139.784, "population": 200000, "commercial": 0.4},
    "Itabashi": {"lat": 35.751, "lon": 139.708, "population": 560000, "commercial": 0.3},
    "Nerima": {"lat": 35.737, "lon": 139.653, "population": 720000, "commercial": 0.2},
    "Adachi": {"lat": 35.776, "lon": 139.804, "population": 690000, "commercial": 0.2},
    "Katsushika": {"lat": 35.745, "lon": 139.846, "population": 440000, "commercial": 0.2},
    "Edogawa": {"lat": 35.703, "lon": 139.871, "population": 690000, "commercial": 0.2},
}


class TokyoSMEGenerator:
    """Generates synthetic Tokyo SME delivery scenarios."""

    def __init__(
        self,
        seed: int | None = None,
        num_customers: int = 20,
        num_vehicles: int = 5,
        vehicle_capacity: float = 100.0,
        depot_ward: str = "Shinjuku",
        time_horizon: float = 480.0,  # 8 hours in minutes
        demand_range: tuple[float, float] = (5.0, 25.0),
        service_time_range: tuple[float, float] = (5.0, 15.0),
        time_window_width: tuple[float, float] = (60.0, 180.0),
        commercial_probability: float = 0.7,
    ) -> None:
        """Initialize the generator.

        Args:
            seed: Random seed for reproducibility
            num_customers: Number of customers to generate
            num_vehicles: Number of vehicles available
            vehicle_capacity: Capacity per vehicle
            depot_ward: Ward where depot is located
            time_horizon: Total time available (minutes)
            demand_range: Min/max demand per customer
            service_time_range: Min/max service time per customer
            time_window_width: Min/max time window width
            commercial_probability: Probability of commercial vs residential
        """
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.num_customers = num_customers
        self.num_vehicles = num_vehicles
        self.vehicle_capacity = vehicle_capacity
        self.depot_ward = depot_ward
        self.time_horizon = time_horizon
        self.demand_range = demand_range
        self.service_time_range = service_time_range
        self.time_window_width = time_window_width
        self.commercial_probability = commercial_probability

    def _select_ward(self) -> str:
        """Select a random ward weighted by population."""
        wards = list(TOKYO_WARDS.keys())
        populations = [TOKYO_WARDS[w]["population"] for w in wards]
        total = sum(populations)
        weights = [p / total for p in populations]
        return self.rng.choices(wards, weights=weights)[0]

    def _generate_location(self, ward: str, spread: float = 0.02) -> tuple[float, float]:
        """Generate a location within or near a ward.

        Args:
            ward: Name of Tokyo ward
            spread: Standard deviation of location spread (degrees)

        Returns:
            (longitude, latitude) tuple
        """
        ward_data = TOKYO_WARDS.get(ward, TOKYO_WARDS["Shinjuku"])
        base_lon = ward_data["lon"]
        base_lat = ward_data["lat"]

        lon = self.np_rng.normal(base_lon, spread)
        lat = self.np_rng.normal(base_lat, spread)

        # Clamp to Tokyo bounds
        lon = np.clip(lon, 139.5, 140.0)
        lat = np.clip(lat, 35.5, 35.9)

        return lon, lat

    def _generate_time_window(
        self,
        customer_idx: int,
        total_customers: int,
    ) -> tuple[float, float]:
        """Generate time window for a customer.

        Distributes time windows across the day to create realistic scenarios.
        """
        window_width = self.rng.uniform(*self.time_window_width)

        # Create some structure - early customers tend to have earlier windows
        base_time = (customer_idx / total_customers) * self.time_horizon * 0.6

        # Add randomness
        center = base_time + self.rng.uniform(0, self.time_horizon * 0.4)
        start = max(0, center - window_width / 2)
        end = min(self.time_horizon, center + window_width / 2)

        return start, end

    def generate_customer(self, idx: int, total_customers: int) -> Customer:
        """Generate a single customer."""
        ward = self._select_ward()
        lon, lat = self._generate_location(ward)

        demand = self.rng.uniform(*self.demand_range)
        service_time = self.rng.uniform(*self.service_time_range)
        tw_start, tw_end = self._generate_time_window(idx, total_customers)

        # Priority based on commercial areas
        ward_data = TOKYO_WARDS.get(ward, TOKYO_WARDS["Shinjuku"])
        is_commercial = self.rng.random() < (
            ward_data["commercial"] * self.commercial_probability
        )
        priority = 1 if is_commercial else 0

        return Customer(
            id=idx + 1,  # 0 is reserved for depot
            name=f"Customer_{idx + 1}",
            x=lon,
            y=lat,
            demand=demand,
            service_time=service_time,
            time_window_start=tw_start,
            time_window_end=tw_end,
            priority=priority,
            ward=ward,
        )

    def generate_depot(self) -> Customer:
        """Generate the depot location."""
        ward_data = TOKYO_WARDS[self.depot_ward]
        return Customer(
            id=0,
            name="Depot",
            x=ward_data["lon"],
            y=ward_data["lat"],
            demand=0.0,
            service_time=0.0,
            time_window_start=0.0,
            time_window_end=self.time_horizon,
            priority=0,
            ward=self.depot_ward,
        )

    def generate_vehicles(self, depot: Customer) -> list[Vehicle]:
        """Generate vehicle fleet."""
        vehicles = []
        depot_loc = (depot.x, depot.y)

        for i in range(self.num_vehicles):
            vehicles.append(
                Vehicle(
                    id=i,
                    capacity=self.vehicle_capacity,
                    start_location=depot_loc,
                    max_route_time=self.time_horizon,
                )
            )
        return vehicles

    def compute_distance_matrix(
        self, locations: list[tuple[float, float]]
    ) -> np.ndarray:
        """Compute Euclidean distance matrix (in km, approximate).

        Tokyo is at ~35.7° latitude, so:
        - 1° latitude ≈ 111 km
        - 1° longitude ≈ 111 * cos(35.7°) ≈ 90 km
        """
        n = len(locations)
        distances = np.zeros((n, n))

        lat_to_km = 111.0
        lon_to_km = 111.0 * np.cos(np.radians(35.7))

        for i in range(n):
            for j in range(i + 1, n):
                dx = (locations[j][0] - locations[i][0]) * lon_to_km
                dy = (locations[j][1] - locations[i][1]) * lat_to_km
                dist = np.sqrt(dx**2 + dy**2)
                distances[i, j] = dist
                distances[j, i] = dist

        return distances

    def compute_time_matrix(
        self,
        distance_matrix: np.ndarray,
        speed: float = 40.0,
    ) -> np.ndarray:
        """Compute travel time matrix (in minutes).

        Args:
            distance_matrix: Distance matrix in km
            speed: Average speed in km/h
        """
        return distance_matrix / speed * 60.0

    def generate(self) -> VRPInstance:
        """Generate complete VRP instance."""
        depot = self.generate_depot()
        customers = [
            self.generate_customer(i, self.num_customers)
            for i in range(self.num_customers)
        ]
        vehicles = self.generate_vehicles(depot)

        # All locations: depot + customers
        all_nodes = [depot] + customers
        locations = [(c.x, c.y) for c in all_nodes]

        distance_matrix = self.compute_distance_matrix(locations)
        time_matrix = self.compute_time_matrix(distance_matrix)

        # Compute total demand
        total_demand = sum(c.demand for c in customers)
        total_capacity = sum(v.capacity for v in vehicles)

        metadata = {
            "total_demand": total_demand,
            "total_capacity": total_capacity,
            "capacity_utilization": total_demand / total_capacity if total_capacity > 0 else 0,
            "avg_time_window_width": np.mean(
                [c.time_window_end - c.time_window_start for c in customers]
            ),
            "depot_ward": self.depot_ward,
            "seed": self.seed,
            "generator_params": {
                "num_customers": self.num_customers,
                "num_vehicles": self.num_vehicles,
                "vehicle_capacity": self.vehicle_capacity,
                "time_horizon": self.time_horizon,
            },
        }

        return VRPInstance(
            name=f"tokyo_sme_n{self.num_customers}_v{self.num_vehicles}",
            customers=customers,
            vehicles=vehicles,
            depot=depot,
            distance_matrix=distance_matrix,
            time_matrix=time_matrix,
            metadata=metadata,
        )


def generate_benchmark_instances(
    seeds: list[int] | None = None,
    sizes: list[tuple[int, int]] | None = None,
) -> list[VRPInstance]:
    """Generate a set of benchmark instances.

    Args:
        seeds: Random seeds for reproducibility
        sizes: List of (num_customers, num_vehicles) tuples

    Returns:
        List of VRPInstance objects
    """
    if seeds is None:
        seeds = [42, 123, 456]
    if sizes is None:
        sizes = [(10, 3), (20, 5), (30, 5), (50, 8)]

    instances = []
    for seed in seeds:
        for num_customers, num_vehicles in sizes:
            generator = TokyoSMEGenerator(
                seed=seed,
                num_customers=num_customers,
                num_vehicles=num_vehicles,
            )
            instances.append(generator.generate())

    return instances