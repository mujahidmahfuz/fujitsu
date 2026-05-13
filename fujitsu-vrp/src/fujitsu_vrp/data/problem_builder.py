"""
VRP Problem Instance Construction.

Builds VRP instances from various data sources and provides utilities
for problem transformation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

if TYPE_CHECKING:
    from .synthetic_generator import Customer, VRPInstance


@dataclass
class ProblemConfig:
    """Configuration for VRP problem generation."""

    name: str
    num_customers: int = 20
    num_vehicles: int = 5
    vehicle_capacity: float = 100.0
    depot_ward: str = "Shinjuku"
    time_horizon: float = 480.0
    demand_range: tuple[float, float] = (5.0, 25.0)
    service_time_range: tuple[float, float] = (5.0, 15.0)
    time_window_width: tuple[float, float] = (60.0, 180.0)
    seed: int | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProblemConfig:
        """Create config from dictionary."""
        return cls(
            name=data.get("name", "vrp_instance"),
            num_customers=data.get("num_customers", 20),
            num_vehicles=data.get("num_vehicles", 5),
            vehicle_capacity=data.get("vehicle_capacity", 100.0),
            depot_ward=data.get("depot_ward", "Shinjuku"),
            time_horizon=data.get("time_horizon", 480.0),
            demand_range=tuple(data.get("demand_range", (5.0, 25.0))),
            service_time_range=tuple(data.get("service_time_range", (5.0, 15.0))),
            time_window_width=tuple(data.get("time_window_width", (60.0, 180.0))),
            seed=data.get("seed"),
        )

    @classmethod
    def from_yaml(cls, path: Path) -> ProblemConfig:
        """Load config from YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)

    def to_yaml(self, path: Path) -> None:
        """Save config to YAML file."""
        data = {
            "name": self.name,
            "num_customers": self.num_customers,
            "num_vehicles": self.num_vehicles,
            "vehicle_capacity": self.vehicle_capacity,
            "depot_ward": self.depot_ward,
            "time_horizon": self.time_horizon,
            "demand_range": list(self.demand_range),
            "service_time_range": list(self.service_time_range),
            "time_window_width": list(self.time_window_width),
            "seed": self.seed,
        }
        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False)


@dataclass
class Solution:
    """Represents a VRP solution."""

    instance_name: str
    routes: list[list[int]]  # List of routes, each route is list of node indices
    total_distance: float
    total_time: float
    total_demand_served: float
    time_window_violations: float
    capacity_violations: float
    metadata: dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    @property
    def num_routes(self) -> int:
        """Number of routes used."""
        return len([r for r in self.routes if len(r) > 2])

    def get_route_lengths(self) -> list[int]:
        """Get number of customers per route."""
        return [len(r) - 2 for r in self.routes]  # Exclude depot at both ends


def build_vrp_instance(
    config: ProblemConfig,
) -> VRPInstance:
    """Build VRP instance from configuration.

    Args:
        config: Problem configuration

    Returns:
        VRPInstance object
    """
    from .synthetic_generator import TokyoSMEGenerator

    generator = TokyoSMEGenerator(
        seed=config.seed,
        num_customers=config.num_customers,
        num_vehicles=config.num_vehicles,
        vehicle_capacity=config.vehicle_capacity,
        depot_ward=config.depot_ward,
        time_horizon=config.time_horizon,
        demand_range=config.demand_range,
        service_time_range=config.service_time_range,
        time_window_width=config.time_window_width,
    )

    return generator.generate()


def build_vrp_from_customers(
    customers: list[Customer],
    depot: Customer,
    num_vehicles: int,
    vehicle_capacity: float,
    name: str = "custom_instance",
) -> VRPInstance:
    """Build VRP instance from customer and depot data.

    Args:
        customers: List of customers
        depot: Depot location
        num_vehicles: Number of vehicles
        vehicle_capacity: Capacity per vehicle
        name: Instance name

    Returns:
        VRPInstance object
    """
    from .synthetic_generator import TokyoSMEGenerator, Vehicle, VRPInstance

    # Create vehicles
    vehicles = [
        Vehicle(
            id=i,
            capacity=vehicle_capacity,
            start_location=(depot.x, depot.y),
            max_route_time=480.0,
        )
        for i in range(num_vehicles)
    ]

    # Compute distance and time matrices
    all_nodes = [depot] + customers
    locations = [(c.x, c.y) for c in all_nodes]

    generator = TokyoSMEGenerator()
    distance_matrix = generator.compute_distance_matrix(locations)
    time_matrix = generator.compute_time_matrix(distance_matrix)

    return VRPInstance(
        name=name,
        customers=customers,
        vehicles=vehicles,
        depot=depot,
        distance_matrix=distance_matrix,
        time_matrix=time_matrix,
    )


def save_instance(instance: VRPInstance, path: Path) -> None:
    """Save VRP instance to file.

    Args:
        instance: VRP instance to save
        path: Output file path (will be .npz)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        path,
        name=instance.name,
        distance_matrix=instance.distance_matrix,
        time_matrix=instance.time_matrix,
        # Customer data
        customer_ids=np.array([c.id for c in instance.customers]),
        customer_x=np.array([c.x for c in instance.customers]),
        customer_y=np.array([c.y for c in instance.customers]),
        customer_demand=np.array([c.demand for c in instance.customers]),
        customer_service_time=np.array([c.service_time for c in instance.customers]),
        customer_tw_start=np.array([c.time_window_start for c in instance.customers]),
        customer_tw_end=np.array([c.time_window_end for c in instance.customers]),
        customer_priority=np.array([c.priority for c in instance.customers]),
        # Vehicle data
        vehicle_ids=np.array([v.id for v in instance.vehicles]),
        vehicle_capacity=np.array([v.capacity for v in instance.vehicles]),
        # Depot
        depot_id=instance.depot.id,
        depot_x=instance.depot.x,
        depot_y=instance.depot.y,
    )


def load_instance(path: Path) -> VRPInstance:
    """Load VRP instance from file.

    Args:
        path: Input file path (.npz)

    Returns:
        VRPInstance object
    """
    from .synthetic_generator import Customer, Vehicle, VRPInstance

    data = np.load(path)

    # Reconstruct customers
    customers = []
    for i in range(len(data["customer_ids"])):
        customers.append(
            Customer(
                id=int(data["customer_ids"][i]),
                name=f"Customer_{data['customer_ids'][i]}",
                x=float(data["customer_x"][i]),
                y=float(data["customer_y"][i]),
                demand=float(data["customer_demand"][i]),
                service_time=float(data["customer_service_time"][i]),
                time_window_start=float(data["customer_tw_start"][i]),
                time_window_end=float(data["customer_tw_end"][i]),
                priority=int(data["customer_priority"][i]),
            )
        )

    # Reconstruct vehicles
    vehicles = []
    for i in range(len(data["vehicle_ids"])):
        vehicles.append(
            Vehicle(
                id=int(data["vehicle_ids"][i]),
                capacity=float(data["vehicle_capacity"][i]),
                start_location=(float(data["depot_x"]), float(data["depot_y"])),
            )
        )

    # Reconstruct depot
    depot = Customer(
        id=int(data["depot_id"]),
        name="Depot",
        x=float(data["depot_x"]),
        y=float(data["depot_y"]),
        demand=0.0,
        service_time=0.0,
        time_window_start=0.0,
        time_window_end=480.0,
    )

    return VRPInstance(
        name=str(data["name"]),
        customers=customers,
        vehicles=vehicles,
        depot=depot,
        distance_matrix=data["distance_matrix"],
        time_matrix=data["time_matrix"],
    )


def save_solution(solution: Solution, path: Path) -> None:
    """Save solution to file.

    Args:
        solution: Solution to save
        path: Output file path (.npz)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert routes to arrays with padding
    max_route_len = max(len(r) for r in solution.routes)
    routes_array = np.zeros((len(solution.routes), max_route_len), dtype=int)
    for i, route in enumerate(solution.routes):
        routes_array[i, : len(route)] = route

    np.savez(
        path,
        instance_name=solution.instance_name,
        routes=routes_array,
        total_distance=solution.total_distance,
        total_time=solution.total_time,
        total_demand_served=solution.total_demand_served,
        time_window_violations=solution.time_window_violations,
        capacity_violations=solution.capacity_violations,
    )


def load_solution(path: Path) -> Solution:
    """Load solution from file.

    Args:
        path: Input file path (.npz)

    Returns:
        Solution object
    """
    data = np.load(path)

    # Convert routes array back to list of lists
    routes = []
    for route_array in data["routes"]:
        # Remove padding zeros
        route = route_array[route_array >= 0].tolist()
        routes.append(route)

    return Solution(
        instance_name=str(data["instance_name"]),
        routes=routes,
        total_distance=float(data["total_distance"]),
        total_time=float(data["total_time"]),
        total_demand_served=float(data["total_demand_served"]),
        time_window_violations=float(data["time_window_violations"]),
        capacity_violations=float(data["capacity_violations"]),
    )