"""
Tokyo Delivery Dataset Generator — Shibuya Ward.

Generates realistic synthetic delivery datasets for the VRP using
a model of Shibuya Ward's road network. Produces stop locations,
demand profiles, time windows, and shortest-path distance matrices.

Since OSMnx requires network access (which may not be available in all
deployment environments), this generator builds a synthetic but
geographically accurate road network based on Shibuya's actual layout.
"""

import numpy as np
import networkx as nx
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict
import time

from src.qubo.vrp_qubo import VRPInstance


# --- Shibuya Ward geographic reference ---
# Bounding box: roughly lat 35.654-35.668, lon 139.693-139.712
SHIBUYA_CENTER = (35.6595, 139.7005)
SHIBUYA_BOUNDS = {
    "lat_min": 35.654, "lat_max": 35.668,
    "lon_min": 139.693, "lon_max": 139.712,
}

# Approximate lat/lon to meters conversion at Tokyo latitude
LAT_TO_M = 111_320.0  # 1 degree latitude ≈ 111.32 km
LON_TO_M = 91_290.0   # 1 degree longitude ≈ 91.29 km at lat 35.66°

# Realistic Tokyo delivery parameters
DELIVERY_PROFILES = {
    "convenience_store": {
        "demand": (1, 3),  # parcels
        "time_window": ("09:00", "21:00"),
        "freq": 0.30,
    },
    "office": {
        "demand": (2, 5),
        "time_window": ("09:00", "17:00"),
        "freq": 0.25,
    },
    "residential": {
        "demand": (1, 2),
        "time_window": ("14:00", "20:00"),
        "freq": 0.25,
    },
    "restaurant": {
        "demand": (3, 8),
        "time_window": ("06:00", "11:00"),
        "freq": 0.15,
    },
    "retail_shop": {
        "demand": (1, 4),
        "time_window": ("10:00", "19:00"),
        "freq": 0.05,
    },
}


def _time_to_minutes(t: str) -> int:
    """Convert 'HH:MM' to minutes since midnight."""
    h, m = t.split(":")
    return int(h) * 60 + int(m)


@dataclass
class DeliveryStop:
    """A single delivery stop with full metadata."""
    id: int
    name: str
    stop_type: str
    lat: float
    lon: float
    demand: int
    time_window_start: int  # minutes since midnight
    time_window_end: int
    address: str


@dataclass
class TokyoDataset:
    """Complete delivery dataset for quantum VRP."""
    name: str
    depot: Dict
    stops: List[Dict]
    distance_matrix: List[List[float]]
    n_stops: int
    total_demand: int
    vehicle_capacity: int
    metadata: Dict


class ShibuyaNetworkGenerator:
    """Generates a synthetic Shibuya Ward road network.

    Creates a grid-based network that models Shibuya's actual layout:
    - Major arterials (Meiji-dori, Inokashira-dori, etc.) as fast, wide roads
    - Residential back-streets as slower, narrower roads
    - One-way streets on selected segments
    - Realistic block sizes (80-120m between intersections)
    """

    def __init__(
        self,
        grid_rows: int = 10,
        grid_cols: int = 12,
        seed: int = 42
    ):
        """
        Args:
            grid_rows: Number of N-S streets.
            grid_cols: Number of E-W streets.
            seed: Random seed for reproducibility.
        """
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.rng = np.random.RandomState(seed)
        self.G: Optional[nx.DiGraph] = None

    def generate(self) -> nx.DiGraph:
        """Build the Shibuya road network.

        Returns:
            NetworkX directed graph with node positions (lat, lon) and
            edge weights (distance in meters).
        """
        G = nx.DiGraph()

        # Generate node positions within Shibuya bounds
        lats = np.linspace(
            SHIBUYA_BOUNDS["lat_min"], SHIBUYA_BOUNDS["lat_max"], self.grid_rows
        )
        lons = np.linspace(
            SHIBUYA_BOUNDS["lon_min"], SHIBUYA_BOUNDS["lon_max"], self.grid_cols
        )

        # Add nodes with lat/lon positions + small random jitter
        node_id = 0
        pos_map = {}
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                lat = lats[r] + self.rng.normal(0, 0.0002)
                lon = lons[c] + self.rng.normal(0, 0.0002)
                G.add_node(node_id, lat=lat, lon=lon)
                pos_map[(r, c)] = node_id
                node_id += 1

        # Add edges (roads) — grid topology with some diagonals
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                node = pos_map[(r, c)]

                # Classify roads
                is_arterial = (r % 3 == 0) or (c % 4 == 0)

                # Horizontal neighbor (east)
                if c + 1 < self.grid_cols:
                    neighbor = pos_map[(r, c + 1)]
                    dist = self._haversine_distance(
                        G.nodes[node]["lat"], G.nodes[node]["lon"],
                        G.nodes[neighbor]["lat"], G.nodes[neighbor]["lon"],
                    )
                    speed = 40 if is_arterial else 20  # km/h
                    travel_time = dist / (speed * 1000 / 3600)  # seconds

                    # Bidirectional for arterials, sometimes one-way for residential
                    G.add_edge(node, neighbor, distance=dist, travel_time=travel_time,
                              road_type="arterial" if is_arterial else "residential",
                              speed_limit=speed)
                    if is_arterial or self.rng.random() > 0.15:
                        G.add_edge(neighbor, node, distance=dist, travel_time=travel_time,
                                  road_type="arterial" if is_arterial else "residential",
                                  speed_limit=speed)

                # Vertical neighbor (south)
                if r + 1 < self.grid_rows:
                    neighbor = pos_map[(r + 1, c)]
                    dist = self._haversine_distance(
                        G.nodes[node]["lat"], G.nodes[node]["lon"],
                        G.nodes[neighbor]["lat"], G.nodes[neighbor]["lon"],
                    )
                    speed = 40 if is_arterial else 20
                    travel_time = dist / (speed * 1000 / 3600)

                    G.add_edge(node, neighbor, distance=dist, travel_time=travel_time,
                              road_type="arterial" if is_arterial else "residential",
                              speed_limit=speed)
                    if is_arterial or self.rng.random() > 0.15:
                        G.add_edge(neighbor, node, distance=dist, travel_time=travel_time,
                                  road_type="arterial" if is_arterial else "residential",
                                  speed_limit=speed)

                # Diagonal (occasional — models angled streets like Meiji-dori)
                if r + 1 < self.grid_rows and c + 1 < self.grid_cols and self.rng.random() < 0.08:
                    neighbor = pos_map[(r + 1, c + 1)]
                    dist = self._haversine_distance(
                        G.nodes[node]["lat"], G.nodes[node]["lon"],
                        G.nodes[neighbor]["lat"], G.nodes[neighbor]["lon"],
                    )
                    speed = 30
                    travel_time = dist / (speed * 1000 / 3600)
                    G.add_edge(node, neighbor, distance=dist, travel_time=travel_time,
                              road_type="diagonal", speed_limit=speed)
                    G.add_edge(neighbor, node, distance=dist, travel_time=travel_time,
                              road_type="diagonal", speed_limit=speed)

        self.G = G
        return G

    def _haversine_distance(self, lat1, lon1, lat2, lon2) -> float:
        """Compute haversine distance in meters between two lat/lon points."""
        dlat = (lat2 - lat1) * LAT_TO_M
        dlon = (lon2 - lon1) * LON_TO_M
        return np.sqrt(dlat**2 + dlon**2)


class TokyoDatasetGenerator:
    """Generate delivery datasets on the Shibuya road network.

    Produces VRPInstance objects ready for QUBO formulation, with
    realistic stop locations, demands, and time windows.
    """

    def __init__(self, seed: int = 42, vehicle_capacity: int = 15):
        """
        Args:
            seed: Random seed for reproducibility.
            vehicle_capacity: Maximum parcels per vehicle.
        """
        self.seed = seed
        self.rng = np.random.RandomState(seed)
        self.vehicle_capacity = vehicle_capacity

        # Generate the road network
        self.network_gen = ShibuyaNetworkGenerator(seed=seed)
        self.G = self.network_gen.generate()
        self.nodes = list(self.G.nodes())

    def generate_stops(self, n_stops: int) -> List[DeliveryStop]:
        """Generate realistic delivery stops on the Shibuya network.

        Args:
            n_stops: Number of customer stops (excluding depot).

        Returns:
            List of DeliveryStop objects.
        """
        # Select random nodes as stop locations (no duplicates)
        available = list(self.nodes)
        self.rng.shuffle(available)
        stop_nodes = available[:n_stops]

        stops = []
        profile_names = list(DELIVERY_PROFILES.keys())
        profile_probs = [DELIVERY_PROFILES[p]["freq"] for p in profile_names]

        for i, node_id in enumerate(stop_nodes):
            # Random stop type weighted by frequency
            stop_type = self.rng.choice(profile_names, p=profile_probs)
            profile = DELIVERY_PROFILES[stop_type]

            # Demand
            d_min, d_max = profile["demand"]
            demand = self.rng.randint(d_min, d_max + 1)

            # Time window
            tw_start = _time_to_minutes(profile["time_window"][0])
            tw_end = _time_to_minutes(profile["time_window"][1])

            # Node position
            node_data = self.G.nodes[node_id]
            lat = node_data["lat"]
            lon = node_data["lon"]

            # Generate Tokyo-style address
            chome = self.rng.randint(1, 6)
            block = self.rng.randint(1, 30)
            address = f"渋谷区 {chome}丁目{block}番"

            name = f"Stop_{i+1}_{stop_type}"

            stops.append(DeliveryStop(
                id=i + 1,
                name=name,
                stop_type=stop_type,
                lat=lat,
                lon=lon,
                demand=demand,
                time_window_start=tw_start,
                time_window_end=tw_end,
                address=address,
            ))

        return stops

    def compute_distance_matrix(
        self, depot_node: int, stop_nodes: List[int]
    ) -> np.ndarray:
        """Compute shortest-path distance matrix on the road graph.

        Args:
            depot_node: Node ID of the depot.
            stop_nodes: Node IDs of delivery stops.

        Returns:
            Distance matrix of shape (n_stops+1, n_stops+1).
            Row/col 0 = depot, rows/cols 1..n = stops.
        """
        all_nodes = [depot_node] + stop_nodes
        n = len(all_nodes)
        dist_matrix = np.zeros((n, n))

        # Compute shortest paths between all pairs
        for i, src in enumerate(all_nodes):
            try:
                lengths = nx.single_source_dijkstra_path_length(
                    self.G, src, weight="distance"
                )
                for j, dst in enumerate(all_nodes):
                    if i != j:
                        dist_matrix[i][j] = lengths.get(dst, float("inf"))
            except nx.NetworkXError:
                # Node not reachable — use euclidean fallback
                for j, dst in enumerate(all_nodes):
                    if i != j:
                        d = self.network_gen._haversine_distance(
                            self.G.nodes[src]["lat"], self.G.nodes[src]["lon"],
                            self.G.nodes[dst]["lat"], self.G.nodes[dst]["lon"],
                        )
                        dist_matrix[i][j] = d * 1.4  # Manhattan factor

        # Handle any infinite distances with large fallback
        max_finite = dist_matrix[dist_matrix < float("inf")].max() if np.any(dist_matrix < float("inf")) else 10000
        dist_matrix[dist_matrix == float("inf")] = max_finite * 3

        return dist_matrix

    def generate_dataset(
        self,
        n_stops: int,
        name: Optional[str] = None,
        depot_lat: float = 35.6595,
        depot_lon: float = 139.6992,
    ) -> TokyoDataset:
        """Generate a complete delivery dataset.

        Args:
            n_stops: Number of delivery stops.
            name: Dataset name.
            depot_lat: Depot latitude (default: Shibuya Station area).
            depot_lon: Depot longitude.

        Returns:
            TokyoDataset with all data for VRP solving.
        """
        if name is None:
            name = f"shibuya_{n_stops}stops"

        # Find nearest node to depot location
        depot_node = self._nearest_node(depot_lat, depot_lon)

        # Generate stops
        stops = self.generate_stops(n_stops)

        # Map stops to nearest network nodes
        stop_nodes = []
        for stop in stops:
            node = self._nearest_node(stop.lat, stop.lon)
            stop_nodes.append(node)
            # Update stop lat/lon to match network node
            stop.lat = self.G.nodes[node]["lat"]
            stop.lon = self.G.nodes[node]["lon"]

        # Compute distance matrix
        dist_matrix = self.compute_distance_matrix(depot_node, stop_nodes)

        # Build depot info
        depot = {
            "lat": self.G.nodes[depot_node]["lat"],
            "lon": self.G.nodes[depot_node]["lon"],
            "node_id": int(depot_node),
            "address": "渋谷区 配送センター",  # Shibuya Distribution Center
        }

        # Build stops dicts
        stops_dicts = []
        demands = []
        time_windows = []
        for stop in stops:
            stops_dicts.append({
                "id": stop.id,
                "name": stop.name,
                "type": stop.stop_type,
                "lat": float(stop.lat),
                "lon": float(stop.lon),
                "demand": stop.demand,
                "time_window": [stop.time_window_start, stop.time_window_end],
                "address": stop.address,
            })
            demands.append(stop.demand)
            time_windows.append((stop.time_window_start, stop.time_window_end))

        total_demand = sum(demands)

        dataset = TokyoDataset(
            name=name,
            depot=depot,
            stops=stops_dicts,
            distance_matrix=dist_matrix.tolist(),
            n_stops=n_stops,
            total_demand=total_demand,
            vehicle_capacity=self.vehicle_capacity,
            metadata={
                "region": "Shibuya Ward, Tokyo",
                "network_nodes": len(self.G.nodes),
                "network_edges": len(self.G.edges),
                "seed": self.seed,
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )

        return dataset

    def to_vrp_instance(self, dataset: TokyoDataset) -> VRPInstance:
        """Convert a TokyoDataset to a VRPInstance for QUBO solving.

        Args:
            dataset: Generated Tokyo delivery dataset.

        Returns:
            VRPInstance ready for the QUBO builder.
        """
        demands = [s["demand"] for s in dataset.stops]
        time_windows = [tuple(s["time_window"]) for s in dataset.stops]

        return VRPInstance(
            n_stops=dataset.n_stops,
            distance_matrix=np.array(dataset.distance_matrix),
            demands=np.array(demands),
            capacity=dataset.vehicle_capacity,
            time_windows=time_windows,
        )

    def save_dataset(self, dataset: TokyoDataset, filepath: str):
        """Save dataset to JSON file.

        Args:
            dataset: The dataset to save.
            filepath: Output file path.
        """
        data = {
            "name": dataset.name,
            "depot": dataset.depot,
            "stops": dataset.stops,
            "distance_matrix": dataset.distance_matrix,
            "n_stops": dataset.n_stops,
            "total_demand": dataset.total_demand,
            "vehicle_capacity": dataset.vehicle_capacity,
            "metadata": dataset.metadata,
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_dataset(filepath: str) -> TokyoDataset:
        """Load a dataset from JSON file.

        Args:
            filepath: Path to JSON file.

        Returns:
            TokyoDataset object.
        """
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return TokyoDataset(**data)

    def _nearest_node(self, lat: float, lon: float) -> int:
        """Find nearest network node to a given lat/lon position."""
        min_dist = float("inf")
        nearest = 0
        for node in self.G.nodes():
            d = self.network_gen._haversine_distance(
                lat, lon,
                self.G.nodes[node]["lat"], self.G.nodes[node]["lon"],
            )
            if d < min_dist:
                min_dist = d
                nearest = node
        return nearest

    def generate_preset_datasets(self) -> Dict[str, TokyoDataset]:
        """Generate all preset datasets (5, 10, 20, 50 stops).

        Returns:
            Dict mapping dataset name to TokyoDataset.
        """
        presets = {}
        for n in [5, 10, 20]:
            ds = self.generate_dataset(n)
            presets[ds.name] = ds
        return presets
