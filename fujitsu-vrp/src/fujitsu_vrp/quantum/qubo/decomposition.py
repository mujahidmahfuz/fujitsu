"""
Problem Decomposition for 40-Qubit Constraint.

Decomposes VRP repair subproblems to fit within Fujitsu's 40-qubit limit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ...data.synthetic_generator import VRPInstance

logger = logging.getLogger(__name__)


@dataclass
class Subproblem:
    """A decomposed subproblem."""

    subproblem_id: int
    customers: list[int]  # Customers in this subproblem
    routes: list[list[int]]  # Routes for this subproblem
    slots: list[tuple[int, int]]  # Available insertion slots
    num_qubits: int  # Required qubits

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Decomposition:
    """Complete decomposition of a VRP repair subproblem."""

    subproblems: list[Subproblem]
    original_customers: list[int]
    original_routes: list[list[int]]
    strategy: str

    # Statistics
    total_qubits: int = 0
    max_subproblem_qubits: int = 0


class ProblemDecomposer:
    """Decomposes VRP repair subproblems to fit qubit constraints.

    Strategies:
    1. Customer-based: Split customers into groups
    2. Route-based: Optimize routes independently
    3. Geographic: Split by geographic regions
    4. Hierarchical: Multi-level decomposition
    """

    def __init__(
        self,
        max_qubits: int = 40,
        strategy: str = "auto",
    ) -> None:
        """Initialize decomposer.

        Args:
            max_qubits: Maximum qubits per subproblem
            strategy: Decomposition strategy ("auto", "customer", "route", "geographic")
        """
        self.max_qubits = max_qubits
        self.strategy = strategy

    def decompose(
        self,
        instance: VRPInstance,
        removed_customers: list[int],
        partial_routes: list[list[int]],
    ) -> Decomposition:
        """Decompose the repair subproblem.

        Args:
            instance: VRP instance
            removed_customers: Customers to be reinserted
            partial_routes: Current partial routes

        Returns:
            Decomposition with subproblems
        """
        # Calculate required qubits
        num_customers = len(removed_customers)
        num_slots = sum(len(r) - 1 for r in partial_routes) + num_customers
        required_qubits = num_customers * num_slots

        if required_qubits <= self.max_qubits:
            # No decomposition needed
            return self._create_single_subproblem(
                removed_customers, partial_routes
            )

        # Choose strategy
        if self.strategy == "auto":
            strategy = self._choose_strategy(removed_customers, partial_routes, instance)
        else:
            strategy = self.strategy

        logger.info(
            f"Decomposing {num_customers} customers with {required_qubits} qubits "
            f"(limit: {self.max_qubits}) using {strategy} strategy"
        )

        if strategy == "customer":
            return self._decompose_by_customer(
                instance, removed_customers, partial_routes
            )
        elif strategy == "route":
            return self._decompose_by_route(
                instance, removed_customers, partial_routes
            )
        elif strategy == "geographic":
            return self._decompose_by_geography(
                instance, removed_customers, partial_routes
            )
        else:
            return self._decompose_by_customer(
                instance, removed_customers, partial_routes
            )

    def _choose_strategy(
        self,
        removed_customers: list[int],
        partial_routes: list[list[int]],
        instance: VRPInstance,
    ) -> str:
        """Choose best decomposition strategy."""
        num_customers = len(removed_customers)
        num_routes = len(partial_routes)

        # If many routes, route-based decomposition works well
        if num_routes >= num_customers:
            return "route"

        # If customers are geographically spread, geographic decomposition
        if self._is_geographically_spread(removed_customers, instance):
            return "geographic"

        # Default to customer-based
        return "customer"

    def _is_geographically_spread(
        self,
        customers: list[int],
        instance: VRPInstance,
    ) -> bool:
        """Check if customers are geographically spread."""
        if len(customers) < 3:
            return False

        # Compute spread
        lons = [instance.customers[c - 1].x for c in customers if c > 0]
        lats = [instance.customers[c - 1].y for c in customers if c > 0]

        lon_range = max(lons) - min(lons) if lons else 0
        lat_range = max(lats) - min(lats) if lats else 0

        # Spread if range > 0.1 degrees (~10km)
        return lon_range > 0.1 or lat_range > 0.1

    def _create_single_subproblem(
        self,
        removed_customers: list[int],
        partial_routes: list[list[int]],
    ) -> Decomposition:
        """Create a single subproblem (no decomposition)."""
        num_slots = sum(len(r) - 1 for r in partial_routes) + len(removed_customers)
        num_qubits = len(removed_customers) * num_slots

        slots = []
        for route_idx, route in enumerate(partial_routes):
            for pos in range(1, len(route)):
                slots.append((route_idx, pos))

        subproblem = Subproblem(
            subproblem_id=0,
            customers=removed_customers,
            routes=partial_routes,
            slots=slots,
            num_qubits=num_qubits,
        )

        return Decomposition(
            subproblems=[subproblem],
            original_customers=removed_customers,
            original_routes=partial_routes,
            strategy="none",
            total_qubits=num_qubits,
            max_subproblem_qubits=num_qubits,
        )

    def _decompose_by_customer(
        self,
        instance: VRPInstance,
        removed_customers: list[int],
        partial_routes: list[list[int]],
    ) -> Decomposition:
        """Decompose by splitting customers into groups."""
        # Calculate customers per subproblem
        num_slots = sum(len(r) - 1 for r in partial_routes)
        customers_per_subproblem = max(1, self.max_qubits // num_slots)

        # Adjust for safety margin
        customers_per_subproblem = max(1, customers_per_subproblem - 1)

        # Group customers
        customer_groups = self._group_customers(
            removed_customers, customers_per_subproblem, instance
        )

        subproblems = []
        for i, group in enumerate(customer_groups):
            # Available slots
            slots = []
            for route_idx, route in enumerate(partial_routes):
                for pos in range(1, len(route)):
                    slots.append((route_idx, pos))

            num_qubits = len(group) * len(slots)

            subproblem = Subproblem(
                subproblem_id=i,
                customers=group,
                routes=[list(r) for r in partial_routes],
                slots=slots,
                num_qubits=num_qubits,
            )
            subproblems.append(subproblem)

        return Decomposition(
            subproblems=subproblems,
            original_customers=removed_customers,
            original_routes=partial_routes,
            strategy="customer",
            total_qubits=sum(s.num_qubits for s in subproblems),
            max_subproblem_qubits=max(s.num_qubits for s in subproblems),
        )

    def _group_customers(
        self,
        customers: list[int],
        group_size: int,
        instance: VRPInstance,
    ) -> list[list[int]]:
        """Group customers using geographic clustering."""
        if len(customers) <= group_size:
            return [customers]

        # Sort by geographic proximity (simplified)
        # Use k-means-like grouping
        groups = []
        remaining = list(customers)

        while remaining:
            # Take next group
            group = remaining[:group_size]
            remaining = remaining[group_size:]
            groups.append(group)

        return groups

    def _decompose_by_route(
        self,
        instance: VRPInstance,
        removed_customers: list[int],
        partial_routes: list[list[int]],
    ) -> Decomposition:
        """Decompose by assigning customers to route groups."""
        subproblems = []

        # Assign each customer to nearest route
        route_customers = [[] for _ in partial_routes]

        for customer in removed_customers:
            cust = instance.customers[customer - 1]

            # Find nearest route
            min_dist = float("inf")
            best_route = 0

            for route_idx, route in enumerate(partial_routes):
                if len(route) <= 2:
                    continue

                # Compute route centroid
                centroid_x = np.mean(
                    [instance.customers[n - 1].x for n in route[1:-1] if n > 0]
                    or [instance.depot.x]
                )
                centroid_y = np.mean(
                    [instance.customers[n - 1].y for n in route[1:-1] if n > 0]
                    or [instance.depot.y]
                )

                dist = np.sqrt((cust.x - centroid_x) ** 2 + (cust.y - centroid_y) ** 2)
                if dist < min_dist:
                    min_dist = dist
                    best_route = route_idx

            route_customers[best_route].append(customer)

        # Create subproblems
        for i, customers in enumerate(route_customers):
            if not customers:
                continue

            route = partial_routes[i]
            slots = [(i, pos) for pos in range(1, len(route))]
            num_qubits = len(customers) * len(slots)

            subproblem = Subproblem(
                subproblem_id=len(subproblems),
                customers=customers,
                routes=[list(route)],
                slots=slots,
                num_qubits=num_qubits,
            )
            subproblems.append(subproblem)

        # Handle unassigned customers (shouldn't happen but safety check)
        assigned = set()
        for sp in subproblems:
            assigned.update(sp.customers)
        unassigned = [c for c in removed_customers if c not in assigned]

        if unassigned and subproblems:
            # Add to first subproblem
            subproblems[0].customers.extend(unassigned)

        return Decomposition(
            subproblems=subproblems,
            original_customers=removed_customers,
            original_routes=partial_routes,
            strategy="route",
            total_qubits=sum(s.num_qubits for s in subproblems),
            max_subproblem_qubits=max(s.num_qubits for s in subproblems),
        )

    def _decompose_by_geography(
        self,
        instance: VRPInstance,
        removed_customers: list[int],
        partial_routes: list[list[int]],
    ) -> Decomposition:
        """Decompose by geographic regions."""
        # Cluster customers by location
        if len(removed_customers) <= 3:
            return self._create_single_subproblem(removed_customers, partial_routes)

        # Get customer locations
        coords = np.array([
            [instance.customers[c - 1].x, instance.customers[c - 1].y]
            for c in removed_customers
        ])

        # Determine number of clusters based on qubit limit
        num_slots = sum(len(r) - 1 for r in partial_routes)
        max_customers_per_cluster = max(1, self.max_qubits // num_slots - 1)
        num_clusters = max(1, len(removed_customers) // max_customers_per_cluster + 1)

        # Simple k-means
        clusters = self._kmeans_cluster(coords, num_clusters)

        # Create subproblems
        subproblems = []
        for cluster_idx in range(num_clusters):
            cluster_customers = [
                removed_customers[i]
                for i, c in enumerate(clusters)
                if c == cluster_idx
            ]

            if not cluster_customers:
                continue

            slots = []
            for route_idx, route in enumerate(partial_routes):
                for pos in range(1, len(route)):
                    slots.append((route_idx, pos))

            num_qubits = len(cluster_customers) * len(slots)

            subproblem = Subproblem(
                subproblem_id=len(subproblems),
                customers=cluster_customers,
                routes=[list(r) for r in partial_routes],
                slots=slots,
                num_qubits=num_qubits,
            )
            subproblems.append(subproblem)

        return Decomposition(
            subproblems=subproblems,
            original_customers=removed_customers,
            original_routes=partial_routes,
            strategy="geographic",
            total_qubits=sum(s.num_qubits for s in subproblems),
            max_subproblem_qubits=max(s.num_qubits for s in subproblems),
        )

    def _kmeans_cluster(
        self,
        coords: np.ndarray,
        k: int,
        max_iterations: int = 20,
    ) -> np.ndarray:
        """Simple k-means clustering."""
        n = len(coords)
        if k >= n:
            return np.arange(n)

        # Initialize centroids randomly
        rng = np.random.default_rng(42)
        centroid_indices = rng.choice(n, k, replace=False)
        centroids = coords[centroid_indices]

        for _ in range(max_iterations):
            # Assign to nearest centroid
            distances = np.zeros((n, k))
            for i in range(k):
                distances[:, i] = np.sqrt(np.sum((coords - centroids[i]) ** 2, axis=1))

            clusters = np.argmin(distances, axis=1)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for i in range(k):
                mask = clusters == i
                if np.any(mask):
                    new_centroids[i] = np.mean(coords[mask], axis=0)
                else:
                    new_centroids[i] = centroids[i]

            if np.allclose(centroids, new_centroids):
                break

            centroids = new_centroids

        return clusters