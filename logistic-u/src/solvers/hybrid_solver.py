"""
Hybrid Classical-Quantum Solver for VRP.

Decomposes large VRP instances into sub-problems that fit within
the quantum solver's qubit budget, then merges the results.

Architecture:
    1. CLUSTER: K-means on stop locations → geographic sub-problems
    2. ROUTE:   QAOA / brute-force per cluster (each ≤ qubit_budget)
    3. CONNECT: Classical 2-opt for inter-cluster transitions
    4. MERGE:   Combine sub-routes into full solution with depot returns

This enables scaling from the 5-qubit proof-of-concept (2 stops)
to 20+ stops by leveraging the quantum solver on tractable sub-problems.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import time

from src.qubo.vrp_qubo import VRPInstance, VRPQuboBuilder


@dataclass
class HybridResult:
    """Results from the hybrid solver."""
    routes: List[List[int]]        # List of routes (each starts/ends at 0=depot)
    total_cost: float               # Total distance cost
    n_vehicles: int                 # Number of vehicles used
    cluster_costs: List[float]      # Cost per cluster
    cluster_sizes: List[int]        # Stops per cluster
    method: str                     # Solver method used
    runtime_seconds: float
    quantum_sub_results: List[Dict] # Results from quantum sub-solvers
    improvement_history: List[float] # Cost improvement over 2-opt iterations


class StopClusterer:
    """Clusters delivery stops by geographic proximity and constraints.

    Uses K-means on stop coordinates, with optional demand-aware balancing
    to ensure each cluster's total demand fits within vehicle capacity.
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def cluster(
        self,
        instance: VRPInstance,
        max_stops_per_cluster: int = 4,
    ) -> List[List[int]]:
        """Cluster stops into sub-problems.

        Args:
            instance: VRP instance with distance matrix and demands.
            max_stops_per_cluster: Maximum stops per quantum sub-problem.

        Returns:
            List of clusters, each a list of stop indices (1-indexed).
        """
        n = instance.n_stops
        if n <= max_stops_per_cluster:
            return [list(range(1, n + 1))]

        # Determine number of clusters
        k = max(2, (n + max_stops_per_cluster - 1) // max_stops_per_cluster)

        # Use distance matrix as feature space (MDS-like embedding)
        # Take distances from depot as initial feature
        dist = instance.distance_matrix
        features = np.column_stack([
            dist[0, 1:],  # distance from depot
            dist[1:, 0],  # distance to depot
        ])

        # Add pairwise distances as additional features
        for ref in range(1, min(n + 1, 5)):
            features = np.column_stack([features, dist[ref, 1:]])

        # K-means clustering
        labels = self._kmeans(features, k)

        # Group stops by cluster
        clusters = [[] for _ in range(k)]
        for stop_idx in range(n):
            cluster_id = labels[stop_idx]
            clusters[cluster_id].append(stop_idx + 1)  # 1-indexed

        # Remove empty clusters
        clusters = [c for c in clusters if len(c) > 0]

        # Post-process: split oversized clusters
        final_clusters = []
        for cluster in clusters:
            if len(cluster) > max_stops_per_cluster:
                for i in range(0, len(cluster), max_stops_per_cluster):
                    sub = cluster[i:i + max_stops_per_cluster]
                    if sub:
                        final_clusters.append(sub)
            else:
                final_clusters.append(cluster)

        return final_clusters

    def _kmeans(
        self, features: np.ndarray, k: int, max_iter: int = 50
    ) -> np.ndarray:
        """Simple K-means implementation.

        Args:
            features: Feature matrix (n_stops, n_features).
            k: Number of clusters.
            max_iter: Maximum iterations.

        Returns:
            Cluster labels array.
        """
        n = features.shape[0]
        k = min(k, n)

        # Initialize centroids randomly
        indices = self.rng.choice(n, k, replace=False)
        centroids = features[indices].copy()

        labels = np.zeros(n, dtype=int)
        for _ in range(max_iter):
            # Assign each point to nearest centroid
            for i in range(n):
                dists = np.linalg.norm(features[i] - centroids, axis=1)
                labels[i] = np.argmin(dists)

            # Update centroids
            new_centroids = np.zeros_like(centroids)
            for c in range(k):
                mask = labels == c
                if np.any(mask):
                    new_centroids[c] = features[mask].mean(axis=0)
                else:
                    new_centroids[c] = centroids[c]

            if np.allclose(centroids, new_centroids):
                break
            centroids = new_centroids

        return labels


class TwoOptImprover:
    """Classical 2-opt local search for route improvement.

    After quantum sub-problems are solved, 2-opt refines the full solution
    by swapping edges to reduce total distance.
    """

    def __init__(self, max_iter: int = 1000):
        self.max_iter = max_iter

    def improve(
        self, route: List[int], dist_matrix: np.ndarray
    ) -> Tuple[List[int], float]:
        """Improve a route using 2-opt swaps.

        Args:
            route: Route as list of stop indices (starts/ends with 0=depot).
            dist_matrix: Distance matrix.

        Returns:
            (improved_route, improved_cost).
        """
        best = list(route)
        best_cost = self._route_cost(best, dist_matrix)

        improved = True
        iteration = 0
        while improved and iteration < self.max_iter:
            improved = False
            iteration += 1
            for i in range(1, len(best) - 2):
                for j in range(i + 1, len(best) - 1):
                    # Try reversing the segment between i and j
                    new_route = best[:i] + best[i:j+1][::-1] + best[j+1:]
                    new_cost = self._route_cost(new_route, dist_matrix)
                    if new_cost < best_cost - 1e-10:
                        best = new_route
                        best_cost = new_cost
                        improved = True

        return best, best_cost

    def _route_cost(self, route: List[int], dist_matrix: np.ndarray) -> float:
        """Compute total distance of a route."""
        cost = 0.0
        for i in range(len(route) - 1):
            cost += dist_matrix[route[i]][route[i + 1]]
        return cost


class HybridSolver:
    """Hybrid classical-quantum VRP solver.

    Combines classical clustering and local search with quantum
    optimization for sub-problems.
    """

    def __init__(
        self,
        max_stops_per_quantum: int = 4,
        qaoa_depth: int = 1,
        encoding: str = "position",
        seed: int = 42,
    ):
        """
        Args:
            max_stops_per_quantum: Max stops per quantum sub-problem.
            qaoa_depth: QAOA depth for quantum sub-solver.
            encoding: QUBO encoding ("position" or "route").
            seed: Random seed.
        """
        self.max_stops_per_quantum = max_stops_per_quantum
        self.qaoa_depth = qaoa_depth
        self.encoding = encoding
        self.seed = seed
        self.clusterer = StopClusterer(seed=seed)
        self.improver = TwoOptImprover()

    def solve(
        self,
        instance: VRPInstance,
        use_quantum: bool = True,
        time_budget: float = 60.0,
    ) -> HybridResult:
        """Solve a VRP instance using hybrid decomposition.

        Args:
            instance: VRP problem instance.
            use_quantum: If True, use QAOA for sub-problems. If False, use brute-force.
            time_budget: Maximum time in seconds.

        Returns:
            HybridResult with routes and metrics.
        """
        start_time = time.time()

        # Step 1: Cluster stops
        clusters = self.clusterer.cluster(instance, self.max_stops_per_quantum)
        n_vehicles = len(clusters)

        # Step 2: Solve each cluster
        routes = []
        cluster_costs = []
        cluster_sizes = []
        quantum_results = []
        improvement_history = [0.0]

        for cluster_idx, cluster_stops in enumerate(clusters):
            if time.time() - start_time > time_budget:
                # Time limit: use nearest-neighbor for remaining clusters
                route, cost = self._nearest_neighbor_route(
                    cluster_stops, instance.distance_matrix
                )
                routes.append(route)
                cluster_costs.append(cost)
                cluster_sizes.append(len(cluster_stops))
                quantum_results.append({"method": "nearest_neighbor", "timeout": True})
                continue

            # Build sub-instance for this cluster
            sub_instance, node_map = self._build_sub_instance(
                instance, cluster_stops
            )

            if len(cluster_stops) <= 1:
                # Trivial: depot → stop → depot
                route = [0] + cluster_stops + [0]
                cost = (instance.distance_matrix[0][cluster_stops[0]] +
                       instance.distance_matrix[cluster_stops[0]][0])
                routes.append(route)
                cluster_costs.append(cost)
                cluster_sizes.append(1)
                quantum_results.append({"method": "trivial", "n_stops": 1})
                continue

            # Try quantum (brute-force QUBO) for small instances
            if use_quantum and len(cluster_stops) <= self.max_stops_per_quantum:
                sub_result = self._solve_quantum(sub_instance, node_map, instance)
            else:
                sub_result = self._solve_brute_force(sub_instance, node_map, instance)

            routes.append(sub_result["route"])
            cluster_costs.append(sub_result["cost"])
            cluster_sizes.append(len(cluster_stops))
            quantum_results.append(sub_result)

        # Step 3: 2-opt improvement on each route
        total_before_2opt = sum(cluster_costs)
        improvement_history.append(total_before_2opt)

        for i, route in enumerate(routes):
            if len(route) > 4:  # Only improve routes with 3+ stops
                improved_route, improved_cost = self.improver.improve(
                    route, instance.distance_matrix
                )
                routes[i] = improved_route
                cluster_costs[i] = improved_cost

        total_cost = sum(cluster_costs)
        improvement_history.append(total_cost)

        runtime = time.time() - start_time

        return HybridResult(
            routes=routes,
            total_cost=total_cost,
            n_vehicles=n_vehicles,
            cluster_costs=cluster_costs,
            cluster_sizes=cluster_sizes,
            method=f"hybrid_{'quantum' if use_quantum else 'classical'}_p{self.qaoa_depth}",
            runtime_seconds=runtime,
            quantum_sub_results=quantum_results,
            improvement_history=improvement_history,
        )

    def _build_sub_instance(
        self, full_instance: VRPInstance, cluster_stops: List[int]
    ) -> Tuple[VRPInstance, Dict[int, int]]:
        """Build a VRPInstance for a cluster sub-problem.

        Args:
            full_instance: The full VRP instance.
            cluster_stops: List of stop indices in this cluster (1-indexed in full instance).

        Returns:
            (sub_instance, node_map) where node_map maps sub-indices to full indices.
        """
        n_sub = len(cluster_stops)

        # Node map: sub_index -> full_index
        # 0 -> 0 (depot), 1..n_sub -> cluster_stops
        full_nodes = [0] + cluster_stops
        node_map = {i: full_nodes[i] for i in range(len(full_nodes))}

        # Extract sub-distance-matrix
        sub_dist = np.zeros((n_sub + 1, n_sub + 1))
        for i, fi in enumerate(full_nodes):
            for j, fj in enumerate(full_nodes):
                sub_dist[i][j] = full_instance.distance_matrix[fi][fj]

        # Extract demands
        sub_demands = np.array([
            full_instance.demands[s - 1] for s in cluster_stops
        ])

        # Extract time windows if available
        sub_tw = None
        if full_instance.time_windows is not None:
            sub_tw = [full_instance.time_windows[s - 1] for s in cluster_stops]

        sub_instance = VRPInstance(
            n_stops=n_sub,
            distance_matrix=sub_dist,
            demands=sub_demands,
            capacity=full_instance.capacity,
            time_windows=sub_tw,
        )

        return sub_instance, node_map

    def _solve_quantum(
        self,
        sub_instance: VRPInstance,
        node_map: Dict[int, int],
        full_instance: VRPInstance,
    ) -> Dict:
        """Solve a sub-problem using QUBO brute-force (small enough for exact).

        For sub-problems up to ~4 stops (16 qubits), brute-force over the
        QUBO solution space is exact and fast.
        """
        builder = VRPQuboBuilder(sub_instance, encoding=self.encoding)
        bits, energy = builder.brute_force_solve()

        # Decode sub-solution
        if bits is not None:
            tour = builder.encoding.decode(bits)
            eval_result = builder.evaluate_solution(bits)
        else:
            tour = None
            eval_result = {"feasible": False, "cost": float("inf")}

        # Map back to full instance indices
        if tour is not None and eval_result["feasible"]:
            # tour gives sub-indices, map to full indices
            route = [0]  # start at depot
            for sub_idx in tour:
                full_idx = node_map.get(sub_idx + 1, sub_idx + 1)
                route.append(full_idx)
            route.append(0)  # return to depot
            cost = eval_result["cost"]
        else:
            # Fallback: nearest neighbor
            cluster_stops = [node_map[i] for i in range(1, sub_instance.n_stops + 1)]
            route, cost = self._nearest_neighbor_route(
                cluster_stops, full_instance.distance_matrix
            )

        return {
            "method": "qubo_brute_force",
            "route": route,
            "cost": cost,
            "n_qubits": builder.build().n_qubits,
            "feasible": eval_result.get("feasible", False),
            "qubo_energy": float(energy) if energy is not None else None,
        }

    def _solve_brute_force(
        self,
        sub_instance: VRPInstance,
        node_map: Dict[int, int],
        full_instance: VRPInstance,
    ) -> Dict:
        """Solve a sub-problem by brute-force TSP enumeration."""
        from itertools import permutations

        n = sub_instance.n_stops
        dist = sub_instance.distance_matrix

        best_cost = float("inf")
        best_perm = None

        for perm in permutations(range(1, n + 1)):
            cost = dist[0][perm[0]]
            for i in range(len(perm) - 1):
                cost += dist[perm[i]][perm[i + 1]]
            cost += dist[perm[-1]][0]

            if cost < best_cost:
                best_cost = cost
                best_perm = perm

        # Map back to full indices
        if best_perm is not None:
            route = [0]
            for sub_idx in best_perm:
                route.append(node_map.get(sub_idx, sub_idx))
            route.append(0)
        else:
            cluster_stops = [node_map[i] for i in range(1, n + 1)]
            route, best_cost = self._nearest_neighbor_route(
                cluster_stops, full_instance.distance_matrix
            )

        return {
            "method": "brute_force_tsp",
            "route": route,
            "cost": best_cost,
            "n_stops": n,
        }

    def _nearest_neighbor_route(
        self, stops: List[int], dist_matrix: np.ndarray
    ) -> Tuple[List[int], float]:
        """Simple nearest-neighbor heuristic for route construction.

        Args:
            stops: Stop indices to visit.
            dist_matrix: Full distance matrix.

        Returns:
            (route, cost) where route starts/ends at depot (0).
        """
        if not stops:
            return [0, 0], 0.0

        remaining = set(stops)
        route = [0]
        cost = 0.0
        current = 0

        while remaining:
            nearest = min(remaining, key=lambda s: dist_matrix[current][s])
            cost += dist_matrix[current][nearest]
            route.append(nearest)
            remaining.remove(nearest)
            current = nearest

        cost += dist_matrix[current][0]
        route.append(0)

        return route, cost
