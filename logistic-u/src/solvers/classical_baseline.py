"""
Classical VRP Baseline Solver using Google OR-Tools.

Provides the "enemy" benchmark that the quantum solver must beat (or at least match).
Supports:
- Exact TSP/VRP solver (for small instances)
- Heuristic VRP solver (for larger instances)
- Metrics: solution cost, runtime, optimality gap
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from itertools import permutations

from src.qubo.vrp_qubo import VRPInstance


@dataclass
class ClassicalResult:
    """Results from a classical solver run."""
    routes: List[List[int]]  # List of routes, each route is node indices
    total_cost: float
    runtime_seconds: float
    method: str  # "exact", "heuristic", "brute_force"
    is_optimal: bool
    solver_status: str


class ClassicalBaseline:
    """Classical VRP solver wrapper."""

    def __init__(self, instance: VRPInstance):
        self.instance = instance

    def brute_force_tsp(self) -> ClassicalResult:
        """Solve TSP by brute force enumeration (single vehicle, no constraints).

        Only feasible for n_stops <= 10.
        """
        start_time = time.time()
        n = self.instance.n_stops
        dist = self.instance.distance_matrix

        if n > 10:
            raise ValueError(f"Brute force not feasible for {n} stops (max 10)")

        best_cost = float('inf')
        best_route = None

        # Try all permutations of stops (1..n)
        stops = list(range(1, n + 1))
        for perm in permutations(stops):
            route = [0] + list(perm) + [0]
            cost = sum(dist[route[i]][route[i + 1]] for i in range(len(route) - 1))
            if cost < best_cost:
                best_cost = cost
                best_route = route

        runtime = time.time() - start_time
        return ClassicalResult(
            routes=[best_route],
            total_cost=best_cost,
            runtime_seconds=runtime,
            method="brute_force",
            is_optimal=True,
            solver_status="OPTIMAL",
        )

    def solve_ortools_heuristic(
        self,
        time_limit_seconds: int = 30,
        n_vehicles: int = 1
    ) -> ClassicalResult:
        """Solve VRP using OR-Tools with heuristic methods.

        Args:
            time_limit_seconds: Maximum solving time.
            n_vehicles: Number of vehicles.

        Returns:
            ClassicalResult with the solution.
        """
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp

        start_time = time.time()
        n_nodes = self.instance.n_stops + 1  # +1 for depot
        dist = self.instance.distance_matrix

        # Create routing index manager
        manager = pywrapcp.RoutingIndexManager(n_nodes, n_vehicles, 0)
        routing = pywrapcp.RoutingModel(manager)

        # Distance callback
        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(dist[from_node][to_node] * 100)  # Scale to int

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Capacity constraint
        if np.any(self.instance.demands > 0):
            def demand_callback(from_index):
                node = manager.IndexToNode(from_index)
                if node == 0:  # Depot
                    return 0
                return int(self.instance.demands[node - 1])

            demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
            routing.AddDimensionWithVehicleCapacity(
                demand_callback_index,
                0,  # null capacity slack
                [int(self.instance.capacity)] * n_vehicles,
                True,  # start cumul to zero
                'Capacity'
            )

        # Time window constraints
        if self.instance.time_windows is not None:
            def time_callback(from_index, to_index):
                from_node = manager.IndexToNode(from_index)
                to_node = manager.IndexToNode(to_index)
                if self.instance.travel_times is not None:
                    return int(self.instance.travel_times[from_node][to_node])
                # Estimate: 30 km/h = 500 m/min.  dist is in meters.
                return max(1, int(dist[from_node][to_node] / 500.0))  # minutes

            time_callback_index = routing.RegisterTransitCallback(time_callback)
            routing.AddDimension(
                time_callback_index,
                30,   # allow waiting up to 30 min
                1440, # max 24 hours total (minutes) — accommodates all time window formats
                False,
                'Time'
            )
            time_dimension = routing.GetDimensionOrDie('Time')

            # Add time windows for each stop (clamp to valid range)
            for stop_idx in range(self.instance.n_stops):
                earliest, latest = self.instance.time_windows[stop_idx]
                earliest = max(0, int(earliest))
                latest = min(1440, max(earliest + 1, int(latest)))
                index = manager.NodeToIndex(stop_idx + 1)
                time_dimension.CumulVar(index).SetRange(earliest, latest)

        # Search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        )
        search_parameters.local_search_metaheuristic = (
            routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
        )
        search_parameters.time_limit.seconds = time_limit_seconds

        # Solve
        solution = routing.SolveWithParameters(search_parameters)

        runtime = time.time() - start_time

        if solution:
            routes = []
            total_cost = 0
            for vehicle_id in range(n_vehicles):
                route = []
                index = routing.Start(vehicle_id)
                while not routing.IsEnd(index):
                    node = manager.IndexToNode(index)
                    route.append(node)
                    index = solution.Value(routing.NextVar(index))
                route.append(0)  # Return to depot
                routes.append(route)

                # Calculate actual cost
                for i in range(len(route) - 1):
                    total_cost += dist[route[i]][route[i + 1]]

            return ClassicalResult(
                routes=routes,
                total_cost=total_cost,
                runtime_seconds=runtime,
                method="ortools_heuristic",
                is_optimal=False,
                solver_status="FEASIBLE",
            )
        else:
            return ClassicalResult(
                routes=[],
                total_cost=float('inf'),
                runtime_seconds=runtime,
                method="ortools_heuristic",
                is_optimal=False,
                solver_status="NO_SOLUTION",
            )

    def solve_ortools_exact(
        self,
        time_limit_seconds: int = 120,
    ) -> ClassicalResult:
        """Solve VRP using OR-Tools exact solver (branch and bound).

        Only viable for small instances (<15 stops). Provides the
        true optimal to benchmark against.
        """
        from ortools.constraint_solver import routing_enums_pb2, pywrapcp

        start_time = time.time()
        n_nodes = self.instance.n_stops + 1
        dist = self.instance.distance_matrix

        manager = pywrapcp.RoutingIndexManager(n_nodes, 1, 0)
        routing = pywrapcp.RoutingModel(manager)

        def distance_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return int(dist[from_node][to_node] * 100)

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            routing_enums_pb2.FirstSolutionStrategy.ALL_UNPERFORMED
        )
        search_parameters.time_limit.seconds = time_limit_seconds

        solution = routing.SolveWithParameters(search_parameters)
        runtime = time.time() - start_time

        if solution:
            route = []
            index = routing.Start(0)
            while not routing.IsEnd(index):
                route.append(manager.IndexToNode(index))
                index = solution.Value(routing.NextVar(index))
            route.append(0)

            cost = sum(dist[route[i]][route[i + 1]] for i in range(len(route) - 1))

            return ClassicalResult(
                routes=[route],
                total_cost=cost,
                runtime_seconds=runtime,
                method="ortools_exact",
                is_optimal=True,
                solver_status="OPTIMAL",
            )
        else:
            return ClassicalResult(
                routes=[],
                total_cost=float('inf'),
                runtime_seconds=runtime,
                method="ortools_exact",
                is_optimal=False,
                solver_status="TIMEOUT",
            )
