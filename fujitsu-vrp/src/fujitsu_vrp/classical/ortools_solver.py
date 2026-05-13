"""
OR-Tools Baseline Solver for VRP.

Provides a classical baseline using Google OR-Tools for solving
Capacitated Vehicle Routing Problems with Time Windows (CVRPTW).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

if TYPE_CHECKING:
    from ..data.synthetic_generator import VRPInstance
    from ..data.problem_builder import Solution

logger = logging.getLogger(__name__)


@dataclass
class SolverConfig:
    """Configuration for OR-Tools solver."""

    # Time limits
    time_limit_seconds: int = 60
    solution_limit: int | None = None  # Max number of solutions

    # Search strategy
    first_solution_strategy: str = "PATH_CHEAPEST_ARC"
    local_search_metaheuristic: str = "GUIDED_LOCAL_SEARCH"

    # Penalty parameters
    penalty_coefficient: float = 1000.0  # For un-served customers
    capacity_penalty: float = 1000.0
    time_window_penalty: float = 1000.0

    # Optimization
    use_full_path_for_first_solution: bool = False
    use_cp_sat: bool = False  # Use CP-SAT solver (more powerful)

    # Logging
    log_search_progress: bool = False

    def get_first_solution_strategy_enum(self) -> int:
        """Convert string to OR-Tools enum."""
        return getattr(
            routing_enums_pb2.FirstSolutionStrategy, self.first_solution_strategy
        )

    def get_local_search_enum(self) -> int:
        """Convert string to OR-Tools enum."""
        return getattr(
            routing_enums_pb2.LocalSearchMetaheuristic, self.local_search_metaheuristic
        )


@dataclass
class ORToolsResult:
    """Result from OR-Tools solver."""

    solution: Solution | None
    status: str
    objective: float
    total_distance: float
    total_time: float
    num_routes_used: int
    unassigned_customers: list[int]
    capacity_violations: float
    time_window_violations: float
    solve_time_ms: int
    metadata: dict[str, Any] = field(default_factory=dict)


class ORToolsSolver:
    """OR-Tools based VRP solver."""

    def __init__(self, config: SolverConfig | None = None) -> None:
        """Initialize solver with configuration.

        Args:
            config: Solver configuration (uses defaults if None)
        """
        self.config = config or SolverConfig()

    def solve(
        self,
        instance: VRPInstance,
        initial_routes: list[list[int]] | None = None,
    ) -> ORToolsResult:
        """Solve the VRP instance.

        Args:
            instance: VRP instance to solve
            initial_routes: Optional initial solution

        Returns:
            ORToolsResult with solution and statistics
        """
        # Create routing model
        manager = pywrapcp.RoutingIndexManager(
            instance.num_customers + 1,  # +1 for depot
            instance.num_vehicles,
            0,  # Depot index
        )
        routing = pywrapcp.RoutingModel(manager)

        # Create distance callback
        def distance_callback(from_index: int, to_index: int) -> int:
            """Returns distance between two nodes."""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            # Scale distance to integers (multiply by 1000 for precision)
            return int(instance.distance_matrix[from_node, to_node] * 1000)

        transit_callback_index = routing.RegisterTransitCallback(distance_callback)

        # Define cost of each arc
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        # Add capacity constraint
        def demand_callback(from_index: int) -> int:
            """Returns demand at node."""
            from_node = manager.IndexToNode(from_index)
            if from_node == 0:  # Depot
                return 0
            return int(instance.customers[from_node - 1].demand)

        demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)

        routing.AddDimensionWithVehicleCapacity(
            demand_callback_index,
            0,  # Null capacity slack
            [int(v.capacity) for v in instance.vehicles],  # Vehicle capacities
            False,  # Start cumul at zero
            "Capacity",
        )

        # Add time window constraint
        def time_callback(from_index: int) -> int:
            """Returns travel time between nodes."""
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            # Scale time to integers (multiply by 100 for precision)
            return int(instance.time_matrix[from_node, to_node] * 100)

        time_callback_index = routing.RegisterTransitCallback(time_callback)

        routing.AddDimension(
            time_callback_index,
            30 * 100,  # Allow 30 minutes waiting time (scaled)
            int(
                max(v.max_route_time for v in instance.vehicles) * 100
            ),  # Max route time
            False,  # Start cumul at zero
            "Time",
        )
        time_dimension = routing.GetDimensionOrDie("Time")

        # Add time windows
        for i, customer in enumerate(instance.customers):
            index = manager.NodeToIndex(i + 1)  # +1 because depot is 0
            start = int(customer.time_window_start * 100)
            end = int(customer.time_window_end * 100)
            time_dimension.CumulVar(index).SetRange(start, end)

        # Set depot time window
        depot_index = manager.NodeToIndex(0)
        time_dimension.CumulVar(depot_index).SetRange(0, int(instance.depot.time_window_end * 100))

        # Set initial solution if provided
        if initial_routes is not None:
            initial_solution = routing.ReadSolutionFromRoutes(
                [
                    [manager.NodeToIndex(n) for n in route]
                    for route in initial_routes
                ],
                True,
            )
            routing.ApplyLockToAllSolutionIndices()

        # Set search parameters
        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = (
            self.config.get_first_solution_strategy_enum()
        )
        search_parameters.local_search_metaheuristic = (
            self.config.get_local_search_enum()
        )
        search_parameters.time_limit.seconds = self.config.time_limit_seconds

        if self.config.solution_limit is not None:
            search_parameters.solution_limit = self.config.solution_limit

        if self.config.log_search_progress:
            search_parameters.log_search = True

        # Solve
        logger.info(f"Starting OR-Tools solve for {instance.name}")
        assignment = routing.SolveWithParameters(search_parameters)
        solve_time_ms = routing.solver().wall_time()

        # Process result
        if assignment is None:
            logger.warning("No solution found")
            return ORToolsResult(
                solution=None,
                status="NO_SOLUTION",
                objective=float("inf"),
                total_distance=0,
                total_time=0,
                num_routes_used=0,
                unassigned_customers=list(range(1, instance.num_customers + 1)),
                capacity_violations=0,
                time_window_violations=0,
                solve_time_ms=solve_time_ms,
            )

        # Extract solution
        routes = self._extract_routes(routing, manager, assignment, instance)
        solution = self._build_solution(instance, routes)

        # Compute statistics
        total_distance = sum(
            self._compute_route_distance(route, instance.distance_matrix)
            for route in routes
        )
        total_time = sum(
            self._compute_route_time(route, instance) for route in routes
        )

        # Check for unassigned customers
        all_served = set()
        for route in routes:
            all_served.update(route[1:-1])  # Exclude depot
        unassigned = [
            i + 1 for i in range(instance.num_customers) if i not in all_served
        ]

        status_map = {
            routing.ROUTING_SUCCESS: "SUCCESS",
            routing.ROUTING_PARTIAL_SUCCESS: "PARTIAL_SUCCESS",
            routing.ROUTING_FAIL: "FAIL",
            routing.ROUTING_NOT_SOLVED: "NOT_SOLVED",
        }

        result = ORToolsResult(
            solution=solution,
            status=status_map.get(routing.status(), "UNKNOWN"),
            objective=assignment.ObjectiveValue() / 1000.0,  # Unscaled
            total_distance=total_distance,
            total_time=total_time,
            num_routes_used=len([r for r in routes if len(r) > 2]),
            unassigned_customers=unassigned,
            capacity_violations=self._check_capacity_violations(
                routes, instance
            ),
            time_window_violations=self._check_time_window_violations(
                routes, instance
            ),
            solve_time_ms=solve_time_ms,
            metadata={
                "routing_status": routing.status(),
                "solver_version": "OR-Tools",
            },
        )

        logger.info(
            f"Solution found: distance={result.total_distance:.2f}km, "
            f"routes={result.num_routes_used}, "
            f"time={solve_time_ms}ms"
        )

        return result

    def _extract_routes(
        self,
        routing: pywrapcp.RoutingModel,
        manager: pywrapcp.RoutingIndexManager,
        assignment: pywrapcp.Assignment,
        instance: VRPInstance,
    ) -> list[list[int]]:
        """Extract routes from OR-Tools solution."""
        routes = []

        for vehicle_id in range(instance.num_vehicles):
            route = []
            index = routing.Start(vehicle_id)

            while not routing.IsEnd(index):
                node_index = manager.IndexToNode(index)
                route.append(node_index)
                index = assignment.Value(routing.NextVar(index))

            # Add end depot
            route.append(manager.IndexToNode(index))
            routes.append(route)

        return routes

    def _build_solution(
        self,
        instance: VRPInstance,
        routes: list[list[int]],
    ) -> Solution:
        """Build Solution object from routes."""
        from ..data.problem_builder import Solution

        total_distance = sum(
            self._compute_route_distance(route, instance.distance_matrix)
            for route in routes
        )
        total_time = sum(
            self._compute_route_time(route, instance) for route in routes
        )

        all_served = set()
        for route in routes:
            all_served.update(route[1:-1])
        total_demand = sum(
            instance.customers[i - 1].demand for i in all_served
        )

        return Solution(
            instance_name=instance.name,
            routes=routes,
            total_distance=total_distance,
            total_time=total_time,
            total_demand_served=total_demand,
            time_window_violations=self._check_time_window_violations(routes, instance),
            capacity_violations=self._check_capacity_violations(routes, instance),
        )

    def _compute_route_distance(
        self,
        route: list[int],
        distance_matrix: np.ndarray,
    ) -> float:
        """Compute total distance of a route."""
        total = 0.0
        for i in range(len(route) - 1):
            total += distance_matrix[route[i], route[i + 1]]
        return total

    def _compute_route_time(
        self,
        route: list[int],
        instance: VRPInstance,
    ) -> float:
        """Compute total time of a route including service."""
        total = 0.0
        all_nodes = [instance.depot] + instance.customers

        for i in range(len(route) - 1):
            # Travel time
            total += instance.time_matrix[route[i], route[i + 1]]
            # Service time (except at final depot)
            if i < len(route) - 2:
                total += all_nodes[route[i + 1]].service_time

        return total

    def _check_capacity_violations(
        self,
        routes: list[list[int]],
        instance: VRPInstance,
    ) -> float:
        """Check total capacity violations."""
        total_violation = 0.0

        for vehicle_id, route in enumerate(routes):
            if vehicle_id >= len(instance.vehicles):
                continue

            capacity = instance.vehicles[vehicle_id].capacity
            demand = sum(
                instance.customers[node - 1].demand
                for node in route[1:-1]
                if node > 0
            )

            if demand > capacity:
                total_violation += demand - capacity

        return total_violation

    def _check_time_window_violations(
        self,
        routes: list[list[int]],
        instance: VRPInstance,
    ) -> float:
        """Check total time window violations."""
        total_violation = 0.0

        for route in routes:
            current_time = 0.0

            for i in range(len(route) - 1):
                # Travel to next node
                current_time += instance.time_matrix[route[i], route[i + 1]]

                # Check time window
                if route[i + 1] > 0:  # Not depot
                    customer = instance.customers[route[i + 1] - 1]
                    if current_time > customer.time_window_end:
                        total_violation += current_time - customer.time_window_end

                    # Wait if early
                    if current_time < customer.time_window_start:
                        current_time = customer.time_window_start

                    # Add service time
                    current_time += customer.service_time

        return total_violation


def solve_vrp(
    instance: VRPInstance,
    time_limit: int = 60,
    strategy: str = "GUIDED_LOCAL_SEARCH",
) -> ORToolsResult:
    """Convenience function to solve VRP with default parameters.

    Args:
        instance: VRP instance
        time_limit: Time limit in seconds
        strategy: Local search strategy

    Returns:
        ORToolsResult
    """
    config = SolverConfig(
        time_limit_seconds=time_limit,
        local_search_metaheuristic=strategy,
    )
    solver = ORToolsSolver(config)
    return solver.solve(instance)