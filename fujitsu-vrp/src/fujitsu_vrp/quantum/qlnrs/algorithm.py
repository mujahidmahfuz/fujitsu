"""
QLNRS (Quantum Large Neighborhood Risk Search) Algorithm.

Main algorithm integrating:
- Chaotic operator selection
- Quantum annealing for repair
- Risk-aware objective
- Lyapunov-adaptive control
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from ...classical.lns_base import (
    GreedyRepair,
    LNSConfig,
    LNSState,
    RegretRepair,
)
from ...data.problem_builder import Solution
from .chaotic_operators import (
    ChaoticOperatorSelector,
    ChaoticMap,
    LogisticMap,
    create_chaotic_map,
)
from .lyapunov import AdaptiveLyapunovController, LyapunovAnalyzer

if TYPE_CHECKING:
    from ...data.synthetic_generator import VRPInstance
    from ..qubo.encoder import QUBOProblem

logger = logging.getLogger(__name__)


@dataclass
class QLNRSConfig:
    """Configuration for QLNRS algorithm."""

    # Iteration parameters
    max_iterations: int = 500
    time_limit_seconds: float = 300.0
    no_improvement_limit: int = 50

    # Destroy parameters
    destroy_fraction: float = 0.3
    min_destroy: int = 2
    max_destroy: int = 10

    # Repair parameters
    use_quantum_repair: bool = True
    num_quantum_samples: int = 10
    classical_repair_fallback: bool = True

    # Chaotic parameters
    chaotic_map_type: str = "logistic"
    chaotic_adaptation: bool = True
    initial_mle: float = 0.5

    # Lyapunov parameters
    target_lyapunov: float = 0.25
    lyapunov_window: int = 50
    adaptation_rate: float = 0.1

    # Acceptance parameters
    initial_temperature: float = 100.0
    final_temperature: float = 0.1
    cooling_rate: float = 0.995

    # Risk parameters
    risk_weight_tw: float = 0.3
    risk_weight_cap: float = 0.3
    risk_weight_op: float = 0.4

    # Seed
    seed: int | None = None


@dataclass
class QLNRSState:
    """State of the QLNRS algorithm."""

    current_solution: Solution
    best_solution: Solution
    current_cost: float
    best_cost: float
    iteration: int = 0
    temperature: float = 100.0
    no_improvement_count: int = 0

    # Lyapunov tracking
    lyapunov_exponent: float = 0.0
    lyapunov_history: list[float] = field(default_factory=list)

    # Chaotic state
    chaotic_mle: float = 0.5
    chaotic_value: float = 0.5

    # Statistics
    quantum_calls: int = 0
    classical_repair_calls: int = 0
    accepted_count: int = 0
    improved_count: int = 0
    solve_time_ms: float = 0.0

    # Operator tracking
    destroy_counts: dict[str, int] = field(default_factory=dict)
    repair_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class QLNRSResult:
    """Result from QLNRS algorithm."""

    best_solution: Solution
    best_cost: float
    iterations: int
    solve_time_ms: float

    # Statistics
    final_lyapunov: float
    lyapunov_history: list[float]
    cost_history: list[float]

    # Operator statistics
    operator_counts: dict[str, int]
    operator_scores: dict[str, float]

    # Quantum statistics
    quantum_calls: int
    classical_repair_calls: int

    metadata: dict[str, Any] = field(default_factory=dict)


class QLNRS:
    """Quantum Large Neighborhood Risk Search algorithm.

    Novel algorithm combining:
    1. Chaotic operator selection for adaptive destroy
    2. Quantum annealing for repair subproblems
    3. Risk-aware objective function
    4. Lyapunov-adaptive parameter control

    Algorithm Overview:
    -------------------
    For each iteration:
    1. Chaotic Operator Selection: Select destroy operator using chaotic map
    2. Destroy: Remove customers based on selected operator
    3. Quantum Repair: Formulate as QUBO, solve with quantum/classical solver
    4. Risk Evaluation: Compute risk-aware objective
    5. Accept: Simulated annealing acceptance
    6. Lyapunov Update: Compute λ from trajectory, adapt parameters
    """

    def __init__(
        self,
        config: QLNRSConfig | None = None,
        quantum_solver: Callable | None = None,
    ) -> None:
        """Initialize QLNRS solver.

        Args:
            config: Algorithm configuration
            quantum_solver: Optional quantum solver (uses SA if None)
        """
        self.config = config or QLNRSConfig()
        self.rng = random.Random(self.config.seed)
        self.np_rng = np.random.default_rng(self.config.seed)

        # Quantum solver
        self.quantum_solver = quantum_solver

        # Lyapunov controller
        self.lyapunov_controller = AdaptiveLyapunovController(
            initial_temperature=self.config.initial_temperature,
            initial_destroy_fraction=self.config.destroy_fraction,
            target_lyapunov=self.config.target_lyapunov,
            adaptation_rate=self.config.adaptation_rate,
        )

        # Initialize destroy operators
        from ...classical.lns_base import (
            RandomDestroy,
            WorstDestroy,
            RelatedDestroy,
            RiskWeightedDestroy,
        )

        self.destroy_operators = [
            RandomDestroy(),
            WorstDestroy(),
            RelatedDestroy(),
            RiskWeightedDestroy(
                alpha=self.config.risk_weight_tw,
                beta=self.config.risk_weight_cap,
                gamma=self.config.risk_weight_op,
            ),
        ]

        # Initialize repair operators
        self.repair_operators = [
            GreedyRepair(),
            RegretRepair(k=2),
            RegretRepair(k=3),
        ]

        # Chaotic operator selector
        self.chaotic_selector = ChaoticOperatorSelector(
            operators=self.destroy_operators,
            chaotic_map=create_chaotic_map(
                self.config.chaotic_map_type,
                self.config.initial_mle,
                self.config.seed,
            ),
            adaptation_mode="lyapunov" if self.config.chaotic_adaptation else "static",
        )

        # Import quantum components
        from ..qubo.encoder import VRP_QUBOEncoder, EncodingConfig
        from ..qubo.decomposition import ProblemDecomposer

        self.qubo_encoder = VRP_QUBOEncoder(
            config=EncodingConfig(max_qubits=40)
        )
        self.decomposer = ProblemDecomposer(max_qubits=40)

    def solve(
        self,
        instance: VRPInstance,
        initial_solution: Solution | None = None,
    ) -> QLNRSResult:
        """Solve the VRP instance using QLNRS.

        Args:
            instance: VRP instance to solve
            initial_solution: Optional initial solution (uses OR-Tools if None)

        Returns:
            QLNRSResult with best solution and statistics
        """
        start_time = time.perf_counter()

        # Get initial solution
        if initial_solution is None:
            initial_solution = self._get_initial_solution(instance)

        # Initialize state
        initial_cost = self._compute_cost(initial_solution, instance)
        state = QLNRSState(
            current_solution=initial_solution,
            best_solution=initial_solution,
            current_cost=initial_cost,
            best_cost=initial_cost,
            temperature=self.config.initial_temperature,
            chaotic_mle=self.config.initial_mle,
        )

        # Initialize tracking
        cost_history = [initial_cost]
        lyapunov_history = []

        # Main loop
        for iteration in range(self.config.max_iterations):
            state.iteration = iteration

            # Check time limit
            elapsed = time.perf_counter() - start_time
            if elapsed > self.config.time_limit_seconds:
                logger.info(f"Time limit reached at iteration {iteration}")
                break

            # Check no improvement limit
            if state.no_improvement_count >= self.config.no_improvement_limit:
                logger.info(f"No improvement limit reached at iteration {iteration}")
                break

            # Get current Lyapunov exponent
            lyapunov = self.lyapunov_controller.analyzer.compute_lyapunov().exponent
            state.lyapunov_exponent = lyapunov
            lyapunov_history.append(lyapunov)

            # --- Step 1: Chaotic Operator Selection ---
            destroy_op = self.chaotic_selector.select(lyapunov)

            # --- Step 2: Destroy ---
            num_remove = self._get_num_remove(instance, state)
            removed_customers, partial_routes = destroy_op.destroy(
                state.current_solution,
                num_remove,
                instance,
                self.rng,
            )

            # --- Step 3: Quantum Repair ---
            if self.config.use_quantum_repair:
                new_routes = self._quantum_repair(
                    instance, partial_routes, removed_customers, state
                )
            else:
                new_routes = self._classical_repair(
                    instance, partial_routes, removed_customers, state
                )

            # --- Step 4: Create new solution ---
            new_solution = self._build_solution(instance, new_routes)
            new_cost = self._compute_cost(new_solution, instance)

            # --- Step 5: Risk-aware evaluation ---
            risk_penalty = self._compute_risk(new_solution, instance)
            new_cost_with_risk = new_cost + risk_penalty

            # --- Step 6: Accept ---
            current_cost_with_risk = state.current_cost + self._compute_risk(
                state.current_solution, instance
            )

            if self._accept_solution(
                current_cost_with_risk, new_cost_with_risk, state.temperature
            ):
                state.current_solution = new_solution
                state.current_cost = new_cost
                state.accepted_count += 1

                # Track operator performance
                improvement = current_cost_with_risk - new_cost_with_risk
                self.chaotic_selector.update_performance(destroy_op, improvement)

                # Check for improvement
                if new_cost < state.best_cost:
                    state.best_solution = new_solution
                    state.best_cost = new_cost
                    state.improved_count += 1
                    state.no_improvement_count = 0
                else:
                    state.no_improvement_count += 1
            else:
                state.no_improvement_count += 1

            # --- Step 7: Lyapunov Update ---
            self.lyapunov_controller.update(
                state.current_solution, iteration, state.current_cost
            )

            # --- Step 8: Adapt chaotic map ---
            if self.config.chaotic_adaptation:
                self.chaotic_selector.adapt_chaotic_map(lyapunov)

            # --- Step 9: Cool down ---
            state.temperature *= self.config.cooling_rate

            # Track history
            cost_history.append(state.best_cost)

            # Update operator tracking
            if destroy_op.name not in state.destroy_counts:
                state.destroy_counts[destroy_op.name] = 0
            state.destroy_counts[destroy_op.name] += 1

        # Finalize
        end_time = time.perf_counter()
        state.solve_time_ms = (end_time - start_time) * 1000

        # Get final Lyapunov
        final_result = self.lyapunov_controller.analyzer.compute_lyapunov()

        return QLNRSResult(
            best_solution=state.best_solution,
            best_cost=state.best_cost,
            iterations=state.iteration + 1,
            solve_time_ms=state.solve_time_ms,
            final_lyapunov=final_result.exponent,
            lyapunov_history=lyapunov_history,
            cost_history=cost_history,
            operator_counts=dict(state.destroy_counts),
            operator_scores=dict(self.chaotic_selector.operator_scores),
            quantum_calls=state.quantum_calls,
            classical_repair_calls=state.classical_repair_calls,
            metadata={
                "config": vars(self.config),
                "chaotic_map": self.config.chaotic_map_type,
                "final_temperature": state.temperature,
            },
        )

    def _get_initial_solution(self, instance: VRPInstance) -> Solution:
        """Get initial solution using OR-Tools."""
        from ...classical.ortools_solver import ORToolsSolver, SolverConfig

        config = SolverConfig(
            time_limit_seconds=30,
            first_solution_strategy="PATH_CHEAPEST_ARC",
            local_search_metaheuristic="GUIDED_LOCAL_SEARCH",
        )
        solver = ORToolsSolver(config)
        result = solver.solve(instance)

        if result.solution is None:
            # Fallback to greedy
            return self._greedy_initial_solution(instance)

        return result.solution

    def _greedy_initial_solution(self, instance: VRPInstance) -> Solution:
        """Create greedy initial solution."""
        routes = [[0] for _ in range(instance.num_vehicles)]

        # Sort customers by distance to depot
        depot = instance.depot
        customer_distances = [
            (i + 1, instance.distance_matrix[0, i + 1])
            for i in range(instance.num_customers)
        ]
        customer_distances.sort(key=lambda x: x[1])

        for customer_id, _ in customer_distances:
            # Find best route
            best_route = 0
            best_cost = float("inf")

            for route_idx, route in enumerate(routes):
                if len(route) <= 1:
                    # Empty route, insert customer
                    cost = (
                        instance.distance_matrix[0, customer_id]
                        + instance.distance_matrix[customer_id, 0]
                    )
                else:
                    # Insert at end
                    last = route[-1]
                    cost = (
                        instance.distance_matrix[last, customer_id]
                        + instance.distance_matrix[customer_id, 0]
                        - instance.distance_matrix[last, 0]
                    )

                # Check capacity
                current_demand = sum(
                    instance.customers[n - 1].demand
                    for n in route[1:]
                    if n > 0
                )
                if current_demand + instance.customers[customer_id - 1].demand <= instance.vehicles[route_idx].capacity:
                    if cost < best_cost:
                        best_cost = cost
                        best_route = route_idx

            routes[best_route].append(customer_id)

        # Close routes
        for route in routes:
            route.append(0)

        return self._build_solution(instance, routes)

    def _get_num_remove(self, instance: VRPInstance, state: QLNRSState) -> int:
        """Determine number of customers to remove."""
        base_num = int(instance.num_customers * self.config.destroy_fraction)
        base_num = max(self.config.min_destroy, min(self.config.max_destroy, base_num))

        # Adapt based on Lyapunov
        if self.config.chaotic_adaptation:
            lyapunov = state.lyapunov_exponent
            # Higher λ = more chaotic = remove more
            # Lower λ = more stable = remove less
            adaptation = (lyapunov - self.config.target_lyapunov) / self.config.target_lyapunov
            adaptation = np.clip(adaptation, -0.5, 0.5)
            base_num = int(base_num * (1 + adaptation))

        return max(1, min(base_num, instance.num_customers - 2))

    def _quantum_repair(
        self,
        instance: VRPInstance,
        partial_routes: list[list[int]],
        removed_customers: list[int],
        state: QLNRSState,
    ) -> list[list[int]]:
        """Repair using quantum/classical solver."""
        state.quantum_calls += 1

        try:
            # Decompose if needed
            decomposition = self.decomposer.decompose(
                instance, removed_customers, partial_routes
            )

            all_routes = [list(r) for r in partial_routes]

            for subproblem in decomposition.subproblems:
                # Encode as QUBO
                qubo = self.qubo_encoder.encode_repair_subproblem(
                    instance, all_routes, subproblem.customers
                )

                # Solve
                if self.quantum_solver is not None:
                    result = self.quantum_solver(qubo)
                else:
                    # Use simulated annealing
                    from ..solvers.quantum_annealing import solve_qubo_sa
                    result = solve_qubo_sa(qubo, num_sweeps=5000)

                # Decode solution
                routes = self.qubo_encoder.decode_solution(
                    qubo, result.best_solution, all_routes, instance
                )

                # Update routes
                all_routes = routes

            return all_routes

        except Exception as e:
            logger.warning(f"Quantum repair failed: {e}, using classical fallback")
            state.classical_repair_calls += 1
            return self._classical_repair(instance, partial_routes, removed_customers, state)

    def _classical_repair(
        self,
        instance: VRPInstance,
        partial_routes: list[list[int]],
        removed_customers: list[int],
        state: QLNRSState,
    ) -> list[list[int]]:
        """Classical greedy repair."""
        state.classical_repair_calls += 1

        repair = self.repair_operators[0]  # Use greedy repair
        return repair.repair(removed_customers, partial_routes, instance, self.rng)

    def _compute_cost(self, solution: Solution, instance: VRPInstance) -> float:
        """Compute solution cost."""
        return solution.total_distance + 10.0 * solution.time_window_violations

    def _compute_risk(self, solution: Solution, instance: VRPInstance) -> float:
        """Compute risk-aware penalty."""
        risk = 0.0

        # Time window risk
        tw_violation = solution.time_window_violations
        risk += self.config.risk_weight_tw * tw_violation

        # Capacity risk
        cap_violation = solution.capacity_violations
        risk += self.config.risk_weight_cap * cap_violation

        # Operational risk (based on route characteristics)
        from ...data.geospatial import compute_route_statistics
        stats = compute_route_statistics(instance, solution.routes)

        # Risk from tight time windows
        avg_tw_width = np.mean([
            instance.customers[c - 1].time_window_end - instance.customers[c - 1].time_window_start
            for route in solution.routes
            for c in route[1:-1]
            if c > 0
        ]) if any(len(r) > 2 for r in solution.routes) else 60.0

        risk += self.config.risk_weight_op * (1.0 / (avg_tw_width + 1.0)) * 10.0

        return risk

    def _accept_solution(
        self,
        current_cost: float,
        new_cost: float,
        temperature: float,
    ) -> bool:
        """Acceptance criterion (Simulated Annealing)."""
        if new_cost < current_cost:
            return True

        delta = new_cost - current_cost
        probability = np.exp(-delta / max(temperature, 0.01))

        return self.rng.random() < probability

    def _build_solution(
        self,
        instance: VRPInstance,
        routes: list[list[int]],
    ) -> Solution:
        """Build Solution object from routes."""
        from ...data.geospatial import compute_route_statistics

        stats = compute_route_statistics(instance, routes)

        return Solution(
            instance_name=instance.name,
            routes=routes,
            total_distance=stats["total_distance"],
            total_time=stats["total_time"],
            total_demand_served=stats["total_demand_served"],
            time_window_violations=stats["total_tw_violation"],
            capacity_violations=stats.get("total_capacity_violation", 0.0),
        )


def solve_qlnrs(
    instance: VRPInstance,
    max_iterations: int = 500,
    use_quantum: bool = True,
    seed: int | None = None,
) -> QLNRSResult:
    """Convenience function to solve VRP with QLNRS.

    Args:
        instance: VRP instance
        max_iterations: Maximum iterations
        use_quantum: Whether to use quantum repair
        seed: Random seed

    Returns:
        QLNRSResult
    """
    config = QLNRSConfig(
        max_iterations=max_iterations,
        use_quantum_repair=use_quantum,
        seed=seed,
    )
    solver = QLNRS(config)
    return solver.solve(instance)