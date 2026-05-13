"""
Benchmark Framework for VRP Solvers.

Provides framework for comparing different VRP algorithms.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

if TYPE_CHECKING:
    from ..data.synthetic_generator import VRPInstance
    from ..data.problem_builder import Solution

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking."""

    # Solver configurations
    solvers: list[str] = field(
        default_factory=lambda: ["ortools", "lns", "qlnrs"]
    )

    # Instance configurations
    num_customers: list[int] = field(
        default_factory=lambda: [10, 20, 30]
    )
    num_vehicles: list[int] = field(
        default_factory=lambda: [3, 5, 5]
    )
    seeds: list[int] = field(
        default_factory=lambda: [42, 123, 456]
    )

    # Time limits
    time_limit_seconds: float = 60.0

    # Repetitions
    num_runs: int = 3

    # Output
    output_dir: str = "results"
    save_solutions: bool = True


@dataclass
class SolverResult:
    """Result from a single solver run."""

    solver_name: str
    instance_name: str
    run_id: int

    # Solution
    total_distance: float
    total_time: float
    num_routes: int
    time_window_violations: float
    capacity_violations: float

    # Timing
    solve_time_ms: float

    # Statistics
    lyapunov_exponent: float | None = None
    quantum_calls: int = 0
    iterations: int = 0

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """Aggregated benchmark results."""

    solver_name: str
    instance_name: str

    # Aggregated metrics
    avg_distance: float
    std_distance: float
    avg_time: float
    std_time: float
    avg_solve_time_ms: float

    # Best solution
    best_distance: float
    best_solution: Solution | None = None

    # Statistics
    num_runs: int
    success_rate: float

    all_results: list[SolverResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "solver_name": self.solver_name,
            "instance_name": self.instance_name,
            "avg_distance": self.avg_distance,
            "std_distance": self.std_distance,
            "avg_time": self.avg_time,
            "std_time": self.std_time,
            "avg_solve_time_ms": self.avg_solve_time_ms,
            "best_distance": self.best_distance,
            "num_runs": self.num_runs,
            "success_rate": self.success_rate,
        }


class BenchmarkRunner:
    """Runs benchmarks for VRP solvers."""

    def __init__(self, config: BenchmarkConfig | None = None) -> None:
        """Initialize benchmark runner."""
        self.config = config or BenchmarkConfig()
        self.results: dict[str, list[SolverResult]] = {}

    def run(
        self,
        instances: list[VRPInstance] | None = None,
        solvers: dict[str, Callable] | None = None,
    ) -> dict[str, dict[str, BenchmarkResult]]:
        """Run benchmarks.

        Args:
            instances: List of VRP instances (generates if None)
            solvers: Dictionary of solver name to solver function

        Returns:
            Dictionary of solver -> instance -> BenchmarkResult
        """
        # Generate instances if not provided
        if instances is None:
            instances = self._generate_instances()

        # Use default solvers if not provided
        if solvers is None:
            solvers = self._get_default_solvers()

        # Create output directory
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Run benchmarks
        all_results: dict[str, dict[str, BenchmarkResult]] = {}

        for solver_name, solver_fn in solvers.items():
            all_results[solver_name] = {}

            for instance in instances:
                instance_results = []

                for run_id in range(self.config.num_runs):
                    logger.info(
                        f"Running {solver_name} on {instance.name} "
                        f"(run {run_id + 1}/{self.config.num_runs})"
                    )

                    try:
                        result = self._run_solver(
                            solver_name, solver_fn, instance, run_id
                        )
                        instance_results.append(result)

                        if self.config.save_solutions and result.metadata.get("solution"):
                            self._save_solution(
                                result, instance, output_dir
                            )

                    except Exception as e:
                        logger.error(
                            f"Error running {solver_name} on {instance.name}: {e}"
                        )

                # Aggregate results
                if instance_results:
                    aggregated = self._aggregate_results(
                        solver_name, instance.name, instance_results
                    )
                    all_results[solver_name][instance.name] = aggregated

        # Save results
        self._save_results(all_results, output_dir)

        return all_results

    def _generate_instances(self) -> list[VRPInstance]:
        """Generate benchmark instances."""
        from ..data.synthetic_generator import TokyoSMEGenerator

        instances = []
        for seed in self.config.seeds:
            for n_cust, n_veh in zip(
                self.config.num_customers, self.config.num_vehicles
            ):
                generator = TokyoSMEGenerator(
                    seed=seed,
                    num_customers=n_cust,
                    num_vehicles=n_veh,
                )
                instances.append(generator.generate())

        return instances

    def _get_default_solvers(self) -> dict[str, Callable]:
        """Get default solver functions."""
        solvers = {}

        # OR-Tools baseline
        def ortools_solve(instance):
            from ..classical.ortools_solver import ORToolsSolver, SolverConfig

            config = SolverConfig(time_limit_seconds=int(self.config.time_limit_seconds))
            solver = ORToolsSolver(config)
            result = solver.solve(instance)
            return {
                "solution": result.solution,
                "total_distance": result.total_distance if result.solution else float("inf"),
                "total_time": result.total_time if result.solution else 0,
                "solve_time_ms": result.solve_time_ms,
                "metadata": {"status": result.status},
            }

        solvers["ortools"] = ortools_solve

        # Classical LNS
        def lns_solve(instance):
            from ..classical.lns_base import LNSBase, LNSConfig
            from ..classical.ortools_solver import ORToolsSolver, SolverConfig

            # Get initial solution from OR-Tools
            ortools_config = SolverConfig(time_limit_seconds=30)
            ortools = ORToolsSolver(ortools_config)
            initial = ortools.solve(instance)

            config = LNSConfig(
                max_iterations=500,
                time_limit_seconds=self.config.time_limit_seconds,
            )
            lns = LNSBase(config)

            if initial.solution is None:
                return {
                    "solution": None,
                    "total_distance": float("inf"),
                    "total_time": 0,
                    "solve_time_ms": 0,
                    "metadata": {"status": "no_initial_solution"},
                }

            # Run LNS (simplified - would need full implementation)
            return {
                "solution": initial.solution,
                "total_distance": initial.total_distance,
                "total_time": initial.total_time,
                "solve_time_ms": initial.solve_time_ms,
                "metadata": {"status": "lns_placeholder"},
            }

        solvers["lns"] = lns_solve

        # QLNRS
        def qlnrs_solve(instance):
            from ..quantum.qlnrs.algorithm import QLNRS, QLNRSConfig
            from ..classical.ortools_solver import ORToolsSolver, SolverConfig

            # Get initial solution
            ortools_config = SolverConfig(time_limit_seconds=30)
            ortools = ORToolsSolver(ortools_config)
            initial = ortools.solve(instance)

            config = QLNRSConfig(
                max_iterations=500,
                time_limit_seconds=self.config.time_limit_seconds,
                use_quantum_repair=False,  # Use SA fallback
            )
            qlnrs = QLNRS(config)

            result = qlnrs.solve(instance, initial.solution)

            return {
                "solution": result.best_solution,
                "total_distance": result.best_cost,
                "total_time": result.best_solution.total_time if result.best_solution else 0,
                "solve_time_ms": result.solve_time_ms,
                "lyapunov_exponent": result.final_lyapunov,
                "iterations": result.iterations,
                "metadata": {
                    "quantum_calls": result.quantum_calls,
                    "lyapunov_history": result.lyapunov_history[-10:] if result.lyapunov_history else [],
                },
            }

        solvers["qlnrs"] = qlnrs_solve

        return solvers

    def _run_solver(
        self,
        solver_name: str,
        solver_fn: Callable,
        instance: VRPInstance,
        run_id: int,
    ) -> SolverResult:
        """Run a single solver."""
        start_time = time.perf_counter()

        result_dict = solver_fn(instance)

        end_time = time.perf_counter()

        solution = result_dict.get("solution")
        metadata = result_dict.get("metadata", {})

        return SolverResult(
            solver_name=solver_name,
            instance_name=instance.name,
            run_id=run_id,
            total_distance=result_dict.get("total_distance", float("inf")),
            total_time=result_dict.get("total_time", 0),
            num_routes=len(solution.routes) if solution else 0,
            time_window_violations=solution.time_window_violations if solution else 0,
            capacity_violations=solution.capacity_violations if solution else 0,
            solve_time_ms=result_dict.get("solve_time_ms", (end_time - start_time) * 1000),
            lyapunov_exponent=result_dict.get("lyapunov_exponent"),
            quantum_calls=result_dict.get("quantum_calls", 0),
            iterations=result_dict.get("iterations", 0),
            metadata=metadata,
        )

    def _aggregate_results(
        self,
        solver_name: str,
        instance_name: str,
        results: list[SolverResult],
    ) -> BenchmarkResult:
        """Aggregate results from multiple runs."""
        distances = [r.total_distance for r in results]
        times = [r.total_time for r in results]
        solve_times = [r.solve_time_ms for r in results]

        # Success rate
        success_count = sum(1 for r in results if r.total_distance < float("inf"))
        success_rate = success_count / len(results) if results else 0

        # Best solution
        best_idx = np.argmin(distances)
        best_result = results[best_idx]

        return BenchmarkResult(
            solver_name=solver_name,
            instance_name=instance_name,
            avg_distance=float(np.mean(distances)),
            std_distance=float(np.std(distances)),
            avg_time=float(np.mean(times)),
            std_time=float(np.std(times)),
            avg_solve_time_ms=float(np.mean(solve_times)),
            best_distance=float(min(distances)),
            num_runs=len(results),
            success_rate=success_rate,
            all_results=results,
        )

    def _save_solution(
        self,
        result: SolverResult,
        instance: VRPInstance,
        output_dir: Path,
    ) -> None:
        """Save solution to file."""
        if result.metadata.get("solution") is None:
            return

        filename = f"{result.solver_name}_{result.instance_name}_run{result.run_id}.json"
        filepath = output_dir / filename

        solution = result.metadata["solution"]
        data = {
            "solver": result.solver_name,
            "instance": result.instance_name,
            "run_id": result.run_id,
            "total_distance": result.total_distance,
            "routes": solution.routes if solution else [],
            "solve_time_ms": result.solve_time_ms,
        }

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def _save_results(
        self,
        all_results: dict[str, dict[str, BenchmarkResult]],
        output_dir: Path,
    ) -> None:
        """Save aggregated results."""
        filepath = output_dir / "benchmark_results.json"

        data = {}
        for solver_name, instance_results in all_results.items():
            data[solver_name] = {}
            for instance_name, result in instance_results.items():
                data[solver_name][instance_name] = result.to_dict()

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

        # Also save summary
        summary_path = output_dir / "summary.txt"
        with open(summary_path, "w") as f:
            f.write("VRP Benchmark Results\n")
            f.write("=" * 50 + "\n\n")

            for solver_name, instance_results in all_results.items():
                f.write(f"\n{solver_name.upper()}\n")
                f.write("-" * 30 + "\n")

                for instance_name, result in instance_results.items():
                    f.write(f"{instance_name}:\n")
                    f.write(f"  Avg Distance: {result.avg_distance:.2f} ± {result.std_distance:.2f}\n")
                    f.write(f"  Best Distance: {result.best_distance:.2f}\n")
                    f.write(f"  Solve Time: {result.avg_solve_time_ms:.0f}ms\n")
                    f.write(f"  Success Rate: {result.success_rate * 100:.1f}%\n\n")