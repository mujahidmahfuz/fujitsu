"""
Benchmark Suite for Quantum VRP Solvers.

Compares all solver methods across problem sizes:
- Classical brute-force (exact baseline)
- OR-Tools heuristic
- QAOA (variational quantum)
- Grover Adaptive Search (exact quantum)
- Hybrid decomposition
- Circuit cutting

Outputs tables and visualization data for the final report.
"""

import numpy as np
import time
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from src.qubo.vrp_qubo import VRPQuboBuilder, VRPInstance
from src.solvers.classical_baseline import ClassicalBaseline
from src.solvers.grover_solver import GroverAdaptiveSearch, QAOAGroverHybrid
from src.solvers.hybrid_solver import HybridSolver
from src.solvers.circuit_cutting import QUBOPartitioner, CircuitCuttingExecutor


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    solver_name: str
    n_stops: int
    n_qubits: int
    cost: float
    runtime_seconds: float
    feasible: bool
    optimality_gap_pct: float  # vs brute-force
    method_detail: str


@dataclass
class BenchmarkReport:
    """Complete benchmark report."""
    results: List[BenchmarkResult]
    problem_sizes: List[int]
    timestamp: str
    system_info: Dict


class VRPBenchmark:
    """Run comprehensive benchmarks across solvers and problem sizes."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def generate_instance(self, n_stops: int) -> VRPInstance:
        """Generate a random VRP instance."""
        n = n_stops + 1  # depot + stops
        dm = self.rng.uniform(5, 50, size=(n, n))
        dm = (dm + dm.T) / 2  # Symmetric
        np.fill_diagonal(dm, 0)
        demands = self.rng.randint(1, 4, size=n_stops)
        capacity = max(int(demands.sum() * 1.2), n_stops * 2)

        return VRPInstance(
            n_stops=n_stops,
            distance_matrix=dm,
            demands=demands,
            capacity=capacity,
        )

    def run_benchmark(
        self,
        sizes: List[int] = None,
        solvers: List[str] = None,
        n_trials: int = 1,
    ) -> BenchmarkReport:
        """Run full benchmark suite.

        Args:
            sizes: Problem sizes to test (default: [2, 3, 4, 5]).
            solvers: Solver names to test (default: all).
            n_trials: Number of trials per config.

        Returns:
            BenchmarkReport with all results.
        """
        if sizes is None:
            sizes = [2, 3, 4, 5]
        if solvers is None:
            solvers = [
                "brute_force", "ortools",
                "grover_gas", "qaoa_grover",
                "hybrid", "circuit_cutting",
            ]

        results = []

        for n_stops in sizes:
            for trial in range(n_trials):
                instance = self.generate_instance(n_stops)

                # Always get brute-force baseline first
                bf_result = self._run_brute_force(instance)
                bf_cost = bf_result.cost if bf_result.feasible else float('inf')

                for solver_name in solvers:
                    try:
                        result = self._run_solver(
                            solver_name, instance, bf_cost
                        )
                        results.append(result)
                    except Exception as e:
                        results.append(BenchmarkResult(
                            solver_name=solver_name,
                            n_stops=n_stops,
                            n_qubits=0,
                            cost=float('inf'),
                            runtime_seconds=0,
                            feasible=False,
                            optimality_gap_pct=float('inf'),
                            method_detail=f"Error: {str(e)[:100]}",
                        ))

        return BenchmarkReport(
            results=results,
            problem_sizes=sizes,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            system_info=self._get_system_info(),
        )

    def _run_solver(
        self, name: str, instance: VRPInstance, bf_cost: float
    ) -> BenchmarkResult:
        """Run a single solver."""
        dispatch = {
            "brute_force": self._run_brute_force,
            "ortools": self._run_ortools,
            "grover_gas": self._run_grover,
            "qaoa_grover": self._run_qaoa_grover,
            "hybrid": self._run_hybrid,
            "circuit_cutting": self._run_circuit_cutting,
        }

        runner = dispatch.get(name)
        if runner is None:
            raise ValueError(f"Unknown solver: {name}")

        result = runner(instance)

        if bf_cost > 0 and result.feasible:
            result.optimality_gap_pct = (
                (result.cost - bf_cost) / bf_cost * 100
            )

        return result

    def _run_brute_force(self, instance: VRPInstance) -> BenchmarkResult:
        t0 = time.time()
        baseline = ClassicalBaseline(instance)
        result = baseline.brute_force_tsp()
        elapsed = time.time() - t0

        return BenchmarkResult(
            solver_name="brute_force",
            n_stops=instance.n_stops,
            n_qubits=0,
            cost=result.total_cost,
            runtime_seconds=elapsed,
            feasible=True,
            optimality_gap_pct=0.0,
            method_detail="Classical exact enumeration",
        )

    def _run_ortools(self, instance: VRPInstance) -> BenchmarkResult:
        t0 = time.time()
        baseline = ClassicalBaseline(instance)
        try:
            result = baseline.solve_ortools_heuristic(time_limit_seconds=10)
            elapsed = time.time() - t0
            return BenchmarkResult(
                solver_name="ortools",
                n_stops=instance.n_stops,
                n_qubits=0,
                cost=result.total_cost,
                runtime_seconds=elapsed,
                feasible=result.solver_status != "NO_SOLUTION",
                optimality_gap_pct=0.0,
                method_detail="OR-Tools guided local search",
            )
        except Exception:
            elapsed = time.time() - t0
            return BenchmarkResult(
                solver_name="ortools",
                n_stops=instance.n_stops,
                n_qubits=0,
                cost=float('inf'),
                runtime_seconds=elapsed,
                feasible=False,
                optimality_gap_pct=0.0,
                method_detail="OR-Tools not available",
            )

    def _run_grover(self, instance: VRPInstance) -> BenchmarkResult:
        builder = VRPQuboBuilder(instance, encoding='position')
        qubo = builder.build()

        t0 = time.time()
        gas = GroverAdaptiveSearch(max_gas_iterations=10, seed=self.seed)
        gas_result = gas.solve(qubo)
        elapsed = time.time() - t0

        # Evaluate the solution
        eval_result = builder.evaluate_solution(gas_result.optimal_bitstring)

        return BenchmarkResult(
            solver_name="grover_gas",
            n_stops=instance.n_stops,
            n_qubits=qubo.n_qubits,
            cost=eval_result["cost"] if eval_result["feasible"] else gas_result.optimal_cost,
            runtime_seconds=elapsed,
            feasible=eval_result["feasible"],
            optimality_gap_pct=0.0,
            method_detail=f"GAS {gas_result.n_iterations} iter, {qubo.n_qubits}q",
        )

    def _run_qaoa_grover(self, instance: VRPInstance) -> BenchmarkResult:
        builder = VRPQuboBuilder(instance, encoding='position')
        qubo = builder.build()

        t0 = time.time()
        hybrid = QAOAGroverHybrid(max_gas_iterations=10, seed=self.seed)
        result = hybrid.solve(qubo)
        elapsed = time.time() - t0

        eval_result = builder.evaluate_solution(result["optimal_bitstring"])

        return BenchmarkResult(
            solver_name="qaoa_grover",
            n_stops=instance.n_stops,
            n_qubits=qubo.n_qubits,
            cost=eval_result["cost"] if eval_result["feasible"] else result["optimal_cost"],
            runtime_seconds=elapsed,
            feasible=eval_result["feasible"],
            optimality_gap_pct=0.0,
            method_detail=f"QAOA→GAS, {result['n_gas_iterations']} iter",
        )

    def _run_hybrid(self, instance: VRPInstance) -> BenchmarkResult:
        t0 = time.time()
        solver = HybridSolver(max_stops_per_quantum=3, seed=self.seed)
        result = solver.solve(instance, use_quantum=True)
        elapsed = time.time() - t0

        return BenchmarkResult(
            solver_name="hybrid",
            n_stops=instance.n_stops,
            n_qubits=0,
            cost=result.total_cost,
            runtime_seconds=elapsed,
            feasible=True,
            optimality_gap_pct=0.0,
            method_detail=f"Hybrid: {result.n_vehicles} vehicles",
        )

    def _run_circuit_cutting(self, instance: VRPInstance) -> BenchmarkResult:
        builder = VRPQuboBuilder(instance, encoding='position')
        qubo = builder.build()

        t0 = time.time()
        partitioner = QUBOPartitioner(max_fragment_qubits=10)
        cut_result = partitioner.partition(qubo.Q, n_fragments=2)

        executor = CircuitCuttingExecutor(seed=self.seed)
        cc_result = executor.solve_fragments(cut_result)
        elapsed = time.time() - t0

        eval_result = builder.evaluate_solution(cc_result["bitstring"])

        return BenchmarkResult(
            solver_name="circuit_cutting",
            n_stops=instance.n_stops,
            n_qubits=qubo.n_qubits,
            cost=eval_result["cost"] if eval_result["feasible"] else cc_result["total_cost"],
            runtime_seconds=elapsed,
            feasible=eval_result["feasible"],
            optimality_gap_pct=0.0,
            method_detail=f"{cut_result.n_cuts} cuts, {cc_result['overhead_factor']}x overhead",
        )

    def _get_system_info(self) -> Dict:
        import platform
        return {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        }

    def save_report(self, report: BenchmarkReport, filepath: str):
        """Save benchmark report to JSON."""
        data = {
            "problem_sizes": report.problem_sizes,
            "timestamp": report.timestamp,
            "system_info": report.system_info,
            "results": [asdict(r) for r in report.results],
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def format_table(self, report: BenchmarkReport) -> str:
        """Format benchmark results as a readable table."""
        lines = []
        lines.append("╔═══════════════════╦════════╦════════╦═══════════╦═══════════╦══════════╗")
        lines.append("║ Solver            ║ Stops  ║ Qubits ║ Cost      ║ Time (ms) ║ Gap %    ║")
        lines.append("╠═══════════════════╬════════╬════════╬═══════════╬═══════════╬══════════╣")

        for r in sorted(report.results, key=lambda x: (x.n_stops, x.solver_name)):
            name = r.solver_name[:17].ljust(17)
            stops = str(r.n_stops).center(6)
            qubits = str(r.n_qubits).center(6)
            cost = f"{r.cost:.1f}" if r.cost < 1e6 else "N/A"
            cost = cost[:9].rjust(9)
            time_ms = f"{r.runtime_seconds * 1000:.1f}".rjust(9)
            gap = f"{r.optimality_gap_pct:.1f}%" if r.feasible else "N/A"
            gap = gap.rjust(8)
            lines.append(f"║ {name} ║ {stops} ║ {qubits} ║ {cost} ║ {time_ms} ║ {gap} ║")

        lines.append("╚═══════════════════╩════════╩════════╩═══════════╩═══════════╩══════════╝")
        return "\n".join(lines)
