"""
Penalty Tuning & Success Probability Analysis.

Systematically evaluates how penalty weights affect:
1. Feasibility rate — % of QUBO ground states that satisfy VRP constraints
2. Solution quality — gap between QUBO optimum and true classical optimum
3. Energy landscape — spectral gap that QAOA/Grover can exploit

This is a critical step for the competition: poor penalties → quantum
solver finds infeasible or suboptimal solutions, regardless of hardware.
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
import json

from src.qubo.vrp_qubo import VRPQuboBuilder, VRPInstance, QUBOResult
from src.qubo.penalty_calibration import PenaltyCalibrator
from src.solvers.grover_solver import GroverAdaptiveSearch


@dataclass
class TuningResult:
    """Result from a single penalty configuration."""
    penalty_multiplier: float
    penalties: Dict[str, float]
    qubo_optimal_energy: float
    qubo_optimal_feasible: bool
    qubo_optimal_cost: float
    classical_optimal_cost: float
    optimality_gap_pct: float
    feasibility_rate: float  # % of low-energy states that are feasible
    spectral_gap: float      # Energy gap between best feasible and best infeasible
    n_feasible_states: int
    n_total_states: int
    success_probability: float  # P(measuring a feasible optimal state)


@dataclass
class TuningReport:
    """Full tuning analysis report."""
    instance_name: str
    n_stops: int
    n_qubits: int
    encoding: str
    results: List[TuningResult]
    best_multiplier: float
    auto_calibrated: Dict[str, float]
    eigenvalue_calibrated: Dict[str, float]
    shot_noise_bound: float


class PenaltyTuningAnalysis:
    """Comprehensive penalty tuning and success probability analysis.

    Runs a systematic sweep of penalty weights and evaluates the
    trade-off between constraint satisfaction and solution quality.
    """

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def analyze(
        self,
        instance: VRPInstance,
        encoding: str = "position",
        multiplier_range: np.ndarray = None,
        instance_name: str = "test",
    ) -> TuningReport:
        """Run complete penalty tuning analysis.

        Args:
            instance: VRP instance to analyze.
            encoding: "position" or "route".
            multiplier_range: Penalty multipliers to sweep.
            instance_name: Name for the report.

        Returns:
            TuningReport with analysis results.
        """
        if multiplier_range is None:
            multiplier_range = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0])

        # Get calibrated baselines
        calibrator = PenaltyCalibrator(instance, encoding)
        auto_penalties = VRPQuboBuilder(instance, encoding)._auto_calibrate_penalties()
        eig_penalties = calibrator.eigenvalue_gap_calibration()
        shot_bound = calibrator.shot_noise_bound(n_shots=8192)

        # Get classical optimal
        classical_opt = self._get_classical_optimal(instance)

        # Sweep multipliers
        results = []
        for mult in multiplier_range:
            scaled = {k: v * mult for k, v in auto_penalties.items()}
            result = self._evaluate_penalty(
                instance, encoding, scaled, mult, classical_opt
            )
            results.append(result)

        # Find best multiplier (highest success prob with feasible optimum)
        feasible_results = [r for r in results if r.qubo_optimal_feasible]
        if feasible_results:
            best = min(feasible_results, key=lambda r: r.optimality_gap_pct)
            best_mult = best.penalty_multiplier
        else:
            best_mult = 1.0

        n_qubits = VRPQuboBuilder(instance, encoding).build().n_qubits

        return TuningReport(
            instance_name=instance_name,
            n_stops=instance.n_stops,
            n_qubits=n_qubits,
            encoding=encoding,
            results=results,
            best_multiplier=best_mult,
            auto_calibrated=auto_penalties,
            eigenvalue_calibrated=eig_penalties,
            shot_noise_bound=shot_bound,
        )

    def _evaluate_penalty(
        self,
        instance: VRPInstance,
        encoding: str,
        penalties: Dict[str, float],
        multiplier: float,
        classical_opt: float,
    ) -> TuningResult:
        """Evaluate a specific penalty configuration."""
        builder = VRPQuboBuilder(instance, encoding=encoding, penalties=penalties)
        qubo = builder.build()
        n = qubo.n_qubits
        N = 2 ** n

        # Skip if too large
        if n > 20:
            return TuningResult(
                penalty_multiplier=multiplier,
                penalties=penalties,
                qubo_optimal_energy=float('inf'),
                qubo_optimal_feasible=False,
                qubo_optimal_cost=float('inf'),
                classical_optimal_cost=classical_opt,
                optimality_gap_pct=float('inf'),
                feasibility_rate=0.0,
                spectral_gap=0.0,
                n_feasible_states=0,
                n_total_states=N,
                success_probability=0.0,
            )

        Q = qubo.Q
        Q_sym = (Q + Q.T) / 2.0

        # Evaluate all states
        best_energy = float('inf')
        best_bits = None
        best_feasible_energy = float('inf')
        best_feasible_bits = None
        best_infeasible_energy = float('inf')

        feasible_count = 0
        feasible_optimal_count = 0
        energies = []

        for idx in range(N):
            bits = np.array([(idx >> k) & 1 for k in range(n)], dtype=float)
            energy = float(bits @ Q_sym @ bits)
            energies.append(energy)

            eval_result = builder.evaluate_solution(bits)

            if energy < best_energy:
                best_energy = energy
                best_bits = bits.copy()

            if eval_result['feasible']:
                feasible_count += 1
                if energy < best_feasible_energy:
                    best_feasible_energy = energy
                    best_feasible_bits = bits.copy()
            else:
                if energy < best_infeasible_energy:
                    best_infeasible_energy = energy

        # Determine if QUBO ground state is feasible
        qubo_gs_eval = builder.evaluate_solution(best_bits)
        qubo_gs_feasible = qubo_gs_eval['feasible']
        qubo_gs_cost = qubo_gs_eval['cost'] if qubo_gs_feasible else float('inf')

        # Optimality gap
        if classical_opt > 0 and qubo_gs_cost < float('inf'):
            gap_pct = (qubo_gs_cost - classical_opt) / classical_opt * 100
        else:
            gap_pct = float('inf')

        # Spectral gap: energy difference between best feasible and best infeasible
        if best_feasible_energy < float('inf') and best_infeasible_energy < float('inf'):
            spectral_gap = best_infeasible_energy - best_feasible_energy
        else:
            spectral_gap = 0.0

        # Success probability: fraction of states within 10% of ground state
        # that are feasible (approximates QAOA/Grover sampling probability)
        energy_threshold = best_energy + abs(best_energy) * 0.1 + 1e-10
        low_energy_total = sum(1 for e in energies if e <= energy_threshold)
        low_energy_feasible = 0
        for idx in range(N):
            if energies[idx] <= energy_threshold:
                bits = np.array([(idx >> k) & 1 for k in range(n)], dtype=float)
                ev = builder.evaluate_solution(bits)
                if ev['feasible']:
                    low_energy_feasible += 1

        success_prob = low_energy_feasible / max(low_energy_total, 1)

        return TuningResult(
            penalty_multiplier=multiplier,
            penalties=penalties,
            qubo_optimal_energy=best_energy,
            qubo_optimal_feasible=qubo_gs_feasible,
            qubo_optimal_cost=qubo_gs_cost,
            classical_optimal_cost=classical_opt,
            optimality_gap_pct=gap_pct,
            feasibility_rate=feasible_count / N,
            spectral_gap=spectral_gap,
            n_feasible_states=feasible_count,
            n_total_states=N,
            success_probability=success_prob,
        )

    def _get_classical_optimal(self, instance: VRPInstance) -> float:
        """Get classical optimal cost via brute force."""
        from src.solvers.classical_baseline import ClassicalBaseline
        baseline = ClassicalBaseline(instance)
        result = baseline.brute_force_tsp()
        return result.total_cost

    def format_report(self, report: TuningReport) -> str:
        """Format tuning report as readable text."""
        lines = []
        lines.append(f"{'='*70}")
        lines.append(f"Penalty Tuning Report: {report.instance_name}")
        lines.append(f"  {report.n_stops} stops, {report.n_qubits} qubits, {report.encoding} encoding")
        lines.append(f"  Shot noise bound: {report.shot_noise_bound:.2f}")
        lines.append(f"  Best multiplier: {report.best_multiplier}")
        lines.append(f"{'='*70}")

        lines.append("")
        lines.append(f"{'Mult':>6} | {'QUBO GS':>9} | {'Feas?':>5} | {'Cost':>8} | "
                      f"{'Gap%':>7} | {'Feas%':>6} | {'P(succ)':>7} | {'Δ(spec)':>8}")
        lines.append("-" * 75)

        for r in report.results:
            feas = "✅" if r.qubo_optimal_feasible else "❌"
            cost = f"{r.qubo_optimal_cost:.0f}" if r.qubo_optimal_cost < float('inf') else "N/A"
            gap = f"{r.optimality_gap_pct:.1f}%" if r.optimality_gap_pct < float('inf') else "N/A"
            lines.append(
                f"{r.penalty_multiplier:>6.2f} | "
                f"{r.qubo_optimal_energy:>9.1f} | "
                f"  {feas}  | "
                f"{cost:>8} | "
                f"{gap:>7} | "
                f"{r.feasibility_rate*100:>5.1f}% | "
                f"{r.success_probability:>6.1%} | "
                f"{r.spectral_gap:>8.1f}"
            )

        lines.append("")
        lines.append("Key:")
        lines.append("  Mult    = Penalty multiplier (relative to auto-calibration)")
        lines.append("  QUBO GS = Ground state energy of QUBO")
        lines.append("  Feas?   = Is the QUBO ground state a feasible VRP solution?")
        lines.append("  Cost    = Route cost of QUBO ground state (meters)")
        lines.append("  Gap%    = Optimality gap vs classical brute-force")
        lines.append("  Feas%   = % of all states that are feasible")
        lines.append("  P(succ) = Probability of sampling a feasible optimal state")
        lines.append("  Δ(spec) = Spectral gap (best infeasible - best feasible energy)")
        lines.append("")
        lines.append("Recommendation:")

        best = next((r for r in report.results
                      if r.penalty_multiplier == report.best_multiplier), None)
        if best and best.qubo_optimal_feasible:
            lines.append(f"  Use penalty multiplier {report.best_multiplier:.1f}")
            lines.append(f"  Expected success probability: {best.success_probability:.1%}")
            lines.append(f"  Expected optimality gap: {best.optimality_gap_pct:.1f}%")
        else:
            lines.append("  ⚠ No multiplier found with feasible QUBO ground state!")
            lines.append("  Consider larger penalties or different encoding.")

        return "\n".join(lines)

    def save_report(self, report: TuningReport, filepath: str):
        """Save report to JSON."""
        from dataclasses import asdict
        data = {
            "instance_name": report.instance_name,
            "n_stops": report.n_stops,
            "n_qubits": report.n_qubits,
            "encoding": report.encoding,
            "best_multiplier": report.best_multiplier,
            "auto_calibrated": report.auto_calibrated,
            "eigenvalue_calibrated": report.eigenvalue_calibrated,
            "shot_noise_bound": report.shot_noise_bound,
            "results": [asdict(r) for r in report.results],
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
