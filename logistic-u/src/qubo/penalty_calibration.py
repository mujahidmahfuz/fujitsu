"""
Penalty Weight Calibration for VRP QUBO.

The quality of QUBO solutions critically depends on penalty weights.
Too low: constraint violations dominate optimal solutions.
Too high: energy landscape becomes flat, QAOA can't distinguish good from bad.

This module provides:
1. Eigenvalue-gap-based calibration
2. Shot-noise-aware bounds
3. Sweep-based analysis tools
"""

import numpy as np
from typing import Dict, List, Tuple
from src.qubo.vrp_qubo import VRPInstance, VRPQuboBuilder, QUBOResult


class PenaltyCalibrator:
    """Automatic penalty weight calibration for VRP QUBO."""

    def __init__(self, instance: VRPInstance, encoding: str = "position"):
        self.instance = instance
        self.encoding = encoding

    def eigenvalue_gap_calibration(self) -> Dict[str, float]:
        """Calibrate penalties using eigenvalue gap analysis.

        Strategy:
        1. Build objective-only QUBO (no constraints)
        2. Compute eigenvalue range of objective
        3. Set penalties such that violating any constraint costs more
           than the worst-to-best objective gap

        Returns:
            Calibrated penalty weights.
        """
        # Build objective-only QUBO
        builder = VRPQuboBuilder(
            self.instance,
            encoding=self.encoding,
            penalties={'visit': 0, 'flow': 0, 'capacity': 0, 'timewindow': 0}
        )

        qubo = builder.build()
        Q_obj = qubo.Q + qubo.Q.T - np.diag(np.diag(qubo.Q))

        # Eigenvalue range of the objective
        if qubo.n_qubits <= 16:
            eigenvalues = np.linalg.eigvalsh(Q_obj)
            eig_range = eigenvalues[-1] - eigenvalues[0]
        else:
            # For larger instances, estimate from matrix norms
            eig_range = np.linalg.norm(Q_obj, ord=2)

        # Set penalty to be 1.5x the eigenvalue range
        # This ensures any single constraint violation dominates the objective
        base_penalty = max(eig_range * 1.5, 1.0)

        return {
            'visit': base_penalty,
            'flow': base_penalty,
            'capacity': base_penalty * 2.0,  # Capacity is critical
            'timewindow': base_penalty * 1.5,
        }

    def shot_noise_bound(self, n_shots: int = 8192) -> float:
        """Compute the minimum penalty that's statistically significant.

        With finite shots, the energy estimate has variance ~ 1/sqrt(n_shots).
        Penalties must be larger than this noise floor.

        Args:
            n_shots: Number of measurement shots.

        Returns:
            Minimum viable penalty weight.
        """
        max_cost = np.max(self.instance.distance_matrix) * self.instance.n_stops
        noise_floor = max_cost / np.sqrt(n_shots)
        return noise_floor * 3.0  # 3-sigma significance

    def penalty_sweep(
        self,
        penalty_range: np.ndarray,
        n_trials: int = 100,
    ) -> List[Dict]:
        """Sweep penalty weights and measure constraint violation rates.

        For small instances (brute-force solvable), this finds the optimal
        penalty that balances solution quality vs constraint satisfaction.

        Args:
            penalty_range: Array of penalty values to test.
            n_trials: Number of random bitstrings to evaluate per penalty.

        Returns:
            List of dicts with penalty, violation_rate, avg_cost, best_cost.
        """
        results = []
        n = self.instance.n_stops

        for P in penalty_range:
            penalties = {'visit': P, 'flow': P, 'capacity': P * 2, 'timewindow': P}
            builder = VRPQuboBuilder(
                self.instance, encoding=self.encoding, penalties=penalties
            )
            qubo = builder.build()
            Q = qubo.Q

            violations = 0
            costs = []
            valid_costs = []

            for _ in range(n_trials):
                # Random bitstring
                bits = np.random.randint(0, 2, qubo.n_qubits).astype(float)
                energy = bits @ Q @ bits

                # Check feasibility
                eval_result = builder.evaluate_solution(bits)
                if not eval_result['feasible']:
                    violations += 1
                else:
                    valid_costs.append(eval_result['cost'])
                costs.append(energy)

            results.append({
                'penalty': P,
                'violation_rate': violations / n_trials,
                'avg_energy': np.mean(costs),
                'valid_solutions_found': len(valid_costs),
                'best_valid_cost': min(valid_costs) if valid_costs else float('inf'),
            })

        return results
