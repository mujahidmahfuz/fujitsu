"""
D-Wave Neal Backend for Simulated Annealing.

Provides interface to D-Wave's neal library for simulated annealing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from ..qubo.encoder import QUBOProblem
    from ..solvers.quantum_annealing import SolverResult

logger = logging.getLogger(__name__)


@dataclass
class NealConfig:
    """Configuration for D-Wave Neal solver."""

    # Annealing
    num_reads: int = 100
    num_sweeps: int = 1000
    beta_range: tuple[float, float] = (0.1, 10.0)

    # Seed
    seed: int | None = None


class NealBackend:
    """D-Wave Neal backend for simulated annealing."""

    def __init__(self, config: NealConfig | None = None) -> None:
        """Initialize Neal backend."""
        self.config = config or NealConfig()

    def solve_qubo(self, qubo: QUBOProblem) -> SolverResult:
        """Solve QUBO using D-Wave Neal.

        Args:
            qubo: QUBO problem

        Returns:
            SolverResult
        """
        import time

        start_time = time.perf_counter()

        try:
            import neal

            # Convert QUBO to BQM format
            bqm = self._qubo_to_bqm(qubo.Q)

            # Create sampler
            sampler = neal.SimulatedAnnealingSampler()

            # Sample
            sampleset = sampler.sample(
                bqm,
                num_reads=self.config.num_reads,
                num_sweeps=self.config.num_sweeps,
                beta_range=self.config.beta_range,
                seed=self.config.seed,
            )

            # Get best solution
            best_sample = sampleset.first.sample
            best_energy = sampleset.first.energy

            # Convert to solution array
            n = qubo.num_variables
            solution = np.zeros(n, dtype=np.int32)
            for i in range(n):
                solution[i] = best_sample.get(i, 0)

            # Get all solutions
            all_solutions = []
            all_energies = []
            for sample, energy in zip(sampleset.record.sample, sampleset.record.energy):
                all_solutions.append(sample)
                all_energies.append(energy)

            end_time = time.perf_counter()

            from ..solvers.quantum_annealing import SolverResult

            return SolverResult(
                best_solution=solution,
                best_energy=float(best_energy) + qubo.offset,
                num_samples=self.config.num_reads,
                solve_time_ms=(end_time - start_time) * 1000,
                all_solutions=all_solutions,
                all_energies=[e + qubo.offset for e in all_energies],
                metadata={
                    "solver": "neal",
                    "config": vars(self.config),
                },
            )

        except ImportError:
            logger.warning("D-Wave neal not available, falling back to custom SA")
            from ..solvers.quantum_annealing import solve_qubo_sa

            return solve_qubo_sa(
                qubo,
                num_sweeps=self.config.num_sweeps,
                seed=self.config.seed,
            )

    def _qubo_to_bqm(self, Q: np.ndarray):
        """Convert QUBO matrix to dimod BQM format."""
        try:
            import dimod

            bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

            n = Q.shape[0]

            # Add linear terms
            for i in range(n):
                if abs(Q[i, i]) > 1e-10:
                    bqm.add_variable(i, Q[i, i])

            # Add quadratic terms
            for i in range(n):
                for j in range(i + 1, n):
                    if abs(Q[i, j]) > 1e-10:
                        bqm.add_interaction(i, j, Q[i, j])

            return bqm

        except ImportError:
            raise ImportError("dimod not available")


def solve_with_neal(
    qubo: QUBOProblem,
    num_reads: int = 100,
    num_sweeps: int = 1000,
    seed: int | None = None,
) -> SolverResult:
    """Convenience function to solve QUBO with Neal.

    Args:
        qubo: QUBO problem
        num_reads: Number of annealing reads
        num_sweeps: Number of sweeps per read
        seed: Random seed

    Returns:
        SolverResult
    """
    config = NealConfig(
        num_reads=num_reads,
        num_sweeps=num_sweeps,
        seed=seed,
    )
    backend = NealBackend(config)
    return backend.solve_qubo(qubo)