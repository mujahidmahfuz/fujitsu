"""
Simulated Quantum Annealing Solver.

Implements simulated quantum annealing for QUBO problems using path-integral
Monte Carlo with transverse field.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ..qubo.encoder import QUBOProblem

logger = logging.getLogger(__name__)


@dataclass
class SAConfig:
    """Configuration for Simulated Annealing."""

    # Temperature schedule
    initial_temperature: float = 100.0
    final_temperature: float = 0.01
    cooling_rate: float = 0.995
    num_sweeps: int = 10000

    # Annealing schedule type
    schedule: str = "exponential"  # "exponential", "linear", "adaptive"

    # Number of restarts
    num_restarts: int = 1

    # Random seed
    seed: int | None = None


@dataclass
class SQAConfig:
    """Configuration for Simulated Quantum Annealing."""

    # Temperature
    temperature: float = 0.1

    # Transverse field schedule
    initial_transverse_field: float = 10.0
    final_transverse_field: float = 0.01
    transverse_decay: float = 0.995

    # Trotter slices
    num_trotter_slices: int = 10

    # Monte Carlo
    num_sweeps: int = 10000

    # Random seed
    seed: int | None = None


@dataclass
class SolverResult:
    """Result from annealing solver."""

    # Best solution found
    best_solution: npt.NDArray[np.int32]

    # Best energy (objective value)
    best_energy: float

    # Number of solutions sampled
    num_samples: int

    # Solve time in milliseconds
    solve_time_ms: float

    # All solutions sampled (if stored)
    all_solutions: list[npt.NDArray[np.int32]] = field(default_factory=list)
    all_energies: list[float] = field(default_factory=list)

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


class SimulatedAnnealingSolver:
    """Simulated Annealing solver for QUBO problems.

    Uses classical simulated annealing with Metropolis updates.
    """

    def __init__(self, config: SAConfig | None = None) -> None:
        """Initialize solver with configuration."""
        self.config = config or SAConfig()
        self.rng = random.Random(self.config.seed)
        self.np_rng = np.random.default_rng(self.config.seed)

    def solve(self, qubo: QUBOProblem) -> SolverResult:
        """Solve the QUBO problem using simulated annealing.

        Args:
            qubo: QUBO problem to solve

        Returns:
            SolverResult with best solution found
        """
        import time

        start_time = time.perf_counter()

        n = qubo.num_variables
        Q = qubo.Q

        best_solution = None
        best_energy = float("inf")
        all_solutions = []
        all_energies = []

        for restart in range(self.config.num_restarts):
            # Initialize random solution
            solution = self.np_rng.integers(0, 2, n)

            # Compute initial energy
            energy = self._compute_energy(solution, Q)

            # Annealing
            temperature = self.config.initial_temperature

            for sweep in range(self.config.num_sweeps):
                # Single sweep: try flipping each variable
                for i in range(n):
                    # Compute energy change from flipping bit i
                    delta = self._compute_delta_energy(solution, Q, i)

                    # Metropolis acceptance
                    if delta < 0 or self.rng.random() < np.exp(-delta / temperature):
                        solution[i] = 1 - solution[i]
                        energy += delta

                # Cool down
                if self.config.schedule == "exponential":
                    temperature *= self.config.cooling_rate
                elif self.config.schedule == "linear":
                    temperature -= (
                        self.config.initial_temperature
                        - self.config.final_temperature
                    ) / self.config.num_sweeps

                # Check for improvement
                if energy < best_energy:
                    best_energy = energy
                    best_solution = solution.copy()

            all_solutions.append(solution.copy())
            all_energies.append(energy)

        end_time = time.perf_counter()
        solve_time_ms = (end_time - start_time) * 1000

        return SolverResult(
            best_solution=best_solution,
            best_energy=best_energy + qubo.offset,
            num_samples=self.config.num_restarts,
            solve_time_ms=solve_time_ms,
            all_solutions=all_solutions,
            all_energies=[e + qubo.offset for e in all_energies],
            metadata={
                "solver": "simulated_annealing",
                "config": vars(self.config),
            },
        )

    def _compute_energy(
        self, solution: npt.NDArray[np.int32], Q: npt.NDArray[np.float64]
    ) -> float:
        """Compute QUBO energy: x^T Q x."""
        return float(solution @ Q @ solution)

    def _compute_delta_energy(
        self,
        solution: npt.NDArray[np.int32],
        Q: npt.NDArray[np.float64],
        i: int,
    ) -> float:
        """Compute energy change from flipping bit i."""
        # Energy change = (1 - 2*x_i) * (Q_ii + sum_j Q_ij * x_j + sum_j Q_ji * x_j)
        # For symmetric Q: = (1 - 2*x_i) * (Q_ii + 2 * sum_{j!=i} Q_ij * x_j)

        old_val = solution[i]
        new_val = 1 - old_val

        # Linear term contribution
        delta = Q[i, i] * (new_val - old_val)

        # Quadratic term contribution
        for j in range(len(solution)):
            if j != i:
                delta += Q[i, j] * (new_val * solution[j] - old_val * solution[j])
                delta += Q[j, i] * (solution[j] * new_val - solution[j] * old_val)

        # Simplify: since Q is symmetric
        # delta = (1 - 2*old_val) * (Q[i,i] + 2 * sum_{j!=i} Q[i,j] * x_j)
        return delta


class SimulatedQuantumAnnealingSolver:
    """Simulated Quantum Annealing solver for QUBO problems.

    Implements path-integral Monte Carlo with transverse field,
    approximating quantum annealing dynamics.
    """

    def __init__(self, config: SQAConfig | None = None) -> None:
        """Initialize solver with configuration."""
        self.config = config or SQAConfig()
        self.rng = random.Random(self.config.seed)
        self.np_rng = np.random.default_rng(self.config.seed)

    def solve(self, qubo: QUBOProblem) -> SolverResult:
        """Solve the QUBO problem using simulated quantum annealing.

        Args:
            qubo: QUBO problem to solve

        Returns:
            SolverResult with best solution found
        """
        import time

        start_time = time.perf_counter()

        n = qubo.num_variables
        Q = qubo.Q
        P = self.config.num_trotter_slices
        T = self.config.temperature

        # Initialize Trotter slices with random spins
        spins = self.np_rng.integers(0, 2, (P, n))

        # Initial transverse field
        Gamma = self.config.initial_transverse_field

        best_solution = None
        best_energy = float("inf")

        # SQA main loop
        for sweep in range(self.config.num_sweeps):
            # Update each Trotter slice
            for p in range(P):
                # Trotter slice to update
                slice_spins = spins[p]

                # Neighboring slices (periodic boundary)
                prev_slice = spins[(p - 1) % P]
                next_slice = spins[(p + 1) % P]

                # Single spin updates
                for i in range(n):
                    # Classical energy contribution
                    classical_energy = self._compute_spin_energy(
                        slice_spins, Q, i
                    )

                    # Quantum coupling from transverse field
                    quantum_energy = -Gamma / P * (
                        (2 * slice_spins[i] - 1) * (2 * prev_slice[i] - 1)
                        + (2 * slice_spins[i] - 1) * (2 * next_slice[i] - 1)
                    )

                    # Total energy change
                    delta = classical_energy + quantum_energy

                    # Metropolis acceptance
                    if delta < 0 or self.rng.random() < np.exp(-delta / T):
                        spins[p, i] = 1 - spins[p, i]

            # Decay transverse field
            Gamma *= self.config.transverse_decay

            # Check for best solution
            for p in range(P):
                energy = self._compute_energy(spins[p], Q)
                if energy < best_energy:
                    best_energy = energy
                    best_solution = spins[p].copy()

        end_time = time.perf_counter()
        solve_time_ms = (end_time - start_time) * 1000

        return SolverResult(
            best_solution=best_solution,
            best_energy=best_energy + qubo.offset,
            num_samples=self.config.num_trotter_slices,
            solve_time_ms=solve_time_ms,
            metadata={
                "solver": "simulated_quantum_annealing",
                "config": vars(self.config),
            },
        )

    def _compute_energy(
        self, spins: npt.NDArray[np.int32], Q: npt.NDArray[np.float64]
    ) -> float:
        """Compute classical energy for a spin configuration."""
        # Convert {0,1} to {-1,+1} for Ising, or keep as {0,1} for QUBO
        # Here we use QUBO formulation directly
        return float(spins @ Q @ spins)

    def _compute_spin_energy(
        self,
        spins: npt.NDArray[np.int32],
        Q: npt.NDArray[np.float64],
        i: int,
    ) -> float:
        """Compute energy change from flipping spin i."""
        old_val = spins[i]
        new_val = 1 - old_val

        # Energy change from QUBO matrix
        delta = Q[i, i] * (new_val - old_val)

        # Interaction with other spins
        for j in range(len(spins)):
            if j != i and spins[j] == 1:
                delta += Q[i, j] + Q[j, i]

        return delta

    def sample_multiple(
        self, qubo: QUBOProblem, num_samples: int = 10
    ) -> SolverResult:
        """Sample multiple solutions from the quantum distribution.

        This is useful for exploring diverse solutions in the repair operator.

        Args:
            qubo: QUBO problem
            num_samples: Number of samples to return

        Returns:
            SolverResult with multiple samples
        """
        import time

        start_time = time.perf_counter()

        n = qubo.num_variables
        Q = qubo.Q

        # Run SQA multiple times with different seeds
        all_solutions = []
        all_energies = []

        for sample_idx in range(num_samples):
            # Reset with new random state
            original_seed = self.config.seed
            self.config.seed = (
                original_seed + sample_idx if original_seed else sample_idx
            )
            self.rng = random.Random(self.config.seed)
            self.np_rng = np.random.default_rng(self.config.seed)

            result = self.solve(qubo)
            all_solutions.append(result.best_solution)
            all_energies.append(result.best_energy)

        # Find best
        best_idx = np.argmin(all_energies)

        end_time = time.perf_counter()

        return SolverResult(
            best_solution=all_solutions[best_idx],
            best_energy=all_energies[best_idx],
            num_samples=num_samples,
            solve_time_ms=(end_time - start_time) * 1000,
            all_solutions=all_solutions,
            all_energies=all_energies,
            metadata={
                "solver": "simulated_quantum_annealing",
                "sampling": True,
            },
        )


def solve_qubo_sa(
    qubo: QUBOProblem,
    num_sweeps: int = 10000,
    initial_temp: float = 100.0,
    final_temp: float = 0.01,
    seed: int | None = None,
) -> SolverResult:
    """Convenience function to solve QUBO with simulated annealing.

    Args:
        qubo: QUBO problem
        num_sweeps: Number of annealing sweeps
        initial_temp: Initial temperature
        final_temp: Final temperature
        seed: Random seed

    Returns:
        SolverResult
    """
    config = SAConfig(
        num_sweeps=num_sweeps,
        initial_temperature=initial_temp,
        final_temperature=final_temp,
        seed=seed,
    )
    solver = SimulatedAnnealingSolver(config)
    return solver.solve(qubo)


def solve_qubo_sqa(
    qubo: QUBOProblem,
    num_sweeps: int = 10000,
    num_trotter_slices: int = 10,
    seed: int | None = None,
) -> SolverResult:
    """Convenience function to solve QUBO with simulated quantum annealing.

    Args:
        qubo: QUBO problem
        num_sweeps: Number of Monte Carlo sweeps
        num_trotter_slices: Number of Trotter slices
        seed: Random seed

    Returns:
        SolverResult
    """
    config = SQAConfig(
        num_sweeps=num_sweeps,
        num_trotter_slices=num_trotter_slices,
        seed=seed,
    )
    solver = SimulatedQuantumAnnealingSolver(config)
    return solver.solve(qubo)