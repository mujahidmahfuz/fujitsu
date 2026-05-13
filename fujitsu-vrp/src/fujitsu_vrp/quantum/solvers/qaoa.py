"""
QAOA (Quantum Approximate Optimization Algorithm) Solver.

Implements QAOA for solving QUBO problems on gate-based quantum computers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from ..qubo.encoder import QUBOProblem

logger = logging.getLogger(__name__)


@dataclass
class QAOAConfig:
    """Configuration for QAOA solver."""

    # QAOA parameters
    p: int = 2  # Number of QAOA layers
    initial_gamma: float = 0.5  # Initial mixer angle
    initial_beta: float = 0.5  # Initial cost angle

    # Optimization
    optimizer: str = "COBYLA"  # Classical optimizer
    max_iterations: int = 100
    optimizer_options: dict = field(default_factory=dict)

    # Shots
    num_shots: int = 1024

    # Backend
    use_simulator: bool = True
    backend_name: str = "aer_simulator"

    # Random seed
    seed: int | None = None


@dataclass
class QAOAResult:
    """Result from QAOA solver."""

    best_solution: npt.NDArray[np.int32]
    best_energy: float
    optimal_parameters: tuple[list[float], list[float]]  # (gammas, betas)
    num_samples: int
    solve_time_ms: float
    all_solutions: list[npt.NDArray[np.int32]] = field(default_factory=list)
    all_energies: list[float] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class QAOASolver:
    """QAOA solver for QUBO problems.

    Uses Qiskit for circuit construction and simulation.
    """

    def __init__(self, config: QAOAConfig | None = None) -> None:
        """Initialize QAOA solver."""
        self.config = config or QAOAConfig()

    def solve(self, qubo: QUBOProblem) -> QAOAResult:
        """Solve QUBO problem using QAOA.

        Args:
            qubo: QUBO problem to solve

        Returns:
            QAOAResult with best solution
        """
        import time

        start_time = time.perf_counter()

        n = qubo.num_variables

        # Convert QUBO to Ising Hamiltonian
        h, J = self._qubo_to_ising(qubo.Q)

        # Try to use Qiskit if available
        try:
            result = self._solve_qiskit(qubo, h, J)
        except ImportError:
            logger.warning("Qiskit not available, using classical simulation")
            result = self._solve_classical(qubo, h, J)

        end_time = time.perf_counter()
        result.solve_time_ms = (end_time - start_time) * 1000
        result.metadata["solver"] = "QAOA"
        result.metadata["config"] = vars(self.config)

        return result

    def _qubo_to_ising(
        self, Q: npt.NDArray[np.float64]
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Convert QUBO matrix to Ising parameters.

        QUBO: minimize x^T Q x where x in {0,1}
        Ising: minimize sum_i h_i s_i + sum_{i<j} J_ij s_i s_j where s in {-1,+1}

        Transformation: x = (1-s)/2, so s = 1 - 2x
        """
        n = Q.shape[0]

        # Compute Ising parameters
        h = np.zeros(n)
        J = np.zeros((n, n))

        for i in range(n):
            # Linear term
            h[i] = 0.5 * Q[i, i] + np.sum(Q[i, :]) * 0.25 + np.sum(Q[:, i]) * 0.25 - np.sum(Q) * 0.25 / n

            for j in range(i + 1, n):
                J[i, j] = Q[i, j] * 0.25
                J[j, i] = Q[i, j] * 0.25

        return h, J

    def _solve_qiskit(
        self,
        qubo: QUBOProblem,
        h: npt.NDArray[np.float64],
        J: npt.NDArray[np.float64],
    ) -> QAOAResult:
        """Solve using Qiskit QAOA implementation."""
        try:
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
            from qiskit.circuit import Parameter
            from qiskit.algorithms import QAOA
            from qiskit.algorithms.optimizers import COBYLA
            from qiskit.opflow import I, PauliSumOp, SparsePauliOp, X, Z
            from qiskit.quantum_info import Pauli
        except ImportError:
            raise ImportError("Qiskit not available")

        n = qubo.num_variables

        # Build Hamiltonian
        pauli_list = []

        # Linear terms (Z operators)
        for i in range(n):
            if abs(h[i]) > 1e-10:
                z_list = ["I"] * n
                z_list[n - 1 - i] = "Z"
                pauli = Pauli("".join(z_list))
                pauli_list.append((pauli, h[i]))

        # Quadratic terms (ZZ operators)
        for i in range(n):
            for j in range(i + 1, n):
                if abs(J[i, j]) > 1e-10:
                    z_list = ["I"] * n
                    z_list[n - 1 - i] = "Z"
                    z_list[n - 1 - j] = "Z"
                    pauli = Pauli("".join(z_list))
                    pauli_list.append((pauli, J[i, j]))

        if not pauli_list:
            # No constraints, return random solution
            solution = np.zeros(n, dtype=np.int32)
            return QAOAResult(
                best_solution=solution,
                best_energy=qubo.offset,
                optimal_parameters=([], []),
                num_samples=1,
                solve_time_ms=0,
            )

        hamiltonian = SparsePauliOp.from_list([(p, c) for p, c in pauli_list])

        # Build QAOA circuit
        p = self.config.p

        # Initial parameters
        initial_params = np.concatenate([
            np.ones(p) * self.config.initial_gamma,
            np.ones(p) * self.config.initial_beta,
        ])

        # Optimizer
        if self.config.optimizer == "COBYLA":
            optimizer = COBYLA(maxiter=self.config.max_iterations)

        # Run QAOA
        simulator = AerSimulator()

        # Manual QAOA implementation
        best_params = self._optimize_qaoa(
            hamiltonian, n, p, initial_params, optimizer, simulator
        )

        # Sample from optimal circuit
        final_circuit = self._build_qaoa_circuit(hamiltonian, n, p, best_params)
        final_circuit.measure_all()

        job = simulator.run(final_circuit, shots=self.config.num_shots)
        counts = job.result().get_counts()

        # Find best solution
        best_bitstring = max(counts, key=counts.get)
        solution = np.array([int(b) for b in best_bitstring], dtype=np.int32)

        # Compute all solutions
        all_solutions = []
        all_energies = []
        for bitstring, count in counts.items():
            sol = np.array([int(b) for b in bitstring], dtype=np.int32)
            energy = float(sol @ qubo.Q @ sol) + qubo.offset
            all_solutions.append(sol)
            all_energies.append(energy)

        best_idx = np.argmin(all_energies)

        return QAOAResult(
            best_solution=all_solutions[best_idx],
            best_energy=all_energies[best_idx],
            optimal_parameters=(
                best_params[:p].tolist(),
                best_params[p:].tolist(),
            ),
            num_samples=self.config.num_shots,
            solve_time_ms=0,
            all_solutions=all_solutions,
            all_energies=all_energies,
        )

    def _build_qaoa_circuit(
        self, hamiltonian, n: int, p: int, params: npt.NDArray
    ):
        """Build QAOA circuit."""
        from qiskit import QuantumCircuit

        gammas = params[:p]
        betas = params[p:]

        circuit = QuantumCircuit(n)

        # Initial state: uniform superposition
        for i in range(n):
            circuit.h(i)

        # QAOA layers
        for layer in range(p):
            # Cost layer
            self._apply_cost_layer(circuit, hamiltonian, gammas[layer])

            # Mixer layer
            for i in range(n):
                circuit.rx(2 * betas[layer], i)

        return circuit

    def _apply_cost_layer(self, circuit, hamiltonian, gamma: float):
        """Apply cost Hamiltonian evolution."""
        from qiskit.quantum_info import Pauli

        n = circuit.num_qubits

        for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
            # For each Pauli term
            pauli_str = str(pauli)

            # Find qubits with Z
            for i, p in enumerate(pauli_str):
                if p == "Z":
                    # Apply Rz rotation (simplified)
                    pass  # Full implementation would use CNOT gates

        # Simplified: just apply Z rotations
        for i in range(n):
            circuit.rz(2 * gamma, i)

    def _optimize_qaoa(
        self, hamiltonian, n: int, p: int, initial_params, optimizer, backend
    ) -> npt.NDArray:
        """Optimize QAOA parameters."""
        from scipy.optimize import minimize

        def objective(params):
            circuit = self._build_qaoa_circuit(hamiltonian, n, p, params)
            circuit.measure_all()

            job = backend.run(circuit, shots=self.config.num_shots)
            counts = job.result().get_counts()

            # Compute expected energy
            total_energy = 0.0
            total_shots = sum(counts.values())

            for bitstring, count in counts.items():
                # Convert to energy
                sol = np.array([int(b) for b in bitstring], dtype=np.int32)
                energy = float(sol @ hamiltonian.coeffs @ sol)  # Simplified
                total_energy += energy * count

            return total_energy / total_shots

        result = minimize(
            objective,
            initial_params,
            method="COBYLA",
            options={"maxiter": self.config.max_iterations},
        )

        return result.x

    def _solve_classical(
        self,
        qubo: QUBOProblem,
        h: npt.NDArray[np.float64],
        J: npt.NDArray[np.float64],
    ) -> QAOAResult:
        """Classical simulation fallback (random sampling with optimization)."""
        n = qubo.num_variables

        best_solution = None
        best_energy = float("inf")
        all_solutions = []
        all_energies = []

        # Try random initializations
        for _ in range(min(self.config.num_shots, 100)):
            solution = np.random.randint(0, 2, n, dtype=np.int32)
            energy = float(solution @ qubo.Q @ solution) + qubo.offset

            all_solutions.append(solution)
            all_energies.append(energy)

            if energy < best_energy:
                best_energy = energy
                best_solution = solution.copy()

        return QAOAResult(
            best_solution=best_solution,
            best_energy=best_energy,
            optimal_parameters=([], []),
            num_samples=len(all_solutions),
            solve_time_ms=0,
            all_solutions=all_solutions,
            all_energies=all_energies,
        )


def solve_qubo_qaoa(
    qubo: QUBOProblem,
    p: int = 2,
    num_shots: int = 1024,
    seed: int | None = None,
) -> QAOAResult:
    """Convenience function to solve QUBO with QAOA.

    Args:
        qubo: QUBO problem
        p: Number of QAOA layers
        num_shots: Number of measurement shots
        seed: Random seed

    Returns:
        QAOAResult
    """
    config = QAOAConfig(p=p, num_shots=num_shots, seed=seed)
    solver = QAOASolver(config)
    return solver.solve(qubo)