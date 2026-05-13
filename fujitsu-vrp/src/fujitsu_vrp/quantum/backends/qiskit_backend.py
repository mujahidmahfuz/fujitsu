"""
Qiskit Backend for Quantum Computing.

Provides interface to Qiskit simulators and quantum devices.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..qubo.encoder import QUBOProblem
    from ..solvers.qaoa import QAOAResult

logger = logging.getLogger(__name__)


@dataclass
class QiskitConfig:
    """Configuration for Qiskit backend."""

    # Backend selection
    backend_name: str = "aer_simulator"
    use_real_device: bool = False
    device_name: str | None = None

    # Simulation
    shots: int = 1024
    optimization_level: int = 1

    # Error mitigation
    use_error_mitigation: bool = False

    # Seed
    seed: int | None = None


class QiskitBackend:
    """Qiskit backend for quantum computing."""

    def __init__(self, config: QiskitConfig | None = None) -> None:
        """Initialize Qiskit backend."""
        self.config = config or QiskitConfig()
        self._backend = None

    def solve_qubo(self, qubo: QUBOProblem) -> QAOAResult:
        """Solve QUBO using Qiskit QAOA."""
        try:
            from ..solvers.qaoa import QAOASolver, QAOAConfig

            config = QAOAConfig(
                num_shots=self.config.shots,
                seed=self.config.seed,
                use_simulator=not self.config.use_real_device,
                backend_name=self.config.backend_name,
            )
            solver = QAOASolver(config)
            return solver.solve(qubo)
        except ImportError:
            raise ImportError(
                "Qiskit not available. Install with: pip install qiskit qiskit-aer"
            )

    def get_backend(self):
        """Get Qiskit backend instance."""
        if self._backend is not None:
            return self._backend

        try:
            from qiskit_aer import AerSimulator

            self._backend = AerSimulator()
            return self._backend
        except ImportError:
            raise ImportError("Qiskit Aer not available")


def run_on_qiskit(
    qubo: QUBOProblem,
    shots: int = 1024,
    seed: int | None = None,
) -> QAOAResult:
    """Convenience function to run QUBO on Qiskit.

    Args:
        qubo: QUBO problem
        shots: Number of measurement shots
        seed: Random seed

    Returns:
        QAOAResult
    """
    config = QiskitConfig(shots=shots, seed=seed)
    backend = QiskitBackend(config)
    return backend.solve_qubo(qubo)