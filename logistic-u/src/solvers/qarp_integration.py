"""
Fujitsu QARP SDK Integration Layer.

Provides an abstraction layer that wraps our quantum solvers with
QARP-compatible interfaces. When running on Fujitsu infrastructure,
this layer translates our QUBO/Ising formulations into QARP QPE
blocks and handles the QARP runtime.

When running locally (without QARP), it falls back to our
native numpy simulators transparently.

QARP QPE Block mapping:
    - QUBO → Ising Hamiltonian → QAOA ansatz or Grover oracle
    - Circuit parameters → QPE configuration
    - Measurement → sampling with QARP's stochastic simulator

This module is designed from the QARP documentation and will
need adjustment once we get access to the actual SDK.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import time
import json

from src.qubo.vrp_qubo import VRPQuboBuilder, VRPInstance, QUBOResult


@dataclass
class QARPConfig:
    """Configuration for QARP simulator."""
    n_qubits: int
    n_shots: int = 1024
    simulator_type: str = "statevector"  # or "mps", "gpu"
    bond_dimension: int = 64             # For MPS simulator
    noise_model: Optional[str] = None
    backend: str = "local"               # "local" or "fujitsu_cloud"
    timeout_seconds: float = 300.0


@dataclass
class QARPResult:
    """Result from QARP execution."""
    bitstrings: List[np.ndarray]
    counts: Dict[str, int]
    best_bitstring: np.ndarray
    best_cost: float
    n_shots: int
    simulator_type: str
    runtime_seconds: float
    metadata: Dict


class QARPInterface:
    """Interface between our solvers and the Fujitsu QARP SDK.

    Provides two execution modes:
    1. Local simulation (numpy-based) — for development/testing
    2. QARP cloud submission — for production runs on Fujitsu hardware

    The interface abstracts away the backend so the caller
    doesn't need to know whether it's running locally or on QARP.
    """

    def __init__(self, config: Optional[QARPConfig] = None):
        self.config = config or QARPConfig(n_qubits=10)
        self.qarp_available = self._check_qarp()

    def _check_qarp(self) -> bool:
        """Check if QARP SDK is available."""
        try:
            import qarp  # noqa: F401
            return True
        except ImportError:
            return False

    def submit_qaoa(
        self,
        qubo_result: QUBOResult,
        depth: int = 1,
        initial_params: Optional[np.ndarray] = None,
    ) -> QARPResult:
        """Submit QAOA job.

        Args:
            qubo_result: QUBO formulation.
            depth: QAOA circuit depth (p).
            initial_params: Warm-start parameters [γ₁,...,γₚ, β₁,...,βₚ].

        Returns:
            QARPResult with sampled solutions.
        """
        start = time.time()

        if self.qarp_available and self.config.backend == "fujitsu_cloud":
            return self._submit_qaoa_qarp(qubo_result, depth, initial_params)
        else:
            return self._simulate_qaoa_local(qubo_result, depth, initial_params)

    def submit_grover(
        self,
        qubo_result: QUBOResult,
        threshold: float,
        n_iterations: Optional[int] = None,
    ) -> QARPResult:
        """Submit Grover Adaptive Search job.

        Args:
            qubo_result: QUBO formulation.
            threshold: Cost threshold for Grover oracle.
            n_iterations: Number of Grover iterations (None = auto).

        Returns:
            QARPResult with marked state amplified.
        """
        start = time.time()

        if self.qarp_available and self.config.backend == "fujitsu_cloud":
            return self._submit_grover_qarp(qubo_result, threshold, n_iterations)
        else:
            return self._simulate_grover_local(qubo_result, threshold, n_iterations)

    def _simulate_qaoa_local(
        self,
        qubo_result: QUBOResult,
        depth: int,
        initial_params: Optional[np.ndarray],
    ) -> QARPResult:
        """Local QAOA simulation using numpy.

        Since QAOASolver.solve() requires a VRPQuboBuilder (not QUBOResult),
        we simulate QAOA-like behaviour directly: compute all state energies,
        apply a Boltzmann-weighted sampling to mimic QAOA output, and return
        the best solution found.
        """
        start = time.time()
        n = qubo_result.n_qubits
        n_shots = self.config.n_shots

        # Build cost vector for all 2^n states
        N = 2 ** n
        Q_sym = (qubo_result.Q + qubo_result.Q.T) / 2.0
        indices = np.arange(N, dtype=np.int64)
        bits = ((indices[:, None] >> np.arange(n)[None, :]) & 1).astype(float)
        costs = np.sum((bits @ Q_sym) * bits, axis=1)

        # Find ground state
        best_idx = int(np.argmin(costs))
        best_bits = np.array([(best_idx >> k) & 1 for k in range(n)], dtype=int)
        best_cost = float(costs[best_idx])

        # Generate Boltzmann-weighted samples (approximate QAOA output)
        std = float(np.std(costs)) + 1e-10
        boltzmann = np.exp(-costs / std)
        probs = boltzmann / boltzmann.sum()
        rng = np.random.RandomState(42)
        samples = rng.choice(N, size=n_shots, p=probs)

        bitstrings = [np.array([(s >> k) & 1 for k in range(n)]) for s in samples]
        counts = {}
        for s in samples:
            key = format(s, f'0{n}b')
            counts[key] = counts.get(key, 0) + 1

        return QARPResult(
            bitstrings=bitstrings,
            counts=counts,
            best_bitstring=best_bits,
            best_cost=best_cost,
            n_shots=n_shots,
            simulator_type="local_numpy",
            runtime_seconds=time.time() - start,
            metadata={"depth": depth, "method": "qaoa_local"},
        )

    def _simulate_grover_local(
        self,
        qubo_result: QUBOResult,
        threshold: float,
        n_iterations: Optional[int],
    ) -> QARPResult:
        """Local Grover simulation."""
        from src.solvers.grover_solver import GroverAdaptiveSearch

        start = time.time()
        gas = GroverAdaptiveSearch(max_gas_iterations=10)
        gas_result = gas.solve(qubo_result, initial_threshold=threshold)

        bitstrings = [gas_result.optimal_bitstring]
        counts = {
            ''.join(str(b) for b in gas_result.optimal_bitstring): 1
        }

        return QARPResult(
            bitstrings=bitstrings,
            counts=counts,
            best_bitstring=gas_result.optimal_bitstring,
            best_cost=gas_result.optimal_cost,
            n_shots=1,
            simulator_type="local_grover",
            runtime_seconds=time.time() - start,
            metadata={
                "method": "grover_local",
                "gas_iterations": gas_result.n_iterations,
                "threshold_history": gas_result.threshold_history,
            },
        )

    def _submit_qaoa_qarp(self, qubo_result, depth, initial_params):
        """Submit to actual QARP SDK (stub for when SDK is available)."""
        raise NotImplementedError(
            "QARP SDK not configured. Install the QARP package and "
            "set config.backend = 'fujitsu_cloud' with credentials."
        )

    def _submit_grover_qarp(self, qubo_result, threshold, n_iterations):
        """Submit to actual QARP SDK (stub)."""
        raise NotImplementedError(
            "QARP SDK not configured. Install the QARP package and "
            "set config.backend = 'fujitsu_cloud' with credentials."
        )

    def to_qarp_format(self, qubo_result: QUBOResult) -> Dict:
        """Convert QUBO to QARP-compatible format.

        Exports the Ising Hamiltonian in JSON format suitable
        for QARP QPE block configuration.

        Returns:
            Dict with QARP-compatible problem specification.
        """
        # Convert QUBO to Ising
        n = qubo_result.n_qubits
        Q = qubo_result.Q
        Q_sym = (Q + Q.T) / 2.0

        # QUBO to Ising: x_i = (1 - z_i)/2
        # H = Σ J_ij z_i z_j + Σ h_i z_i + const
        h = np.zeros(n)
        J = np.zeros((n, n))
        offset = 0.0

        for i in range(n):
            for j in range(n):
                if i == j:
                    h[i] -= Q_sym[i, i] / 2.0
                    offset += Q_sym[i, i] / 4.0
                elif i < j:
                    J[i, j] = Q_sym[i, j] / 4.0
                    h[i] -= Q_sym[i, j] / 4.0
                    h[j] -= Q_sym[i, j] / 4.0
                    offset += Q_sym[i, j] / 4.0

        # QARP format
        return {
            "format": "qarp_ising_v1",
            "n_qubits": n,
            "ising": {
                "h": h.tolist(),
                "J": {
                    f"{i},{j}": float(J[i, j])
                    for i in range(n) for j in range(i + 1, n)
                    if abs(J[i, j]) > 1e-10
                },
                "offset": float(offset),
            },
            "algorithm": "qaoa",
            "config": {
                "depth": 1,
                "n_shots": self.config.n_shots,
                "optimizer": "COBYLA",
                "simulator": self.config.simulator_type,
            },
        }

    def export_problem(
        self, qubo_result: QUBOResult, filepath: str
    ) -> str:
        """Export problem to JSON file for QARP submission.

        Args:
            qubo_result: QUBO formulation.
            filepath: Output file path.

        Returns:
            Path to the exported file.
        """
        data = self.to_qarp_format(qubo_result)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return filepath
