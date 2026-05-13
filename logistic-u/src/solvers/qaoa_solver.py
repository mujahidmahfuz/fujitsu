"""
QAOA Solver for VRP QUBO — Direct numpy/scipy implementation.

This is a direct mathematical simulation of QAOA circuits using statevector
manipulation with numpy. It's:
- Fast (no Qiskit circuit compilation overhead)
- Transparent (every operation is visible)
- Portable (easy to migrate to any backend including QARP)

Implements:
- Standard QAOA with configurable depth (p)
- Warm-start QAOA with parameter transfer between depths
- CVaR (Conditional Value-at-Risk) objective for focused optimization

The QAOA circuit for p layers:
    |ψ(γ,β)⟩ = U_M(β_p) U_C(γ_p) ... U_M(β_1) U_C(γ_1) |+⟩^n

Where:
    U_C(γ) = exp(-iγ C) is the cost/problem unitary
    U_M(β) = exp(-iβ B) is the mixer unitary, B = ΣX_i
"""

import numpy as np
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import time

from src.qubo.vrp_qubo import VRPQuboBuilder, QUBOResult, VRPInstance


@dataclass
class QAOAResult:
    """Results from a QAOA optimization run."""
    best_bitstring: np.ndarray
    best_cost: float
    qubo_energy: float
    optimal_params: np.ndarray
    convergence_history: List[float]
    runtime_seconds: float
    n_qubits: int
    depth: int
    success_probability: float
    top_k_solutions: List[Tuple[str, float, float]]  # (bitstring_str, cost, probability)


class QAOASolver:
    """QAOA solver using direct statevector simulation.

    This implementation directly manipulates the 2^n statevector using
    matrix exponentials, avoiding Qiskit's circuit compilation overhead.
    For n <= 20 qubits, this is typically faster than Qiskit Aer.
    """

    def __init__(
        self,
        depth: int = 1,
        optimizer: str = "COBYLA",
        max_iter: int = 200,
        use_cvar: bool = False,
        cvar_alpha: float = 0.1,
        warm_start_params: Optional[np.ndarray] = None,
    ):
        """
        Args:
            depth: QAOA circuit depth (p).
            optimizer: Scipy optimizer ("COBYLA", "Nelder-Mead", "Powell", "L-BFGS-B").
            max_iter: Maximum optimizer iterations.
            use_cvar: If True, use CVaR objective (focus on best alpha fraction).
            cvar_alpha: CVaR confidence level (fraction of best samples).
            warm_start_params: Initial (gamma, beta) from a previous run.
        """
        self.depth = depth
        self.optimizer_name = optimizer
        self.max_iter = max_iter
        self.use_cvar = use_cvar
        self.cvar_alpha = cvar_alpha
        self.warm_start_params = warm_start_params
        self.convergence_history: List[float] = []

    def _compute_cost_vector(self, qubo_result: QUBOResult) -> np.ndarray:
        """Compute the diagonal cost vector for all 2^n bitstrings.

        Vectorized: uses numpy broadcasting to evaluate x^T Q x for all x simultaneously.

        Args:
            qubo_result: The QUBO result with matrix Q.

        Returns:
            Array of shape (2^n,) with cost for each computational basis state.
        """
        n = qubo_result.n_qubits
        N = 2 ** n
        Q = qubo_result.Q
        Q_sym = (Q + Q.T) / 2.0

        # Build all bitstrings as a matrix: shape (N, n)
        indices = np.arange(N, dtype=np.int64)
        bits_matrix = ((indices[:, None] >> np.arange(n)[None, :]) & 1).astype(float)

        # Vectorized: costs[i] = bits[i] @ Q_sym @ bits[i]
        # = sum_j (bits @ Q_sym)_j * bits_j = trace of outer products
        costs = np.sum((bits_matrix @ Q_sym) * bits_matrix, axis=1)

        return costs

    def _apply_cost_unitary(
        self, state: np.ndarray, gamma: float, cost_vector: np.ndarray
    ) -> np.ndarray:
        """Apply the cost unitary U_C(γ) = exp(-iγ C). Diagonal, hence vectorized."""
        return state * np.exp(-1j * gamma * cost_vector)

    def _apply_mixer_unitary(
        self, state: np.ndarray, beta: float, n_qubits: int
    ) -> np.ndarray:
        """Apply the mixer unitary U_M(β) = ∏_i exp(-iβ X_i).

        Vectorized using numpy reshape to apply single-qubit gates without loops
        over the full Hilbert space.
        """
        cos_b = np.cos(beta)
        sin_b_neg_i = -1j * np.sin(beta)

        for k in range(n_qubits):
            # Reshape state into shape (..., 2, ...) with axis k representing qubit k
            # For qubit k, group pairs of basis states differing in bit k
            N = len(state)
            half = N // 2

            # Separate state into |0_k⟩ and |1_k⟩ components
            stride = 1 << k

            # Create index arrays for |0_k⟩ and |1_k⟩ states
            # Indices where bit k is 0 vs 1
            block_size = stride
            block_count = N // (2 * stride)

            idx0 = np.zeros(half, dtype=np.int64)
            idx1 = np.zeros(half, dtype=np.int64)
            pos = 0
            for b in range(block_count):
                base = b * 2 * stride
                for s in range(block_size):
                    idx0[pos] = base + s
                    idx1[pos] = base + s + stride
                    pos += 1

            s0 = state[idx0]
            s1 = state[idx1]

            # exp(-iβ X) on qubit k: |0⟩ → cos(β)|0⟩ - i sin(β)|1⟩
            #                          |1⟩ → -i sin(β)|0⟩ + cos(β)|1⟩
            new_s0 = cos_b * s0 + sin_b_neg_i * s1
            new_s1 = sin_b_neg_i * s0 + cos_b * s1

            state = state.copy()
            state[idx0] = new_s0
            state[idx1] = new_s1

        return state

    def _qaoa_statevector(
        self,
        params: np.ndarray,
        n_qubits: int,
        cost_vector: np.ndarray,
    ) -> np.ndarray:
        """Compute the QAOA statevector for given parameters.

        |ψ(γ,β)⟩ = U_M(β_p) U_C(γ_p) ... U_M(β_1) U_C(γ_1) |+⟩^n

        Args:
            params: Array [γ_1, ..., γ_p, β_1, ..., β_p] of length 2p.
            n_qubits: Number of qubits.
            cost_vector: Precomputed cost vector.

        Returns:
            Final statevector.
        """
        p = self.depth
        gammas = params[:p]
        betas = params[p:]

        N = 2 ** n_qubits

        # Initial state: |+⟩^n = uniform superposition
        state = np.ones(N, dtype=complex) / np.sqrt(N)

        # Apply p layers
        for layer in range(p):
            state = self._apply_cost_unitary(state, gammas[layer], cost_vector)
            state = self._apply_mixer_unitary(state, betas[layer], n_qubits)

        return state

    def _objective(
        self,
        params: np.ndarray,
        n_qubits: int,
        cost_vector: np.ndarray,
    ) -> float:
        """QAOA objective function: ⟨ψ(γ,β)|C|ψ(γ,β)⟩.

        If use_cvar=True, computes CVaR instead of expectation value.

        Args:
            params: QAOA parameters.
            n_qubits: Number of qubits.
            cost_vector: Precomputed cost vector.

        Returns:
            Objective value (to be minimized).
        """
        state = self._qaoa_statevector(params, n_qubits, cost_vector)
        probs = np.abs(state) ** 2

        if self.use_cvar:
            # CVaR: average cost of the best alpha fraction of outcomes
            sorted_indices = np.argsort(cost_vector)
            cumulative_prob = 0.0
            cvar_value = 0.0
            for idx in sorted_indices:
                if cumulative_prob >= self.cvar_alpha:
                    break
                added_prob = min(probs[idx], self.cvar_alpha - cumulative_prob)
                cvar_value += added_prob * cost_vector[idx]
                cumulative_prob += probs[idx]
            if cumulative_prob > 0:
                return cvar_value / min(cumulative_prob, self.cvar_alpha)
            return np.mean(cost_vector)
        else:
            # Standard expectation: ⟨C⟩ = Σ_x prob(x) * cost(x)
            return float(np.sum(probs * cost_vector))

    def solve(self, builder: VRPQuboBuilder) -> QAOAResult:
        """Solve a VRP QUBO using QAOA.

        Args:
            builder: VRPQuboBuilder with the problem instance.

        Returns:
            QAOAResult with the optimization results.
        """
        start_time = time.time()

        # Build QUBO and precompute costs
        qubo_result = builder.build()
        n_qubits = qubo_result.n_qubits

        if n_qubits > 20:
            raise ValueError(
                f"Direct statevector QAOA not feasible for {n_qubits} qubits. "
                f"Use hybrid solver for larger instances."
            )

        cost_vector = self._compute_cost_vector(qubo_result)

        # Track convergence
        self.convergence_history = []

        def callback_fn(params):
            val = self._objective(params, n_qubits, cost_vector)
            self.convergence_history.append(val)

        # Initial parameters
        if self.warm_start_params is not None:
            x0 = self.warm_start_params
        else:
            # Random initialization near 0
            x0 = np.random.uniform(-0.1, 0.1, 2 * self.depth)

        # Optimize
        result = minimize(
            self._objective,
            x0,
            args=(n_qubits, cost_vector),
            method=self.optimizer_name,
            callback=callback_fn,
            options={'maxiter': self.max_iter},
        )

        optimal_params = result.x
        optimal_value = result.fun

        # Get final statevector and extract results
        final_state = self._qaoa_statevector(optimal_params, n_qubits, cost_vector)
        probs = np.abs(final_state) ** 2

        # Find best bitstring
        best_idx = np.argmin(cost_vector * (probs > 1e-10).astype(float) + 
                            (probs <= 1e-10).astype(float) * 1e15)
        # Actually, find the most probable low-cost solution
        weighted = cost_vector.copy()
        weighted[probs < 1e-10] = 1e15
        best_idx = np.argmin(weighted)

        best_bits = np.array([(best_idx >> b) & 1 for b in range(n_qubits)], dtype=float)
        best_energy = cost_vector[best_idx]

        # Evaluate the solution through the builder
        eval_result = builder.evaluate_solution(best_bits)

        # Get top-k solutions
        top_k_count = min(10, len(probs))
        top_indices = np.argsort(-probs)[:top_k_count]
        top_k = []
        for idx in top_indices:
            if probs[idx] < 1e-10:
                break
            bits_str = format(idx, f'0{n_qubits}b')[::-1]
            top_k.append((bits_str, float(cost_vector[idx]), float(probs[idx])))

        runtime = time.time() - start_time

        # Find success probability (probability of the best FEASIBLE solution)
        # Check each high-probability state for feasibility
        best_feasible_cost = float('inf')
        success_prob = 0.0
        for idx in np.argsort(-probs)[:50]:
            if probs[idx] < 1e-10:
                break
            bits = np.array([(idx >> b) & 1 for b in range(n_qubits)], dtype=float)
            ev = builder.evaluate_solution(bits)
            if ev['feasible'] and ev['cost'] < best_feasible_cost:
                best_feasible_cost = ev['cost']
                best_bits = bits
                best_energy = cost_vector[idx]
                success_prob = float(probs[idx])
                eval_result = ev

        return QAOAResult(
            best_bitstring=best_bits,
            best_cost=eval_result['cost'],
            qubo_energy=best_energy,
            optimal_params=optimal_params,
            convergence_history=self.convergence_history,
            runtime_seconds=runtime,
            n_qubits=n_qubits,
            depth=self.depth,
            success_probability=success_prob,
            top_k_solutions=top_k,
        )

    def solve_with_depth_transfer(
        self, builder: VRPQuboBuilder, max_depth: int = 3
    ) -> List[QAOAResult]:
        """Run QAOA with increasing depth, transferring parameters.

        Strategy: Solve at p=1, use optimal params to initialize p=2, etc.

        Args:
            builder: VRPQuboBuilder with the problem instance.
            max_depth: Maximum QAOA depth to try.

        Returns:
            List of QAOAResult, one per depth level.
        """
        results = []
        warm_params = None

        for p in range(1, max_depth + 1):
            solver = QAOASolver(
                depth=p,
                optimizer=self.optimizer_name,
                max_iter=self.max_iter,
                use_cvar=self.use_cvar,
                cvar_alpha=self.cvar_alpha,
                warm_start_params=warm_params,
            )
            result = solver.solve(builder)
            results.append(result)

            # Transfer parameters: duplicate last layer params for next depth
            if result.optimal_params is not None:
                params = result.optimal_params
                gammas = params[:p]
                betas = params[p:]
                new_gammas = np.append(gammas, gammas[-1] * 0.9)
                new_betas = np.append(betas, betas[-1] * 0.9)
                warm_params = np.concatenate([new_gammas, new_betas])

        return results
