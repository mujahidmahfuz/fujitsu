"""
Tests for Fujitsu VRP Core Components.

Tests the fundamental building blocks of the QLNRS algorithm.
"""

import numpy as np
import pytest


class TestSyntheticGenerator:
    """Tests for Tokyo SME data generation."""

    def test_generator_creation(self):
        """Test generator initialization."""
        from fujitsu_vrp.data.synthetic_generator import TokyoSMEGenerator

        generator = TokyoSMEGenerator(seed=42, num_customers=10)
        assert generator.num_customers == 10
        assert generator.seed == 42

    def test_instance_generation(self):
        """Test VRP instance generation."""
        from fujitsu_vrp.data.synthetic_generator import TokyoSMEGenerator

        generator = TokyoSMEGenerator(seed=42, num_customers=10, num_vehicles=3)
        instance = generator.generate()

        assert instance.num_customers == 10
        assert instance.num_vehicles == 3
        assert len(instance.customers) == 10
        assert len(instance.vehicles) == 3
        assert instance.distance_matrix.shape == (11, 11)  # Depot + 10 customers

    def test_distance_matrix_symmetry(self):
        """Test that distance matrix is symmetric."""
        from fujitsu_vrp.data.synthetic_generator import TokyoSMEGenerator

        generator = TokyoSMEGenerator(seed=42, num_customers=10)
        instance = generator.generate()

        # Check symmetry
        assert np.allclose(instance.distance_matrix, instance.distance_matrix.T)

    def test_customer_locations(self):
        """Test that customer locations are within Tokyo bounds."""
        from fujitsu_vrp.data.synthetic_generator import TokyoSMEGenerator, TOKYO_BOUNDS

        generator = TokyoSMEGenerator(seed=42, num_customers=20)
        instance = generator.generate()

        for customer in instance.customers:
            assert TOKYO_BOUNDS.min_lon <= customer.x <= TOKYO_BOUNDS.max_lon
            assert TOKYO_BOUNDS.min_lat <= customer.y <= TOKYO_BOUNDS.max_lat


class TestORToolsSolver:
    """Tests for OR-Tools baseline solver."""

    def test_solver_creation(self):
        """Test solver initialization."""
        from fujitsu_vrp.classical.ortools_solver import ORToolsSolver, SolverConfig

        config = SolverConfig(time_limit_seconds=30)
        solver = ORToolsSolver(config)
        assert solver.config.time_limit_seconds == 30

    def test_solve_small_instance(self):
        """Test solving a small VRP instance."""
        from fujitsu_vrp.data.synthetic_generator import TokyoSMEGenerator
        from fujitsu_vrp.classical.ortools_solver import ORToolsSolver, SolverConfig

        generator = TokyoSMEGenerator(seed=42, num_customers=5, num_vehicles=2)
        instance = generator.generate()

        config = SolverConfig(time_limit_seconds=10)
        solver = ORToolsSolver(config)
        result = solver.solve(instance)

        assert result.status == "SUCCESS"
        assert result.solution is not None
        assert len(result.solution.routes) == 2
        assert result.total_distance > 0


class TestChaoticOperators:
    """Tests for chaotic operator selection."""

    def test_logistic_map(self):
        """Test logistic map chaotic dynamics."""
        from fujitsu_vrp.quantum.qlnrs.chaotic_operators import LogisticMap

        cmap = LogisticMap(r=4.0, seed=42)

        # Generate sequence
        sequence = cmap.get_sequence(100)
        assert len(sequence) == 100

        # Values should be in (0, 1)
        assert all(0 < x < 1 for x in sequence)

        # MLE should be ln(2)
        assert abs(cmap.mle - np.log(2)) < 0.01

    def test_tent_map(self):
        """Test tent map chaotic dynamics."""
        from fujitsu_vrp.quantum.qlnrs.chaotic_operators import TentMap

        cmap = TentMap(seed=42)
        sequence = cmap.get_sequence(100)

        assert len(sequence) == 100
        assert all(0 < x < 1 for x in sequence)
        assert abs(cmap.mle - np.log(2)) < 0.01

    def test_chebyshev_map(self):
        """Test Chebyshev map chaotic dynamics."""
        from fujitsu_vrp.quantum.qlnrs.chaotic_operators import ChebyshevMap

        cmap = ChebyshevMap(k=3, seed=42)
        sequence = cmap.get_sequence(100)

        assert len(sequence) == 100
        # MLE = ln(k)
        assert abs(cmap.mle - np.log(3)) < 0.01

    def test_operator_selector(self):
        """Test chaotic operator selection."""
        from fujitsu_vrp.quantum.qlnrs.chaotic_operators import (
            ChaoticOperatorSelector,
            LogisticMap,
        )
        from fujitsu_vrp.classical.lns_base import RandomDestroy, WorstDestroy

        operators = [RandomDestroy(), WorstDestroy()]
        selector = ChaoticOperatorSelector(
            operators=operators,
            chaotic_map=LogisticMap(seed=42),
        )

        # Select multiple times
        selected = [selector.select() for _ in range(10)]

        # Should select from available operators
        assert all(op in operators for op in selected)

        # Update performance
        selector.update_performance(selected[0], 1.0)
        assert selector.operator_scores[selected[0].name] > 0


class TestLyapunovAnalyzer:
    """Tests for Lyapunov exponent computation."""

    def test_lyapunov_computation(self):
        """Test Lyapunov exponent computation."""
        from fujitsu_vrp.quantum.qlnrs.lyapunov import LyapunovAnalyzer, TrajectoryPoint

        analyzer = LyapunovAnalyzer()

        # Add trajectory points
        for i in range(20):
            point = TrajectoryPoint(
                iteration=i,
                cost=100.0 - i * 0.5 + np.random.randn() * 0.1,
                distance=50.0 - i * 0.2,
                time_violation=0.0,
                capacity_violation=0.0,
            )
            analyzer.add_point(point)

        result = analyzer.compute_lyapunov()

        assert result.exponent >= 0
        assert result.stability in ["stable", "periodic", "chaotic", "unknown"]

    def test_adaptive_controller(self):
        """Test adaptive Lyapunov controller."""
        from fujitsu_vrp.quantum.qlnrs.lyapunov import AdaptiveLyapunovController
        from fujitsu_vrp.data.problem_builder import Solution

        controller = AdaptiveLyapunovController(
            initial_temperature=100.0,
            target_lyapunov=0.25,
        )

        # Create a simple solution
        solution = Solution(
            instance_name="test",
            routes=[[0, 1, 2, 0]],
            total_distance=100.0,
            total_time=50.0,
            total_demand_served=30.0,
            time_window_violations=0.0,
            capacity_violations=0.0,
        )

        # Update with solution
        params = controller.update(solution, 0, 100.0)

        assert "temperature" in params
        assert "lyapunov" in params


class TestQUBOEncoder:
    """Tests for QUBO encoding."""

    def test_encoder_creation(self):
        """Test QUBO encoder initialization."""
        from fujitsu_vrp.quantum.qubo.encoder import VRP_QUBOEncoder, EncodingConfig

        config = EncodingConfig(max_qubits=40)
        encoder = VRP_QUBOEncoder(config)
        assert encoder.config.max_qubits == 40

    def test_small_encoding(self):
        """Test encoding a small repair subproblem."""
        from fujitsu_vrp.data.synthetic_generator import TokyoSMEGenerator
        from fujitsu_vrp.quantum.qubo.encoder import VRP_QUBOEncoder, EncodingConfig

        # Create small instance
        generator = TokyoSMEGenerator(seed=42, num_customers=5, num_vehicles=2)
        instance = generator.generate()

        # Create partial solution
        routes = [[0, 1, 2, 0], [0, 3, 4, 0]]
        removed_customers = [5]

        # Encode
        config = EncodingConfig(max_qubits=40)
        encoder = VRP_QUBOEncoder(config)
        qubo = encoder.encode_repair_subproblem(instance, routes, removed_customers)

        assert qubo.num_variables > 0
        assert qubo.num_variables <= config.max_qubits
        assert qubo.Q.shape[0] == qubo.Q.shape[1]


class TestSimulatedAnnealing:
    """Tests for simulated annealing solver."""

    def test_sa_solve(self):
        """Test SA solver on small QUBO."""
        from fujitsu_vrp.quantum.qubo.encoder import QUBOProblem
        from fujitsu_vrp.quantum.solvers.quantum_annealing import (
            SimulatedAnnealingSolver,
            SAConfig,
        )

        # Create simple QUBO
        n = 10
        Q = np.random.randn(n, n)
        Q = (Q + Q.T) / 2  # Make symmetric

        qubo = QUBOProblem(
            Q=Q,
            variable_mapping=[(i, 0, 0) for i in range(n)],
            num_variables=n,
            penalties={},
            num_customers=n,
            num_routes=1,
            removed_customers=list(range(1, n + 1)),
        )

        config = SAConfig(num_sweeps=1000, seed=42)
        solver = SimulatedAnnealingSolver(config)
        result = solver.solve(qubo)

        assert result.best_solution is not None
        assert len(result.best_solution) == n
        assert all(x in [0, 1] for x in result.best_solution)


class TestLNSBase:
    """Tests for LNS base implementation."""

    def test_destroy_operators(self):
        """Test destroy operators."""
        from fujitsu_vrp.classical.lns_base import (
            RandomDestroy,
            WorstDestroy,
            RelatedDestroy,
            RiskWeightedDestroy,
        )
        from fujitsu_vrp.data.synthetic_generator import TokyoSMEGenerator
        from fujitsu_vrp.data.problem_builder import Solution
        import random

        # Create instance and solution
        generator = TokyoSMEGenerator(seed=42, num_customers=10, num_vehicles=2)
        instance = generator.generate()

        solution = Solution(
            instance_name=instance.name,
            routes=[[0, 1, 2, 3, 0], [0, 4, 5, 6, 7, 8, 9, 10, 0]],
            total_distance=100.0,
            total_time=50.0,
            total_demand_served=50.0,
            time_window_violations=0.0,
            capacity_violations=0.0,
        )

        rng = random.Random(42)

        # Test RandomDestroy
        destroy = RandomDestroy()
        removed, routes = destroy.destroy(solution, 3, instance, rng)
        assert len(removed) == 3
        assert len(routes) == 2

        # Test WorstDestroy
        destroy = WorstDestroy()
        removed, routes = destroy.destroy(solution, 3, instance, rng)
        assert len(removed) == 3

        # Test RelatedDestroy
        destroy = RelatedDestroy()
        removed, routes = destroy.destroy(solution, 3, instance, rng)
        assert len(removed) == 3

        # Test RiskWeightedDestroy
        destroy = RiskWeightedDestroy()
        removed, routes = destroy.destroy(solution, 3, instance, rng)
        assert len(removed) == 3

    def test_repair_operators(self):
        """Test repair operators."""
        from fujitsu_vrp.classical.lns_base import GreedyRepair, RegretRepair
        from fujitsu_vrp.data.synthetic_generator import TokyoSMEGenerator
        import random

        # Create instance
        generator = TokyoSMEGenerator(seed=42, num_customers=5, num_vehicles=2)
        instance = generator.generate()

        # Partial solution (customers removed)
        removed_customers = [1, 2, 3]
        partial_routes = [[0, 0], [0, 4, 5, 0]]

        rng = random.Random(42)

        # Test GreedyRepair
        repair = GreedyRepair()
        routes = repair.repair(removed_customers, partial_routes, instance, rng)

        # All customers should be in routes
        all_in_routes = [c for route in routes for c in route]
        for c in removed_customers:
            assert c in all_in_routes

        # Test RegretRepair
        repair = RegretRepair(k=2)
        routes = repair.repair(removed_customers, partial_routes, instance, rng)

        all_in_routes = [c for route in routes for c in route]
        for c in removed_customers:
            assert c in all_in_routes


class TestQLNRSAlgorithm:
    """Tests for QLNRS algorithm."""

    def test_qlnrs_solve_small(self):
        """Test QLNRS on small instance."""
        from fujitsu_vrp.quantum.qlnrs.algorithm import QLNRS, QLNRSConfig
        from fujitsu_vrp.data.synthetic_generator import TokyoSMEGenerator

        # Create small instance
        generator = TokyoSMEGenerator(seed=42, num_customers=5, num_vehicles=2)
        instance = generator.generate()

        # Configure QLNRS
        config = QLNRSConfig(
            max_iterations=10,
            use_quantum_repair=False,  # Use classical repair for testing
            seed=42,
        )

        solver = QLNRS(config)
        result = solver.solve(instance)

        assert result.best_solution is not None
        assert result.iterations > 0
        assert result.best_cost > 0
        assert len(result.lyapunov_history) > 0


class TestRiskMetrics:
    """Tests for risk evaluation."""

    def test_risk_evaluation(self):
        """Test risk metric computation."""
        from fujitsu_vrp.analysis.risk_metrics import RiskEvaluator, compute_solution_risk
        from fujitsu_vrp.data.synthetic_generator import TokyoSMEGenerator
        from fujitsu_vrp.data.problem_builder import Solution

        generator = TokyoSMEGenerator(seed=42, num_customers=5, num_vehicles=2)
        instance = generator.generate()

        solution = Solution(
            instance_name=instance.name,
            routes=[[0, 1, 2, 0], [0, 3, 4, 5, 0]],
            total_distance=50.0,
            total_time=30.0,
            total_demand_served=30.0,
            time_window_violations=0.0,
            capacity_violations=0.0,
        )

        metrics = compute_solution_risk(solution, instance)

        assert metrics.total_risk >= 0
        assert metrics.time_window_risk >= 0
        assert metrics.capacity_risk >= 0
        assert metrics.operational_risk >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])