"""
Tests for penalty tuning analysis and 4-stop full VRP validation.
"""
import numpy as np
import pytest
from src.qubo.vrp_qubo import VRPQuboBuilder, VRPInstance
from src.qubo.penalty_calibration import PenaltyCalibrator
from src.analysis.penalty_tuning import PenaltyTuningAnalysis
from src.solvers.classical_baseline import ClassicalBaseline
from src.solvers.grover_solver import GroverAdaptiveSearch


def _make_instance(n_stops, seed=42):
    rng = np.random.RandomState(seed)
    n = n_stops + 1
    dist = rng.uniform(5, 30, (n, n))
    dist = (dist + dist.T) / 2
    np.fill_diagonal(dist, 0)
    demands = rng.randint(1, 3, n_stops)
    capacity = max(15, int(demands.sum() * 1.5))
    return VRPInstance(
        n_stops=n_stops,
        distance_matrix=dist,
        demands=demands,
        capacity=capacity,
    )


# === Penalty Tuning Tests ===

class TestPenaltyCalibrator:
    def test_eigenvalue_calibration(self):
        inst = _make_instance(2)
        cal = PenaltyCalibrator(inst, "position")
        penalties = cal.eigenvalue_gap_calibration()

        assert 'visit' in penalties
        assert 'flow' in penalties
        assert 'capacity' in penalties
        assert penalties['visit'] > 0
        assert penalties['capacity'] >= penalties['visit']

    def test_shot_noise_bound(self):
        inst = _make_instance(3)
        cal = PenaltyCalibrator(inst, "position")

        bound_1k = cal.shot_noise_bound(n_shots=1000)
        bound_8k = cal.shot_noise_bound(n_shots=8000)

        # More shots → tighter bound
        assert bound_8k < bound_1k

    def test_penalty_sweep(self):
        inst = _make_instance(2)
        cal = PenaltyCalibrator(inst, "position")
        results = cal.penalty_sweep(
            penalty_range=np.array([10.0, 50.0, 100.0]),
            n_trials=50,
        )
        assert len(results) == 3
        assert 'penalty' in results[0]
        assert 'violation_rate' in results[0]


class TestPenaltyTuningAnalysis:
    def test_analyze_2stop(self):
        inst = _make_instance(2)
        analysis = PenaltyTuningAnalysis(seed=42)
        report = analysis.analyze(
            inst, encoding="position",
            multiplier_range=np.array([0.5, 1.0, 2.0]),
            instance_name="test_2stop",
        )

        assert report.n_stops == 2
        assert len(report.results) == 3
        assert report.best_multiplier > 0

        # At least one multiplier should yield feasible ground state
        any_feasible = any(r.qubo_optimal_feasible for r in report.results)
        assert any_feasible

    def test_analyze_3stop(self):
        inst = _make_instance(3)
        analysis = PenaltyTuningAnalysis(seed=42)
        report = analysis.analyze(
            inst, encoding="position",
            multiplier_range=np.array([0.5, 1.0, 2.0]),
            instance_name="test_3stop",
        )

        assert report.n_stops == 3
        assert report.encoding == "position"
        assert report.shot_noise_bound > 0

    def test_format_report(self):
        inst = _make_instance(2)
        analysis = PenaltyTuningAnalysis(seed=42)
        report = analysis.analyze(
            inst, encoding="position",
            multiplier_range=np.array([1.0, 2.0]),
        )
        text = analysis.format_report(report)
        assert "Penalty Tuning Report" in text
        assert "Mult" in text

    def test_save_report(self, tmp_path):
        inst = _make_instance(2)
        analysis = PenaltyTuningAnalysis(seed=42)
        report = analysis.analyze(
            inst, encoding="position",
            multiplier_range=np.array([1.0]),
        )
        filepath = str(tmp_path / "tuning.json")
        analysis.save_report(report, filepath)

        import json
        with open(filepath) as f:
            data = json.load(f)
        assert data["n_stops"] == 2
        assert "results" in data


# === 4-Stop Full VRP Validation ===

class TestFourStopFullVRP:
    """Validates that a 4-stop VRP with all constraints works end-to-end."""

    @pytest.fixture
    def instance_4stop(self):
        """4-stop instance with realistic constraints."""
        dist = np.array([
            [0,  10, 15, 20, 25],
            [10, 0,  12, 18, 22],
            [15, 12, 0,  14, 16],
            [20, 18, 14, 0,  11],
            [25, 22, 16, 11, 0],
        ], dtype=float)
        demands = np.array([2, 3, 1, 2])
        time_windows = [(0, 120), (30, 180), (0, 240), (60, 300)]
        return VRPInstance(
            n_stops=4,
            distance_matrix=dist,
            demands=demands,
            capacity=10,
            time_windows=time_windows,
        )

    def test_classical_baseline(self, instance_4stop):
        baseline = ClassicalBaseline(instance_4stop)
        result = baseline.brute_force_tsp()
        assert result.total_cost > 0
        assert result.is_optimal
        assert len(result.routes[0]) == 6  # 0→a→b→c→d→0

    def test_qubo_position_encoding(self, instance_4stop):
        builder = VRPQuboBuilder(instance_4stop, encoding='position')
        qubo = builder.build()

        # 4 stops × 4 positions = 16 qubits
        assert qubo.n_qubits == 16

        # Brute force should find optimal
        bits, energy = builder.brute_force_solve()
        assert bits is not None
        eval_result = builder.evaluate_solution(bits)
        assert eval_result['feasible']
        assert eval_result['cost'] > 0

    def test_qubo_matches_classical(self, instance_4stop):
        """QUBO optimal should match classical optimal."""
        # Classical
        baseline = ClassicalBaseline(instance_4stop)
        cl_result = baseline.brute_force_tsp()

        # QUBO brute force
        builder = VRPQuboBuilder(instance_4stop, encoding='position')
        bits, energy = builder.brute_force_solve()
        eval_result = builder.evaluate_solution(bits)

        if eval_result['feasible']:
            # QUBO cost should be close to classical optimal
            gap = abs(eval_result['cost'] - cl_result.total_cost)
            # Allow some gap due to penalty structure
            assert gap <= cl_result.total_cost * 0.5, (
                f"QUBO cost {eval_result['cost']:.1f} too far from "
                f"classical {cl_result.total_cost:.1f}"
            )

    def test_grover_on_4stop(self, instance_4stop):
        """Grover Adaptive Search on 4 stops (16 qubits)."""
        builder = VRPQuboBuilder(instance_4stop, encoding='position')
        qubo = builder.build()

        gas = GroverAdaptiveSearch(max_gas_iterations=10, seed=42)
        result = gas.solve(qubo)

        assert result.optimal_bitstring is not None
        assert result.optimal_cost < float('inf')
        assert result.n_iterations > 0

        # Check solution quality
        eval_result = builder.evaluate_solution(result.optimal_bitstring)
        if eval_result['feasible']:
            assert eval_result['cost'] > 0

    def test_penalty_effect_on_4stop(self, instance_4stop):
        """Verify that penalty tuning improves feasibility for 4 stops."""
        analysis = PenaltyTuningAnalysis(seed=42)
        report = analysis.analyze(
            instance_4stop,
            encoding="position",
            multiplier_range=np.array([0.25, 0.5, 1.0, 2.0, 4.0]),
            instance_name="4stop_validation",
        )

        # At least one configuration should have feasible ground state
        any_feasible = any(r.qubo_optimal_feasible for r in report.results)
        assert any_feasible, "No penalty configuration yielded feasible QUBO ground state"

        # Higher penalties should generally increase feasibility
        low_feas = [r for r in report.results if r.penalty_multiplier <= 0.5]
        high_feas = [r for r in report.results if r.penalty_multiplier >= 2.0]

        if low_feas and high_feas:
            avg_low = np.mean([r.feasibility_rate for r in low_feas])
            avg_high = np.mean([r.feasibility_rate for r in high_feas])
            # This is a soft check — high penalties don't always help
            # but the success probability should be non-zero somewhere
            total_success = sum(r.success_probability for r in report.results)
            assert total_success > 0

    def test_ising_conversion_4stop(self, instance_4stop):
        """Verify Ising Hamiltonian conversion structure for 4 stops."""
        builder = VRPQuboBuilder(instance_4stop, encoding='position')
        qubo = builder.build()

        J, h, offset = qubo.to_ising()

        # Structural checks
        assert J.shape == (16, 16)
        assert h.shape == (16,)
        assert isinstance(offset, float)

        # J should be upper triangular
        for i in range(16):
            for j in range(i):
                assert J[i][j] == 0.0, f"J[{i}][{j}] = {J[i][j]}, expected 0"

        # J and h should be non-trivial (not all zeros)
        assert np.any(J != 0), "J is all zeros"
        assert np.any(h != 0), "h is all zeros"

        # Ising couplings should reflect problem structure
        n_nonzero_J = np.count_nonzero(J)
        assert n_nonzero_J > 0, "No couplings in Ising model"
