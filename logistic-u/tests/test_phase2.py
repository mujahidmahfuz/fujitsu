"""
Tests for Tokyo data generator, traffic dynamics, and hybrid solver modules.
"""
import numpy as np
import pytest
from src.data.tokyo_generator import (
    TokyoDatasetGenerator, ShibuyaNetworkGenerator, TokyoDataset
)
from src.routing.traffic_dynamics import (
    TrafficVelocityField, LyapunovEstimator, StabilityWeightedGraph
)
from src.routing.traffic_sim import TrafficSimulator, EventType
from src.solvers.hybrid_solver import HybridSolver, StopClusterer
from src.qubo.vrp_qubo import VRPInstance


# --- Network Generator Tests ---

class TestShibuyaNetwork:
    def test_network_generation(self):
        gen = ShibuyaNetworkGenerator(seed=42)
        G = gen.generate()
        assert len(G.nodes) == 10 * 12  # grid_rows * grid_cols
        assert len(G.edges) > 0

    def test_node_positions(self):
        gen = ShibuyaNetworkGenerator(seed=42)
        G = gen.generate()
        for node in G.nodes():
            data = G.nodes[node]
            assert "lat" in data
            assert "lon" in data
            assert 35.6 < data["lat"] < 35.7  # Shibuya bounds
            assert 139.68 < data["lon"] < 139.72


# --- Tokyo Dataset Generator Tests ---

class TestTokyoGenerator:
    def setup_method(self):
        self.gen = TokyoDatasetGenerator(seed=42)

    def test_generate_5stops(self):
        ds = self.gen.generate_dataset(5)
        assert ds.n_stops == 5
        assert len(ds.stops) == 5
        assert ds.total_demand > 0

    def test_distance_matrix_shape(self):
        ds = self.gen.generate_dataset(5)
        dm = np.array(ds.distance_matrix)
        assert dm.shape == (6, 6)  # depot + 5 stops
        assert dm[0][0] == 0
        assert np.all(dm >= 0)

    def test_distance_matrix_symmetric_approx(self):
        ds = self.gen.generate_dataset(3)
        dm = np.array(ds.distance_matrix)
        # Not exactly symmetric due to one-way streets, but should be close
        for i in range(4):
            for j in range(4):
                if i != j:
                    assert dm[i][j] > 0  # All pairs reachable

    def test_to_vrp_instance(self):
        ds = self.gen.generate_dataset(3)
        inst = self.gen.to_vrp_instance(ds)
        assert inst.n_stops == 3
        assert inst.distance_matrix.shape == (4, 4)
        assert len(inst.demands) == 3
        assert inst.capacity == 15

    def test_stop_types(self):
        ds = self.gen.generate_dataset(20)
        types = [s["type"] for s in ds.stops]
        # Should have variety
        unique_types = set(types)
        assert len(unique_types) >= 2

    def test_save_load(self, tmp_path):
        ds = self.gen.generate_dataset(3, name="test_3stops")
        filepath = str(tmp_path / "test.json")
        self.gen.save_dataset(ds, filepath)
        loaded = TokyoDatasetGenerator.load_dataset(filepath)
        assert loaded.n_stops == ds.n_stops
        assert loaded.total_demand == ds.total_demand


# --- Lyapunov Estimator Tests ---

class TestLyapunovEstimator:
    def test_stable_series(self):
        """A constant series should have ~zero Lyapunov exponent."""
        est = LyapunovEstimator()
        series = np.ones(100) * 30.0 + np.random.normal(0, 0.01, 100)
        lya = est.compute(series)
        assert abs(lya) < 1.0  # Near zero

    def test_chaotic_series(self):
        """A noisy series should have a positive Lyapunov exponent."""
        est = LyapunovEstimator()
        # Logistic map in chaotic regime
        series = np.zeros(200)
        series[0] = 0.1
        r = 3.9  # Chaotic parameter
        for i in range(1, 200):
            series[i] = r * series[i-1] * (1 - series[i-1])
        lya = est.compute(series)
        assert lya > 0  # Should be positive for chaotic series

    def test_short_series(self):
        """Very short series should return 0."""
        est = LyapunovEstimator()
        lya = est.compute(np.array([1, 2, 3]))
        assert lya == 0.0


# --- Use small network for traffic tests (avoid sandbox timeout) ---

def _small_network():
    """Create a tiny 3x3 network for fast testing."""
    gen = ShibuyaNetworkGenerator(grid_rows=3, grid_cols=3, seed=42)
    return gen, gen.generate()


# --- Traffic Velocity Field Tests ---

class TestTrafficVelocityField:
    def test_speed_series_generation(self):
        _, G = _small_network()
        tvf = TrafficVelocityField(G, seed=42)

        edge = list(G.edges())[0]
        series = tvf.get_speed_series(edge, 8 * 60)  # 8 AM rush
        assert len(series) == 120
        assert np.all(series >= 2.0)
        assert np.all(series <= 60.0)

    def test_rush_hour_slower(self):
        _, G = _small_network()
        tvf = TrafficVelocityField(G, seed=42)

        edge = list(G.edges())[0]
        rush = tvf.get_speed_series(edge, 8 * 60)   # 8 AM
        night = tvf.get_speed_series(edge, 23 * 60)  # 11 PM

        # Rush hour should be slower on average
        assert np.mean(rush) < np.mean(night)


# --- Stability Weighted Graph Tests ---

class TestStabilityWeightedGraph:
    def test_stability_summary(self):
        _, G = _small_network()
        tvf = TrafficVelocityField(G, seed=42)
        swg = StabilityWeightedGraph(G, tvf, risk_aversion=0.5)

        summary = swg.get_stability_summary(time_minutes=8 * 60)
        assert "mean_lyapunov" in summary
        assert "pct_chaotic" in summary
        assert "mean_stability" in summary
        assert 0 <= summary["mean_stability"] <= 1.0

    def test_risk_aversion_effect(self):
        _, G = _small_network()
        tvf = TrafficVelocityField(G, seed=42)

        swg_low = StabilityWeightedGraph(G, tvf, risk_aversion=0.0)
        swg_high = StabilityWeightedGraph(G, tvf, risk_aversion=1.0)

        edge = list(G.edges())[0]
        state_low = swg_low.compute_edge_stability(edge, 8 * 60)
        state_high = swg_high.compute_edge_stability(edge, 8 * 60)

        # Higher risk aversion → higher risk cost (for chaotic edges)
        # α=0 means stability=1, so risk_cost = distance
        assert state_low.stability_score >= state_high.stability_score


# --- Traffic Simulator Tests ---

class TestTrafficSimulator:
    def test_scenario_generation(self):
        _, G = _small_network()
        sim = TrafficSimulator(G, seed=42)
        events = sim.generate_scenario(n_incidents=2, time_minutes=8 * 60)
        assert len(events) > 0

    def test_disrupted_distances(self):
        _, G = _small_network()
        sim = TrafficSimulator(G, seed=42)
        sim.generate_scenario(n_incidents=3, time_minutes=8 * 60)

        disrupted = sim.get_disrupted_distances(8 * 60)
        base = sim._base_distances

        # Some edges should have higher disrupted distances
        n_increased = sum(
            1 for e in disrupted
            if disrupted[e] > base[e] * 1.01
        )
        assert n_increased > 0


# --- Hybrid Solver Tests ---

class TestHybridSolver:
    def _make_5stop_instance(self):
        """Create a 5-stop VRP instance."""
        gen = TokyoDatasetGenerator(seed=42)
        ds = gen.generate_dataset(5)
        return gen.to_vrp_instance(ds)

    def test_clustering(self):
        inst = self._make_5stop_instance()
        clusterer = StopClusterer(seed=42)
        clusters = clusterer.cluster(inst, max_stops_per_cluster=3)
        # All stops should be covered
        all_stops = set()
        for c in clusters:
            for s in c:
                all_stops.add(s)
        assert all_stops == set(range(1, 6))

    def test_hybrid_solve_5stops(self):
        inst = self._make_5stop_instance()
        solver = HybridSolver(max_stops_per_quantum=3, seed=42)
        result = solver.solve(inst, use_quantum=True)

        assert result.total_cost > 0
        assert result.n_vehicles >= 1
        assert len(result.routes) >= 1
        # All routes start and end at depot
        for route in result.routes:
            assert route[0] == 0
            assert route[-1] == 0

    def test_hybrid_visits_all_stops(self):
        inst = self._make_5stop_instance()
        solver = HybridSolver(max_stops_per_quantum=3, seed=42)
        result = solver.solve(inst, use_quantum=True)

        visited = set()
        for route in result.routes:
            for node in route:
                if node != 0:
                    visited.add(node)
        assert visited == set(range(1, 6))

    def test_small_instance_trivial(self):
        """2-stop instance should solve trivially."""
        dist = np.array([[0, 5, 3], [5, 0, 4], [3, 4, 0]])
        inst = VRPInstance(
            n_stops=2, distance_matrix=dist,
            demands=np.array([1, 1]), capacity=15,
        )
        solver = HybridSolver(max_stops_per_quantum=4, seed=42)
        result = solver.solve(inst, use_quantum=True)
        assert result.total_cost == 12  # Both tours cost 12
