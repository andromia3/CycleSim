"""Tests for the full simulation orchestrator."""

import numpy as np
import pytest
import time

from cyclesim.market import calibrate_ar2
from cyclesim.losses import LossParams
from cyclesim.insurer import InsurerParams
from cyclesim.simulator import SimulationConfig, run_simulation
from cyclesim.defaults import INSURER_A_PRESET, INSURER_B_PRESET


@pytest.fixture
def config():
    return SimulationConfig(
        n_paths=200,
        n_years=20,
        random_seed=42,
        market_params=calibrate_ar2(),
        loss_params=LossParams.from_defaults(),
        insurers=[
            InsurerParams.from_dict(INSURER_A_PRESET),
            InsurerParams.from_dict(INSURER_B_PRESET),
        ],
    )


class TestSimulatorOutput:
    def test_output_shapes(self, config):
        results = run_simulation(config)
        for key in ["gwp", "nwp", "gross_lr", "net_lr", "combined_ratio",
                     "uw_profit", "total_profit", "capital", "rorac",
                     "cumulative_profit", "is_ruined"]:
            assert results.insurer_a[key].shape == (200, 20), f"Shape mismatch for {key}"
            assert results.insurer_b[key].shape == (200, 20), f"Shape mismatch for {key}"

    def test_determinism(self, config):
        r1 = run_simulation(config)
        r2 = run_simulation(config)
        np.testing.assert_allclose(
            r1.insurer_a["cumulative_profit"],
            r2.insurer_a["cumulative_profit"],
            rtol=1e-10,
        )

    def test_gwp_starts_at_initial(self, config):
        results = run_simulation(config)
        mean_gwp_0 = results.insurer_a["gwp"][:, 0].mean()
        initial = config.insurer_a.initial_gwp
        assert abs(mean_gwp_0 / initial - 1) < 0.3, (
            f"Year 0 mean GWP {mean_gwp_0:.0f} too far from initial {initial:.0f}"
        )

    def test_cumulative_profit_increasing(self, config):
        results = run_simulation(config)
        mean_cum = results.insurer_a["cumulative_profit"].mean(axis=0)
        assert mean_cum[-1] > 0, "Mean cumulative profit should be positive"

    def test_summary_metrics_present(self, config):
        results = run_simulation(config)
        required_keys = [
            "mean_through_cycle_rorac", "prob_ruin", "var_95_cumulative",
            "tvar_95_cumulative", "profit_to_risk_ratio", "mean_combined_ratio",
            "attribution", "yearly_means", "percentile_bands",
        ]
        for key in required_keys:
            assert key in results.summary_a, f"Missing summary key: {key}"
            assert key in results.summary_b, f"Missing summary key: {key}"

    def test_signal_arrays_stored(self, config):
        """Verify strategy signal arrays are populated (decision transparency)."""
        results = run_simulation(config)
        for key in ["strategy_signal", "signal_own_lr", "signal_market_lr",
                     "signal_rate_adequacy", "signal_rate_change", "signal_capital"]:
            arr = results.insurer_a[key]
            assert arr.shape == (200, 20), f"Signal {key} wrong shape"
            assert np.all(np.isfinite(arr)), f"Signal {key} has NaN/Inf"


class TestSanity:
    def test_rorac_in_plausible_range(self, config):
        results = run_simulation(config)
        rorac_a = results.summary_a["mean_through_cycle_rorac"]
        rorac_b = results.summary_b["mean_through_cycle_rorac"]
        assert -0.05 < rorac_a < 0.80, f"RORAC A {rorac_a:.1%} out of range"
        assert -0.05 < rorac_b < 0.80, f"RORAC B {rorac_b:.1%} out of range"

    def test_combined_ratio_in_plausible_range(self, config):
        results = run_simulation(config)
        cr_a = results.summary_a["mean_combined_ratio"]
        cr_b = results.summary_b["mean_combined_ratio"]
        assert 0.70 < cr_a < 1.30, f"CR A {cr_a:.1%} out of range"
        assert 0.70 < cr_b < 1.30, f"CR B {cr_b:.1%} out of range"

    def test_ruin_probability_low(self, config):
        results = run_simulation(config)
        assert results.summary_a["prob_ruin"] < 0.10
        assert results.summary_b["prob_ruin"] < 0.10


class TestNStrategy:
    def test_three_strategies(self):
        """Verify N>2 strategies work correctly."""
        from cyclesim.defaults import INSURER_DEFAULTS
        config = SimulationConfig(
            n_paths=100,
            n_years=10,
            random_seed=42,
            market_params=calibrate_ar2(),
            loss_params=LossParams.from_defaults(),
            insurers=[
                InsurerParams.from_dict(INSURER_A_PRESET),
                InsurerParams.from_dict(INSURER_B_PRESET),
                InsurerParams.from_dict({**INSURER_DEFAULTS, "name": "Strategy C"}),
            ],
        )
        results = run_simulation(config)
        assert len(results.insurers) == 3
        assert len(results.summaries) == 3
        for i in range(3):
            assert results.insurers[i]["gwp"].shape == (100, 10)
            assert "mean_through_cycle_rorac" in results.summaries[i]

    def test_backward_compat_properties(self):
        config = SimulationConfig(
            n_paths=50,
            n_years=5,
            random_seed=42,
            market_params=calibrate_ar2(),
            loss_params=LossParams.from_defaults(),
            insurers=[
                InsurerParams.from_dict(INSURER_A_PRESET),
                InsurerParams.from_dict(INSURER_B_PRESET),
            ],
        )
        results = run_simulation(config)
        # Properties should work
        assert results.insurer_a is results.insurers[0]
        assert results.insurer_b is results.insurers[1]
        assert results.summary_a is results.summaries[0]
        assert results.summary_b is results.summaries[1]
        assert config.insurer_a is config.insurers[0]
        assert config.insurer_b is config.insurers[1]
        assert config.n_strategies == 2


class TestPerformance:
    def test_1000_paths_under_5_seconds(self):
        config = SimulationConfig(
            n_paths=1000,
            n_years=30,
            random_seed=42,
            market_params=calibrate_ar2(),
            loss_params=LossParams.from_defaults(),
            insurers=[
                InsurerParams.from_dict(INSURER_A_PRESET),
                InsurerParams.from_dict(INSURER_B_PRESET),
            ],
        )
        t0 = time.perf_counter()
        run_simulation(config)
        elapsed = time.perf_counter() - t0
        assert elapsed < 5.0, f"Simulation took {elapsed:.2f}s, target < 5s"


class TestCapitalModel:
    def test_with_imported_data(self):
        from cyclesim.io import import_capital_model
        from cyclesim.losses import CapitalModelData
        import pandas as pd

        rng = np.random.default_rng(99)
        df = pd.DataFrame({
            "gross_lr": rng.lognormal(np.log(0.55), 0.10, 1000),
            "net_lr": rng.lognormal(np.log(0.45), 0.08, 1000),
        })
        from io import BytesIO, StringIO
        buf = StringIO()
        df.to_csv(buf, index=False)
        buf.seek(0)

        cm = import_capital_model(
            BytesIO(buf.read().encode()),
            gross_column="gross_lr",
            net_column="net_lr",
        )

        config = SimulationConfig(
            n_paths=100,
            n_years=10,
            random_seed=42,
            market_params=calibrate_ar2(),
            loss_params=LossParams.from_defaults(),
            insurers=[
                InsurerParams.from_dict(INSURER_A_PRESET),
                InsurerParams.from_dict(INSURER_B_PRESET),
            ],
            capital_model=cm,
        )
        results = run_simulation(config)
        assert results.insurer_a["gwp"].shape == (100, 10)
        assert results.summary_a["mean_cumulative_profit"] != 0
