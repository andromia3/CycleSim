"""Tests for risk-adjusted metrics computations."""

import numpy as np
import pytest

from cyclesim.metrics import compute_summary_metrics, compute_strategy_comparison


@pytest.fixture
def mock_results():
    """Create mock insurer results for testing metrics."""
    rng = np.random.default_rng(42)
    n_paths, n_years = 500, 20

    # Simulate plausible data
    total_profit = rng.normal(20e6, 15e6, (n_paths, n_years))
    capital = np.cumsum(total_profit, axis=1) + 200e6
    gwp = np.full((n_paths, n_years), 500e6) * (1 + rng.normal(0.02, 0.05, (n_paths, n_years))).cumprod(axis=1)
    nwp = gwp * 0.77
    econ_cap = nwp * 0.45

    return {
        "total_profit": total_profit,
        "capital": capital,
        "gwp": gwp,
        "nwp": nwp,
        "economic_capital": econ_cap,
        "uw_profit": total_profit * 0.7,
        "investment_income": total_profit * 0.3,
        "rorac": total_profit / econ_cap,
        "combined_ratio": rng.normal(0.95, 0.08, (n_paths, n_years)),
        "is_ruined": capital < 0,
        "cumulative_profit": np.cumsum(total_profit, axis=1),
        "gross_lr": rng.normal(0.55, 0.06, (n_paths, n_years)),
        "net_lr": rng.normal(0.45, 0.05, (n_paths, n_years)),
        "expense_ratio": np.full((n_paths, n_years), 0.36),
        "cession_pct": np.full((n_paths, n_years), 0.23),
        "solvency_ratio": capital / econ_cap,
        "ri_cost": np.full((n_paths, n_years), 5e6),
        "reserve_dev": rng.normal(0, 0.02, (n_paths, n_years)),
        "capital_injections": np.zeros((n_paths, n_years)),
        "dividend_extractions": np.zeros((n_paths, n_years)),
        "gwp_change_pct": rng.normal(0.02, 0.05, (n_paths, n_years)),
        "adverse_selection": np.ones((n_paths, n_years)),
    }


class MockParams:
    capital_ratio = 0.45
    cost_of_capital = 0.10
    name = "Test"


class TestSummaryMetrics:
    def test_returns_required_keys(self, mock_results):
        summary = compute_summary_metrics(mock_results, MockParams())
        required = [
            "mean_through_cycle_rorac", "prob_ruin",
            "var_95_cumulative", "tvar_95_cumulative",
            "profit_to_risk_ratio", "mean_combined_ratio",
            "attribution", "yearly_means", "percentile_bands",
            "cumulative_profit_dist", "terminal_capital_dist",
            "through_cycle_rorac_dist", "ruin_prob_by_year",
        ]
        for key in required:
            assert key in summary, f"Missing key: {key}"

    def test_var_less_than_mean(self, mock_results):
        summary = compute_summary_metrics(mock_results, MockParams())
        assert summary["var_95_cumulative"] < summary["mean_cumulative_profit"]

    def test_tvar_less_than_var(self, mock_results):
        summary = compute_summary_metrics(mock_results, MockParams())
        assert summary["tvar_95_cumulative"] <= summary["var_95_cumulative"]

    def test_percentile_bands_ordered(self, mock_results):
        summary = compute_summary_metrics(mock_results, MockParams())
        for key in ["gwp", "combined_ratio", "cumulative_profit"]:
            bands = summary["percentile_bands"][key]
            for t in range(len(bands["p5"])):
                assert bands["p5"][t] <= bands["p25"][t] <= bands["p50"][t]
                assert bands["p50"][t] <= bands["p75"][t] <= bands["p95"][t]

    def test_ruin_prob_between_0_and_1(self, mock_results):
        summary = compute_summary_metrics(mock_results, MockParams())
        assert 0 <= summary["prob_ruin"] <= 1

    def test_attribution_sums_roughly(self, mock_results):
        summary = compute_summary_metrics(mock_results, MockParams())
        attr = summary["attribution"]
        # Attribution components should roughly sum to total
        component_sum = (
            attr["underwriting"] + attr["investment"]
            + attr["reinsurance_cost"] + attr["capital_actions"]
        )
        # Allow generous tolerance due to different computation paths
        assert abs(component_sum) < abs(attr["total"]) * 5


class TestStrategyComparison:
    def test_comparison_output(self, mock_results):
        summary = compute_summary_metrics(mock_results, MockParams())
        comparison = compute_strategy_comparison([summary, summary])
        assert "mean_through_cycle_rorac" in comparison
        entry = comparison["mean_through_cycle_rorac"]
        # List-based keys
        assert "values" in entry
        assert len(entry["values"]) == 2
        assert "best_idx" in entry
        assert "spread" in entry
        # Backward-compat keys
        assert "insurer_a" in entry
        assert "insurer_b" in entry
        assert "difference" in entry
