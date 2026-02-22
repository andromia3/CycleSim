"""Tests for the AR(2) + regime-switching market cycle engine."""

import numpy as np
import pytest

from cyclesim.market import calibrate_ar2, simulate_market_paths


@pytest.fixture
def default_params():
    return calibrate_ar2()


@pytest.fixture
def rng():
    return np.random.default_rng(42)


class TestCalibration:
    def test_default_calibration_is_stationary(self, default_params):
        assert default_params.is_stationary

    def test_default_calibration_is_oscillatory(self, default_params):
        assert default_params.is_oscillatory

    def test_implied_cycle_period_near_target(self, default_params):
        period = default_params.implied_cycle_period
        assert 5 <= period <= 12, f"Period {period} outside expected range"

    def test_direct_override(self):
        params = calibrate_ar2(phi_1_override=1.1, phi_2_override=-0.45, sigma_override=0.10)
        assert params.phi_1 == 1.1
        assert params.phi_2 == -0.45
        assert params.sigma_epsilon == 0.10

    def test_nondefault_period(self):
        params = calibrate_ar2(cycle_period=6.0)
        assert params.is_oscillatory
        period = params.implied_cycle_period
        assert 4 <= period <= 9


class TestSimulation:
    def test_output_shapes(self, default_params, rng):
        result = simulate_market_paths(default_params, n_paths=100, n_years=20, rng=rng)
        assert result["rate_adequacy"].shape == (100, 20)
        assert result["market_loss_ratio"].shape == (100, 20)
        assert result["market_rate_change"].shape == (100, 20)
        assert result["regime"].shape == (100, 20)
        assert result["shock_mask"].shape == (100, 20)

    def test_determinism(self, default_params):
        r1 = simulate_market_paths(default_params, 50, 10, np.random.default_rng(99))
        r2 = simulate_market_paths(default_params, 50, 10, np.random.default_rng(99))
        np.testing.assert_array_equal(r1["rate_adequacy"], r2["rate_adequacy"])

    def test_loss_ratio_bounds(self, default_params, rng):
        result = simulate_market_paths(default_params, 500, 25, rng)
        lr = result["market_loss_ratio"]
        # Allow some headroom beyond stated bounds due to regime multipliers
        assert lr.min() >= 0.2, f"Min LR {lr.min():.3f} unreasonably low"
        assert lr.max() <= 1.0, f"Max LR {lr.max():.3f} unreasonably high"

    def test_mean_loss_ratio_near_long_run(self, default_params, rng):
        result = simulate_market_paths(default_params, 2000, 50, rng)
        mean_lr = result["market_loss_ratio"].mean()
        # Should be within ~5pp of long-run target
        assert abs(mean_lr - default_params.long_run_loss_ratio) < 0.08, (
            f"Mean LR {mean_lr:.3f} too far from {default_params.long_run_loss_ratio}"
        )

    def test_rate_change_bounds(self, default_params, rng):
        result = simulate_market_paths(default_params, 500, 25, rng)
        rc = result["market_rate_change"]
        assert rc.min() >= default_params.min_rate_change_pa
        assert rc.max() <= default_params.max_rate_change_pa

    def test_regimes_are_valid(self, default_params, rng):
        result = simulate_market_paths(default_params, 100, 20, rng)
        assert np.all(result["regime"] >= 0)
        assert np.all(result["regime"] <= 3)

    def test_shocks_occur(self, default_params, rng):
        result = simulate_market_paths(default_params, 500, 50, rng)
        shock_rate = result["shock_mask"].mean()
        # Should be roughly near shock_prob (8%)
        assert 0.02 < shock_rate < 0.20, f"Shock rate {shock_rate:.3f} unexpected"
