"""
AR(2) + Hidden Markov Regime-Switching market cycle engine.

Generates N simulated market paths, each spanning M years.
The AR(2) process provides the base cycle dynamics; a hidden Markov
regime overlay modulates volatility, loss ratio multipliers, and
introduces regime-dependent shocks (crisis states).

Fully vectorized across paths for performance.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .defaults import MARKET_DEFAULTS, REGIMES


@dataclass
class MarketParams:
    """Fully resolved market cycle parameters."""
    phi_0: float
    phi_1: float
    phi_2: float
    sigma_epsilon: float
    shock_prob: float
    shock_magnitude_std: float
    long_run_loss_ratio: float
    min_loss_ratio: float
    max_loss_ratio: float
    market_expense_ratio: float
    hard_market_rate_change_pa: float
    soft_market_rate_change_pa: float
    max_rate_change_pa: float
    min_rate_change_pa: float
    regime_transition_matrix: np.ndarray   # (4, 4)
    regime_lr_multipliers: np.ndarray      # (4,) indexed by REGIMES
    regime_vol_multipliers: np.ndarray     # (4,)

    @property
    def implied_cycle_period(self) -> float:
        disc = self.phi_1 ** 2 + 4 * self.phi_2
        if disc >= 0:
            return float("inf")
        cos_arg = np.clip(self.phi_1 / (2 * np.sqrt(-self.phi_2)), -1, 1)
        return 2 * np.pi / np.arccos(cos_arg)

    @property
    def is_stationary(self) -> bool:
        return (
            self.phi_1 + self.phi_2 < 1
            and self.phi_2 - self.phi_1 < 1
            and abs(self.phi_2) < 1
        )

    @property
    def is_oscillatory(self) -> bool:
        return self.phi_1 ** 2 + 4 * self.phi_2 < 0

    @property
    def long_run_mean(self) -> float:
        denom = 1 - self.phi_1 - self.phi_2
        if abs(denom) < 1e-10:
            return 0.0
        return self.phi_0 / denom


def calibrate_ar2(
    cycle_period: float = MARKET_DEFAULTS["cycle_period_years"],
    min_lr: float = MARKET_DEFAULTS["min_loss_ratio"],
    max_lr: float = MARKET_DEFAULTS["max_loss_ratio"],
    long_run_lr: float = MARKET_DEFAULTS["long_run_loss_ratio"],
    shock_prob: float = MARKET_DEFAULTS["shock_prob"],
    shock_magnitude_std: float = MARKET_DEFAULTS["shock_magnitude_std"],
    market_expense_ratio: float = MARKET_DEFAULTS["market_expense_ratio"],
    hard_rate: float = MARKET_DEFAULTS["hard_market_rate_change_pa"],
    soft_rate: float = MARKET_DEFAULTS["soft_market_rate_change_pa"],
    max_rate: float = MARKET_DEFAULTS["max_rate_change_pa"],
    min_rate: float = MARKET_DEFAULTS["min_rate_change_pa"],
    regime_transition: Optional[list] = None,
    regime_lr_mult: Optional[dict] = None,
    regime_vol_mult: Optional[dict] = None,
    phi_1_override: Optional[float] = None,
    phi_2_override: Optional[float] = None,
    sigma_override: Optional[float] = None,
) -> MarketParams:
    """
    Calibrate AR(2) + regime parameters from user-friendly inputs.

    Two modes:
    - User-friendly: set cycle_period, min_lr, max_lr → engine solves AR(2) coefficients
    - Direct: set phi_1_override, phi_2_override, sigma_override
    """
    omega = 2 * np.pi / cycle_period

    if phi_2_override is not None:
        phi_2 = phi_2_override
    else:
        # Solve phi_2 from desired period, starting from a reasonable anchor
        # For period T: cos(omega) = phi_1 / (2*sqrt(-phi_2))
        # Choose phi_2 to give a stable, oscillatory process
        phi_2 = -0.5 * (omega / np.pi) ** 0.5
        phi_2 = np.clip(phi_2, -0.75, -0.15)

    if phi_1_override is not None:
        phi_1 = phi_1_override
    else:
        phi_1 = 2 * np.sqrt(-phi_2) * np.cos(omega)
        # Ensure stationarity
        if phi_1 + phi_2 >= 0.99:
            phi_1 = 0.99 - phi_2

    # phi_0 = 0: the AR(2) output is deviation from equilibrium
    phi_0 = 0.0

    if sigma_override is not None:
        sigma_epsilon = sigma_override
    else:
        # Calibrate sigma so that the stationary 95% range maps to [min_lr, max_lr]
        # Stationary variance: Var(y) = sigma^2 / ((1-phi_2)*((1+phi_2)^2 - phi_1^2))
        denom = (1 - phi_2) * ((1 + phi_2) ** 2 - phi_1 ** 2)
        if denom <= 0.001:
            denom = 0.001
        amplitude = (max_lr - min_lr) / 2
        target_std = amplitude / 1.96  # 95% range
        sigma_epsilon = np.sqrt(target_std ** 2 * denom)
        sigma_epsilon = max(sigma_epsilon, 0.01)

    # Regime matrices
    if regime_transition is None:
        regime_transition = MARKET_DEFAULTS["regime_transition_matrix"]
    trans_matrix = np.array(regime_transition, dtype=np.float64)
    # Normalize rows (guard against zero-sum rows)
    row_sums = trans_matrix.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums > 0, row_sums, 1.0)
    trans_matrix /= row_sums

    lr_mult_dict = regime_lr_mult or MARKET_DEFAULTS["regime_lr_multipliers"]
    vol_mult_dict = regime_vol_mult or MARKET_DEFAULTS["regime_vol_multipliers"]
    lr_mult = np.array([lr_mult_dict[r] for r in REGIMES], dtype=np.float64)
    vol_mult = np.array([vol_mult_dict[r] for r in REGIMES], dtype=np.float64)

    return MarketParams(
        phi_0=phi_0,
        phi_1=phi_1,
        phi_2=phi_2,
        sigma_epsilon=sigma_epsilon,
        shock_prob=shock_prob,
        shock_magnitude_std=shock_magnitude_std,
        long_run_loss_ratio=long_run_lr,
        min_loss_ratio=min_lr,
        max_loss_ratio=max_lr,
        market_expense_ratio=market_expense_ratio,
        hard_market_rate_change_pa=hard_rate,
        soft_market_rate_change_pa=soft_rate,
        max_rate_change_pa=max_rate,
        min_rate_change_pa=min_rate,
        regime_transition_matrix=trans_matrix,
        regime_lr_multipliers=lr_mult,
        regime_vol_multipliers=vol_mult,
    )


def simulate_market_paths(
    params: MarketParams,
    n_paths: int,
    n_years: int,
    rng: np.random.Generator,
) -> dict:
    """
    Generate n_paths independent market cycle simulations.

    Returns dict of arrays, each shape (n_paths, n_years):
        - rate_adequacy: raw AR(2) output (centered at 0)
        - market_loss_ratio: mapped to [min_lr, max_lr] with regime overlay
        - market_rate_change: annual rate change %
        - regime: integer regime index (0=soft, 1=firming, 2=hard, 3=crisis)
        - shock_mask: boolean mask of shock years
    """
    # --- Pre-allocate ---
    y = np.zeros((n_paths, n_years), dtype=np.float64)
    regimes = np.zeros((n_paths, n_years), dtype=np.int32)
    shock_mask = np.zeros((n_paths, n_years), dtype=bool)

    # --- Initialize regime from stationary distribution ---
    P = params.regime_transition_matrix
    stationary_dist = compute_stationary_distribution(P)

    # Draw initial regimes
    regimes[:, 0] = rng.choice(len(REGIMES), size=n_paths, p=stationary_dist)

    # --- Initialize AR(2) from stationary variance ---
    denom = (1 - params.phi_2) * (
        (1 + params.phi_2) ** 2 - params.phi_1 ** 2
    )
    if denom > 0.001:
        stationary_var = params.sigma_epsilon ** 2 / denom
    else:
        stationary_var = params.sigma_epsilon ** 2
    stationary_std = np.sqrt(max(stationary_var, 1e-6))

    y[:, 0] = rng.normal(0, stationary_std, n_paths)
    if n_years > 1:
        y[:, 1] = rng.normal(0, stationary_std, n_paths)
        # Draw regime for t=1
        for r in range(len(REGIMES)):
            mask = regimes[:, 0] == r
            if mask.any():
                regimes[mask, 1] = _draw_regime(
                    P[r], mask.sum(), rng
                )

    # --- Generate base innovations ---
    base_eps = rng.normal(0, 1, (n_paths, n_years))

    # --- Shock draws ---
    shock_draws = rng.random((n_paths, n_years))
    shock_innovations = rng.normal(0, params.shock_magnitude_std, (n_paths, n_years))

    # --- Simulate AR(2) with regime modulation ---
    for t in range(2, n_years):
        # Regime transition
        for r in range(len(REGIMES)):
            mask = regimes[:, t - 1] == r
            if mask.any():
                regimes[mask, t] = _draw_regime(P[r], mask.sum(), rng)

        # Crisis override: shocks can trigger crisis
        shock_mask[:, t] = shock_draws[:, t] < params.shock_prob
        # Shocked paths have elevated chance of crisis
        crisis_boost = shock_mask[:, t] & (regimes[:, t] != 3)
        if crisis_boost.any():
            # 50% chance shocked paths flip to crisis
            flip = rng.random(crisis_boost.sum()) < 0.50
            crisis_indices = np.where(crisis_boost)[0][flip]
            regimes[crisis_indices, t] = 3  # crisis

        # Regime-dependent volatility
        vol_mult = params.regime_vol_multipliers[regimes[:, t]]
        eps = base_eps[:, t] * params.sigma_epsilon * vol_mult
        eps += shock_mask[:, t] * shock_innovations[:, t]

        y[:, t] = (
            params.phi_0
            + params.phi_1 * y[:, t - 1]
            + params.phi_2 * y[:, t - 2]
            + eps
        )

    # --- Map to observables ---
    lr_range = params.max_loss_ratio - params.min_loss_ratio
    lr_midpoint = params.long_run_loss_ratio

    # Normalize y to roughly [-1, +1] using stationary std
    y_normalized = y / (2 * stationary_std) if stationary_std > 0.001 else y

    # Base market loss ratio (from AR(2))
    base_lr = lr_midpoint + y_normalized * (lr_range / 2)

    # Apply regime multiplier
    regime_mult = params.regime_lr_multipliers[regimes]
    market_loss_ratio = base_lr * regime_mult

    # Clip to reasonable bounds
    market_loss_ratio = np.clip(
        market_loss_ratio,
        params.min_loss_ratio * 0.8,
        params.max_loss_ratio * 1.5,
    )

    # Rate change: inverse of loss ratio movement, mapped to rate range
    # Positive rate adequacy → hard market → positive rate change
    rate_range = params.hard_market_rate_change_pa - params.soft_market_rate_change_pa
    market_rate_change = (
        params.soft_market_rate_change_pa
        + (1 - (market_loss_ratio - params.min_loss_ratio) / lr_range) * rate_range
    )
    market_rate_change = np.clip(
        market_rate_change, params.min_rate_change_pa, params.max_rate_change_pa
    )

    # --- AR(2) residuals for model validation ---
    residuals = np.zeros_like(y)
    for t in range(2, n_years):
        residuals[:, t] = y[:, t] - (
            params.phi_0 + params.phi_1 * y[:, t - 1] + params.phi_2 * y[:, t - 2]
        )

    return {
        "rate_adequacy": y,
        "market_loss_ratio": market_loss_ratio,
        "market_rate_change": market_rate_change,
        "regime": regimes,
        "shock_mask": shock_mask,
        "params": params,
        "residuals": residuals,
        "ar2_y": y,
    }


def _draw_regime(
    transition_probs: np.ndarray, n: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw n next-regime samples from transition probability vector."""
    return rng.choice(len(transition_probs), size=n, p=transition_probs)


def get_regime_labels(regime_array: np.ndarray) -> list:
    """Convert integer regime array to string labels."""
    return [REGIMES[r] for r in regime_array.flat]


def compute_stationary_distribution(transition_matrix: np.ndarray) -> np.ndarray:
    """
    Compute stationary distribution of the Markov regime chain.

    Solves pi * P = pi, sum(pi) = 1 using eigendecomposition.
    Returns array of shape (4,) with regime probabilities.
    """
    try:
        eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
        stationary_idx = np.argmin(np.abs(eigenvalues - 1.0))
        pi = np.real(eigenvectors[:, stationary_idx])
        pi = np.abs(pi)
        total = pi.sum()
        if total > 0:
            pi /= total
        else:
            # Fallback to uniform
            pi = np.ones(len(pi)) / len(pi)
    except np.linalg.LinAlgError:
        pi = np.ones(transition_matrix.shape[0]) / transition_matrix.shape[0]
    return pi


def compute_regime_forecast(
    transition_matrix: np.ndarray,
    current_regime: int,
    n_years: int = 10,
) -> np.ndarray:
    """
    Compute forward regime probability distribution for each future year.

    Uses matrix powers: P(regime at t+k | current) = e_current @ P^k.

    Args:
        transition_matrix: (4, 4) Markov transition matrix
        current_regime: integer index of current regime (0-3)
        n_years: number of years to forecast

    Returns:
        (n_years, 4) array: probability of each regime at each future year.
        Row 0 = year 1 (one transition from now).
    """
    P = transition_matrix
    n_regimes = P.shape[0]
    state = np.zeros(n_regimes)
    current_regime = max(0, min(current_regime, n_regimes - 1))
    state[current_regime] = 1.0

    forecast = np.zeros((n_years, 4))
    for k in range(n_years):
        state = state @ P
        forecast[k] = state

    return forecast
