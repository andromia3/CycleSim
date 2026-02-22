"""
Insurer strategy model.

Implements the full per-year simulation logic including:
- Multi-signal strategy reactions (own LR, market LR, rate adequacy,
  rate change momentum, capital position)
- Adverse selection on rapid growth
- Dynamic expense ratio (growth/shrink penalties)
- Component-specific reinsurance with its own pricing cycle
- Capital management (injection/extraction triggers)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .defaults import INSURER_DEFAULTS


@dataclass
class InsurerParams:
    """All parameters defining an insurer's strategy."""
    name: str
    initial_gwp: float
    expense_ratio: float
    base_cession_pct: float
    investment_return: float
    capital_ratio: float
    cost_of_capital: float

    # Signal weights
    signal_weights: dict

    # Growth/shrink
    expected_lr: float
    growth_rate_when_profitable: float
    shrink_rate_when_unprofitable: float
    max_gwp_growth_pa: float
    max_gwp_shrink_pa: float

    # Adverse selection
    adverse_selection_sensitivity: float

    # Expense dynamics
    expense_growth_penalty: float
    expense_shrink_penalty: float
    expense_stability_bonus: float

    # Reinsurance
    cession_cycle_sensitivity: float
    min_cession_pct: float
    max_cession_pct: float

    # RI pricing
    ri_cost_base: float
    ri_cost_cycle_sensitivity: float
    ri_cost_lag_years: int

    # Capital
    capital_injection_trigger: float
    dividend_extraction_trigger: float
    ruin_threshold: float

    # Fixed/variable expense split (operating leverage)
    expense_fixed_pct: float = 0.60
    expense_base_gwp: float = None

    @classmethod
    def from_dict(cls, d: dict) -> "InsurerParams":
        defaults = {**INSURER_DEFAULTS}
        defaults.update(d)
        return cls(**{k: defaults[k] for k in cls.__dataclass_fields__})


def compute_strategy_signal(
    params: InsurerParams,
    own_lr: np.ndarray,              # (n_paths,) last year's own gross LR
    market_lr: np.ndarray,           # (n_paths,) last year's market LR
    rate_adequacy: np.ndarray,       # (n_paths,) current rate adequacy index
    solvency_ratio: np.ndarray,     # (n_paths,) current solvency ratio
    rate_change: np.ndarray = None,  # (n_paths,) current year market rate change
) -> np.ndarray:
    """
    Compute blended strategy signal from multiple inputs.

    Positive signal → "things are good" → grow.
    Negative signal → "things are bad" → shrink.

    Five signals: own loss ratio, market loss ratio, rate adequacy,
    rate change momentum (hardening → grow into it), capital position.

    Returns: (n_paths,) signal array.
    """
    # Normalize signal weights to sum to 1.0
    raw_w = params.signal_weights
    w_own = raw_w.get("own_lr", 0.35)
    w_mkt = raw_w.get("market_lr", 0.20)
    w_ra = raw_w.get("rate_adequacy", 0.20)
    w_rc = raw_w.get("rate_change", 0.15)
    w_cap = raw_w.get("capital_position", 0.10)
    w_sum = w_own + w_mkt + w_ra + w_rc + w_cap
    if w_sum > 0:
        w_own /= w_sum
        w_mkt /= w_sum
        w_ra /= w_sum
        w_rc /= w_sum
        w_cap /= w_sum

    # Own LR signal: below expected = positive, above = negative
    safe_expected_lr = max(params.expected_lr, 0.01)
    own_signal = (safe_expected_lr - own_lr) / safe_expected_lr

    # Market LR signal: below long-run mean = positive
    market_signal = (0.55 - market_lr) / 0.55  # normalized to long-run ~55%

    # Rate adequacy: positive = hard market = good for writing
    rate_signal = rate_adequacy

    # Rate change momentum: positive rate increase (hardening) = grow into it
    if rate_change is not None:
        # Normalize: +10% rate increase → signal ~1.0
        rate_chg_signal = rate_change / 0.10
    else:
        rate_chg_signal = np.zeros_like(own_lr)

    # Capital signal: high solvency = room to grow
    capital_signal = (solvency_ratio - 1.5) / 1.5  # centered at 1.5x

    blended = (
        w_own * own_signal
        + w_mkt * market_signal
        + w_ra * rate_signal
        + w_rc * rate_chg_signal
        + w_cap * capital_signal
    )

    return blended


def compute_gwp_change(
    params: InsurerParams,
    signal: np.ndarray,              # (n_paths,) blended strategy signal
) -> np.ndarray:
    """
    Determine GWP growth/shrink rate from strategy signal.

    Positive signal → grow at growth_rate, negative → shrink at shrink_rate.
    Scaled by signal magnitude, capped at max growth/shrink.
    """
    # Scale growth/shrink by signal magnitude (0-1 range)
    growth = np.where(
        signal > 0,
        params.growth_rate_when_profitable * np.minimum(np.abs(signal) * 2, 1),
        params.shrink_rate_when_unprofitable * np.minimum(np.abs(signal) * 2, 1),
    )
    return np.clip(growth, params.max_gwp_shrink_pa, params.max_gwp_growth_pa)


def compute_dynamic_expense_ratio(
    params: InsurerParams,
    gwp_change: np.ndarray,          # (n_paths,) GWP change rate this year
) -> np.ndarray:
    """
    Compute expense ratio adjusted for growth/shrink dynamics.

    - Rapid growth → higher expenses (hiring, systems, acquisition costs)
    - Rapid shrinkage → higher expenses (fixed costs over smaller base)
    - Stability → bonus (efficient operations)
    """
    base = params.expense_ratio

    # Growth penalty: +2pp per 10% growth
    growth_penalty = np.where(
        gwp_change > 0.03,
        params.expense_growth_penalty * gwp_change / 0.10,
        0.0,
    )

    # Shrink penalty: +1pp per 10% shrinkage
    shrink_penalty = np.where(
        gwp_change < -0.03,
        params.expense_shrink_penalty * np.abs(gwp_change) / 0.10,
        0.0,
    )

    # Stability bonus
    stability = np.where(
        np.abs(gwp_change) < 0.03,
        params.expense_stability_bonus,
        0.0,
    )

    return base + growth_penalty + shrink_penalty + stability


def compute_fixed_variable_expense(
    params: InsurerParams,
    gwp: np.ndarray,                 # (n_paths,) current year GWP
    nwp: np.ndarray,                 # (n_paths,) current year NWP
    gwp_change: np.ndarray,          # (n_paths,) GWP change rate this year
) -> np.ndarray:
    """
    Compute expense ratio with fixed/variable cost split (operating leverage).

    Fixed costs stay constant regardless of volume changes, so shrinking
    the book concentrates fixed costs over fewer premiums. Growing dilutes them.

    Falls back to compute_dynamic_expense_ratio when expense_fixed_pct == 0.

    Returns: (n_paths,) expense ratio relative to NWP.
    """
    fp = params.expense_fixed_pct
    if fp is None or fp <= 0:
        return compute_dynamic_expense_ratio(params, gwp_change)

    base_gwp = params.expense_base_gwp or params.initial_gwp
    base_er = params.expense_ratio

    # Fixed cost = absolute £ amount that doesn't scale with volume
    fixed_cost = base_er * fp * base_gwp
    # Variable cost = scales proportionally with current GWP
    variable_cost = base_er * (1 - fp) * gwp

    # Base expense ratio = total cost / NWP
    # Guard: when NWP is near zero (ruined paths), fall back to base rate
    safe_nwp = np.where(nwp > 1000.0, nwp, 1.0)
    base_expense_ratio = np.where(
        nwp > 1000.0,
        (fixed_cost + variable_cost) / safe_nwp,
        base_er,  # fallback for dead paths
    )

    # Apply growth/shrink penalties on top (same as dynamic model)
    growth_penalty = np.where(
        gwp_change > 0.03,
        params.expense_growth_penalty * gwp_change / 0.10,
        0.0,
    )
    shrink_penalty = np.where(
        gwp_change < -0.03,
        params.expense_shrink_penalty * np.abs(gwp_change) / 0.10,
        0.0,
    )
    stability = np.where(
        np.abs(gwp_change) < 0.03,
        params.expense_stability_bonus,
        0.0,
    )

    return base_expense_ratio + growth_penalty + shrink_penalty + stability


def compute_adverse_selection_penalty(
    params: InsurerParams,
    gwp_change: np.ndarray,          # (n_paths,) insurer's GWP change
    market_gwp_change: float,        # scalar: average market GWP change
) -> np.ndarray:
    """
    Compute adverse selection penalty on loss ratio.

    When insurer grows faster than market, they attract marginal risks.
    Penalty is multiplicative on loss ratio.

    Returns: (n_paths,) multiplicative factor >= 1.0.
    """
    excess_growth = np.maximum(gwp_change - market_gwp_change, 0)
    # 10% sensitivity: 10% excess growth → 1% worse LR
    penalty = 1.0 + params.adverse_selection_sensitivity * excess_growth
    return penalty


def compute_ri_cost(
    params: InsurerParams,
    cession_pct: np.ndarray,         # (n_paths,)
    gwp: np.ndarray,                 # (n_paths,)
    rate_adequacy_lagged: np.ndarray,# (n_paths,) lagged by ri_cost_lag_years
) -> np.ndarray:
    """
    Compute reinsurance cost (premium paid to reinsurers net of expected recovery).

    RI cost varies with the reinsurance cycle (lagged from primary).
    Hard RI market → higher cost; soft RI market → lower cost.

    Returns: (n_paths,) RI cost in absolute terms.
    """
    # RI cost rate: base + cycle adjustment (RI hardens after primary)
    ri_cost_rate = params.ri_cost_base + (
        params.ri_cost_cycle_sensitivity * rate_adequacy_lagged
    )
    ri_cost_rate = np.clip(ri_cost_rate, 0.15, 0.55)

    # Total RI cost = ceded premium * cost rate
    ceded_premium = gwp * cession_pct
    return ceded_premium * ri_cost_rate
