"""
Loss ratio simulation engine.

Two modes:
1. Parametric: LogNormal attritional + CompoundPoisson large + CompoundPoisson cat
   with cycle-dependent mean shift AND tail thickness modulation.
2. Capital model: draw from imported 10,000 sims with cycle adjustment.

Includes prior-year reserve development model.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from .defaults import LOSS_DEFAULTS


@dataclass
class LossParams:
    """Parameters for parametric loss ratio generation."""
    attritional_mean: float
    attritional_cv: float
    cycle_sensitivity: float
    tail_thickness_sensitivity: float
    large_loss_frequency: float
    large_loss_pareto_alpha: float
    large_loss_pareto_xmin: float
    large_loss_cap: float
    cat_frequency: float
    cat_severity_mean: float
    cat_severity_cv: float
    reserve_dev_mean: float
    reserve_dev_std: float
    reserve_dev_cycle_lag: int
    reserve_dev_soft_penalty: float
    ri_effectiveness_attritional: float
    ri_effectiveness_large: float
    ri_effectiveness_cat: float

    @classmethod
    def from_defaults(cls, overrides: Optional[dict] = None) -> "LossParams":
        d = {**LOSS_DEFAULTS}
        if overrides:
            d.update(overrides)
        return cls(**d)


@dataclass
class CapitalModelData:
    """Imported capital model simulation data."""
    gross_loss_ratios: np.ndarray   # (n_sims,)
    net_loss_ratios: np.ndarray     # (n_sims,)
    ri_recoveries: Optional[np.ndarray] = None  # (n_sims,) if separate
    ri_spend: Optional[np.ndarray] = None       # (n_sims,) if separate


def sample_parametric_losses(
    params: LossParams,
    rate_adequacy: np.ndarray,      # (n_paths,)
    n_paths: int,
    rng: np.random.Generator,
) -> dict:
    """
    Sample loss ratios for n_paths from parametric distributions.

    rate_adequacy: positive = hard market (lower losses), negative = soft market.

    Returns dict with keys: 'attritional', 'large', 'cat', 'gross_total'
    Each value is np.ndarray of shape (n_paths,).
    """
    # --- Attritional: cycle-adjusted lognormal with variable tail thickness ---
    # Mean shifts with cycle
    adj_mean = params.attritional_mean * (
        1 - params.cycle_sensitivity * rate_adequacy
    )
    adj_mean = np.clip(adj_mean, 0.20, 0.85)

    # CV increases in soft markets (tail thickness)
    # rate_adequacy < 0 = soft → fatter tails
    adj_cv = params.attritional_cv * (
        1 + params.tail_thickness_sensitivity * np.maximum(-rate_adequacy, 0)
    )

    # Convert mean/cv to lognormal mu/sigma (per-path)
    sigma_ln = np.sqrt(np.log(1 + adj_cv ** 2))
    mu_ln = np.log(adj_mean) - sigma_ln ** 2 / 2
    attritional = rng.lognormal(mu_ln, sigma_ln)

    # --- Large losses: compound Poisson-Pareto ---
    # Frequency increases in soft market
    adj_freq = params.large_loss_frequency * (
        1 + 0.15 * np.maximum(-rate_adequacy, 0)
    )
    n_events = rng.poisson(adj_freq)
    large = np.zeros(n_paths, dtype=np.float64)
    max_events = n_events.max() if n_events.max() > 0 else 0
    if max_events > 0 and params.large_loss_pareto_alpha > 0:
        # Vectorized: generate max_events severities for all paths,
        # then mask out excess events
        all_severities = (
            rng.pareto(params.large_loss_pareto_alpha, (n_paths, max_events)) + 1
        ) * params.large_loss_pareto_xmin
        all_severities = np.minimum(all_severities, params.large_loss_cap)
        # Mask: zero out events beyond n_events[i]
        event_idx = np.arange(max_events)[np.newaxis, :]
        mask = event_idx < n_events[:, np.newaxis]
        large = (all_severities * mask).sum(axis=1)

    # --- Catastrophe losses: compound Poisson-LogNormal ---
    n_cat = rng.poisson(params.cat_frequency, n_paths)
    cat = np.zeros(n_paths, dtype=np.float64)
    max_cat = n_cat.max() if n_cat.max() > 0 else 0
    if max_cat > 0 and params.cat_severity_mean > 0:
        cat_sigma = np.sqrt(np.log(1 + max(params.cat_severity_cv, 0.01) ** 2))
        cat_mu = np.log(params.cat_severity_mean) - cat_sigma ** 2 / 2
        all_cat = rng.lognormal(cat_mu, cat_sigma, (n_paths, max_cat))
        cat_idx = np.arange(max_cat)[np.newaxis, :]
        cat_mask = cat_idx < n_cat[:, np.newaxis]
        cat = (all_cat * cat_mask).sum(axis=1)

    return {
        "attritional": attritional,
        "large": large,
        "cat": cat,
        "gross_total": attritional + large + cat,
    }


def sample_capital_model_losses(
    capital_model: CapitalModelData,
    rate_adequacy: np.ndarray,      # (n_paths,)
    cycle_sensitivity: float,
    n_paths: int,
    rng: np.random.Generator,
) -> dict:
    """
    Draw from imported capital model sims with quantile-dependent cycle
    adjustment.

    Mirrors parametric model behaviour where attritional losses are highly
    cycle-sensitive, large losses are moderately sensitive (frequency-driven),
    and catastrophe losses are largely independent:
      - Below median (attritional-dominated): full cycle_sensitivity
      - Median to 90th pctile (large-loss territory): 0.7x sensitivity
      - Above 90th pctile (cat/extreme): 0.3x sensitivity
    """
    n_sims = len(capital_model.gross_loss_ratios)
    indices = rng.integers(0, n_sims, n_paths)
    gross = capital_model.gross_loss_ratios[indices].copy()
    net = capital_model.net_loss_ratios[indices].copy()

    # Pre-compute quantile thresholds from the full capital model distribution
    p50 = np.percentile(capital_model.gross_loss_ratios, 50)
    p90 = np.percentile(capital_model.gross_loss_ratios, 90)

    # Quantile-dependent sensitivity: attritional=full, large=0.7, cat=0.3
    sens_multiplier = np.where(
        gross < p50, 1.0,
        np.where(gross < p90, 0.7, 0.3)
    )

    cycle_factor = 1 - (cycle_sensitivity * sens_multiplier) * rate_adequacy
    gross *= cycle_factor
    net *= cycle_factor

    result = {"gross_total": gross, "net_total": net}

    if capital_model.ri_recoveries is not None:
        result["ri_recoveries"] = capital_model.ri_recoveries[indices] * cycle_factor
    if capital_model.ri_spend is not None:
        result["ri_spend"] = capital_model.ri_spend[indices].copy()

    return result


def compute_net_loss_ratio(
    gross_components: dict,
    cession_pct: np.ndarray,         # (n_paths,)
    params: LossParams,
) -> np.ndarray:
    """
    Compute net loss ratio from gross components and reinsurance structure.

    Different RI effectiveness by component:
    - Attritional: QS-like, high recovery
    - Large: XL-like, moderate recovery
    - Cat: Cat XL, high recovery
    """
    net_attritional = gross_components["attritional"] * (
        1 - cession_pct * params.ri_effectiveness_attritional
    )
    net_large = gross_components["large"] * (
        1 - cession_pct * params.ri_effectiveness_large
    )
    net_cat = gross_components["cat"] * (
        1 - cession_pct * params.ri_effectiveness_cat
    )
    return net_attritional + net_large + net_cat


def compute_reserve_development(
    historical_rate_adequacy: np.ndarray,  # (n_paths, n_years_so_far)
    current_year: int,
    params: LossParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute prior-year reserve development for current year.

    Soft-market underwriting (from `lag` years ago) develops adversely.
    Returns reserve development as a loss ratio impact (can be + or -).
    Shape: (n_paths,).
    """
    n_paths = historical_rate_adequacy.shape[0]

    # Base random development
    dev = rng.normal(params.reserve_dev_mean, params.reserve_dev_std, n_paths)

    # Add soft-market penalty from `lag` years ago
    lag = params.reserve_dev_cycle_lag
    if current_year >= lag and historical_rate_adequacy.shape[1] > current_year - lag:
        past_adequacy = historical_rate_adequacy[:, current_year - lag]
        # Negative rate_adequacy (soft market) → positive penalty
        soft_penalty = params.reserve_dev_soft_penalty * np.maximum(-past_adequacy, 0)
        dev += soft_penalty

    return dev
