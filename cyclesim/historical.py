"""
Historical Lloyd's Market backtest engine (2001-2024).

Deterministic single-path replay of insurer strategies against actual
Lloyd's market history. No Monte Carlo — uses real combined ratios
from Lloyd's Annual Reports (public data).

Includes counterfactual decomposition to attribute performance gaps
to specific strategic factors using sequential factor-swap methodology.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .defaults import LOSS_DEFAULTS, REGIMES
from .insurer import (
    InsurerParams, compute_strategy_signal, compute_gwp_change,
    compute_dynamic_expense_ratio, compute_adverse_selection_penalty,
    compute_ri_cost,
)
from .losses import LossParams


# ---------------------------------------------------------------------------
# Lloyd's Market Data 2001-2024 (from publicly available Annual Reports)
# ---------------------------------------------------------------------------
REGIME_MAP = {name: idx for idx, name in enumerate(REGIMES)}

LLOYDS_HISTORICAL = [
    {"year": 2001, "market_lr": 0.78, "market_cr": 1.15, "rate_change": -0.03, "regime": "crisis",  "event": "9/11 + Enron"},
    {"year": 2002, "market_lr": 0.58, "market_cr": 0.98, "rate_change":  0.12, "regime": "hard",    "event": "Post-9/11 hardening"},
    {"year": 2003, "market_lr": 0.52, "market_cr": 0.91, "rate_change":  0.08, "regime": "hard",    "event": "Rate momentum"},
    {"year": 2004, "market_lr": 0.59, "market_cr": 0.99, "rate_change":  0.03, "regime": "firming", "event": "Hurricane Ivan"},
    {"year": 2005, "market_lr": 0.72, "market_cr": 1.07, "rate_change": -0.02, "regime": "crisis",  "event": "Katrina/Rita/Wilma"},
    {"year": 2006, "market_lr": 0.50, "market_cr": 0.83, "rate_change":  0.15, "regime": "hard",    "event": "Post-Katrina hardening"},
    {"year": 2007, "market_lr": 0.52, "market_cr": 0.87, "rate_change":  0.05, "regime": "hard",    "event": "Benign year"},
    {"year": 2008, "market_lr": 0.59, "market_cr": 0.97, "rate_change": -0.04, "regime": "firming", "event": "GFC + Hurricane Ike"},
    {"year": 2009, "market_lr": 0.58, "market_cr": 0.98, "rate_change": -0.02, "regime": "soft",    "event": "Recession softening"},
    {"year": 2010, "market_lr": 0.65, "market_cr": 1.07, "rate_change": -0.05, "regime": "soft",    "event": "Chile/NZ earthquakes"},
    {"year": 2011, "market_lr": 0.73, "market_cr": 1.14, "rate_change": -0.06, "regime": "crisis",  "event": "Tohoku/Thai floods"},
    {"year": 2012, "market_lr": 0.55, "market_cr": 0.91, "rate_change":  0.04, "regime": "firming", "event": "Superstorm Sandy"},
    {"year": 2013, "market_lr": 0.52, "market_cr": 0.87, "rate_change":  0.02, "regime": "firming", "event": "Benign year"},
    {"year": 2014, "market_lr": 0.54, "market_cr": 0.90, "rate_change": -0.01, "regime": "soft",    "event": "Capital influx"},
    {"year": 2015, "market_lr": 0.56, "market_cr": 0.93, "rate_change": -0.03, "regime": "soft",    "event": "Tianjin + softening"},
    {"year": 2016, "market_lr": 0.58, "market_cr": 0.98, "rate_change": -0.04, "regime": "soft",    "event": "Deepening soft market"},
    {"year": 2017, "market_lr": 0.72, "market_cr": 1.15, "rate_change": -0.07, "regime": "crisis",  "event": "HIM hurricanes"},
    {"year": 2018, "market_lr": 0.55, "market_cr": 0.95, "rate_change":  0.06, "regime": "firming", "event": "Post-HIM adjustment"},
    {"year": 2019, "market_lr": 0.56, "market_cr": 0.98, "rate_change":  0.03, "regime": "soft",    "event": "Social inflation"},
    {"year": 2020, "market_lr": 0.68, "market_cr": 1.10, "rate_change":  0.05, "regime": "crisis",  "event": "COVID-19"},
    {"year": 2021, "market_lr": 0.54, "market_cr": 0.91, "rate_change":  0.09, "regime": "hard",    "event": "Rate hardening"},
    {"year": 2022, "market_lr": 0.53, "market_cr": 0.91, "rate_change":  0.08, "regime": "hard",    "event": "Ukraine + hard market"},
    {"year": 2023, "market_lr": 0.48, "market_cr": 0.84, "rate_change":  0.06, "regime": "hard",    "event": "Strong discipline"},
    {"year": 2024, "market_lr": 0.50, "market_cr": 0.86, "rate_change":  0.03, "regime": "hard",    "event": "Peak returns"},
]


@dataclass
class BacktestResult:
    """Complete backtest output for one insurer strategy."""
    years: np.ndarray
    gwp: np.ndarray
    nwp: np.ndarray
    cession_pct: np.ndarray
    gross_lr: np.ndarray
    net_lr: np.ndarray
    expense_ratio: np.ndarray
    combined_ratio: np.ndarray
    uw_profit: np.ndarray
    investment_income: np.ndarray
    ri_cost: np.ndarray
    total_profit: np.ndarray
    capital: np.ndarray
    solvency_ratio: np.ndarray
    rorac: np.ndarray
    cumulative_profit: np.ndarray
    gwp_change: np.ndarray
    # Terminal summary
    total_cumulative: float = 0.0
    cagr: float = 0.0
    max_drawdown_pct: float = 0.0
    worst_year: int = 0
    worst_year_profit: float = 0.0
    mean_rorac: float = 0.0
    ruin_year: Optional[int] = None


@dataclass
class CounterfactualDecomposition:
    """Attribution of performance gap to strategic factors."""
    total_gap: float
    growth_timing: float
    ri_purchasing: float
    shrink_discipline: float
    expense_efficiency: float
    descriptions: dict = field(default_factory=dict)


def derive_rate_adequacy(historical: list = None) -> np.ndarray:
    """
    Derive rate adequacy index from historical market data.

    Rate adequacy positive in hard markets (low LR, rates rising),
    negative in soft markets (high LR, rates falling).
    Normalized to roughly same scale as the AR(2) engine output.
    """
    if historical is None:
        historical = LLOYDS_HISTORICAL

    long_run_lr = 0.545  # Lloyd's 15-year average
    n = len(historical)
    ra = np.zeros(n)
    for i, h in enumerate(historical):
        ra[i] = (long_run_lr - h["market_lr"]) / long_run_lr
    return ra


def replay_historical(
    insurer_params: InsurerParams,
    loss_params: LossParams = None,
    historical: list = None,
) -> BacktestResult:
    """
    Deterministic single-path replay through historical Lloyd's data.

    Uses the same strategy logic as the MC engine (compute_strategy_signal,
    compute_gwp_change, etc.) but with actual market loss ratios instead
    of sampled ones.
    """
    if historical is None:
        historical = LLOYDS_HISTORICAL
    if loss_params is None:
        loss_params = LossParams.from_defaults()

    n_years = len(historical)
    rate_adequacy = derive_rate_adequacy(historical)
    years = np.array([h["year"] for h in historical])
    params = insurer_params

    # Output arrays
    gwp = np.zeros(n_years)
    nwp = np.zeros(n_years)
    cession_pct_arr = np.zeros(n_years)
    gross_lr_arr = np.zeros(n_years)
    net_lr_arr = np.zeros(n_years)
    expense_ratio_arr = np.zeros(n_years)
    combined_ratio_arr = np.zeros(n_years)
    uw_profit_arr = np.zeros(n_years)
    inv_income_arr = np.zeros(n_years)
    ri_cost_arr = np.zeros(n_years)
    total_profit_arr = np.zeros(n_years)
    capital_arr = np.zeros(n_years)
    solvency_arr = np.zeros(n_years)
    rorac_arr = np.zeros(n_years)
    cum_profit_arr = np.zeros(n_years)
    gwp_change_arr = np.zeros(n_years)

    # State
    curr_gwp = float(params.initial_gwp)
    initial_nwp = params.initial_gwp * (1 - params.base_cession_pct)
    curr_capital = initial_nwp * params.capital_ratio
    curr_cumulative = 0.0
    last_own_lr = params.expected_lr
    ruined = False
    ruin_year = None

    for t in range(n_years):
        h = historical[t]
        ra_t = np.array([rate_adequacy[t]])
        mkt_lr_t = np.array([h["market_lr"]])
        rate_chg_t = np.array([h["rate_change"]])

        # --- 1. Strategy signal ---
        denom_solv = curr_gwp * (1 - params.base_cession_pct) * params.capital_ratio
        denom_solv = max(denom_solv, 1.0)
        curr_solv = curr_capital / denom_solv

        signal = compute_strategy_signal(
            params,
            own_lr=np.array([last_own_lr]),
            market_lr=mkt_lr_t,
            rate_adequacy=ra_t,
            solvency_ratio=np.array([curr_solv]),
            rate_change=rate_chg_t,
        )

        # --- 2. GWP change ---
        if ruined:
            gwp_chg = 0.0
            new_gwp = 0.0
        else:
            gwp_chg = float(compute_gwp_change(params, signal)[0])
            new_gwp = curr_gwp * (1 + gwp_chg)

        # --- 3. Cession % ---
        cess = params.base_cession_pct + (
            params.cession_cycle_sensitivity * max(-rate_adequacy[t], 0)
            - params.cession_cycle_sensitivity * 0.5 * max(rate_adequacy[t], 0)
        )
        cess = float(np.clip(cess, params.min_cession_pct, params.max_cession_pct))
        new_nwp = new_gwp * (1 - cess)

        # --- 4. Loss ratios from actual history ---
        gross = h["market_lr"]

        # Adverse selection penalty
        market_gwp_change = gwp_chg
        adv_sel = float(compute_adverse_selection_penalty(
            params, np.array([gwp_chg]), market_gwp_change
        )[0])
        gross *= adv_sel

        # Net LR using component-weighted average RI effectiveness
        avg_ri_eff = (
            loss_params.ri_effectiveness_attritional * 0.65
            + loss_params.ri_effectiveness_large * 0.20
            + loss_params.ri_effectiveness_cat * 0.15
        )
        net = gross * (1 - cess * avg_ri_eff)

        # --- 5. Dynamic expense ratio ---
        dyn_expense = float(compute_dynamic_expense_ratio(
            params, np.array([gwp_chg])
        )[0])

        # --- 6. RI cost ---
        lagged_ra = rate_adequacy[max(t - params.ri_cost_lag_years, 0)]
        ri_cost_val = float(compute_ri_cost(
            params,
            np.array([cess]),
            np.array([new_gwp]),
            np.array([lagged_ra]),
        )[0])

        # --- 7. Financials ---
        net_claims = new_nwp * net
        expenses = new_nwp * dyn_expense
        uw_pft = new_nwp - net_claims - expenses - ri_cost_val

        float_base = new_nwp * 0.5 + curr_capital
        inv_income = max(float_base, 0) * params.investment_return
        total_pft = uw_pft + inv_income

        # --- 8. Combined ratio ---
        if new_nwp > 1.0:
            cr = (net_claims + expenses + ri_cost_val) / new_nwp
        else:
            cr = float('nan')

        # --- 9. Capital update ---
        new_capital = curr_capital + total_pft
        econ_cap = max(new_nwp * params.capital_ratio, 1.0)
        solv = new_capital / econ_cap

        if solv < params.capital_injection_trigger:
            new_capital += econ_cap * 0.25

        if solv > params.dividend_extraction_trigger:
            excess = max(0, (new_capital - econ_cap * 1.5) * 0.5)
            new_capital -= excess

        solv = new_capital / econ_cap
        r = total_pft / econ_cap if econ_cap > 0 else 0.0

        # Ruin check
        if new_capital < params.ruin_threshold and not ruined:
            ruined = True
            ruin_year = h["year"]

        curr_cumulative += total_pft

        # --- Store ---
        gwp[t] = new_gwp
        nwp[t] = new_nwp
        cession_pct_arr[t] = cess
        gross_lr_arr[t] = gross
        net_lr_arr[t] = net
        expense_ratio_arr[t] = dyn_expense
        combined_ratio_arr[t] = cr
        uw_profit_arr[t] = uw_pft
        inv_income_arr[t] = inv_income
        ri_cost_arr[t] = ri_cost_val
        total_profit_arr[t] = total_pft
        capital_arr[t] = new_capital
        solvency_arr[t] = solv
        rorac_arr[t] = r
        cum_profit_arr[t] = curr_cumulative
        gwp_change_arr[t] = gwp_chg

        # Update state
        curr_gwp = new_gwp
        curr_capital = new_capital
        last_own_lr = gross

    # --- Terminal summary ---
    worst_idx = int(np.argmin(total_profit_arr))

    peak = np.maximum.accumulate(cum_profit_arr)
    drawdowns = peak - cum_profit_arr
    peak_safe = np.where(peak > 0, peak, 1.0)
    dd_pct = drawdowns / peak_safe
    max_dd = float(np.max(dd_pct)) if np.any(peak > 0) else 0.0

    if gwp[0] > 0 and gwp[-1] > 0:
        cagr = (gwp[-1] / gwp[0]) ** (1 / n_years) - 1
    else:
        cagr = 0.0

    return BacktestResult(
        years=years, gwp=gwp, nwp=nwp, cession_pct=cession_pct_arr,
        gross_lr=gross_lr_arr, net_lr=net_lr_arr, expense_ratio=expense_ratio_arr,
        combined_ratio=combined_ratio_arr, uw_profit=uw_profit_arr,
        investment_income=inv_income_arr, ri_cost=ri_cost_arr,
        total_profit=total_profit_arr, capital=capital_arr,
        solvency_ratio=solvency_arr, rorac=rorac_arr,
        cumulative_profit=cum_profit_arr, gwp_change=gwp_change_arr,
        total_cumulative=float(curr_cumulative),
        cagr=cagr, max_drawdown_pct=max_dd,
        worst_year=int(years[worst_idx]),
        worst_year_profit=float(total_profit_arr[worst_idx]),
        mean_rorac=float(np.nanmean(rorac_arr)),
        ruin_year=ruin_year,
    )


def counterfactual_decomposition(
    bt_a: BacktestResult,
    bt_b: BacktestResult,
    params_a: InsurerParams,
    params_b: InsurerParams,
    loss_params: LossParams = None,
    historical: list = None,
) -> CounterfactualDecomposition:
    """
    Decompose performance gap into strategic factors via sequential swap.

    Starting from B's parameters, sequentially swap in A's parameters
    one factor group at a time to isolate each factor's contribution.
    """
    if loss_params is None:
        loss_params = LossParams.from_defaults()

    base_gap = bt_a.total_cumulative - bt_b.total_cumulative

    # Factor groups
    growth_keys = ["growth_rate_when_profitable", "max_gwp_growth_pa"]
    ri_keys = [
        "base_cession_pct", "cession_cycle_sensitivity",
        "min_cession_pct", "max_cession_pct",
    ]
    shrink_keys = [
        "shrink_rate_when_unprofitable", "max_gwp_shrink_pa", "expected_lr",
    ]

    def make_hybrid(base_p, donor_p, keys):
        d = {}
        for f in InsurerParams.__dataclass_fields__:
            d[f] = getattr(base_p, f)
        for k in keys:
            if hasattr(donor_p, k):
                d[k] = getattr(donor_p, k)
        return InsurerParams(**d)

    # Sequential swap: B → B+A_growth → B+A_growth+A_ri → ... → A
    bt_1 = replay_historical(
        make_hybrid(params_b, params_a, growth_keys),
        loss_params, historical,
    )
    bt_2 = replay_historical(
        make_hybrid(params_b, params_a, growth_keys + ri_keys),
        loss_params, historical,
    )
    bt_3 = replay_historical(
        make_hybrid(params_b, params_a, growth_keys + ri_keys + shrink_keys),
        loss_params, historical,
    )

    growth_timing = bt_1.total_cumulative - bt_b.total_cumulative
    ri_purchasing = bt_2.total_cumulative - bt_1.total_cumulative
    shrink_discipline = bt_3.total_cumulative - bt_2.total_cumulative
    expense_efficiency = bt_a.total_cumulative - bt_3.total_cumulative

    descriptions = {
        "growth_timing": _describe_factor("Growth timing", growth_timing),
        "ri_purchasing": _describe_factor("RI purchasing", ri_purchasing),
        "shrink_discipline": _describe_factor("Shrink discipline", shrink_discipline),
        "expense_efficiency": _describe_factor("Expense efficiency", expense_efficiency),
    }

    return CounterfactualDecomposition(
        total_gap=base_gap,
        growth_timing=growth_timing,
        ri_purchasing=ri_purchasing,
        shrink_discipline=shrink_discipline,
        expense_efficiency=expense_efficiency,
        descriptions=descriptions,
    )


def _describe_factor(name: str, amount: float) -> str:
    abs_m = abs(amount) / 1e6
    fmt = f"£{abs_m:.0f}m" if abs_m >= 1 else f"£{abs_m:.1f}m"
    if amount > 0:
        return f"{name}: +{fmt} favouring A"
    elif amount < 0:
        return f"{name}: +{fmt} favouring B"
    return f"{name}: neutral"
