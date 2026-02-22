"""
Main orchestrator: runs the full Monte Carlo simulation.

Coordinates market cycle generation, loss sampling, and insurer strategy
simulation. Vectorized across paths, sequential across years.

Supports 2-6 insurer strategies compared simultaneously.
"""

import numpy as np
import time
from dataclasses import dataclass, field
from typing import Optional

from .market import MarketParams, simulate_market_paths, calibrate_ar2
from .losses import (
    LossParams, CapitalModelData, sample_parametric_losses,
    sample_capital_model_losses, compute_net_loss_ratio,
    compute_reserve_development,
)
from .insurer import (
    InsurerParams, compute_strategy_signal, compute_gwp_change,
    compute_dynamic_expense_ratio, compute_fixed_variable_expense,
    compute_adverse_selection_penalty, compute_ri_cost,
)
from .metrics import compute_summary_metrics


@dataclass
class SimulationConfig:
    """Top-level simulation configuration."""
    n_paths: int
    n_years: int
    random_seed: int
    market_params: MarketParams
    loss_params: LossParams
    insurers: list = field(default_factory=list)  # list[InsurerParams]
    capital_model: Optional[CapitalModelData] = None
    discount_rate: float = 0.0

    # Backward-compat properties
    @property
    def insurer_a(self) -> InsurerParams:
        return self.insurers[0]

    @property
    def insurer_b(self) -> InsurerParams:
        return self.insurers[1] if len(self.insurers) > 1 else self.insurers[0]

    @property
    def n_strategies(self) -> int:
        return len(self.insurers)


@dataclass
class SimulationResults:
    """Complete output of a simulation run."""
    config: SimulationConfig
    market: dict                  # From simulate_market_paths
    insurers: list = field(default_factory=list)   # list[dict], per-metric arrays
    summaries: list = field(default_factory=list)   # list[dict], scalar aggregates
    elapsed_seconds: float = 0.0

    # Backward-compat properties
    @property
    def insurer_a(self) -> dict:
        return self.insurers[0] if self.insurers else {}

    @property
    def insurer_b(self) -> dict:
        return self.insurers[1] if len(self.insurers) > 1 else (self.insurers[0] if self.insurers else {})

    @property
    def summary_a(self) -> dict:
        return self.summaries[0] if self.summaries else {}

    @property
    def summary_b(self) -> dict:
        return self.summaries[1] if len(self.summaries) > 1 else (self.summaries[0] if self.summaries else {})


def run_simulation(config: SimulationConfig) -> SimulationResults:
    """Execute the full Monte Carlo simulation for all strategies."""
    t0 = time.perf_counter()
    rng = np.random.default_rng(config.random_seed)

    # Step 1: Generate market paths
    market = simulate_market_paths(
        config.market_params, config.n_paths, config.n_years, rng
    )

    # Step 2: Simulate all insurers
    insurers = []
    summaries = []
    for params in config.insurers:
        ins_data = _simulate_insurer(params, config, market, rng)
        insurers.append(ins_data)
        summaries.append(compute_summary_metrics(ins_data, params))

    elapsed = time.perf_counter() - t0

    return SimulationResults(
        config=config,
        market=market,
        insurers=insurers,
        summaries=summaries,
        elapsed_seconds=elapsed,
    )


def _simulate_insurer(
    params: InsurerParams,
    config: SimulationConfig,
    market: dict,
    rng: np.random.Generator,
) -> dict:
    """
    Vectorized insurer simulation across all paths.

    Year loop (sequential due to state dependency),
    path operations (vectorized via NumPy).
    """
    n_paths = config.n_paths
    n_years = config.n_years
    lp = config.loss_params
    rate_adequacy = market["rate_adequacy"]
    market_lr = market["market_loss_ratio"]

    # --- Pre-allocate output arrays ---
    gwp = np.zeros((n_paths, n_years))
    nwp = np.zeros((n_paths, n_years))
    cession_pct = np.zeros((n_paths, n_years))
    gross_lr = np.zeros((n_paths, n_years))
    net_lr = np.zeros((n_paths, n_years))
    attritional_lr = np.full((n_paths, n_years), np.nan)
    large_lr = np.full((n_paths, n_years), np.nan)
    cat_lr = np.full((n_paths, n_years), np.nan)
    expense_ratio = np.zeros((n_paths, n_years))
    combined_ratio = np.zeros((n_paths, n_years))
    uw_profit = np.zeros((n_paths, n_years))
    investment_income = np.zeros((n_paths, n_years))
    ri_cost = np.zeros((n_paths, n_years))
    total_profit = np.zeros((n_paths, n_years))
    capital = np.zeros((n_paths, n_years))
    economic_capital = np.zeros((n_paths, n_years))
    solvency_ratio = np.zeros((n_paths, n_years))
    rorac = np.zeros((n_paths, n_years))
    cumulative_profit = np.zeros((n_paths, n_years))
    pv_total_profit = np.zeros((n_paths, n_years))
    pv_cumulative_profit = np.zeros((n_paths, n_years))
    reserve_dev = np.zeros((n_paths, n_years))
    adverse_selection = np.zeros((n_paths, n_years))
    capital_injections = np.zeros((n_paths, n_years))
    dividend_extractions = np.zeros((n_paths, n_years))
    gwp_change_pct = np.zeros((n_paths, n_years))
    is_ruined = np.zeros((n_paths, n_years), dtype=bool)

    # Strategy signal arrays (for decision transparency)
    strategy_signal = np.zeros((n_paths, n_years))
    signal_own_lr = np.zeros((n_paths, n_years))
    signal_market_lr = np.zeros((n_paths, n_years))
    signal_rate_adequacy = np.zeros((n_paths, n_years))
    signal_rate_change = np.zeros((n_paths, n_years))
    signal_capital = np.zeros((n_paths, n_years))

    # --- Initialize state vectors ---
    curr_gwp = np.full(n_paths, params.initial_gwp)
    initial_nwp = params.initial_gwp * (1 - params.base_cession_pct)
    curr_capital = np.full(n_paths, initial_nwp * params.capital_ratio)
    curr_cumulative = np.zeros(n_paths)
    last_own_lr = np.full(n_paths, params.expected_lr)
    ruined = np.zeros(n_paths, dtype=bool)
    last_gwp_change = np.zeros(n_paths)

    for t in range(n_years):
        # --- 1. Strategy signal ---
        denom_solv = curr_gwp * (1 - params.base_cession_pct) * params.capital_ratio
        denom_solv = np.where(denom_solv > 1.0, denom_solv, 1.0)
        curr_solv = curr_capital / denom_solv

        mkt_lr_prev = market_lr[:, max(t - 1, 0)]
        ra_t = rate_adequacy[:, t]
        rc_t = market["market_rate_change"][:, t]

        signal = compute_strategy_signal(
            params,
            own_lr=last_own_lr,
            market_lr=mkt_lr_prev,
            rate_adequacy=ra_t,
            solvency_ratio=curr_solv,
            rate_change=rc_t,
        )

        # Store signal components for decision transparency
        strategy_signal[:, t] = signal
        safe_elr = max(params.expected_lr, 0.01)
        signal_own_lr[:, t] = (safe_elr - last_own_lr) / safe_elr
        signal_market_lr[:, t] = (0.55 - mkt_lr_prev) / 0.55
        signal_rate_adequacy[:, t] = ra_t
        signal_rate_change[:, t] = rc_t / 0.10
        signal_capital[:, t] = (curr_solv - 1.5) / 1.5

        # --- 2. GWP change ---
        gwp_chg = compute_gwp_change(params, signal)
        gwp_chg[ruined] = 0.0
        new_gwp = curr_gwp * (1 + gwp_chg)
        new_gwp[ruined] = 0.0

        # --- 3. Cession % (buy more RI when soft) ---
        cess = params.base_cession_pct + (
            params.cession_cycle_sensitivity * np.maximum(-rate_adequacy[:, t], 0)
            - params.cession_cycle_sensitivity * 0.5 * np.maximum(rate_adequacy[:, t], 0)
        )
        cess = np.clip(cess, params.min_cession_pct, params.max_cession_pct)
        new_nwp = new_gwp * (1 - cess)

        # --- 4. Sample gross loss ratios ---
        if config.capital_model is not None:
            loss_data = sample_capital_model_losses(
                config.capital_model,
                rate_adequacy[:, t],
                lp.cycle_sensitivity,
                n_paths, rng,
            )
            gross_lr_sample = loss_data["gross_total"]

            if loss_data.get("ri_recoveries") is not None:
                base_cess = params.base_cession_pct
                cess_ratio = cess / np.maximum(base_cess, 0.01)
                net_lr_sample = gross_lr_sample - loss_data["ri_recoveries"] * cess_ratio
            else:
                net_lr_sample = loss_data.get(
                    "net_total",
                    gross_lr_sample * (1 - cess * lp.ri_effectiveness_attritional),
                )

            if loss_data.get("ri_spend") is not None:
                base_cess = params.base_cession_pct
                cess_ratio = cess / np.maximum(base_cess, 0.01)
                cm_ri_cost = loss_data["ri_spend"] * cess_ratio * new_gwp
            else:
                cm_ri_cost = None

            market_gwp_change = float(np.mean(gwp_chg))
            adv_sel = compute_adverse_selection_penalty(
                params, gwp_chg, market_gwp_change
            )
            gross_lr_sample = gross_lr_sample * adv_sel
            net_lr_sample = net_lr_sample * adv_sel
        else:
            loss_components = sample_parametric_losses(
                lp, rate_adequacy[:, t], n_paths, rng
            )
            gross_lr_sample = loss_components["gross_total"]

            # --- 5. Adverse selection penalty ---
            market_gwp_change = float(np.mean(gwp_chg))
            adv_sel = compute_adverse_selection_penalty(
                params, gwp_chg, market_gwp_change
            )
            gross_lr_sample = gross_lr_sample * adv_sel

            # --- 6. Net loss ratio (component-specific RI) ---
            for key in ("attritional", "large", "cat"):
                loss_components[key] = loss_components[key] * adv_sel

            # Store component losses for tail decomposition
            attritional_lr[:, t] = loss_components["attritional"]
            large_lr[:, t] = loss_components["large"]
            cat_lr[:, t] = loss_components["cat"]

            net_lr_sample = compute_net_loss_ratio(
                loss_components, cess, lp
            )

        # --- 7. Reserve development ---
        res_dev = compute_reserve_development(
            rate_adequacy[:, :t + 1], t, lp, rng
        )

        # --- 8. Dynamic expense ratio (with optional fixed/variable split) ---
        if 0 < (params.expense_fixed_pct or 0) < 1:
            dyn_expense = compute_fixed_variable_expense(
                params, new_gwp, new_nwp, gwp_chg,
            )
        else:
            dyn_expense = compute_dynamic_expense_ratio(params, gwp_chg)

        # --- 9. RI cost ---
        if config.capital_model is not None and cm_ri_cost is not None:
            ri_cost_amount = cm_ri_cost
        else:
            lagged_ra = rate_adequacy[:, max(t - params.ri_cost_lag_years, 0)]
            ri_cost_amount = compute_ri_cost(params, cess, new_gwp, lagged_ra)

        # --- 10. Financials ---
        net_claims = new_nwp * net_lr_sample
        reserve_impact = new_nwp * res_dev
        expenses = new_nwp * dyn_expense
        uw_pft = new_nwp - net_claims - reserve_impact - expenses - ri_cost_amount

        float_base = new_nwp * 0.5 + curr_capital
        inv_income = np.maximum(float_base, 0) * params.investment_return
        total_pft = uw_pft + inv_income

        # --- 11. Combined ratio ---
        safe_nwp = np.where(new_nwp > 1.0, new_nwp, 1.0)
        cr = np.where(
            new_nwp > 1.0,
            (net_claims + reserve_impact + expenses + ri_cost_amount) / safe_nwp,
            np.nan,
        )

        # --- 12. Capital update ---
        new_capital = curr_capital + total_pft
        econ_cap = np.maximum(new_nwp * params.capital_ratio, 1.0)
        solv = new_capital / econ_cap

        # Injection
        need_inject = solv < params.capital_injection_trigger
        injection = np.where(need_inject, econ_cap * 0.25, 0.0)
        new_capital += injection

        # Extraction
        can_extract = solv > params.dividend_extraction_trigger
        extraction = np.where(
            can_extract,
            np.maximum(0, (new_capital - econ_cap * 1.5) * 0.5),
            0.0,
        )
        new_capital -= extraction

        # Recompute solvency
        solv = new_capital / econ_cap

        # RORAC
        r = np.where(econ_cap > 0, total_pft / econ_cap, 0.0)

        # Ruin check
        ruined |= new_capital < params.ruin_threshold

        # Cumulative
        new_cumulative = curr_cumulative + total_pft

        # Present value
        dr = config.discount_rate
        pv_factor = 1.0 / (1.0 + dr) ** (t + 1) if dr > 0 else 1.0
        pv_pft = total_pft * pv_factor

        # --- Store ---
        gwp[:, t] = new_gwp
        nwp[:, t] = new_nwp
        cession_pct[:, t] = cess
        gross_lr[:, t] = gross_lr_sample
        net_lr[:, t] = net_lr_sample
        expense_ratio[:, t] = dyn_expense
        combined_ratio[:, t] = cr
        uw_profit[:, t] = uw_pft
        investment_income[:, t] = inv_income
        ri_cost[:, t] = ri_cost_amount
        total_profit[:, t] = total_pft
        capital[:, t] = new_capital
        economic_capital[:, t] = econ_cap
        solvency_ratio[:, t] = solv
        rorac[:, t] = r
        cumulative_profit[:, t] = new_cumulative
        pv_total_profit[:, t] = pv_pft
        pv_cumulative_profit[:, t] = (
            pv_cumulative_profit[:, t - 1] + pv_pft if t > 0 else pv_pft
        )
        reserve_dev[:, t] = res_dev
        adverse_selection[:, t] = adv_sel
        capital_injections[:, t] = injection
        dividend_extractions[:, t] = extraction
        gwp_change_pct[:, t] = gwp_chg
        is_ruined[:, t] = ruined

        # --- Update state for next year ---
        curr_gwp = new_gwp
        curr_capital = new_capital
        curr_cumulative = new_cumulative
        last_own_lr = gross_lr_sample
        last_gwp_change = gwp_chg

    return {
        "gwp": gwp,
        "nwp": nwp,
        "cession_pct": cession_pct,
        "gross_lr": gross_lr,
        "net_lr": net_lr,
        "expense_ratio": expense_ratio,
        "combined_ratio": combined_ratio,
        "uw_profit": uw_profit,
        "investment_income": investment_income,
        "ri_cost": ri_cost,
        "total_profit": total_profit,
        "capital": capital,
        "economic_capital": economic_capital,
        "solvency_ratio": solvency_ratio,
        "rorac": rorac,
        "cumulative_profit": cumulative_profit,
        "pv_total_profit": pv_total_profit,
        "pv_cumulative_profit": pv_cumulative_profit,
        "reserve_dev": reserve_dev,
        "adverse_selection": adverse_selection,
        "capital_injections": capital_injections,
        "dividend_extractions": dividend_extractions,
        "gwp_change_pct": gwp_change_pct,
        "is_ruined": is_ruined,
        "attritional_lr": attritional_lr,
        "large_lr": large_lr,
        "cat_lr": cat_lr,
        "strategy_signal": strategy_signal,
        "signal_own_lr": signal_own_lr,
        "signal_market_lr": signal_market_lr,
        "signal_rate_adequacy": signal_rate_adequacy,
        "signal_rate_change": signal_rate_change,
        "signal_capital": signal_capital,
    }
