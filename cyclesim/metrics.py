"""
Risk-adjusted metrics computed from simulation results.

Includes:
- Through-cycle RORAC
- VaR/TVaR on cumulative profit
- Probability of ruin
- Max drawdown distribution
- Profit attribution (timing, selection, RI value, capital efficiency)
- Sensitivity analysis helpers
"""

import numpy as np
from typing import Optional, Tuple


def compute_summary_metrics(results: dict, params) -> dict:
    """
    Compute aggregate metrics across all simulated paths.

    Input: results dict from _simulate_insurer (arrays of shape (n_paths, n_years))
    Output: dict of scalar and distribution metrics.
    """
    n_paths, n_years = results["total_profit"].shape
    cumulative = results["cumulative_profit"][:, -1]
    terminal_capital = results["capital"][:, -1]
    annual_rorac = results["rorac"]
    ruined_any = results["is_ruined"].any(axis=1)

    # Handle edge case: all profits identical
    std_cumulative = float(np.std(cumulative))
    if std_cumulative < 1e-10:
        std_cumulative = 1.0

    # --- VaR / TVaR ---
    var_95 = float(np.nanpercentile(cumulative, 5))
    var_99 = float(np.nanpercentile(cumulative, 1))
    var_995 = float(np.nanpercentile(cumulative, 0.5))
    below_var_95 = cumulative[cumulative <= var_95]
    tvar_95 = float(np.mean(below_var_95)) if len(below_var_95) > 0 else var_95
    below_var_995 = cumulative[cumulative <= var_995]
    tvar_995 = float(np.mean(below_var_995)) if len(below_var_995) > 0 else var_995

    # --- VaR-based economic capital (distribution-derived SCR) ---
    # Annual profit 0.5th percentile across all path-years
    annual_profits = results["total_profit"]
    annual_var_995 = float(np.nanpercentile(annual_profits, 0.5))
    var_based_econ_cap = max(0.0, -annual_var_995)  # capital = worst loss at 99.5%

    # --- Through-cycle RORAC (annualized) ---
    total_profit_sum = results["total_profit"].sum(axis=1)
    avg_econ_cap = results["economic_capital"].mean(axis=1)
    # Annualize: divide cumulative by years to get average annual, then by capital
    through_cycle_rorac = np.where(
        avg_econ_cap > 0, (total_profit_sum / n_years) / avg_econ_cap, 0.0
    )

    # --- Combined ratio: filter out inf/nan from ruined paths ---
    cr_clean = results["combined_ratio"].copy()
    cr_clean[~np.isfinite(cr_clean)] = np.nan

    # --- Max drawdown ---
    max_drawdowns = _compute_max_drawdowns(results["capital"])

    # --- Profit attribution ---
    attribution = compute_profit_attribution(results, params)

    # --- Year-by-year means for time series charts ---
    # Use nanmean for metrics that may have NaN (ruined paths)
    nan_keys = {"combined_ratio", "rorac", "solvency_ratio"}
    yearly_means = {}
    for key in [
        "gwp", "nwp", "cession_pct", "gross_lr", "net_lr",
        "expense_ratio", "combined_ratio", "uw_profit",
        "total_profit", "capital", "rorac", "cumulative_profit",
        "solvency_ratio", "ri_cost", "reserve_dev",
    ]:
        data = cr_clean if key == "combined_ratio" else results[key]
        if key in nan_keys:
            yearly_means[key] = np.nanmean(data, axis=0).tolist()
        else:
            yearly_means[key] = data.mean(axis=0).tolist()

    # --- Percentile bands for fan charts ---
    percentile_bands = {}
    for key in [
        "gwp", "nwp", "combined_ratio", "cumulative_profit", "capital",
        "rorac", "cession_pct", "expense_ratio", "net_lr",
    ]:
        data = cr_clean if key == "combined_ratio" else results[key]
        percentile_bands[key] = {
            "p5": np.nanpercentile(data, 5, axis=0).tolist(),
            "p25": np.nanpercentile(data, 25, axis=0).tolist(),
            "p50": np.nanpercentile(data, 50, axis=0).tolist(),
            "p75": np.nanpercentile(data, 75, axis=0).tolist(),
            "p95": np.nanpercentile(data, 95, axis=0).tolist(),
        }

    summary = {
        # Central tendency
        "mean_annual_rorac": float(np.mean(annual_rorac)),
        "median_annual_rorac": float(np.median(annual_rorac)),
        "mean_through_cycle_rorac": float(np.mean(through_cycle_rorac)),
        "std_through_cycle_rorac": float(np.std(through_cycle_rorac)),

        # Cumulative profit
        "mean_cumulative_profit": float(np.mean(cumulative)),
        "median_cumulative_profit": float(np.median(cumulative)),
        "std_cumulative_profit": std_cumulative,

        # Terminal wealth
        "mean_terminal_capital": float(np.mean(terminal_capital)),
        "median_terminal_capital": float(np.median(terminal_capital)),

        # Risk
        "prob_ruin": float(np.mean(ruined_any)),
        "var_95_cumulative": var_95,
        "tvar_95_cumulative": tvar_95,
        "var_99_cumulative": var_99,
        "var_995_cumulative": var_995,
        "tvar_995_cumulative": tvar_995,
        "var_based_econ_cap": var_based_econ_cap,
        "profit_to_risk_ratio": float(np.mean(cumulative) / std_cumulative),

        # Max drawdown
        "mean_max_drawdown": float(np.mean(max_drawdowns)),
        "median_max_drawdown": float(np.median(max_drawdowns)),
        "p95_max_drawdown": float(np.percentile(max_drawdowns, 95)),

        # Combined ratio (excluding ruined paths)
        "mean_combined_ratio": float(np.nanmean(cr_clean)),
        "std_combined_ratio": float(np.nanstd(cr_clean)),

        # GWP
        "mean_terminal_gwp": float(np.mean(results["gwp"][:, -1])),
        "mean_gwp_cagr": float(np.nanmean(
            np.where(
                results["gwp"][:, 0] > 1.0,
                (np.maximum(results["gwp"][:, -1], 0) / results["gwp"][:, 0]) ** (1 / n_years) - 1,
                0.0,
            )
        )),

        # Expense
        "mean_expense_ratio": float(np.mean(results["expense_ratio"])),

        # RI
        "mean_cession_pct": float(np.mean(results["cession_pct"])),
        "total_ri_cost": float(np.mean(results["ri_cost"].sum(axis=1))),

        # Capital efficiency
        "mean_solvency_ratio": float(np.mean(results["solvency_ratio"])),
        "total_injections": float(np.mean(results["capital_injections"].sum(axis=1))),
        "total_dividends": float(np.mean(results["dividend_extractions"].sum(axis=1))),

        # Attribution
        "attribution": attribution,

        # Time series
        "yearly_means": yearly_means,
        "percentile_bands": percentile_bands,

        # Distribution data (for histograms)
        "cumulative_profit_dist": cumulative.tolist(),
        "terminal_capital_dist": terminal_capital.tolist(),
        "through_cycle_rorac_dist": through_cycle_rorac.tolist(),
        "max_drawdown_dist": max_drawdowns.tolist(),

        # Ruin probability over time
        "ruin_prob_by_year": results["is_ruined"].any(axis=1)[:, np.newaxis]
        if False else _compute_ruin_prob_by_year(results["is_ruined"]),
    }

    # --- Bootstrap confidence intervals (F1) ---
    if n_paths >= 200:
        try:
            ci = compute_bootstrap_ci(results, n_bootstrap=500, ci_level=0.90)
            summary["confidence_intervals"] = ci
        except Exception:
            pass

    # --- GPD tail extrapolation (F4) ---
    if n_paths >= 500:
        try:
            gpd = fit_gpd_tail(cumulative, threshold_pctl=10.0)
            summary["gpd_tail"] = gpd
        except Exception:
            pass

    # --- PV metrics (F3) ---
    pv_cum = results.get("pv_cumulative_profit")
    if pv_cum is not None and pv_cum.shape[1] > 0:
        pv_terminal = pv_cum[:, -1]
        pv_std = float(np.std(pv_terminal))
        if pv_std < 1e-10:
            pv_std = 1.0
        summary["pv_mean_cumulative_profit"] = float(np.mean(pv_terminal))
        summary["pv_var_995_cumulative"] = float(np.nanpercentile(pv_terminal, 0.5))
        summary["pv_tvar_995_cumulative"] = float(
            np.mean(pv_terminal[pv_terminal <= np.nanpercentile(pv_terminal, 0.5)])
            if (pv_terminal <= np.nanpercentile(pv_terminal, 0.5)).sum() > 0
            else np.nanpercentile(pv_terminal, 0.5)
        )
        summary["pv_yearly_means"] = pv_cum.mean(axis=0).tolist()

    return summary


def _compute_max_drawdowns(capital: np.ndarray) -> np.ndarray:
    """Compute max drawdown for each path. Shape: (n_paths,)."""
    running_max = np.maximum.accumulate(capital, axis=1)
    drawdowns = running_max - capital
    return np.max(drawdowns, axis=1)


def _compute_ruin_prob_by_year(is_ruined: np.ndarray) -> list:
    """Compute cumulative ruin probability at each year."""
    # is_ruined is (n_paths, n_years), True once ruined
    # Cumulative: probability of being ruined by year t
    ruined_by_year = np.zeros(is_ruined.shape[1])
    for t in range(is_ruined.shape[1]):
        ruined_by_year[t] = float(np.mean(is_ruined[:, t]))
    return ruined_by_year.tolist()


def compute_profit_attribution(results: dict, params) -> dict:
    """
    Decompose cumulative profit into attribution components.

    Components:
    1. Underwriting timing: profit from growing in hard market / shrinking in soft
    2. Risk selection: own loss ratio performance vs market (incl. adverse selection)
    3. Reinsurance value: net benefit/cost of RI program
    4. Capital efficiency: investment income and capital management
    5. Reserve development: impact of prior-year development
    """
    n_paths, n_years = results["total_profit"].shape

    # Total profit for reference
    total = results["cumulative_profit"][:, -1]

    # 1. UW profit component
    uw_total = results["uw_profit"].sum(axis=1)

    # 2. Investment income component
    inv_total = results["investment_income"].sum(axis=1)

    # 3. RI cost impact (negative = cost)
    ri_total = -results["ri_cost"].sum(axis=1)

    # 4. Reserve development impact (negative = adverse)
    # Reserve dev is a loss ratio fraction — multiply per-year before summing
    res_total = -(results["reserve_dev"] * results["nwp"]).sum(axis=1)

    # 5. Capital actions (injections reduce return, extractions are return)
    cap_actions = (
        results["dividend_extractions"].sum(axis=1)
        - results["capital_injections"].sum(axis=1)
    )

    return {
        "underwriting": float(np.mean(uw_total)),
        "investment": float(np.mean(inv_total)),
        "reinsurance_cost": float(np.mean(ri_total)),
        "reserve_development": float(np.mean(res_total)),
        "capital_actions": float(np.mean(cap_actions)),
        "total": float(np.mean(total)),
    }


def compute_return_period_table(results_list: list, params_list: list) -> list:
    """
    Compute return period metrics for N insurers.

    Return periods (1-in-N) are the standard actuarial/regulatory framework
    for communicating tail risk. Lloyd's Solvency Capital Requirement uses 1-in-200.

    Returns list of dicts, one per return period. Each row has a 'strategies'
    list with per-insurer metrics.
    """
    n_strat = len(results_list)

    # Pre-compute per-strategy vectors
    per_strat = []
    for res in results_list:
        cum = res["cumulative_profit"][:, -1]
        worst_annual = res["total_profit"].min(axis=1)
        cr_clean = res["combined_ratio"].copy()
        cr_clean[~np.isfinite(cr_clean)] = np.nan
        worst_cr = np.nanmax(cr_clean, axis=1)
        min_cap = res["capital"].min(axis=1)
        per_strat.append({
            "cum": cum, "worst_annual": worst_annual,
            "worst_cr": worst_cr, "min_cap": min_cap,
        })

    return_periods = [
        (10, 10.0),
        (50, 2.0),
        (100, 1.0),
        (200, 0.5),
    ]

    rows = []
    for rp, pctl in return_periods:
        strats = []
        for s in per_strat:
            strats.append({
                "cum_profit": float(np.percentile(s["cum"], pctl)),
                "worst_annual": float(np.percentile(s["worst_annual"], pctl)),
                "worst_cr": float(np.nanpercentile(s["worst_cr"], 100 - pctl)),
                "min_capital": float(np.percentile(s["min_cap"], pctl)),
            })
        row = {
            "rp": rp,
            "label": f"1-in-{rp}",
            "percentile": pctl,
            "strategies": strats,
            # Backward compat for 2-strategy consumers
            "cum_profit_a": strats[0]["cum_profit"],
            "worst_annual_a": strats[0]["worst_annual"],
            "worst_cr_a": strats[0]["worst_cr"],
            "min_capital_a": strats[0]["min_capital"],
        }
        if n_strat > 1:
            row["cum_profit_b"] = strats[1]["cum_profit"]
            row["worst_annual_b"] = strats[1]["worst_annual"]
            row["worst_cr_b"] = strats[1]["worst_cr"]
            row["min_capital_b"] = strats[1]["min_capital"]
        rows.append(row)

    return rows


def compute_strategy_comparison(summaries: list) -> dict:
    """
    Compute comparative metrics across N insurer strategies.

    For each metric, identifies the best strategy and the spread.
    Returns dict keyed by metric name, each with:
      values: list of floats (one per strategy)
      best_idx: index of the best strategy
      spread: max - min across strategies
    Also includes backward-compat 'insurer_a'/'insurer_b' keys for 2-strategy case.
    """
    comparison = {}
    scalar_keys = [
        "mean_through_cycle_rorac", "prob_ruin", "var_95_cumulative",
        "tvar_95_cumulative", "var_995_cumulative", "tvar_995_cumulative",
        "var_based_econ_cap", "profit_to_risk_ratio", "mean_cumulative_profit",
        "mean_terminal_capital", "mean_combined_ratio", "mean_max_drawdown",
        "mean_terminal_gwp", "mean_gwp_cagr", "mean_expense_ratio",
        "mean_cession_pct", "total_ri_cost",
    ]

    def _safe(v):
        """Coerce to float, replace NaN/None with 0."""
        if v is None:
            return 0.0
        try:
            v = float(v)
        except (TypeError, ValueError):
            return 0.0
        return v if np.isfinite(v) else 0.0

    for key in scalar_keys:
        vals = [_safe(s.get(key, 0)) for s in summaries]
        # Determine best: for most metrics higher is better,
        # but for prob_ruin, combined_ratio, max_drawdown, expense_ratio lower is better
        lower_better = key in (
            "prob_ruin", "mean_combined_ratio", "mean_max_drawdown",
            "mean_expense_ratio", "total_ri_cost",
        )
        best_idx = int(np.argmin(vals) if lower_better else np.argmax(vals))

        entry = {
            "values": vals,
            "best_idx": best_idx,
            "spread": max(vals) - min(vals),
        }
        # Backward compat
        if len(vals) >= 2:
            entry["insurer_a"] = vals[0]
            entry["insurer_b"] = vals[1]
            entry["difference"] = vals[0] - vals[1]
            entry["a_better"] = _is_a_better(key, vals[0], vals[1])
        comparison[key] = entry

    return comparison


def compute_tail_decomposition(
    results: dict,
    params,
    return_periods: list = None,
) -> list:
    """
    Euler-style marginal attribution of tail losses by risk factor.

    For each return period, identifies the worst-N% paths by cumulative
    profit, then decomposes their average loss into component contributions.

    Components (parametric mode only — NaN for capital model):
    - attritional: attritional loss ratio contribution
    - large_loss: large loss contribution
    - cat: catastrophe contribution
    - reserve_dev: prior-year development impact
    - ri_cost: reinsurance cost drag
    - expense: expense ratio impact

    Returns list of dicts, one per return period.
    """
    if return_periods is None:
        return_periods = [10, 50, 100, 200]

    n_paths, n_years = results["total_profit"].shape
    cum_profit = results["cumulative_profit"][:, -1]

    # Check if component data is available (parametric mode only)
    att_lr = results.get("attritional_lr")
    has_components = (
        att_lr is not None
        and att_lr.size > 0
        and not np.all(np.isnan(att_lr))
    )

    rows = []
    for rp in return_periods:
        pctl = 100.0 / rp  # 1-in-100 → 1st percentile
        threshold = np.percentile(cum_profit, pctl)
        tail_mask = cum_profit <= threshold

        if tail_mask.sum() == 0:
            tail_mask = np.zeros(n_paths, dtype=bool)
            tail_mask[np.argmin(cum_profit)] = True

        n_tail = tail_mask.sum()

        # Average P&L components over tail paths
        tail_nwp = results["nwp"][tail_mask].mean()

        # Total loss = NWP - UW profit (what was consumed)
        tail_total_loss = -results["uw_profit"][tail_mask].sum(axis=1).mean()

        row = {
            "rp": rp,
            "label": f"1-in-{rp}",
            "n_tail_paths": int(n_tail),
            "tail_cum_profit": float(cum_profit[tail_mask].mean()),
        }

        if has_components and n_tail >= 2:
            # Component contributions (sum over years, average over tail paths)
            nwp_tail = results["nwp"][tail_mask]
            att_total = (results["attritional_lr"][tail_mask] * nwp_tail).sum(axis=1).mean()
            large_total = (results["large_lr"][tail_mask] * nwp_tail).sum(axis=1).mean()
            cat_total = (results["cat_lr"][tail_mask] * nwp_tail).sum(axis=1).mean()
            res_total = (results["reserve_dev"][tail_mask] * nwp_tail).sum(axis=1).mean()
            ri_total = results["ri_cost"][tail_mask].sum(axis=1).mean()
            exp_total = (results["expense_ratio"][tail_mask] * nwp_tail).sum(axis=1).mean()

            # Total claims + costs for normalization
            total_drain = att_total + large_total + cat_total + abs(res_total) + ri_total + exp_total
            total_drain = max(total_drain, 1.0)

            row.update({
                "attritional_pct": float(att_total / total_drain),
                "large_pct": float(large_total / total_drain),
                "cat_pct": float(cat_total / total_drain),
                "reserve_pct": float(abs(res_total) / total_drain),
                "ri_pct": float(ri_total / total_drain),
                "expense_pct": float(exp_total / total_drain),
                "has_components": True,
            })
        else:
            row.update({
                "attritional_pct": 0, "large_pct": 0, "cat_pct": 0,
                "reserve_pct": 0, "ri_pct": 0, "expense_pct": 0,
                "has_components": False,
            })

        rows.append(row)

    return rows


def compute_capital_allocation(results: dict) -> dict:
    """
    Euler-style capital allocation by risk component.

    Identifies tail paths (cumulative profit <= VaR(99.5%)) and decomposes
    their average loss into component contributions. Returns allocation
    as percentages of total tail loss.

    Returns dict with component percentages, or empty dict if component
    data not available (capital model mode).
    """
    att_lr = results.get("attritional_lr")
    has_components = (
        att_lr is not None
        and att_lr.size > 0
        and not np.all(np.isnan(att_lr))
    )
    if not has_components:
        return {}

    cum_profit = results["cumulative_profit"][:, -1]
    var_995 = float(np.nanpercentile(cum_profit, 0.5))
    tail_mask = cum_profit <= var_995
    if tail_mask.sum() < 2:
        tail_mask = np.zeros(len(cum_profit), dtype=bool)
        tail_mask[np.argmin(cum_profit)] = True

    nwp_tail = results["nwp"][tail_mask]
    att_total = float((results["attritional_lr"][tail_mask] * nwp_tail).sum(axis=1).mean())
    large_total = float((results["large_lr"][tail_mask] * nwp_tail).sum(axis=1).mean())
    cat_total = float((results["cat_lr"][tail_mask] * nwp_tail).sum(axis=1).mean())
    res_total = float(abs((results["reserve_dev"][tail_mask] * nwp_tail).sum(axis=1).mean()))
    ri_total = float(results["ri_cost"][tail_mask].sum(axis=1).mean())
    exp_total = float((results["expense_ratio"][tail_mask] * nwp_tail).sum(axis=1).mean())

    total_drain = att_total + large_total + cat_total + res_total + ri_total + exp_total
    total_drain = max(total_drain, 1.0)

    return {
        "attritional_pct": att_total / total_drain,
        "large_pct": large_total / total_drain,
        "cat_pct": cat_total / total_drain,
        "reserve_pct": res_total / total_drain,
        "ri_pct": ri_total / total_drain,
        "expense_pct": exp_total / total_drain,
        "total_abs": total_drain,
    }


def _is_a_better(metric: str, a: float, b: float) -> bool:
    """Determine if insurer A's metric value is better than B's."""
    # Higher is better
    higher_better = {
        "mean_through_cycle_rorac", "profit_to_risk_ratio",
        "mean_cumulative_profit", "mean_terminal_capital",
        "mean_terminal_gwp", "mean_gwp_cagr", "var_95_cumulative",
        "tvar_95_cumulative", "var_995_cumulative", "tvar_995_cumulative",
    }
    # Lower is better
    lower_better = {
        "prob_ruin", "mean_combined_ratio", "mean_max_drawdown",
        "mean_expense_ratio", "total_ri_cost", "var_based_econ_cap",
    }

    if metric in higher_better:
        return a > b
    elif metric in lower_better:
        return a < b
    return a > b  # default: higher is better


def compute_bootstrap_ci(
    results: dict,
    n_bootstrap: int = 500,
    ci_level: float = 0.90,
) -> dict:
    """
    Nonparametric bootstrap confidence intervals for key summary metrics.

    Resamples full path indices (not just terminal values) to preserve
    RORAC computation integrity. Returns dict mapping metric names to
    (lower, upper) tuples at the specified confidence level.
    """
    n_paths, n_years = results["total_profit"].shape
    alpha = (1 - ci_level) / 2

    cum = results["cumulative_profit"][:, -1]
    total_sum = results["total_profit"].sum(axis=1)
    avg_ecap = results["economic_capital"].mean(axis=1)
    ruined_any = results["is_ruined"].any(axis=1)

    # Pre-draw bootstrap indices for speed
    rng = np.random.default_rng(12345)
    boot_indices = rng.integers(0, n_paths, size=(n_bootstrap, n_paths))

    boot_var_95 = np.empty(n_bootstrap)
    boot_var_995 = np.empty(n_bootstrap)
    boot_tvar_995 = np.empty(n_bootstrap)
    boot_ruin = np.empty(n_bootstrap)
    boot_rorac = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx = boot_indices[b]
        bc = cum[idx]
        boot_var_95[b] = np.nanpercentile(bc, 5)
        boot_var_995[b] = np.nanpercentile(bc, 0.5)
        below = bc[bc <= boot_var_995[b]]
        boot_tvar_995[b] = float(np.mean(below)) if len(below) > 0 else boot_var_995[b]
        boot_ruin[b] = float(np.mean(ruined_any[idx]))
        bt_sum = total_sum[idx]
        bt_cap = avg_ecap[idx]
        tc_rorac = np.where(bt_cap > 0, (bt_sum / n_years) / bt_cap, 0.0)
        boot_rorac[b] = float(np.mean(tc_rorac))

    def _ci(arr):
        return (float(np.percentile(arr, alpha * 100)), float(np.percentile(arr, (1 - alpha) * 100)))

    return {
        "var_95": _ci(boot_var_95),
        "var_995": _ci(boot_var_995),
        "tvar_995": _ci(boot_tvar_995),
        "prob_ruin": _ci(boot_ruin),
        "rorac": _ci(boot_rorac),
    }


def fit_gpd_tail(
    cumulative_profit: np.ndarray,
    threshold_pctl: float = 10.0,
) -> dict:
    """
    Fit Generalized Pareto Distribution to tail losses for VaR extrapolation.

    Converts profits to losses (negate), fits GPD to exceedances above the
    threshold (90th percentile of losses), and extrapolates VaR(99.5%) and
    VaR(99.9%).

    Returns dict with GPD VaR estimates, shape/scale params, and fit diagnostics.
    """
    from scipy.stats import genpareto

    # Convert to losses (positive = bad)
    losses = -cumulative_profit
    n = len(losses)

    # Threshold: the (100 - threshold_pctl)th percentile of losses
    # = the threshold_pctl-th percentile of profits, but in loss space
    threshold = np.percentile(losses, 100 - threshold_pctl)
    exceedances = losses[losses > threshold] - threshold
    n_exceed = len(exceedances)

    if n_exceed < 20:
        return {"fit_successful": False, "reason": "insufficient_exceedances"}

    try:
        shape, loc, scale = genpareto.fit(exceedances, floc=0)
    except Exception:
        return {"fit_successful": False, "reason": "fit_failed"}

    # Sanity: reject clearly pathological fits
    if not np.isfinite(shape) or not np.isfinite(scale) or scale <= 0:
        return {"fit_successful": False, "reason": "invalid_params"}

    # GPD quantile: for exceedance probability p_exceed
    # VaR at overall probability p: p_exceed = p * n / n_exceed
    def _gpd_var(p):
        """VaR at overall probability p (proportion in tail)."""
        # We want the (1-p)-th quantile of the loss distribution
        # Tail probability beyond threshold: n_exceed / n
        tail_frac = n_exceed / n
        # Conditional exceedance probability
        cond_p = 1 - p / tail_frac
        if cond_p <= 0 or cond_p >= 1:
            return float(np.percentile(losses, (1 - p) * 100))
        q = genpareto.ppf(cond_p, shape, loc=0, scale=scale)
        return float(threshold + q)

    var_995_gpd = _gpd_var(0.005)
    var_999_gpd = _gpd_var(0.001)

    # Empirical VaR for comparison
    var_995_emp = float(np.percentile(losses, 99.5))

    return {
        "fit_successful": True,
        "var_995_gpd": -var_995_gpd,   # Convert back to profit space
        "var_999_gpd": -var_999_gpd,
        "var_995_empirical": -var_995_emp,
        "shape": float(shape),
        "scale": float(scale),
        "threshold": float(threshold),
        "n_exceedances": n_exceed,
    }
