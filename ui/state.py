"""
Simulation state management and caching for the Dash app.

Caches results by parameter hash to avoid re-running on every callback.
"""

import hashlib
import json
import numpy as np
from typing import Optional

from cyclesim.market import calibrate_ar2, MarketParams
from cyclesim.losses import LossParams
from cyclesim.insurer import InsurerParams
from cyclesim.simulator import SimulationConfig, SimulationResults, run_simulation
from cyclesim.defaults import (
    MARKET_DEFAULTS, LOSS_DEFAULTS, INSURER_A_PRESET, INSURER_B_PRESET,
    SIMULATION_DEFAULTS, INSURER_PRESETS, STRATEGY_PREFIXES,
)


# Module-level cache
_cache: dict = {"hash": None, "results": None}

# Capital model storage (separate from simulation cache)
_capital_model: dict = {"data": None, "info": None}


def set_capital_model(data, info: str):
    """Store imported capital model data."""
    _capital_model["data"] = data
    _capital_model["info"] = info


def clear_capital_model():
    """Remove any imported capital model."""
    _capital_model["data"] = None
    _capital_model["info"] = None
    # Invalidate sim cache so next run uses parametric
    _cache["hash"] = None
    _cache["results"] = None


def get_capital_model():
    """Return (data, info) tuple."""
    return _capital_model["data"], _capital_model["info"]


def build_config(params: dict) -> SimulationConfig:
    """Build a SimulationConfig from flat UI parameter dict."""
    # Market
    mp_kwargs = {
        "cycle_period": params.get("cycle_period", MARKET_DEFAULTS["cycle_period_years"]),
        "min_lr": params.get("min_lr", MARKET_DEFAULTS["min_loss_ratio"]),
        "max_lr": params.get("max_lr", MARKET_DEFAULTS["max_loss_ratio"]),
        "long_run_lr": params.get("long_run_lr", MARKET_DEFAULTS["long_run_loss_ratio"]),
        "shock_prob": params.get("shock_prob", MARKET_DEFAULTS["shock_prob"]),
        "shock_magnitude_std": params.get("shock_magnitude", MARKET_DEFAULTS["shock_magnitude_std"]),
        "market_expense_ratio": params.get("market_expense_ratio", MARKET_DEFAULTS["market_expense_ratio"]),
    }

    if params.get("ar2_mode") == "direct":
        mp_kwargs["phi_1_override"] = params.get("phi_1", 1.1)
        mp_kwargs["phi_2_override"] = params.get("phi_2", -0.45)
        mp_kwargs["sigma_override"] = params.get("sigma_epsilon", 0.10)

    market_params = calibrate_ar2(**mp_kwargs)

    # Loss
    loss_params = LossParams.from_defaults({
        "attritional_mean": params.get("attritional_mean", LOSS_DEFAULTS["attritional_mean"]),
        "attritional_cv": params.get("attritional_cv", LOSS_DEFAULTS["attritional_cv"]),
        "cycle_sensitivity": params.get("cycle_sensitivity", LOSS_DEFAULTS["cycle_sensitivity"]),
        "tail_thickness_sensitivity": params.get("tail_sensitivity", LOSS_DEFAULTS["tail_thickness_sensitivity"]),
        "large_loss_frequency": params.get("large_freq", LOSS_DEFAULTS["large_loss_frequency"]),
        "cat_frequency": params.get("cat_freq", LOSS_DEFAULTS["cat_frequency"]),
        "cat_severity_mean": params.get("cat_severity", LOSS_DEFAULTS["cat_severity_mean"]),
        "ri_effectiveness_attritional": params.get("ri_eff_att", LOSS_DEFAULTS["ri_effectiveness_attritional"]),
        "ri_effectiveness_large": params.get("ri_eff_large", LOSS_DEFAULTS["ri_effectiveness_large"]),
        "ri_effectiveness_cat": params.get("ri_eff_cat", LOSS_DEFAULTS["ri_effectiveness_cat"]),
    })

    # Insurers — build N strategies based on n_strategies param
    n_strategies = max(2, min(int(params.get("n_strategies", 2)), len(STRATEGY_PREFIXES)))

    def build_insurer(prefix: str, preset: dict) -> InsurerParams:
        # Read signal weights from UI params, falling back to preset
        preset_sw = preset.get("signal_weights", {})
        signal_weights = {
            "own_lr": params.get(f"{prefix}_sw_own_lr", preset_sw.get("own_lr", 0.35)),
            "market_lr": params.get(f"{prefix}_sw_market_lr", preset_sw.get("market_lr", 0.20)),
            "rate_adequacy": params.get(f"{prefix}_sw_rate_adequacy", preset_sw.get("rate_adequacy", 0.20)),
            "rate_change": params.get(f"{prefix}_sw_rate_change", preset_sw.get("rate_change", 0.15)),
            "capital_position": params.get(f"{prefix}_sw_capital", preset_sw.get("capital_position", 0.10)),
        }
        return InsurerParams.from_dict({
            **preset,
            "name": params.get(f"{prefix}_name", preset["name"]),
            "initial_gwp": params.get(f"{prefix}_gwp", preset["initial_gwp"]),
            "expense_ratio": params.get(f"{prefix}_expense", preset["expense_ratio"]),
            "base_cession_pct": params.get(f"{prefix}_cession", preset["base_cession_pct"]),
            "investment_return": params.get(f"{prefix}_inv_return", preset["investment_return"]),
            "capital_ratio": params.get(f"{prefix}_cap_ratio", preset["capital_ratio"]),
            "cost_of_capital": params.get(f"{prefix}_coc", preset["cost_of_capital"]),
            "expected_lr": params.get(f"{prefix}_expected_lr", preset["expected_lr"]),
            "growth_rate_when_profitable": params.get(f"{prefix}_growth", preset["growth_rate_when_profitable"]),
            "shrink_rate_when_unprofitable": params.get(f"{prefix}_shrink", preset["shrink_rate_when_unprofitable"]),
            "max_gwp_growth_pa": params.get(f"{prefix}_max_growth", preset["max_gwp_growth_pa"]),
            "max_gwp_shrink_pa": params.get(f"{prefix}_max_shrink", preset["max_gwp_shrink_pa"]),
            "cession_cycle_sensitivity": params.get(f"{prefix}_cess_sens", preset["cession_cycle_sensitivity"]),
            "min_cession_pct": params.get(f"{prefix}_min_cess", preset["min_cession_pct"]),
            "max_cession_pct": params.get(f"{prefix}_max_cess", preset["max_cession_pct"]),
            "adverse_selection_sensitivity": params.get(f"{prefix}_adv_sel", preset["adverse_selection_sensitivity"]),
            "expense_fixed_pct": params.get(f"{prefix}_expense_fixed_pct", preset.get("expense_fixed_pct", 0.60)),
            "signal_weights": signal_weights,
        })

    insurers = []
    for i in range(n_strategies):
        prefix = STRATEGY_PREFIXES[i]
        preset = INSURER_PRESETS[prefix]
        insurers.append(build_insurer(prefix, preset))

    # Check for imported capital model
    cm_data, _ = get_capital_model()

    return SimulationConfig(
        n_paths=params.get("n_paths", SIMULATION_DEFAULTS["n_paths"]),
        n_years=params.get("n_years", SIMULATION_DEFAULTS["n_years"]),
        random_seed=params.get("seed", SIMULATION_DEFAULTS["random_seed"]),
        market_params=market_params,
        loss_params=loss_params,
        insurers=insurers,
        capital_model=cm_data,
        discount_rate=float(params.get("discount_rate", SIMULATION_DEFAULTS.get("discount_rate", 0.0))),
    )


def get_or_run(params: dict) -> SimulationResults:
    """Run simulation, caching by parameter hash."""
    param_hash = _hash_params(params)
    if _cache["hash"] == param_hash and _cache["results"] is not None:
        return _cache["results"]

    config = build_config(params)
    results = run_simulation(config)

    _cache["hash"] = param_hash
    _cache["results"] = results
    return results


def _run_single_perturbation(args: tuple) -> dict:
    """
    Module-level worker for parallel sensitivity analysis.

    Picklable on Windows spawn. Accepts (perturbed_params_dict,) and
    returns dict with RORAC results for all N strategies.
    """
    perturbed_params, key, label, tag = args
    cfg = build_config(perturbed_params)
    res = run_simulation(cfg)
    rorac_list = [s["mean_through_cycle_rorac"] for s in res.summaries]
    result = {
        "key": key, "label": label, "tag": tag,
        "rorac": rorac_list,
    }
    # Backward compat
    if len(rorac_list) >= 1:
        result["rorac_a"] = rorac_list[0]
    if len(rorac_list) >= 2:
        result["rorac_b"] = rorac_list[1]
    return result


def run_sensitivity(params: dict, n_paths_sensitivity: int = 1000) -> list:
    """
    Run one-at-a-time sensitivity analysis with parallel evaluation.

    Perturbs key parameters ±20% (or meaningful range) and re-runs simulation
    at reduced path count. Uses ProcessPoolExecutor for parallel perturbation runs.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    perturbations = [
        ("cycle_period", "Cycle Period", 0.75, 1.25),
        ("min_lr", "Min LR (Hard)", 0.85, 1.15),
        ("max_lr", "Max LR (Soft)", 0.85, 1.15),
        ("shock_prob", "Shock Probability", 0.5, 2.0),
        ("attritional_mean", "Attritional Mean", 0.90, 1.10),
        ("cycle_sensitivity", "Cycle Sensitivity", 0.5, 1.5),
        ("large_freq", "Large Loss Freq", 0.5, 1.5),
        ("cat_freq", "Cat Frequency", 0.5, 2.0),
        ("a_growth", "A: Growth Rate", 0.5, 1.5),
        ("a_cession", "A: Base Cession", 0.80, 1.20),
        ("b_growth", "B: Growth Rate", 0.5, 1.5),
        ("b_cession", "B: Base Cession", 0.80, 1.20),
    ]

    base_params = {**params, "n_paths": n_paths_sensitivity, "seed": 99}

    # Build all worker args
    worker_args = []
    # Base case
    worker_args.append((base_params, "__base__", "Base", "base"))
    for key, label, low_mult, high_mult in perturbations:
        if key not in params:
            continue
        base_val = params[key]
        # Guard: skip non-numeric or zero values
        try:
            base_val = float(base_val)
        except (TypeError, ValueError):
            continue
        if base_val == 0 or not np.isfinite(base_val):
            continue
        worker_args.append(({**base_params, key: base_val * low_mult}, key, label, "low"))
        worker_args.append(({**base_params, key: base_val * high_mult}, key, label, "high"))

    # Run in parallel (fall back to sequential if multiprocessing fails on Windows)
    max_workers = min(os.cpu_count() or 8, 16)
    results_map = {}
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_single_perturbation, a): a for a in worker_args}
            for future in as_completed(futures):
                try:
                    r = future.result()
                    results_map[(r["key"], r["tag"])] = r
                except Exception:
                    pass
    except RuntimeError:
        # Windows spawn can fail if caller isn't guarded by __name__ == '__main__'
        for a in worker_args:
            try:
                r = _run_single_perturbation(a)
                results_map[(r["key"], r["tag"])] = r
            except Exception:
                pass

    base = results_map.get(("__base__", "base"))
    if not base:
        return []
    n_strat = len(base["rorac"])
    base_rorac = base["rorac"]

    rows = []
    for key, label, _, _ in perturbations:
        low = results_map.get((key, "low"))
        high = results_map.get((key, "high"))
        if not low or not high:
            continue
        # Compute per-strategy swings
        swings = []
        valid = True
        for i in range(n_strat):
            s = abs(high["rorac"][i] - low["rorac"][i])
            if not np.isfinite(s):
                valid = False
                break
            swings.append(s)
        if not valid:
            continue
        row = {
            "param": key, "label": label,
            "base_rorac": base_rorac,
            "low_rorac": low["rorac"],
            "high_rorac": high["rorac"],
            "swings": swings,
            "avg_swing": sum(swings) / len(swings),
        }
        # Backward compat for 2-strategy consumers
        if n_strat >= 1:
            row["base_a"] = base_rorac[0]
            row["low_a"] = low["rorac"][0]
            row["high_a"] = high["rorac"][0]
            row["swing_a"] = swings[0]
        if n_strat >= 2:
            row["base_b"] = base_rorac[1]
            row["low_b"] = low["rorac"][1]
            row["high_b"] = high["rorac"][1]
            row["swing_b"] = swings[1]
        rows.append(row)

    rows.sort(key=lambda r: r["avg_swing"], reverse=True)
    return rows


def build_optimizer_base_config(params: dict) -> dict:
    """
    Build a serializable (picklable) config dict for the optimizer workers.

    Extracts market + loss parameters from the UI params dict.
    Must be plain Python types — no numpy arrays or dataclass instances.
    """
    base = {
        "cycle_period": float(params.get("cycle_period", MARKET_DEFAULTS["cycle_period_years"])),
        "min_lr": float(params.get("min_lr", MARKET_DEFAULTS["min_loss_ratio"])),
        "max_lr": float(params.get("max_lr", MARKET_DEFAULTS["max_loss_ratio"])),
        "long_run_lr": float(params.get("long_run_lr", MARKET_DEFAULTS["long_run_loss_ratio"])),
        "shock_prob": float(params.get("shock_prob", MARKET_DEFAULTS["shock_prob"])),
        "shock_magnitude": float(params.get("shock_magnitude", MARKET_DEFAULTS["shock_magnitude_std"])),
        # Loss params (pass through for LossParams.from_defaults)
        "attritional_mean": float(params.get("attritional_mean", LOSS_DEFAULTS["attritional_mean"])),
        "attritional_cv": float(LOSS_DEFAULTS["attritional_cv"]),
        "cycle_sensitivity": float(params.get("cycle_sensitivity", LOSS_DEFAULTS["cycle_sensitivity"])),
        "tail_thickness_sensitivity": float(params.get("tail_sensitivity", LOSS_DEFAULTS["tail_thickness_sensitivity"])),
        "large_loss_frequency": float(params.get("large_freq", LOSS_DEFAULTS["large_loss_frequency"])),
        "cat_frequency": float(params.get("cat_freq", LOSS_DEFAULTS["cat_frequency"])),
        "cat_severity_mean": float(params.get("cat_severity", LOSS_DEFAULTS["cat_severity_mean"])),
    }

    # Insurer overrides (non-optimized params from current UI settings)
    insurer_overrides = {
        "initial_gwp": float(params.get("a_gwp", INSURER_A_PRESET["initial_gwp"])),
        "expense_ratio": float(params.get("a_expense", INSURER_A_PRESET["expense_ratio"])),
        "investment_return": float(params.get("a_inv_return", INSURER_A_PRESET["investment_return"])),
        "capital_ratio": float(params.get("a_cap_ratio", INSURER_A_PRESET["capital_ratio"])),
        "cost_of_capital": float(params.get("a_coc", INSURER_A_PRESET["cost_of_capital"])),
        "min_cession_pct": float(params.get("a_min_cess", INSURER_A_PRESET["min_cession_pct"])),
        "max_cession_pct": float(params.get("a_max_cess", INSURER_A_PRESET["max_cession_pct"])),
    }
    base["insurer_overrides"] = insurer_overrides
    return base


def run_ri_sweep(params: dict, n_points: int = 11, n_paths_sweep: int = 300) -> list:
    """
    Sweep first strategy's cession % from 0% to 50%, evaluating risk-return at each level.

    Returns list of dicts: {cession_pct, rorac, var_95, var_995, ruin_prob}
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    cession_levels = [i / (n_points - 1) * 0.50 for i in range(n_points)]
    base_params = {**params, "n_paths": n_paths_sweep, "seed": 77}

    worker_args = []
    for cess in cession_levels:
        perturbed = {**base_params, "a_cession": cess}
        worker_args.append((perturbed, f"cess_{cess:.2f}", f"Cess {cess:.0%}", "sweep"))

    results_list = []
    max_workers = min(os.cpu_count() or 8, 12)
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_single_perturbation, a): a[1] for a in worker_args}
            results_map = {}
            for future in as_completed(futures):
                key = futures[future]
                try:
                    r = future.result()
                    results_map[key] = r
                except Exception:
                    pass
    except RuntimeError:
        results_map = {}
        for a in worker_args:
            try:
                r = _run_single_perturbation(a)
                results_map[a[1]] = r
            except Exception:
                pass

    for cess in cession_levels:
        key = f"cess_{cess:.2f}"
        r = results_map.get(key)
        if not r:
            continue
        # Re-run to get full metrics (the perturbation worker only returns RORAC)
        perturbed = {**base_params, "a_cession": cess}
        try:
            cfg = build_config(perturbed)
            res = run_simulation(cfg)
            s = res.summaries[0] if res.summaries else {}
            results_list.append({
                "cession_pct": cess,
                "rorac": s.get("mean_through_cycle_rorac", 0),
                "var_95": s.get("var_95_cumulative", 0),
                "var_995": s.get("var_995_cumulative", 0),
                "ruin_prob": s.get("prob_ruin", 0),
            })
        except Exception:
            pass

    return sorted(results_list, key=lambda x: x["cession_pct"])


def _run_seed_worker(args: tuple) -> dict:
    """Worker for parallel seed stability runs."""
    params_dict, seed_val = args
    perturbed = {**params_dict, "seed": seed_val}
    cfg = build_config(perturbed)
    res = run_simulation(cfg)
    strategies = []
    for s in res.summaries:
        strategies.append({
            "rorac": s.get("mean_through_cycle_rorac", 0),
            "var_995": s.get("var_995_cumulative", 0),
            "prob_ruin": s.get("prob_ruin", 0),
            "combined_ratio": s.get("mean_combined_ratio", 1),
        })
    return {"seed": seed_val, "strategies": strategies}


def run_seed_stability(params: dict, n_seeds: int = 5, n_paths_per_seed: int = 300) -> list:
    """
    Run multiple seeds at reduced path count to measure metric stability.

    Returns list of dicts: {seed, strategies: [{rorac, var_995, prob_ruin, combined_ratio}]}
    """
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed

    base_seed = int(params.get("seed", 42))
    base_params = {**params, "n_paths": n_paths_per_seed}
    seeds = [base_seed + i for i in range(n_seeds)]

    worker_args = [(base_params, s) for s in seeds]
    results = []
    max_workers = min(os.cpu_count() or 8, n_seeds)

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_run_seed_worker, a): a[1] for a in worker_args}
            for future in as_completed(futures):
                try:
                    results.append(future.result())
                except Exception:
                    pass
    except RuntimeError:
        for a in worker_args:
            try:
                results.append(_run_seed_worker(a))
            except Exception:
                pass

    return sorted(results, key=lambda x: x["seed"])


def _hash_params(params: dict) -> str:
    """Deterministic hash of parameter dict, including capital model state."""
    serializable = {
        k: (v if not isinstance(v, np.ndarray) else v.tolist())
        for k, v in sorted(params.items())
        if not callable(v)
    }
    # Include capital model presence in hash
    cm_data, cm_info = get_capital_model()
    serializable["__capital_model__"] = cm_info or "none"
    raw = json.dumps(serializable, sort_keys=True, default=str)
    return hashlib.md5(raw.encode()).hexdigest()
