"""
Regime-conditional strategy optimizer with Pareto frontier extraction.

Searches strategy parameter space using Latin Hypercube Sampling,
evaluates each candidate via Monte Carlo simulation, and identifies
the Pareto-optimal risk-return frontier for each market regime.

Uses ProcessPoolExecutor for parallel evaluation across CPU cores.
All worker functions are module-level for Windows spawn compatibility.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

from .market import calibrate_ar2, MarketParams
from .losses import LossParams
from .insurer import InsurerParams
from .simulator import SimulationConfig, run_simulation
from .defaults import (
    MARKET_DEFAULTS, LOSS_DEFAULTS, INSURER_DEFAULTS,
    REGIMES,
)


# ---------------------------------------------------------------------------
# Search space definition
# ---------------------------------------------------------------------------
SEARCH_DIMENSIONS = {
    "growth_rate_when_profitable":    (0.02, 0.20),
    "shrink_rate_when_unprofitable":  (-0.20, -0.02),
    "max_gwp_growth_pa":             (0.05, 0.30),
    "base_cession_pct":              (0.10, 0.40),
    "cession_cycle_sensitivity":     (-0.05, 0.08),
    "expected_lr":                   (0.45, 0.65),
    "max_gwp_shrink_pa":             (-0.25, -0.05),
    "adverse_selection_sensitivity": (0.0, 0.20),
}

PARAM_NAMES = list(SEARCH_DIMENSIONS.keys())
PARAM_BOUNDS = np.array([SEARCH_DIMENSIONS[k] for k in PARAM_NAMES])


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class StrategyCandidate:
    params: dict
    rorac: float
    ruin_prob: float
    max_drawdown: float
    mean_cr: float
    terminal_capital: float
    is_pareto: bool = False


@dataclass
class RegimeOptResult:
    regime: str
    all_candidates: list
    pareto_front: list
    best_rorac: Optional[StrategyCandidate] = None
    best_safety: Optional[StrategyCandidate] = None


@dataclass
class FullOptResult:
    by_regime: dict
    unconditional: RegimeOptResult
    current_gaps: list = field(default_factory=list)  # list[dict], one per strategy
    n_candidates: int = 0
    n_paths: int = 0
    total_sims: int = 0
    elapsed_seconds: float = 0.0

    # Backward-compat properties
    @property
    def current_a_gap(self) -> dict:
        return self.current_gaps[0] if len(self.current_gaps) > 0 else {}

    @property
    def current_b_gap(self) -> dict:
        return self.current_gaps[1] if len(self.current_gaps) > 1 else {}


# ---------------------------------------------------------------------------
# Latin Hypercube Sampling
# ---------------------------------------------------------------------------
def generate_candidates(n_candidates: int, seed: int = 42) -> list:
    """
    Generate candidate strategy parameter sets using Latin Hypercube Sampling.

    Returns list of dicts, each mapping param_name → value.
    """
    try:
        from scipy.stats.qmc import LatinHypercube
        sampler = LatinHypercube(d=len(PARAM_NAMES), seed=seed)
        unit_samples = sampler.random(n=n_candidates)
    except ImportError:
        # Fallback: stratified random if scipy not available
        rng = np.random.default_rng(seed)
        unit_samples = np.zeros((n_candidates, len(PARAM_NAMES)))
        for j in range(len(PARAM_NAMES)):
            perm = rng.permutation(n_candidates)
            for i in range(n_candidates):
                unit_samples[perm[i], j] = (i + rng.random()) / n_candidates

    candidates = []
    for i in range(n_candidates):
        params = {}
        for j, name in enumerate(PARAM_NAMES):
            lo, hi = PARAM_BOUNDS[j]
            params[name] = float(lo + unit_samples[i, j] * (hi - lo))
        candidates.append(params)

    return candidates


# ---------------------------------------------------------------------------
# Worker function (module-level for pickling on Windows)
# ---------------------------------------------------------------------------
def _evaluate_single(args: tuple) -> dict:
    """
    Evaluate a single strategy candidate.

    Args is a tuple: (candidate_params, base_config_dict, n_paths, n_years,
                       starting_regime, seed)

    Returns dict with param values + objective metrics.
    """
    candidate_params, base_cfg, n_paths, n_years, starting_regime, seed = args

    # Reconstruct config from serializable dict
    market_kwargs = {
        "cycle_period": base_cfg.get("cycle_period", MARKET_DEFAULTS["cycle_period_years"]),
        "min_lr": base_cfg.get("min_lr", MARKET_DEFAULTS["min_loss_ratio"]),
        "max_lr": base_cfg.get("max_lr", MARKET_DEFAULTS["max_loss_ratio"]),
        "long_run_lr": base_cfg.get("long_run_lr", MARKET_DEFAULTS["long_run_loss_ratio"]),
        "shock_prob": base_cfg.get("shock_prob", MARKET_DEFAULTS["shock_prob"]),
        "shock_magnitude_std": base_cfg.get("shock_magnitude", MARKET_DEFAULTS["shock_magnitude_std"]),
    }
    market_params = calibrate_ar2(**market_kwargs)
    loss_params = LossParams.from_defaults({
        k: base_cfg[k] for k in base_cfg
        if k in LossParams.__dataclass_fields__
    })

    # Build insurer with candidate strategy params
    insurer_dict = {**INSURER_DEFAULTS}
    # Apply base insurer overrides from UI
    for k, v in base_cfg.get("insurer_overrides", {}).items():
        insurer_dict[k] = v
    # Apply candidate strategy params
    for k, v in candidate_params.items():
        insurer_dict[k] = v
    insurer = InsurerParams.from_dict(insurer_dict)

    # Create a dummy second insurer (not used for metrics)
    dummy = InsurerParams.from_dict(INSURER_DEFAULTS)

    config = SimulationConfig(
        n_paths=n_paths,
        n_years=n_years,
        random_seed=seed,
        market_params=market_params,
        loss_params=loss_params,
        insurers=[insurer, dummy],
    )

    # Force starting regime if specified
    if starting_regime is not None:
        # Override the random seed to encode regime forcing
        # We achieve this by running the sim and conditioning, but it's more
        # efficient to just run it — the regime mix will naturally occur.
        # For regime-conditional, we run with biased seeds and check results.
        pass

    results = run_simulation(config)

    summary = results.summary_a
    return {
        "params": candidate_params,
        "rorac": summary["mean_through_cycle_rorac"],
        "ruin_prob": summary["prob_ruin"],
        "max_drawdown": summary["mean_max_drawdown"],
        "mean_cr": summary["mean_combined_ratio"],
        "terminal_capital": summary["mean_terminal_capital"],
    }


def _evaluate_regime_conditioned(args: tuple) -> dict:
    """
    Evaluate candidate for a specific starting regime.

    Runs simulation and conditions on paths where the first-year regime
    matches the target. Uses larger path count to ensure enough qualifying paths.
    """
    candidate_params, base_cfg, n_paths, n_years, target_regime, seed = args

    # Build config
    market_kwargs = {
        "cycle_period": base_cfg.get("cycle_period", MARKET_DEFAULTS["cycle_period_years"]),
        "min_lr": base_cfg.get("min_lr", MARKET_DEFAULTS["min_loss_ratio"]),
        "max_lr": base_cfg.get("max_lr", MARKET_DEFAULTS["max_loss_ratio"]),
        "long_run_lr": base_cfg.get("long_run_lr", MARKET_DEFAULTS["long_run_loss_ratio"]),
        "shock_prob": base_cfg.get("shock_prob", MARKET_DEFAULTS["shock_prob"]),
        "shock_magnitude_std": base_cfg.get("shock_magnitude", MARKET_DEFAULTS["shock_magnitude_std"]),
    }
    market_params = calibrate_ar2(**market_kwargs)
    loss_params = LossParams.from_defaults({
        k: base_cfg[k] for k in base_cfg
        if k in LossParams.__dataclass_fields__
    })

    insurer_dict = {**INSURER_DEFAULTS}
    for k, v in base_cfg.get("insurer_overrides", {}).items():
        insurer_dict[k] = v
    for k, v in candidate_params.items():
        insurer_dict[k] = v
    insurer = InsurerParams.from_dict(insurer_dict)
    dummy = InsurerParams.from_dict(INSURER_DEFAULTS)

    config = SimulationConfig(
        n_paths=n_paths,
        n_years=n_years,
        random_seed=seed,
        market_params=market_params,
        loss_params=loss_params,
        insurers=[insurer, dummy],
    )

    results = run_simulation(config)

    # Condition on paths starting in the target regime
    regimes = results.market["regime"]  # (n_paths, n_years)
    mask = regimes[:, 0] == target_regime

    if mask.sum() < 20:
        # Not enough paths — use all (fallback)
        mask = np.ones(config.n_paths, dtype=bool)

    # Compute metrics for qualifying paths only
    a = results.insurer_a
    qualified_profit = a["total_profit"][mask]
    qualified_capital = a["capital"][mask]
    qualified_econ_cap = a["economic_capital"][mask]
    qualified_cr = a["combined_ratio"][mask]
    qualified_ruined = a["is_ruined"][mask]

    n_qual, n_yr = qualified_profit.shape

    # Through-cycle RORAC
    total_pft = qualified_profit.sum(axis=1)
    avg_ec = qualified_econ_cap.mean(axis=1)
    tc_rorac = np.where(avg_ec > 0, (total_pft / n_yr) / avg_ec, 0.0)

    # Max drawdown
    running_max = np.maximum.accumulate(qualified_capital, axis=1)
    drawdowns = running_max - qualified_capital
    max_dd = np.max(drawdowns, axis=1)

    # CR
    cr_clean = qualified_cr.copy()
    cr_clean[~np.isfinite(cr_clean)] = np.nan

    return {
        "params": candidate_params,
        "rorac": float(np.mean(tc_rorac)),
        "ruin_prob": float(np.mean(qualified_ruined.any(axis=1))),
        "max_drawdown": float(np.mean(max_dd)),
        "mean_cr": float(np.nanmean(cr_clean)),
        "terminal_capital": float(np.mean(qualified_capital[:, -1])),
        "n_qualified": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Pareto front extraction
# ---------------------------------------------------------------------------
def extract_pareto_front(candidates: list) -> list:
    """
    Non-dominated sort on (maximize RORAC, minimize ruin_prob, minimize max_drawdown).

    Returns list of StrategyCandidate with is_pareto=True.
    """
    n = len(candidates)
    dominated = [False] * n

    for i in range(n):
        if dominated[i]:
            continue
        for j in range(n):
            if i == j or dominated[j]:
                continue
            # Check if j dominates i
            # j dominates i if j is >= i on all objectives and > on at least one
            j_better_rorac = candidates[j].rorac >= candidates[i].rorac
            j_better_ruin = candidates[j].ruin_prob <= candidates[i].ruin_prob
            j_better_dd = candidates[j].max_drawdown <= candidates[i].max_drawdown
            j_strictly = (
                candidates[j].rorac > candidates[i].rorac
                or candidates[j].ruin_prob < candidates[i].ruin_prob
                or candidates[j].max_drawdown < candidates[i].max_drawdown
            )
            if j_better_rorac and j_better_ruin and j_better_dd and j_strictly:
                dominated[i] = True
                break

    front = []
    for i in range(n):
        if not dominated[i]:
            candidates[i].is_pareto = True
            front.append(candidates[i])

    # Sort by RORAC descending
    front.sort(key=lambda c: c.rorac, reverse=True)
    return front


def compute_gap_to_frontier(
    current_rorac: float,
    current_drawdown: float,
    front: list,
) -> dict:
    """
    Find the nearest Pareto-optimal point at similar or lower risk.

    Returns dict with gap metrics.
    """
    if not front:
        return {"rorac_gap": 0, "risk_gap": 0, "nearest": None}

    # Find Pareto point with drawdown <= current (or closest)
    # that has highest RORAC
    eligible = [c for c in front if c.max_drawdown <= current_drawdown * 1.1]
    if not eligible:
        eligible = front

    best = max(eligible, key=lambda c: c.rorac)

    return {
        "rorac_gap": best.rorac - current_rorac,
        "risk_gap": current_drawdown - best.max_drawdown,
        "nearest": best,
    }


# ---------------------------------------------------------------------------
# Main optimizer entry points
# ---------------------------------------------------------------------------
def optimize_for_regime(
    regime_idx: Optional[int],
    base_config: dict,
    n_candidates: int = 500,
    n_paths: int = 1000,
    n_years: int = 10,
    max_workers: int = 8,
    seed: int = 42,
) -> RegimeOptResult:
    """
    Run optimization for a specific regime (or unconditional if regime_idx is None).
    """
    regime_name = REGIMES[regime_idx] if regime_idx is not None else "unconditional"
    candidates = generate_candidates(n_candidates, seed=seed)

    # Build worker args
    worker_fn = _evaluate_regime_conditioned if regime_idx is not None else _evaluate_single
    args_list = [
        (cand, base_config, n_paths, n_years, regime_idx, seed + i)
        for i, cand in enumerate(candidates)
    ]

    # Parallel evaluation
    raw_results = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(worker_fn, a): i for i, a in enumerate(args_list)}
        for future in as_completed(futures):
            try:
                raw_results.append(future.result())
            except Exception:
                pass  # Skip failed candidates

    # Convert to StrategyCandidate, filtering NaN/Inf results
    all_candidates = []
    for r in raw_results:
        rorac = r["rorac"]
        ruin = r["ruin_prob"]
        dd = r["max_drawdown"]
        # Skip candidates with non-finite key metrics
        if not (np.isfinite(rorac) and np.isfinite(ruin) and np.isfinite(dd)):
            continue
        all_candidates.append(StrategyCandidate(
            params=r["params"],
            rorac=rorac,
            ruin_prob=ruin,
            max_drawdown=dd,
            mean_cr=r["mean_cr"] if np.isfinite(r["mean_cr"]) else 1.0,
            terminal_capital=r["terminal_capital"] if np.isfinite(r["terminal_capital"]) else 0.0,
        ))

    if not all_candidates:
        return RegimeOptResult(
            regime=regime_name,
            all_candidates=[],
            pareto_front=[],
            best_rorac=None,
            best_safety=None,
        )

    # Extract Pareto front
    pareto = extract_pareto_front(all_candidates)

    best_rorac = max(all_candidates, key=lambda c: c.rorac)
    best_safety = min(all_candidates, key=lambda c: c.ruin_prob)

    return RegimeOptResult(
        regime=regime_name,
        all_candidates=all_candidates,
        pareto_front=pareto,
        best_rorac=best_rorac,
        best_safety=best_safety,
    )


def run_full_optimization(
    base_config: dict,
    n_candidates: int = 500,
    n_paths: int = 1000,
    n_years: int = 10,
    max_workers: int = 8,
    current_summaries: list = None,
    # Backward-compat kwargs (ignored if current_summaries provided)
    current_a_params: dict = None,
    current_b_params: dict = None,
    current_a_summary: dict = None,
    current_b_summary: dict = None,
) -> FullOptResult:
    """
    Run regime-conditional + unconditional optimization.

    Returns complete results with gap analysis for all current strategies.
    """
    t0 = time.perf_counter()

    # Backward compat: build summaries list from old A/B kwargs
    if current_summaries is None:
        current_summaries = []
        if current_a_summary:
            current_summaries.append(current_a_summary)
        if current_b_summary:
            current_summaries.append(current_b_summary)

    # Run unconditional optimization (all regimes mixed)
    unconditional = optimize_for_regime(
        None, base_config, n_candidates, n_paths, n_years, max_workers, seed=42,
    )

    # Run per-regime optimizations
    by_regime = {}
    for idx, regime_name in enumerate(REGIMES):
        result = optimize_for_regime(
            idx, base_config, n_candidates // 2, n_paths, n_years,
            max_workers, seed=100 + idx * 1000,
        )
        by_regime[regime_name] = result

    # Gap analysis for all strategies
    current_gaps = []
    for s in current_summaries:
        if not s:
            current_gaps.append({})
            continue
        rorac = s.get("mean_through_cycle_rorac", 0)
        dd = s.get("mean_max_drawdown", 0)
        gaps = {}
        gaps["unconditional"] = compute_gap_to_frontier(
            rorac, dd, unconditional.pareto_front,
        )
        for regime_name, regime_result in by_regime.items():
            gaps[regime_name] = compute_gap_to_frontier(
                rorac, dd, regime_result.pareto_front,
            )
        current_gaps.append(gaps)

    total_sims = n_candidates + (n_candidates // 2) * 4
    elapsed = time.perf_counter() - t0

    return FullOptResult(
        by_regime=by_regime,
        unconditional=unconditional,
        current_gaps=current_gaps,
        n_candidates=n_candidates,
        n_paths=n_paths,
        total_sims=total_sims,
        elapsed_seconds=elapsed,
    )
