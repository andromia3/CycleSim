"""
Import/export utilities for capital model data and simulation results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
from io import BytesIO

from .losses import CapitalModelData


def import_capital_model(
    filepath: str | Path | BytesIO,
    gross_column: str = "gross_lr",
    net_column: str = "net_lr",
    ri_recoveries_column: Optional[str] = None,
    ri_spend_column: Optional[str] = None,
) -> CapitalModelData:
    """
    Import capital model output from CSV or Excel.

    Expects at least 'gross_lr' and 'net_lr' columns with ~10,000 rows.
    """
    if isinstance(filepath, BytesIO):
        # Try Excel first, then CSV
        try:
            df = pd.read_excel(filepath)
        except Exception:
            filepath.seek(0)
            df = pd.read_csv(filepath)
    else:
        path = Path(filepath)
        if path.suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)

    if gross_column not in df.columns:
        raise ValueError(
            f"Column '{gross_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if net_column not in df.columns:
        raise ValueError(
            f"Column '{net_column}' not found. "
            f"Available columns: {list(df.columns)}"
        )
    if len(df) < 100:
        raise ValueError(f"Need at least 100 sims, got {len(df)}")

    # Coerce to numeric safely — non-numeric values become NaN, then drop them
    gross_vals = pd.to_numeric(df[gross_column], errors="coerce")
    net_vals = pd.to_numeric(df[net_column], errors="coerce")
    valid_mask = gross_vals.notna() & net_vals.notna()
    if valid_mask.sum() < 100:
        raise ValueError(
            f"Only {valid_mask.sum()} valid numeric rows after cleaning. "
            f"Need at least 100."
        )
    result = CapitalModelData(
        gross_loss_ratios=gross_vals[valid_mask].values.astype(np.float64),
        net_loss_ratios=net_vals[valid_mask].values.astype(np.float64),
    )

    if ri_recoveries_column and ri_recoveries_column in df.columns:
        result.ri_recoveries = pd.to_numeric(
            df.loc[valid_mask, ri_recoveries_column], errors="coerce"
        ).fillna(0).values.astype(np.float64)
    if ri_spend_column and ri_spend_column in df.columns:
        result.ri_spend = pd.to_numeric(
            df.loc[valid_mask, ri_spend_column], errors="coerce"
        ).fillna(0).values.astype(np.float64)

    return result


def export_results_to_excel(results, filepath) -> None:
    """
    Export simulation results to a multi-sheet Excel workbook.

    filepath can be a string/Path for disk or BytesIO for in-memory.

    Sheets:
    - Summary: side-by-side scalar metrics
    - Market: mean market path
    - Insurer A: year-by-year mean metrics
    - Insurer B: year-by-year mean metrics
    """
    years = list(range(1, results.config.n_years + 1))

    n_strategies = len(results.config.insurers)

    with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
        # --- Summary sheet (N strategy columns) ---
        summary_data = []
        if results.summaries:
            for key in sorted(results.summaries[0].keys()):
                val_0 = results.summaries[0].get(key)
                if isinstance(val_0, (int, float)):
                    row = {"Metric": key}
                    for i in range(n_strategies):
                        name = results.config.insurers[i].name
                        row[name] = results.summaries[i].get(key)
                    summary_data.append(row)
        if summary_data:
            pd.DataFrame(summary_data).to_excel(
                writer, sheet_name="Summary", index=False
            )

        # --- Market sheet ---
        market_df = pd.DataFrame({
            "Year": years,
            "Mean Loss Ratio": results.market["market_loss_ratio"].mean(axis=0),
            "Mean Rate Change": results.market["market_rate_change"].mean(axis=0),
            "P5 Loss Ratio": np.percentile(
                results.market["market_loss_ratio"], 5, axis=0
            ),
            "P95 Loss Ratio": np.percentile(
                results.market["market_loss_ratio"], 95, axis=0
            ),
        })
        market_df.to_excel(writer, sheet_name="Market", index=False)

        # --- Insurer sheets (one per strategy) ---
        for i in range(n_strategies):
            name = results.config.insurers[i].name
            data = results.insurers[i]
            ins_df = pd.DataFrame({
                "Year": years,
                "Mean GWP": data["gwp"].mean(axis=0),
                "Mean NWP": data["nwp"].mean(axis=0),
                "Mean Cession %": data["cession_pct"].mean(axis=0),
                "Mean Gross LR": data["gross_lr"].mean(axis=0),
                "Mean Net LR": data["net_lr"].mean(axis=0),
                "Mean Expense Ratio": data["expense_ratio"].mean(axis=0),
                "Mean Combined Ratio": data["combined_ratio"].mean(axis=0),
                "Mean UW Profit": data["uw_profit"].mean(axis=0),
                "Mean Total Profit": data["total_profit"].mean(axis=0),
                "Mean Capital": data["capital"].mean(axis=0),
                "Mean RORAC": data["rorac"].mean(axis=0),
                "Mean Cumulative Profit": data["cumulative_profit"].mean(axis=0),
            })
            sheet_name = name[:31]  # Excel sheet name limit
            ins_df.to_excel(writer, sheet_name=sheet_name, index=False)

        # --- Metadata sheet (Pass 26: data provenance) ---
        import datetime
        has_cm = results.config.capital_model is not None
        market_params = results.market.get("params")
        strategy_names = ", ".join(results.config.insurers[i].name for i in range(n_strategies))
        meta_data = [
            {"Field": "Generated", "Value": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            {"Field": "CycleSim Version", "Value": "Pass 31"},
            {"Field": "Loss Model", "Value": "Uploaded Capital Model" if has_cm else "Parametric (Lloyd's-calibrated)"},
            {"Field": "N Paths", "Value": results.config.n_paths},
            {"Field": "N Years", "Value": results.config.n_years},
            {"Field": "N Strategies", "Value": n_strategies},
            {"Field": "Random Seed", "Value": results.config.random_seed},
            {"Field": "Strategies", "Value": strategy_names},
            {"Field": "Cycle Period", "Value": getattr(market_params, "implied_cycle_period", "N/A") if market_params else "N/A"},
            {"Field": "Long-Run LR", "Value": getattr(market_params, "long_run_loss_ratio", "N/A") if market_params else "N/A"},
            {"Field": "Disclaimer", "Value": (
                "ILLUSTRATIVE ONLY. Results generated using synthetic Lloyd's-calibrated parameters. "
                "Not based on proprietary data. For strategic analysis and educational purposes."
                if not has_cm else
                "Results generated using your uploaded capital model data."
            )},
        ]
        pd.DataFrame(meta_data).to_excel(writer, sheet_name="Metadata", index=False)


def generate_sample_capital_model(
    n_sims: int = 10000,
    seed: int = 123,
) -> pd.DataFrame:
    """Generate synthetic capital model output for testing."""
    rng = np.random.default_rng(seed)

    # Attritional
    attritional = rng.lognormal(np.log(0.50) - 0.06**2 / 2, 0.06, n_sims)
    # Large
    n_large = rng.poisson(3.0, n_sims)
    large = np.zeros(n_sims)
    for i in range(n_sims):
        if n_large[i] > 0:
            sevs = (rng.pareto(1.8, n_large[i]) + 1) * 0.01
            large[i] = min(sevs.sum(), 0.25)
    # Cat
    n_cat = rng.poisson(0.3, n_sims)
    cat = np.zeros(n_sims)
    for i in range(n_sims):
        if n_cat[i] > 0:
            cat[i] = rng.lognormal(np.log(0.08) - 1.2**2 / 2, 1.2, n_cat[i]).sum()

    gross = attritional + large + cat
    # Net: apply ~23% cession with 85% effectiveness
    cession = 0.23
    eff = 0.85
    net = gross * (1 - cession * eff)

    return pd.DataFrame({
        "gross_lr": gross,
        "net_lr": net,
        "attritional": attritional,
        "large": large,
        "cat": cat,
    })


def export_raw_paths_csv(results, filepath, max_paths: int = 1000) -> None:
    """
    Export raw path-level data to CSV for independent verification.

    Includes per-path, per-year: market LR, regime, GWP, NWP, gross LR,
    net LR, combined ratio, profit, capital, solvency, RORAC for each insurer.

    Args:
        results: SimulationResults object
        filepath: string/Path or BytesIO
        max_paths: limit paths exported to control file size
    """
    n_paths = min(results.config.n_paths, max_paths)
    n_years = results.config.n_years
    n_strategies = len(results.config.insurers)

    rows = []
    for p in range(n_paths):
        for t in range(n_years):
            row = {
                "path": p,
                "year": t + 1,
                "market_lr": float(results.market["market_loss_ratio"][p, t]),
                "rate_adequacy": float(results.market["rate_adequacy"][p, t]),
                "market_rate_change": float(results.market["market_rate_change"][p, t]),
                "regime": int(results.market["regime"][p, t]),
            }
            for i in range(n_strategies):
                name = results.config.insurers[i].name
                prefix = name.replace(" ", "_")[:12]
                ins = results.insurers[i]
                row[f"{prefix}_gwp"] = float(ins["gwp"][p, t])
                row[f"{prefix}_nwp"] = float(ins["nwp"][p, t])
                row[f"{prefix}_gross_lr"] = float(ins["gross_lr"][p, t])
                row[f"{prefix}_net_lr"] = float(ins["net_lr"][p, t])
                row[f"{prefix}_combined_ratio"] = float(ins["combined_ratio"][p, t])
                row[f"{prefix}_total_profit"] = float(ins["total_profit"][p, t])
                row[f"{prefix}_capital"] = float(ins["capital"][p, t])
                row[f"{prefix}_solvency"] = float(ins["solvency_ratio"][p, t])
                row[f"{prefix}_rorac"] = float(ins["rorac"][p, t])
            rows.append(row)

    df = pd.DataFrame(rows)
    if isinstance(filepath, BytesIO):
        df.to_csv(filepath, index=False)
    else:
        df.to_csv(filepath, index=False)
