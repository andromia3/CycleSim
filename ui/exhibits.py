"""
Summary tables, KPI cards, and formatted output components.

Professional actuarial formatting: color-coded winners, delta indicators,
contextual tooltips, and worst-path analysis tables.
"""

import numpy as np
import dash_mantine_components as dmc
from dash import html

from cyclesim.defaults import STRATEGY_COLORS


def _fmt(val, fmt_type="number"):
    """Format a value for display. Handles None, NaN, and Inf gracefully."""
    if val is None:
        return "\u2014"
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "\u2014"
    if not np.isfinite(val):
        return "\u2014"
    if fmt_type == "pct":
        return f"{val:.1%}"
    elif fmt_type == "pct2":
        return f"{val:.2%}"
    elif fmt_type == "money":
        if abs(val) >= 1e9:
            return f"\u00a3{val/1e9:.1f}bn"
        elif abs(val) >= 1e6:
            return f"\u00a3{val/1e6:.0f}m"
        else:
            return f"\u00a3{val/1e3:.0f}k"
    elif fmt_type == "ratio":
        return f"{val:.2f}x"
    elif fmt_type == "multiple":
        return f"{val:.2f}"
    return f"{val:.2f}"


def _safe_num(val, fallback=0.0):
    """Coerce a value to float, returning fallback for None/NaN/Inf."""
    if val is None:
        return fallback
    try:
        val = float(val)
    except (TypeError, ValueError):
        return fallback
    return val if np.isfinite(val) else fallback


def _get_colors(colors, n):
    c = colors or STRATEGY_COLORS
    return [c[i % len(c)] for i in range(n)]


# ---------------------------------------------------------------------------
# Metric tooltips (F10) — hover explanations for actuarial terms
# ---------------------------------------------------------------------------
METRIC_TOOLTIPS = {
    "Through-Cycle RORAC (ann.)": "Mean annual profit divided by mean economic capital, averaged across all Monte Carlo paths. The primary risk-adjusted return metric.",
    "Mean Combined Ratio": "Average (net claims + expenses + RI cost) / NWP across all paths and years. Below 100% indicates underwriting profit.",
    "Mean Cumulative Profit": "Average total profit accumulated over the full projection horizon across all paths.",
    "Profit-to-Risk Ratio": "Mean cumulative profit divided by standard deviation of cumulative profit. Higher = more profit per unit of uncertainty.",
    "Probability of Ruin": "Fraction of Monte Carlo paths where capital falls below zero at any point during the projection.",
    "VaR(95%) Cumulative": "5th percentile of the cumulative profit distribution. 1-in-20 year downside scenario.",
    "VaR(99.5%) Cumulative": "0.5th percentile of cumulative profit. Lloyd's Solvency Capital Requirement (SCR) calibration standard.",
    "TVaR(99.5%) Cumulative": "Expected cumulative profit conditional on being below VaR(99.5%). Captures average severity beyond the SCR threshold.",
    "VaR-Based Econ Capital": "Capital needed to absorb a 1-in-200 annual loss: max(0, -VaR₉₉.₅%(annual profit)). Distribution-derived SCR.",
    "Mean Max Drawdown": "Average of the maximum peak-to-trough capital decline across all paths. Measures worst interim loss.",
    "Mean Terminal GWP": "Average gross written premium at the end of the projection period.",
    "GWP CAGR": "Compound annual growth rate of GWP from year 1 to final year, averaged across paths.",
    "Mean Expense Ratio": "Average expense ratio including dynamic adjustments for growth/shrinkage penalties.",
    "Mean Cession %": "Average proportion of GWP ceded to reinsurers across all paths and years.",
    "Mean Solvency Ratio": "Average available capital / economic capital. Above 1.5x is typically comfortable; below 1.2x triggers injection.",
    "Total RI Cost": "Mean cumulative reinsurance cost (premium paid to reinsurers net of expected recovery).",
    "Total Capital Injections": "Mean cumulative capital injected by shareholders when solvency falls below trigger.",
    "Total Dividends Extracted": "Mean cumulative dividends paid when solvency exceeds the extraction trigger.",
}


# ---------------------------------------------------------------------------
# Conditional formatting thresholds (F9) — RAG status for metrics
# ---------------------------------------------------------------------------
_RAG_THRESHOLDS = {
    "mean_combined_ratio": [
        (0.95, "rgba(16, 185, 129, 0.08)"),    # green: < 95%
        (1.02, "rgba(245, 158, 11, 0.08)"),     # amber: 95-102%
        (float("inf"), "rgba(220, 38, 38, 0.08)"),  # red: > 102%
    ],
    "prob_ruin": [
        (0.01, "rgba(16, 185, 129, 0.08)"),
        (0.05, "rgba(245, 158, 11, 0.08)"),
        (float("inf"), "rgba(220, 38, 38, 0.08)"),
    ],
    "mean_through_cycle_rorac": [
        (0.05, "rgba(220, 38, 38, 0.08)"),
        (0.10, "rgba(245, 158, 11, 0.08)"),
        (float("inf"), "rgba(16, 185, 129, 0.08)"),
    ],
    "mean_solvency_ratio": [
        (1.2, "rgba(220, 38, 38, 0.08)"),
        (1.5, "rgba(245, 158, 11, 0.08)"),
        (float("inf"), "rgba(16, 185, 129, 0.08)"),
    ],
    "mean_expense_ratio": [
        (0.32, "rgba(16, 185, 129, 0.08)"),
        (0.38, "rgba(245, 158, 11, 0.08)"),
        (float("inf"), "rgba(220, 38, 38, 0.08)"),
    ],
}


def _rag_bg(key: str, val: float) -> str:
    """Return RAG background color for a metric value, or empty string."""
    thresholds = _RAG_THRESHOLDS.get(key)
    if not thresholds:
        return ""
    for limit, color in thresholds:
        if val < limit:
            return color
    return ""


def _delta_badge(a_val, b_val, fmt_type, a_is_better):
    """Small delta indicator showing the gap."""
    a_val = _safe_num(a_val)
    b_val = _safe_num(b_val)
    diff = a_val - b_val
    if fmt_type == "pct" or fmt_type == "pct2":
        delta_text = f"{abs(diff):.1%}pts"
    elif fmt_type == "money":
        if abs(diff) >= 1e9:
            delta_text = f"\u00a3{abs(diff)/1e9:.1f}bn"
        elif abs(diff) >= 1e6:
            delta_text = f"\u00a3{abs(diff)/1e6:.0f}m"
        else:
            delta_text = f"\u00a3{abs(diff)/1e3:.0f}k"
    else:
        delta_text = f"{abs(diff):.2f}"

    color = "green" if a_is_better else "orange"
    arrow = "\u25b2" if a_is_better else "\u25bc"

    return dmc.Badge(
        f"{arrow} {delta_text}",
        size="xs", variant="light", color=color,
        style={"fontWeight": 500, "fontSize": "10px"},
    )


def summary_table(summaries: list, names: list, colors=None) -> dmc.Stack:
    """Build the main comparison summary table with section headers."""
    n = len(names)
    clrs = _get_colors(colors, n)

    sections = [
        ("Profitability", [
            ("Through-Cycle RORAC (ann.)", "mean_through_cycle_rorac", "pct"),
            ("Mean Combined Ratio", "mean_combined_ratio", "pct"),
            ("Mean Cumulative Profit", "mean_cumulative_profit", "money"),
            ("Profit-to-Risk Ratio", "profit_to_risk_ratio", "multiple"),
        ]),
        ("Risk", [
            ("Probability of Ruin", "prob_ruin", "pct2"),
            ("VaR(95%) Cumulative", "var_95_cumulative", "money"),
            ("VaR(99.5%) Cumulative", "var_995_cumulative", "money"),
            ("TVaR(99.5%) Cumulative", "tvar_995_cumulative", "money"),
            ("VaR-Based Econ Capital", "var_based_econ_cap", "money"),
            ("Mean Max Drawdown", "mean_max_drawdown", "money"),
        ]),
        ("Growth & Scale", [
            ("Mean Terminal GWP", "mean_terminal_gwp", "money"),
            ("GWP CAGR", "mean_gwp_cagr", "pct"),
            ("Mean Expense Ratio", "mean_expense_ratio", "pct"),
            ("Mean Cession %", "mean_cession_pct", "pct"),
        ]),
        ("Capital", [
            ("Mean Solvency Ratio", "mean_solvency_ratio", "ratio"),
            ("Total RI Cost", "total_ri_cost", "money"),
            ("Total Capital Injections", "total_injections", "money"),
            ("Total Dividends Extracted", "total_dividends", "money"),
        ]),
    ]

    higher_better = {
        "mean_through_cycle_rorac", "mean_cumulative_profit",
        "var_95_cumulative", "tvar_95_cumulative",
        "var_995_cumulative", "tvar_995_cumulative",
        "profit_to_risk_ratio", "mean_terminal_gwp", "mean_gwp_cagr",
        "mean_solvency_ratio", "total_dividends",
    }
    lower_better = {
        "mean_combined_ratio", "prob_ruin", "mean_max_drawdown",
        "mean_expense_ratio", "total_ri_cost", "total_injections",
        "var_based_econ_cap",
    }

    # Header
    header_cells = [dmc.TableTh("Metric", style={"width": "35%"})]
    for i, name in enumerate(names):
        header_cells.append(dmc.TableTh(name, style={"textAlign": "right"}))
    header = dmc.TableThead(dmc.TableTr(header_cells))

    body_rows = []
    wins = [0] * n
    total_metrics = 0

    for section_title, metrics in sections:
        section_cells = [
            dmc.TableTd(
                dmc.Text(section_title, fw=700, size="xs", c="dimmed", tt="uppercase"),
                style={"backgroundColor": "#f9fafb", "paddingTop": "10px", "paddingBottom": "4px"},
            ),
        ]
        for _ in names:
            section_cells.append(dmc.TableTd(style={"backgroundColor": "#f9fafb"}))
        body_rows.append(dmc.TableTr(section_cells))

        for label, key, fmt in metrics:
            vals = [_safe_num(s.get(key, 0)) for s in summaries]

            # Find best
            if key in higher_better:
                best_idx = int(np.argmax(vals))
            elif key in lower_better:
                best_idx = int(np.argmin(vals))
            else:
                best_idx = int(np.argmax(vals))

            wins[best_idx] += 1
            total_metrics += 1

            # Metric label with optional tooltip
            tooltip_text = METRIC_TOOLTIPS.get(label)
            if tooltip_text:
                label_el = dmc.Tooltip(
                    dmc.Text(label, size="sm", td="underline", style={"textDecorationStyle": "dotted", "cursor": "help"}),
                    label=tooltip_text, position="right", multiline=True, w=300,
                )
            else:
                label_el = dmc.Text(label, size="sm")

            # CI key mapping for bootstrap intervals
            _CI_KEY_MAP = {
                "var_95_cumulative": "var_95",
                "var_995_cumulative": "var_995",
                "tvar_995_cumulative": "tvar_995",
                "prob_ruin": "prob_ruin",
                "mean_through_cycle_rorac": "rorac",
            }

            row_cells = [dmc.TableTd(label_el)]
            for i, v in enumerate(vals):
                is_best = (i == best_idx)
                bg = _rag_bg(key, v)
                cell_style = {"textAlign": "right"}
                if bg:
                    cell_style["backgroundColor"] = bg

                cell_children = [dmc.Text(
                    _fmt(v, fmt),
                    size="sm", fw=600 if is_best else 400,
                    c=clrs[i] if is_best else "#6b7280",
                    ff="'JetBrains Mono', monospace",
                )]

                # Add CI annotation if available
                ci_key = _CI_KEY_MAP.get(key)
                if ci_key:
                    ci_data = summaries[i].get("confidence_intervals", {})
                    ci_bounds = ci_data.get(ci_key)
                    if ci_bounds:
                        lo, hi = ci_bounds
                        if fmt == "money":
                            ci_text = f"[{_fmt(lo, 'money')}, {_fmt(hi, 'money')}]"
                        elif fmt in ("pct", "pct2"):
                            ci_text = f"[{lo:.1%}, {hi:.1%}]"
                        else:
                            ci_text = f"[{lo:.3f}, {hi:.3f}]"
                        cell_children.append(dmc.Text(
                            ci_text, size="xs", c="#9ca3af",
                            ff="'JetBrains Mono', monospace",
                        ))

                row_cells.append(dmc.TableTd(
                    dmc.Stack(cell_children, gap=0) if len(cell_children) > 1
                    else cell_children[0],
                    style=cell_style,
                ))
            body_rows.append(dmc.TableTr(row_cells))

    # PV section (only if discount_rate > 0 and PV data exists)
    has_pv = any(s.get("pv_mean_cumulative_profit") is not None for s in summaries)
    if has_pv:
        pv_metrics = [
            ("PV Mean Cumulative Profit", "pv_mean_cumulative_profit", "money"),
            ("PV VaR(99.5%) Cumulative", "pv_var_995_cumulative", "money"),
            ("PV TVaR(99.5%) Cumulative", "pv_tvar_995_cumulative", "money"),
        ]
        section_cells = [
            dmc.TableTd(
                dmc.Text("PRESENT VALUE", fw=700, size="xs", c="dimmed", tt="uppercase"),
                style={"backgroundColor": "#f9fafb", "paddingTop": "10px", "paddingBottom": "4px"},
            ),
        ]
        for _ in names:
            section_cells.append(dmc.TableTd(style={"backgroundColor": "#f9fafb"}))
        body_rows.append(dmc.TableTr(section_cells))

        for label, key, fmt in pv_metrics:
            vals = [_safe_num(s.get(key, 0)) for s in summaries]
            best_idx_pv = int(np.argmax(vals))
            row_cells = [dmc.TableTd(dmc.Text(label, size="sm"))]
            for i, v in enumerate(vals):
                is_best = (i == best_idx_pv)
                row_cells.append(dmc.TableTd(
                    dmc.Text(
                        _fmt(v, fmt),
                        size="sm", fw=600 if is_best else 400,
                        c=clrs[i] if is_best else "#6b7280",
                        ff="'JetBrains Mono', monospace",
                    ),
                    style={"textAlign": "right"},
                ))
            body_rows.append(dmc.TableTr(row_cells))

    body = dmc.TableTbody(body_rows)

    # Score summary
    best_strat = int(np.argmax(wins))
    if max(wins) > total_metrics / n:
        verdict_text = f"{names[best_strat]} leads {wins[best_strat]}/{total_metrics} metrics"
        verdict_color = clrs[best_strat]
    else:
        verdict_text = f"Close contest: " + ", ".join(f"{names[i]} {wins[i]}" for i in range(n))
        verdict_color = "#6b7280"

    return dmc.Stack([
        dmc.Table(
            [header, body],
            striped=True, highlightOnHover=True,
            withTableBorder=True, withColumnBorders=True,
        ),
        dmc.Text(verdict_text, size="xs", c=verdict_color, fw=600, ta="center"),
    ], gap="xs")


def stat_card(
    title, value: str,
    subtitle: str = "", color: str = "blue",
    delta: str = "", delta_positive: bool = True,
) -> dmc.Paper:
    """Single statistic card with optional delta indicator."""
    title_style = {"fontSize": "10px", "letterSpacing": "0.04em", "textTransform": "uppercase"}
    if isinstance(title, str):
        title_el = dmc.Text(title, size="xs", c="#6b7280", fw=600, style=title_style)
    else:
        # Component title (e.g. wrapped in Tooltip)
        title_el = html.Div(title, style={**title_style, "color": "#6b7280", "fontWeight": 600})
    children = [
        title_el,
        dmc.Text(value, fw=700,
                 ff="'JetBrains Mono', monospace",
                 style={"fontSize": "1.15rem", "color": "#111827"}),
    ]
    if delta:
        arrow = "\u25b2" if delta_positive else "\u25bc"
        children.append(
            dmc.Text(
                f"{arrow} {delta}",
                size="xs", fw=500,
                c="#059669" if delta_positive else "#dc2626",
            )
        )
    if subtitle:
        children.append(dmc.Text(subtitle, size="xs", c="#adb5bd"))
    return dmc.Paper(
        dmc.Stack(children, gap=2, align="center"),
        p="md", radius="md", withBorder=True,
        style={"textAlign": "center"},
    )


def kpi_row(summaries: list, names: list, elapsed: float, colors=None) -> dmc.SimpleGrid:
    """Top-level KPI cards for N strategies."""
    n = len(names)
    clrs = _get_colors(colors, n)

    cards = []
    # RORAC cards per strategy
    roracs = [_safe_num(s.get("mean_through_cycle_rorac", 0)) for s in summaries]
    best_rorac_idx = int(np.argmax(roracs))
    rorac_tip = METRIC_TOOLTIPS.get("Through-Cycle RORAC (ann.)", "")
    for i in range(n):
        delta_text = ""
        delta_pos = True
        if i != best_rorac_idx:
            delta_text = f"vs {_fmt(roracs[best_rorac_idx], 'pct')}"
            delta_pos = False
        cards.append(stat_card(
            dmc.Tooltip(
                dmc.Text(f"{names[i]} RORAC", td="underline",
                         style={"textDecorationStyle": "dotted", "cursor": "help"}),
                label=rorac_tip, position="bottom", multiline=True, w=300,
            ) if rorac_tip else f"{names[i]} RORAC",
            _fmt(roracs[i], "pct"), names[i],
            delta=delta_text, delta_positive=delta_pos,
        ))

    # Combined ratio card
    cr_tip = METRIC_TOOLTIPS.get("Mean Combined Ratio", "")
    cr_parts = " | ".join(_fmt(_safe_num(s.get("mean_combined_ratio", 1)), "pct") for s in summaries)
    name_parts = " | ".join(names)
    cards.append(stat_card(
        dmc.Tooltip(
            dmc.Text("Combined Ratio", td="underline",
                     style={"textDecorationStyle": "dotted", "cursor": "help"}),
            label=cr_tip, position="bottom", multiline=True, w=300,
        ) if cr_tip else "Combined Ratio",
        cr_parts, name_parts,
    ))

    # Sim time
    cards.append(stat_card("Sim Time", f"{elapsed:.2f}s",
                           f"{summaries[0].get('n_paths', '?')} paths" if 'n_paths' in summaries[0] else ""))

    return dmc.SimpleGrid(cols=min(n + 2, 6), spacing="md", children=cards)


def worst_paths_table(rows: list, names: list, colors=None) -> dmc.Table:
    """Table of worst-performing paths."""
    n = len(names)
    clrs = _get_colors(colors, n)

    header_cells = [
        dmc.TableTh("#", style={"width": "5%"}),
        dmc.TableTh("Path", style={"width": "8%"}),
    ]
    for name in names:
        header_cells.append(dmc.TableTh(f"Cum. Profit ({name})", style={"textAlign": "right"}))
    for name in names:
        header_cells.append(dmc.TableTh(f"Peak CR ({name})", style={"textAlign": "right"}))
    header_cells.append(dmc.TableTh("Ruin?", style={"textAlign": "center"}))
    header = dmc.TableThead(dmc.TableTr(header_cells))

    body_rows = []
    for r in rows:
        strategies = r.get("strategies", [])
        ruined = []
        for j, s in enumerate(strategies):
            if s.get("ruined") == "Yes":
                ruined.append(names[j] if j < len(names) else f"S{j}")
        ruin_text = ", ".join(ruined) if ruined else "\u2014"
        ruin_color = "red" if ruined else "dimmed"

        row_cells = [
            dmc.TableTd(dmc.Text(str(r["rank"]), size="sm", c="dimmed")),
            dmc.TableTd(dmc.Text(f"#{r['path']}", size="sm", fw=500)),
        ]
        for s in strategies:
            row_cells.append(dmc.TableTd(
                dmc.Text(s["cum_profit"], size="sm", ff="monospace"), style={"textAlign": "right"}))
        for s in strategies:
            row_cells.append(dmc.TableTd(
                dmc.Text(s["max_cr"], size="sm", ff="monospace"), style={"textAlign": "right"}))
        row_cells.append(dmc.TableTd(
            dmc.Text(ruin_text, size="sm", c=ruin_color, fw=500), style={"textAlign": "center"}))

        body_rows.append(dmc.TableTr(row_cells))

    return dmc.Table(
        [header, dmc.TableTbody(body_rows)],
        striped=True, highlightOnHover=True,
        withTableBorder=True, withColumnBorders=True,
    )


def drilldown_table(
    market: dict, ins_list: list,
    path_idx: int, n_years: int,
    names: list, colors=None,
) -> dmc.Table:
    """Year-by-year table for a single path drill-down."""
    n = len(names)
    clrs = _get_colors(colors, n)

    header_cells = [dmc.TableTh("Year", style={"width": "5%"}),
                    dmc.TableTh("Market LR", style={"textAlign": "right"})]
    for name in names:
        header_cells.append(dmc.TableTh(f"GWP ({name})", style={"textAlign": "right"}))
    for name in names:
        header_cells.append(dmc.TableTh(f"CR ({name})", style={"textAlign": "right"}))
    for name in names:
        header_cells.append(dmc.TableTh(f"Profit ({name})", style={"textAlign": "right"}))
    header = dmc.TableThead(dmc.TableTr(header_cells))

    body_rows = []
    n_paths_available = market["market_loss_ratio"].shape[0]
    safe_path = min(path_idx, n_paths_available - 1) if n_paths_available > 0 else 0
    for t in range(n_years):
        mlr = _safe_num(market["market_loss_ratio"][safe_path, t])
        row_cells = [
            dmc.TableTd(dmc.Text(str(t + 1), size="xs", c="dimmed")),
            dmc.TableTd(dmc.Text(_fmt(mlr, "pct"), size="xs", ff="monospace"), style={"textAlign": "right"}),
        ]
        # GWP
        for ins in ins_list:
            v = _safe_num(ins["gwp"][safe_path, t])
            row_cells.append(dmc.TableTd(
                dmc.Text(_fmt(v, "money"), size="xs", ff="monospace"), style={"textAlign": "right"}))
        # CR
        for ins in ins_list:
            v = _safe_num(ins["combined_ratio"][safe_path, t])
            row_cells.append(dmc.TableTd(
                dmc.Text(_fmt(v, "pct"), size="xs", ff="monospace",
                         c="red" if v > 1.0 else "inherit"), style={"textAlign": "right"}))
        # Profit
        for ins in ins_list:
            v = _safe_num(ins["total_profit"][safe_path, t])
            row_cells.append(dmc.TableTd(
                dmc.Text(_fmt(v, "money"), size="xs", ff="monospace",
                         c="red" if v < 0 else "inherit"), style={"textAlign": "right"}))

        body_rows.append(dmc.TableTr(row_cells))

    return dmc.Table(
        [header, dmc.TableTbody(body_rows)],
        striped=True, highlightOnHover=True,
        withTableBorder=True, withColumnBorders=True,
        style={"fontSize": "12px"},
    )


# Regime colors for display
_REGIME_COLORS = {0: "#60a5fa", 1: "#fbbf24", 2: "#34d399", 3: "#f87171"}


def regime_performance_table(
    rows: list, names: list, colors=None,
) -> dmc.Table:
    """Regime-conditional performance breakdown table."""
    n = len(names)
    clrs = _get_colors(colors, n)

    header_cells = [
        dmc.TableTh("Regime", style={"width": "12%"}),
        dmc.TableTh("Obs.", style={"textAlign": "right", "width": "8%"}),
    ]
    for name in names:
        header_cells.append(dmc.TableTh(f"CR ({name})", style={"textAlign": "right"}))
    for name in names:
        header_cells.append(dmc.TableTh(f"RORAC ({name})", style={"textAlign": "right"}))
    header = dmc.TableThead(dmc.TableTr(header_cells))

    body_rows = []
    for r in rows:
        r_color = _REGIME_COLORS.get(r["regime_idx"], "#666")
        strategies = r.get("strategies", [])

        # Find best CR and RORAC
        crs = [_safe_num(s.get("cr", 1)) for s in strategies]
        roracs = [_safe_num(s.get("rorac", 0)) for s in strategies]
        best_cr_idx = int(np.argmin(crs)) if crs else 0
        best_rorac_idx = int(np.argmax(roracs)) if roracs else 0

        row_cells = [
            dmc.TableTd(dmc.Badge(r["regime"], color=r_color, variant="light", size="sm")),
            dmc.TableTd(dmc.Text(f"{r.get('n_obs', 0):,}", size="xs", c="dimmed", ff="monospace"),
                        style={"textAlign": "right"}),
        ]
        for j, cr in enumerate(crs):
            is_best = (j == best_cr_idx)
            cr_bg = (
                "rgba(16, 185, 129, 0.08)" if cr < 0.95
                else "rgba(245, 158, 11, 0.08)" if cr < 1.0
                else "rgba(220, 38, 38, 0.08)"
            )
            row_cells.append(dmc.TableTd(dmc.Text(
                _fmt(cr, "pct"), size="xs", ff="monospace",
                fw=600 if is_best else 400,
                c=clrs[j] if is_best else "dimmed",
            ), style={"textAlign": "right", "backgroundColor": cr_bg}))
        for j, rorac in enumerate(roracs):
            is_best = (j == best_rorac_idx)
            rorac_bg = _rag_bg("mean_through_cycle_rorac", rorac)
            r_style = {"textAlign": "right"}
            if rorac_bg:
                r_style["backgroundColor"] = rorac_bg
            row_cells.append(dmc.TableTd(dmc.Text(
                _fmt(rorac, "pct"), size="xs", ff="monospace",
                fw=600 if is_best else 400,
                c=clrs[j] if is_best else "dimmed",
            ), style=r_style))

        body_rows.append(dmc.TableTr(row_cells))

    return dmc.Table(
        [header, dmc.TableTbody(body_rows)],
        striped=True, highlightOnHover=True,
        withTableBorder=True, withColumnBorders=True,
    )


def executive_summary(
    summaries: list, names: list,
    n_paths: int, n_years: int, elapsed: float,
    regime_rows: list,
    colors=None,
    risk_appetite: dict = None,
) -> dmc.Paper:
    """Auto-generated narrative interpreting the simulation results."""
    n = len(names)
    clrs = _get_colors(colors, n)

    roracs = [_safe_num(s.get("mean_through_cycle_rorac", 0)) for s in summaries]
    ruins = [_safe_num(s.get("prob_ruin", 0)) for s in summaries]
    crs = [_safe_num(s.get("mean_combined_ratio", 1)) for s in summaries]
    cums = [_safe_num(s.get("mean_cumulative_profit", 0)) for s in summaries]

    best_rorac_idx = int(np.argmax(roracs))
    winner = names[best_rorac_idx]
    w_rorac = roracs[best_rorac_idx]

    paragraphs = []

    # Opening
    if n == 2:
        rorac_gap = abs(roracs[0] - roracs[1])
        if rorac_gap < 0.005:
            paragraphs.append(
                f"The two strategies produce broadly similar risk-adjusted returns "
                f"({names[0]}: {roracs[0]:.1%}, {names[1]}: {roracs[1]:.1%} through-cycle RORAC). "
                f"Differentiation comes primarily from their risk profiles."
            )
        else:
            loser_idx = 1 - best_rorac_idx
            paragraphs.append(
                f"{winner} delivers a {w_rorac:.1%} through-cycle RORAC versus "
                f"{roracs[loser_idx]:.1%} for {names[loser_idx]} \u2014 a "
                f"{rorac_gap:.1%}pt advantage over the {n_years}-year projection horizon."
            )
    else:
        rorac_strs = ", ".join(f"{names[i]}: {roracs[i]:.1%}" for i in range(n))
        paragraphs.append(
            f"Among {n} strategies, {winner} achieves the highest through-cycle RORAC "
            f"at {w_rorac:.1%}. Breakdown: {rorac_strs}."
        )

    # Profitability
    best_cum_idx = int(np.argmax(cums))
    paragraphs.append(
        f"{names[best_cum_idx]} accumulates the highest mean cumulative profit at "
        f"{_fmt(cums[best_cum_idx], 'money')}. Combined ratios range from "
        f"{_fmt(min(crs), 'pct')} to {_fmt(max(crs), 'pct')}."
    )

    # Risk
    max_ruin = max(ruins)
    if max_ruin > 0.001:
        riskiest_idx = int(np.argmax(ruins))
        safest_idx = int(np.argmin(ruins))
        paragraphs.append(
            f"{names[riskiest_idx]} has the highest ruin probability at {ruins[riskiest_idx]:.1%}, "
            f"while {names[safest_idx]} is safest at {ruins[safest_idx]:.1%}."
        )
    else:
        paragraphs.append("All strategies have near-zero ruin probabilities under these conditions.")

    # Regime insight
    if regime_rows:
        crisis_rows = [r for r in regime_rows if r["regime"] == "Crisis"]
        if crisis_rows:
            cr_row = crisis_rows[0]
            strategies = cr_row.get("strategies", [])
            if strategies:
                crisis_crs = [_safe_num(s.get("cr", 1)) for s in strategies]
                best_crisis = int(np.argmin(crisis_crs))
                paragraphs.append(
                    f"Under crisis conditions, {names[best_crisis]} achieves best underwriting "
                    f"discipline ({_fmt(crisis_crs[best_crisis], 'pct')} combined ratio)."
                )

    # Risk-return trade-off (F7)
    if n >= 2:
        best_ruin_idx = int(np.argmin(ruins))
        worst_ruin_idx = int(np.argmax(ruins))
        if best_rorac_idx == worst_ruin_idx and ruins[worst_ruin_idx] > 0.001:
            # Highest RORAC also has highest ruin
            safe_idx = best_ruin_idx
            safe_rorac_pct = roracs[safe_idx] / max(w_rorac, 0.001) * 100
            ruin_ratio = ruins[worst_ruin_idx] / max(ruins[safe_idx], 0.0001)
            if safe_rorac_pct >= 85:
                paragraphs.append(
                    f"Note: {winner} delivers the highest RORAC but also carries the highest ruin "
                    f"probability ({ruins[worst_ruin_idx]:.2%}). {names[safe_idx]} achieves "
                    f"{roracs[safe_idx]:.1%} RORAC ({safe_rorac_pct:.0f}% of the leader) with "
                    f"{ruin_ratio:.0f}x lower ruin risk \u2014 a more conservative alternative."
                )
            else:
                paragraphs.append(
                    f"Note: {winner} has both the highest RORAC and the highest ruin probability "
                    f"({ruins[worst_ruin_idx]:.2%}). The risk-return trade-off should be evaluated "
                    f"against the firm's risk appetite."
                )

        # Efficiency check: profit-to-risk ratio
        ptrs = [_safe_num(s.get("profit_to_risk_ratio", 0)) for s in summaries]
        best_ptr_idx = int(np.argmax(ptrs))
        if best_ptr_idx != best_rorac_idx and ptrs[best_ptr_idx] > ptrs[best_rorac_idx] * 1.1:
            paragraphs.append(
                f"{names[best_ptr_idx]} has the highest profit-to-risk ratio ({ptrs[best_ptr_idx]:.2f} "
                f"vs {ptrs[best_rorac_idx]:.2f}), indicating superior risk-adjusted efficiency "
                f"despite not leading on absolute RORAC."
            )

    # Risk appetite assessment (F7)
    if risk_appetite and n >= 2:
        ra = risk_appetite
        ruin_max = ra.get("ruin_max", 0.02)
        solvency_min = ra.get("solvency_min", 1.5)
        cr_max = ra.get("cr_max", 1.05)
        rorac_min = ra.get("rorac_min", 0.05)

        def _compliant(s):
            return (
                _safe_num(s.get("prob_ruin", 1)) <= ruin_max
                and _safe_num(s.get("mean_solvency_ratio", 0)) >= solvency_min
                and _safe_num(s.get("mean_combined_ratio", 2)) <= cr_max
                and _safe_num(s.get("mean_through_cycle_rorac", 0)) >= rorac_min
            )

        compliant = [_compliant(s) for s in summaries]
        n_compliant = sum(compliant)
        winner_compliant = compliant[best_rorac_idx]

        if not winner_compliant and n_compliant > 0:
            alt_idx = next(i for i, c in enumerate(compliant) if c)
            paragraphs.append(
                f"However, {winner} breaches at least one risk appetite threshold. "
                f"{names[alt_idx]} is the strongest compliant strategy."
            )
        elif not winner_compliant and n_compliant == 0:
            paragraphs.append(
                "No strategy fully satisfies all risk appetite thresholds under these assumptions. "
                "Consider adjusting parameters or risk appetite limits."
            )

    # Closing
    paragraphs.append(
        f"Based on {n_paths:,} Monte Carlo paths over {n_years} years "
        f"(simulated in {elapsed:.2f}s), {winner} offers the most attractive "
        f"risk-adjusted profile under these market assumptions."
    )

    return dmc.Paper(
        dmc.Stack([
            dmc.Text("EXECUTIVE SUMMARY", fw=600, size="xs",
                     style={"color": "#6b7280", "letterSpacing": "0.06em"}),
            *[dmc.Text(p, size="sm", c="#374151", lh=1.65) for p in paragraphs],
        ], gap="sm"),
        p="lg", radius="md", withBorder=True,
        style={"borderLeft": f"3px solid {clrs[best_rorac_idx]}", "borderColor": "#e4e7ec"},
    )


def strategy_profile_card(
    name: str,
    growth: float, shrink: float,
    max_growth: float, max_shrink: float,
    expected_lr: float,
    cess_sens: float,
    base_cession: float,
    adv_sel: float,
    cap_ratio: float,
) -> dmc.Paper:
    """Plain-English description of an insurer's strategy from its parameters."""
    if growth >= 0.12:
        growth_label = "aggressive"
        growth_desc = f"targets {growth:.0%} annual growth when profitable"
    elif growth >= 0.06:
        growth_label = "moderate"
        growth_desc = f"targets {growth:.0%} annual growth when profitable"
    else:
        growth_label = "conservative"
        growth_desc = f"targets only {growth:.0%} growth even in favourable conditions"

    if abs(shrink) >= 0.08:
        shrink_desc = f"de-risks quickly ({shrink:.0%} shrinkage) when unprofitable"
    elif abs(shrink) >= 0.04:
        shrink_desc = f"shrinks moderately ({shrink:.0%}) when unprofitable"
    else:
        shrink_desc = f"reluctant to shrink ({shrink:.0%}), maintaining book through downturns"

    if cess_sens > 0.02:
        ri_desc = (
            f"Defensive RI buyer: increases cession in soft markets "
            f"(+{cess_sens:.0%}pt sensitivity) from a {base_cession:.0%} base."
        )
    elif cess_sens < -0.01:
        ri_desc = (
            f"Opportunistic RI buyer: reduces cession when market softens "
            f"({cess_sens:.0%}pt sensitivity), accepting more net risk."
        )
    else:
        ri_desc = f"Stable RI program: maintains ~{base_cession:.0%} cession regardless of cycle."

    if expected_lr <= 0.52:
        lr_desc = "Conservative profitability threshold \u2014 reacts quickly to deterioration."
    elif expected_lr >= 0.58:
        lr_desc = "High pain tolerance \u2014 continues operating in marginal conditions."
    else:
        lr_desc = "Moderate profitability threshold."

    bullets = [
        f"Growth posture: {growth_label} \u2014 {growth_desc}",
        f"Downside discipline: {shrink_desc}",
        ri_desc,
        lr_desc,
        f"Growth guardrails: {max_growth:.0%} max growth, {max_shrink:.0%} max shrink p.a.",
        f"Adverse selection sensitivity: {adv_sel:.0%} LR penalty per unit excess growth.",
        f"Capital requirement: {cap_ratio:.0%} of NWP.",
    ]

    accent = "#2563eb" if "Disciplined" in name or "Conservative" in name else "#dc5c0c"
    return dmc.Paper(
        dmc.Stack([
            dmc.Text(f"STRATEGY PROFILE \u2014 {name}", fw=600, size="xs",
                     style={"color": "#6b7280", "letterSpacing": "0.04em"}),
            dmc.List(
                [dmc.ListItem(dmc.Text(b, size="xs", c="#374151")) for b in bullets],
                size="xs", spacing="xs",
            ),
        ], gap="xs"),
        p="md", radius="md", withBorder=True,
        style={"borderLeft": f"3px solid {accent}", "borderColor": "#e4e7ec"},
    )


def return_period_table(
    rows: list, names: list, colors=None,
) -> dmc.Table:
    """Return period analysis table for N strategies."""
    n = len(names)
    clrs = _get_colors(colors, n)

    header_cells = [dmc.TableTh("Return Period", style={"width": "10%"})]
    for name in names:
        header_cells.append(dmc.TableTh(f"Cum. Profit ({name})", style={"textAlign": "right"}))
    for name in names:
        header_cells.append(dmc.TableTh(f"Peak CR ({name})", style={"textAlign": "right"}))
    header = dmc.TableThead(dmc.TableTr(header_cells))

    body_rows = []
    for r in rows:
        is_scr = r["rp"] == 200
        row_style = {"backgroundColor": "#fff5f5"} if is_scr else {}
        rp_badge_color = (
            "gray" if r["rp"] == 10 else
            "yellow" if r["rp"] == 50 else
            "orange" if r["rp"] == 100 else "red"
        )

        strategies = r.get("strategies", [])
        row_cells = [
            dmc.TableTd(dmc.Group([
                dmc.Badge(r["label"], color=rp_badge_color, variant="light", size="sm"),
                dmc.Text("SCR", size="xs", c="red", fw=700) if is_scr else html.Span(),
            ], gap=4)),
        ]
        for s in strategies:
            cp = _safe_num(s.get("cum_profit", 0))
            row_cells.append(dmc.TableTd(
                dmc.Text(_fmt(cp, "money"), size="xs", ff="monospace",
                         c="red" if cp < 0 else "inherit"),
                style={"textAlign": "right"}))
        for s in strategies:
            wcr = _safe_num(s.get("worst_cr", 1.0), 1.0)
            row_cells.append(dmc.TableTd(
                dmc.Text(_fmt(wcr, "pct"), size="xs", ff="monospace",
                         c="red" if wcr > 1.2 else "inherit"),
                style={"textAlign": "right"}))

        body_rows.append(dmc.TableTr(row_cells, style=row_style))

    return dmc.Table(
        [header, dmc.TableTbody(body_rows)],
        striped=True, highlightOnHover=True,
        withTableBorder=True, withColumnBorders=True,
    )


def stress_scenario_cards(
    rows: list, names: list, colors=None,
) -> dmc.SimpleGrid:
    """Visual stress cards for key return periods."""
    n = len(names)
    clrs = _get_colors(colors, n)
    cards = []

    for r in rows:
        if r["rp"] not in (100, 200):
            continue

        severity_color = "red" if r["rp"] == 200 else "orange"
        label = f"1-in-{r['rp']} Year Scenario"
        sublabel = "Lloyd's SCR Standard" if r["rp"] == 200 else "Regulatory Threshold"

        strategies = r.get("strategies", [])
        strategy_cards = []
        for j, s in enumerate(strategies):
            if j >= n:
                break
            cp = _safe_num(s.get("cum_profit", 0))
            wcr = _safe_num(s.get("worst_cr", 1.0), 1.0)
            mc = _safe_num(s.get("min_capital", 0))
            strategy_cards.append(dmc.Stack([
                dmc.Text(names[j], size="xs", c=clrs[j], fw=600),
                dmc.Text(_fmt(cp, "money"), size="sm", fw=700,
                         ff="'JetBrains Mono', monospace",
                         c="red" if cp < 0 else "inherit"),
                dmc.Text(f"Peak CR: {_fmt(wcr, 'pct')}", size="xs", c="dimmed"),
                dmc.Text(f"Min Capital: {_fmt(mc, 'money')}", size="xs",
                         c="red" if mc < 0 else "dimmed"),
            ], gap=2))

        # Find best survivor
        min_caps = [_safe_num(s.get("min_capital", 0)) for s in strategies[:n]]
        best_idx = int(np.argmax(min_caps)) if min_caps else 0

        cards.append(dmc.Paper([
            dmc.Group([
                dmc.Badge(label, color=severity_color, variant="filled", size="sm"),
                dmc.Text(sublabel, size="xs", c="dimmed"),
            ], justify="space-between", mb="xs"),
            dmc.SimpleGrid(cols=min(n, 3), spacing="xs", children=strategy_cards),
            dmc.Text(
                f"{names[best_idx]} maintains strongest capital under this scenario.",
                size="xs", c="dimmed", mt="xs", fs="italic",
            ),
        ], p="md", radius="md", withBorder=True,
           style={"borderLeft": f"3px solid {'#dc2626' if r['rp'] == 200 else '#dc5c0c'}",
                  "borderColor": "#e4e7ec"}))

    return dmc.SimpleGrid(cols=2, spacing="md", children=cards)


def convergence_indicator(
    rorac_dists: list, cum_profit_dists: list, n_paths: int,
) -> dmc.Group:
    """Monte Carlo convergence diagnostic for N strategies."""
    # Use first strategy for convergence assessment
    rorac = np.array(rorac_dists[0], dtype=float)
    rorac = rorac[np.isfinite(rorac)] if len(rorac) > 0 else np.array([0.0])
    n_eff = max(len(rorac), 1)

    se_rorac = float(np.std(rorac) / np.sqrt(n_eff))
    mean_rorac = float(np.mean(rorac))
    cv_rorac = abs(se_rorac / mean_rorac) if abs(mean_rorac) > 1e-8 else 1.0

    if cv_rorac < 0.03:
        color = "green"
        label = "Converged"
        desc = f"SE of RORAC mean: {se_rorac:.2%}pt"
    elif cv_rorac < 0.10:
        color = "yellow"
        label = "Adequate"
        desc = f"SE of RORAC mean: {se_rorac:.2%}pt \u2014 consider more paths"
    else:
        color = "red"
        label = "Unstable"
        desc = f"SE of RORAC mean: {se_rorac:.2%}pt \u2014 increase paths"

    return dmc.Group([
        dmc.Badge(label, color=color, variant="light", size="sm"),
        dmc.Text(desc, size="xs", c="dimmed"),
        dmc.Text(f"{n_paths:,} paths", size="xs", c="dimmed", ff="monospace"),
    ], gap="xs")


# ---------------------------------------------------------------------------
# Historical Backtest Exhibits
# ---------------------------------------------------------------------------
def backtest_summary_cards(bt_list, names, colors=None):
    """Terminal summary cards for the 2001-2024 backtest."""
    n = len(names)
    clrs = _get_colors(colors, n)

    def _card(bt, name, color):
        tc = _safe_num(bt.total_cumulative)
        cagr = _safe_num(bt.cagr)
        mr = _safe_num(bt.mean_rorac)
        mdd = _safe_num(bt.max_drawdown_pct)
        return dmc.Paper(
            dmc.Stack([
                dmc.Text(name, fw=600, size="sm", c=color),
                dmc.Group([
                    dmc.Stack([
                        dmc.Text("Cumulative Profit", size="xs", c="dimmed"),
                        dmc.Text(_fmt(tc, "money"), fw=700, ff="monospace", size="lg"),
                    ], gap=2),
                    dmc.Stack([
                        dmc.Text("CAGR", size="xs", c="dimmed"),
                        dmc.Text(_fmt(cagr, "pct"), fw=600, ff="monospace", size="lg"),
                    ], gap=2),
                    dmc.Stack([
                        dmc.Text("Mean RORAC", size="xs", c="dimmed"),
                        dmc.Text(_fmt(mr, "pct"), fw=600, ff="monospace", size="lg"),
                    ], gap=2),
                    dmc.Stack([
                        dmc.Text("Max Drawdown", size="xs", c="dimmed"),
                        dmc.Text(_fmt(mdd, "pct"), fw=600, ff="monospace", size="lg",
                                **({"c": "red"} if mdd > 0.3 else {})),
                    ], gap=2),
                    dmc.Stack([
                        dmc.Text("Worst Year", size="xs", c="dimmed"),
                        dmc.Text(f"{bt.worst_year}", fw=600, ff="monospace", size="lg"),
                    ], gap=2),
                ], grow=True),
            ], gap="xs"),
            p="md", radius="md",
            style={"borderLeft": f"3px solid {color}"},
        )

    return dmc.SimpleGrid(
        [_card(bt, name, clrs[i]) for i, (bt, name) in enumerate(zip(bt_list, names))],
        cols=min(n, 3), spacing="md",
    )


def backtest_year_table(bt_list, historical_data, names, colors=None):
    """24-row year-by-year table with conditional coloring."""
    n = len(names)
    clrs = _get_colors(colors, n)

    header_cells = [html.Th("Year"), html.Th("Event"), html.Th("Regime"), html.Th("Market CR")]
    for name in names:
        header_cells.append(html.Th(f"{name} CR"))
    for name in names:
        header_cells.append(html.Th(f"{name} Profit"))
    header = html.Tr(header_cells)

    n_bt = min(*(len(bt.combined_ratio) for bt in bt_list), len(historical_data))
    regime_colors = {"soft": "#60a5fa", "firming": "#fbbf24", "hard": "#34d399", "crisis": "#f87171"}
    rows = []
    for i in range(n_bt):
        h = historical_data[i]
        r_color = regime_colors.get(h.get("regime", ""), "#6b7280")
        mcr = _safe_num(h.get("market_cr", 1.0), 1.0)

        row_cells = [
            html.Td(str(h.get("year", "")), style={"fontFamily": "monospace"}),
            html.Td(h.get("event", ""), style={"fontSize": "11px", "maxWidth": "120px",
                     "overflow": "hidden", "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
            html.Td(dmc.Badge(h.get("regime", "").title(), size="xs",
                             style={"backgroundColor": r_color, "color": "white"})),
            html.Td(_fmt(mcr, "pct"), style={"fontFamily": "monospace",
                    "color": "#dc2626" if mcr > 1.0 else "#059669"}),
        ]
        for bt in bt_list:
            cr = _safe_num(bt.combined_ratio[i], 1.0)
            row_cells.append(html.Td(_fmt(cr, "pct"), style={"fontFamily": "monospace",
                    "color": "#dc2626" if cr > 1.0 else "#059669"}))
        for bt in bt_list:
            pft = _safe_num(bt.total_profit[i])
            row_cells.append(html.Td(_fmt(pft, "money"), style={"fontFamily": "monospace",
                    **({"color": "#dc2626"} if pft < 0 else {})}))
        rows.append(html.Tr(row_cells))

    return dmc.Paper(
        html.Table(
            [html.Thead(header), html.Tbody(rows)],
            style={"width": "100%", "fontSize": "12px", "borderCollapse": "collapse"},
            className="cycle-table",
        ),
        p="sm", radius="md",
    )


def counterfactual_verdict(decomposition, name_a="A", name_b="B"):
    """Narrative verdict on the counterfactual decomposition (2-strategy only)."""
    gap = _safe_num(getattr(decomposition, "total_gap", 0))
    winner = name_a if gap > 0 else name_b
    loser = name_b if gap > 0 else name_a
    abs_gap = _fmt(abs(gap), "money")

    factors = {
        "growth timing": abs(_safe_num(getattr(decomposition, "growth_timing", 0))),
        "RI purchasing": abs(_safe_num(getattr(decomposition, "ri_purchasing", 0))),
        "shrink discipline": abs(_safe_num(getattr(decomposition, "shrink_discipline", 0))),
        "expense efficiency": abs(_safe_num(getattr(decomposition, "expense_efficiency", 0))),
    }
    dominant = max(factors, key=factors.get)
    dominant_val = _fmt(factors[dominant], "money")

    text = (
        f"{winner} outperformed {loser} by {abs_gap} over 2001-2024. "
        f"The primary driver was {dominant} ({dominant_val}). "
    )

    for desc in decomposition.descriptions.values():
        text += desc + ". "

    return dmc.Paper(
        dmc.Text(text, size="sm", style={"lineHeight": 1.6}),
        p="md", radius="md",
        style={"borderLeft": "3px solid #60a5fa", "backgroundColor": "#f8fafc"},
    )


# ---------------------------------------------------------------------------
# Regime Forecast Exhibits
# ---------------------------------------------------------------------------
def regime_outlook_table(forecast, regime_perfs, names, colors=None):
    """
    Forward regime probability + expected performance table.
    regime_perfs: list of dicts, one per strategy, each mapping regime_name -> {"cr", "rorac"}
    """
    n = len(names)
    clrs = _get_colors(colors, n)
    regime_names = ["Soft", "Firming", "Hard", "Crisis"]

    header_cells = [html.Th("Year")]
    for r in regime_names:
        header_cells.append(html.Th(r, style={"fontSize": "11px"}))
    for name in names:
        header_cells.append(html.Th(f"E[RORAC] {name}"))
    header = html.Tr(header_cells)

    rows = []
    for yr in range(min(5, forecast.shape[0])):
        probs = forecast[yr]
        row_cells = [html.Td(f"Year {yr + 1}", style={"fontFamily": "monospace"})]
        for i in range(4):
            row_cells.append(html.Td(
                _fmt(_safe_num(probs[i]), "pct"),
                style={"fontFamily": "monospace", "fontSize": "11px"}))

        for j in range(n):
            perf = regime_perfs[j] if j < len(regime_perfs) else {}
            e_rorac = sum(
                _safe_num(probs[i]) * _safe_num(perf.get(regime_names[i].lower(), {}).get("rorac", 0))
                for i in range(4)
            )
            row_cells.append(html.Td(
                _fmt(e_rorac, "pct"),
                style={"fontFamily": "monospace", "color": clrs[j]}))

        rows.append(html.Tr(row_cells))

    return dmc.Paper(
        html.Table(
            [html.Thead(header), html.Tbody(rows)],
            style={"width": "100%", "fontSize": "12px", "borderCollapse": "collapse"},
            className="cycle-table",
        ),
        p="sm", radius="md",
    )


# ---------------------------------------------------------------------------
# Tail Risk Decomposition Exhibits
# ---------------------------------------------------------------------------
def tail_decomposition_table(decomp_list, names, colors=None):
    """Grid: return periods x risk factors, values as percentages."""
    n = len(names)
    clrs = _get_colors(colors, n)

    if not decomp_list or not decomp_list[0] or not decomp_list[0][0].get("has_components", False):
        return dmc.Text(
            "Tail decomposition not available \u2014 requires parametric loss model",
            size="sm", c="dimmed",
        )

    components = [
        ("attritional_pct", "Attritional"),
        ("large_pct", "Large"),
        ("cat_pct", "Cat"),
        ("reserve_pct", "Reserves"),
        ("ri_pct", "RI Cost"),
        ("expense_pct", "Expenses"),
    ]

    header_cells = [html.Th("Return Period")]
    for j, name in enumerate(names):
        for _, comp_label in components:
            header_cells.append(html.Th(f"{name}\n{comp_label}", style={"fontSize": "10px"}))
        if j < n - 1:
            header_cells.append(html.Th("", style={"width": "8px"}))
    header = html.Tr(header_cells)

    rows = []
    n_rps = len(decomp_list[0])
    for i in range(n_rps):
        is_scr = decomp_list[0][i]["rp"] == 200

        cells = [html.Td(
            dmc.Text(decomp_list[0][i]["label"], fw=700 if is_scr else 400, size="sm"),
            style={**({"backgroundColor": "#fef2f2"} if is_scr else {})},
        )]

        for j in range(n):
            d = decomp_list[j][i]
            for key, _ in components:
                v = _safe_num(d.get(key, 0))
                is_max = v == max(_safe_num(d.get(k, 0)) for k, _ in components)
                cells.append(html.Td(
                    _fmt(v, "pct"),
                    style={"fontFamily": "monospace", "fontWeight": "bold" if is_max else "normal",
                           **({"backgroundColor": "#fef2f2"} if is_scr else {})},
                ))
            if j < n - 1:
                cells.append(html.Td(""))

        rows.append(html.Tr(cells))

    return dmc.Paper(
        html.Table(
            [html.Thead(header), html.Tbody(rows)],
            style={"width": "100%", "fontSize": "12px", "borderCollapse": "collapse"},
            className="cycle-table",
        ),
        p="sm", radius="md",
    )


# ---------------------------------------------------------------------------
# Optimizer Exhibits
# ---------------------------------------------------------------------------
def playbook_table(full_opt, current_strategies=None, names=None,
                    # Backward-compat kwargs
                    current_a=None, current_b=None, name_a="A", name_b="B"):
    """CUO Decision Matrix: optimal strategy params per regime + gap analysis."""
    # Backward compat: convert old A/B args to list
    if current_strategies is None:
        current_strategies = []
        if current_a is not None:
            current_strategies.append(current_a)
        if current_b is not None:
            current_strategies.append(current_b)
    if names is None:
        names = [name_a, name_b]

    regime_names = ["Soft", "Firming", "Hard", "Crisis", "Overall"]
    regime_keys = ["soft", "firming", "hard", "crisis"]

    param_labels = {
        "growth_rate_when_profitable": "Growth %",
        "shrink_rate_when_unprofitable": "Shrink %",
        "base_cession_pct": "Cession %",
        "cession_cycle_sensitivity": "Cess Sens",
        "expected_lr": "Exp LR",
    }

    # Use first strategy name for gap column header
    gap_label = f"Gap vs {names[0]}" if names else "Gap"
    header = html.Tr([
        html.Th("Regime"),
        *[html.Th(v, style={"fontSize": "10px"}) for v in param_labels.values()],
        html.Th("RORAC"), html.Th("Ruin %"), html.Th(gap_label),
    ])

    rows = []
    results_list = [full_opt.by_regime.get(k) for k in regime_keys] + [full_opt.unconditional]

    for i, (regime_label, opt_result) in enumerate(zip(regime_names, results_list)):
        if opt_result is None or not opt_result.best_rorac:
            continue

        best = opt_result.best_rorac
        regime_badge_colors = {"Soft": "#60a5fa", "Firming": "#fbbf24", "Hard": "#34d399",
                               "Crisis": "#f87171", "Overall": "#6b7280"}

        cells = [html.Td(dmc.Badge(
            regime_label, size="xs",
            style={"backgroundColor": regime_badge_colors.get(regime_label, "#6b7280"), "color": "white"},
        ))]

        for key in param_labels:
            v = _safe_num(best.params.get(key, 0))
            cells.append(html.Td(_fmt(v, "pct") if abs(v) < 1 else f"{v:.2f}",
                                style={"fontFamily": "monospace", "fontSize": "11px"}))

        rorac_val = _safe_num(getattr(best, "rorac", 0))
        ruin_val = _safe_num(getattr(best, "ruin_prob", 0))
        cells.append(html.Td(_fmt(rorac_val, "pct"),
                            style={"fontFamily": "monospace", "fontWeight": "bold", "color": "#059669"}))
        cells.append(html.Td(_fmt(ruin_val, "pct"),
                            style={"fontFamily": "monospace"}))

        # Use first strategy's gap to frontier
        first_gaps = full_opt.current_gaps[0] if full_opt.current_gaps else {}
        gap_info = first_gaps.get(
            regime_keys[i] if i < 4 else "unconditional", {}
        )
        gap_val = _safe_num(gap_info.get("rorac_gap", 0))
        cells.append(html.Td(
            f"+{_fmt(gap_val, 'pct')}" if gap_val > 0 else _fmt(gap_val, "pct"),
            style={"fontFamily": "monospace",
                   "color": "#059669" if gap_val > 0.005 else "#6b7280"},
        ))

        rows.append(html.Tr(cells))

    return dmc.Paper(
        dmc.Stack([
            dmc.Text("CUO Decision Matrix \u2014 Optimal Strategy by Market Regime",
                    fw=600, size="sm"),
            html.Table(
                [html.Thead(header), html.Tbody(rows)],
                style={"width": "100%", "fontSize": "12px", "borderCollapse": "collapse"},
                className="cycle-table",
            ),
        ], gap="sm"),
        p="md", radius="md",
    )


def optimizer_gap_cards(full_opt, names, colors=None):
    """Cards showing RORAC gap to frontier for each strategy."""
    n = len(names)
    clrs = _get_colors(colors, n)

    # Support both old (current_a_gap/current_b_gap) and new (current_gaps list) formats
    gaps_list = getattr(full_opt, "current_gaps", None)
    if not gaps_list:
        gaps_list = [
            getattr(full_opt, "current_a_gap", {}),
            getattr(full_opt, "current_b_gap", {}),
        ]
    # Pad to match n strategies (missing strategies get empty gap)
    while len(gaps_list) < n:
        gaps_list.append({})

    def _gap_card(gaps, name, color):
        unc_gap = _safe_num(gaps.get("unconditional", {}).get("rorac_gap", 0))
        regime_gaps = [
            (k, _safe_num(v.get("rorac_gap", 0)))
            for k, v in gaps.items() if k != "unconditional"
        ]
        worst_regime = max(regime_gaps, key=lambda x: x[1], default=("", 0))

        return dmc.Paper(
            dmc.Stack([
                dmc.Text(name, fw=600, size="sm", c=color),
                dmc.Group([
                    dmc.Stack([
                        dmc.Text("RORAC Gap to Frontier", size="xs", c="dimmed"),
                        dmc.Text(
                            f"+{unc_gap:.1%}" if unc_gap > 0 else "At frontier",
                            fw=700, ff="monospace", size="xl",
                            c="#059669" if unc_gap > 0.005 else "#6b7280",
                        ),
                    ], gap=2),
                    dmc.Stack([
                        dmc.Text("Biggest Opportunity", size="xs", c="dimmed"),
                        dmc.Text(
                            f"{worst_regime[0].title()}: +{worst_regime[1]:.1%}"
                            if worst_regime[1] > 0.005 else "Near optimal",
                            fw=600, ff="monospace", size="md",
                        ),
                    ], gap=2),
                ], grow=True),
            ], gap="xs"),
            p="md", radius="md",
            style={"borderLeft": f"3px solid {color}"},
        )

    cards = []
    for j in range(min(n, len(gaps_list))):
        cards.append(_gap_card(gaps_list[j], names[j], clrs[j]))

    return dmc.SimpleGrid(cards, cols=min(n, 3), spacing="md")


def optimizer_stats_badge(full_opt):
    """Compute stats badge for the optimizer run."""
    total_sims = getattr(full_opt, "total_sims", 0) or 0
    n_paths = getattr(full_opt, "n_paths", 0) or 0
    elapsed = max(getattr(full_opt, "elapsed_seconds", 0.01) or 0.01, 0.01)
    total_path_years = total_sims * n_paths * 10
    throughput = total_path_years / elapsed

    return dmc.Group([
        dmc.Badge(f"{elapsed:.0f}s", color="violet", variant="light", size="sm"),
        dmc.Text(
            f"Searched {total_sims:,} strategies | "
            f"{total_path_years/1e6:.1f}M path-years | "
            f"{throughput/1e6:.1f}M path-years/sec",
            size="xs", c="dimmed",
        ),
    ], gap="xs")


# ---------------------------------------------------------------------------
# Data Provenance
# ---------------------------------------------------------------------------
def data_provenance_badge(has_capital_model=False):
    """Badge indicating whether results use synthetic or uploaded data."""
    if has_capital_model:
        return dmc.Badge(
            "YOUR CAPITAL MODEL",
            color="green", variant="light", size="sm",
            leftSection=html.Span("\u2713"),
        )
    return dmc.Badge(
        "ILLUSTRATIVE \u2014 SYNTHETIC PARAMETERS",
        color="orange", variant="light", size="sm",
    )


# ---------------------------------------------------------------------------
# F4: Year-1 Business Plan Cards
# ---------------------------------------------------------------------------
def year1_plan_cards(ins_list: list, names: list, colors=None) -> dmc.Paper:
    """Board-level Year 1 projection summary with key probability metrics."""
    n = len(names)
    clrs = _get_colors(colors, n)
    strategy_blocks = []

    for i in range(n):
        ins = ins_list[i]
        yr1_cr = ins["combined_ratio"][:, 0]
        yr1_profit = ins["total_profit"][:, 0]
        yr1_gwp = ins["gwp"][:, 0]

        mean_cr = float(np.nanmean(yr1_cr))
        mean_profit = float(np.mean(yr1_profit))
        mean_gwp = float(np.mean(yr1_gwp))
        p_profitable = float((yr1_profit > 0).mean())
        p_cr_above_100 = float((yr1_cr > 1.0).mean())
        p_cr_above_110 = float((yr1_cr > 1.1).mean())

        cr_color = "#059669" if mean_cr < 1.0 else "#dc2626"

        strategy_blocks.append(dmc.Paper(
            dmc.Stack([
                dmc.Text(names[i], fw=700, size="sm", c=clrs[i]),
                dmc.SimpleGrid(cols=3, spacing="xs", children=[
                    dmc.Stack([
                        dmc.Text("Expected GWP", size="xs", c="dimmed"),
                        dmc.Text(_fmt(mean_gwp, "money"), fw=600, ff="monospace"),
                    ], gap=2),
                    dmc.Stack([
                        dmc.Text("Expected CR", size="xs", c="dimmed"),
                        dmc.Text(_fmt(mean_cr, "pct"), fw=700, ff="monospace", c=cr_color),
                    ], gap=2),
                    dmc.Stack([
                        dmc.Text("Expected Profit", size="xs", c="dimmed"),
                        dmc.Text(_fmt(mean_profit, "money"), fw=600, ff="monospace",
                                 c="#dc2626" if mean_profit < 0 else "inherit"),
                    ], gap=2),
                ]),
                dmc.Group([
                    dmc.Badge(f"P(profit>0) = {p_profitable:.0%}",
                              color="green" if p_profitable > 0.7 else "orange", size="sm", variant="light"),
                    dmc.Badge(f"P(CR>100%) = {p_cr_above_100:.0%}",
                              color="red" if p_cr_above_100 > 0.4 else "green", size="sm", variant="light"),
                    dmc.Badge(f"P(CR>110%) = {p_cr_above_110:.0%}",
                              color="red" if p_cr_above_110 > 0.1 else "gray", size="sm", variant="light"),
                ], gap="xs"),
            ], gap="xs"),
            p="md", radius="md",
            style={"borderLeft": f"3px solid {clrs[i]}", "borderColor": "#e4e7ec"},
        ))

    return dmc.Paper(
        dmc.Stack([
            dmc.Text("YEAR 1 BUSINESS PLAN", fw=600, size="xs",
                     style={"color": "#6b7280", "letterSpacing": "0.06em"}),
            dmc.Text("Next-year projection based on current parameterisation.",
                     size="xs", c="dimmed"),
            dmc.SimpleGrid(cols=min(n, 3), spacing="md", children=strategy_blocks),
        ], gap="sm"),
        p="lg", radius="md", withBorder=True,
    )


# ---------------------------------------------------------------------------
# F8: Audit Log Table
# ---------------------------------------------------------------------------
def audit_log_table(entries: list) -> dmc.Table:
    """Render the run history audit log as a compact table."""
    if not entries:
        return dmc.Text("No simulation runs recorded yet.", size="sm", c="dimmed")

    header_cells = [
        dmc.TableTh("#"), dmc.TableTh("Time"),
        dmc.TableTh("Paths"), dmc.TableTh("Yrs"),
        dmc.TableTh("Strategies"), dmc.TableTh("RORAC"),
        dmc.TableTh("CR"), dmc.TableTh("Duration"),
    ]
    header = dmc.TableThead(dmc.TableTr(header_cells))

    body_rows = []
    for e in reversed(entries):  # most recent first
        roracs = e.get("roracs", [])
        crs = e.get("combined_ratios", [])
        rorac_str = " | ".join(_fmt(r, "pct") for r in roracs) if roracs else "\u2014"
        cr_str = " | ".join(_fmt(c, "pct") for c in crs) if crs else "\u2014"
        strat_names = e.get("strategy_names", [])
        strat_str = ", ".join(strat_names) if strat_names else f"{e.get('n_strategies', '?')} strategies"

        body_rows.append(dmc.TableTr([
            dmc.TableTd(dmc.Text(str(e.get("gen", "")), size="xs", c="dimmed")),
            dmc.TableTd(dmc.Text(e.get("timestamp", ""), size="xs", ff="monospace")),
            dmc.TableTd(dmc.Text(f"{e.get('n_paths', ''):,}", size="xs", ff="monospace")),
            dmc.TableTd(dmc.Text(str(e.get("n_years", "")), size="xs", ff="monospace")),
            dmc.TableTd(dmc.Text(strat_str, size="xs", style={"maxWidth": "120px", "overflow": "hidden",
                        "textOverflow": "ellipsis", "whiteSpace": "nowrap"})),
            dmc.TableTd(dmc.Text(rorac_str, size="xs", ff="monospace"), style={"textAlign": "right"}),
            dmc.TableTd(dmc.Text(cr_str, size="xs", ff="monospace"), style={"textAlign": "right"}),
            dmc.TableTd(dmc.Text(f"{e.get('elapsed', 0):.1f}s", size="xs", ff="monospace")),
        ]))

    return dmc.Table(
        [header, dmc.TableTbody(body_rows)],
        striped=True, highlightOnHover=True,
        withTableBorder=True, withColumnBorders=True,
        style={"fontSize": "12px"},
    )


# ---------------------------------------------------------------------------
# F1: Scenario Delta Table
# ---------------------------------------------------------------------------
def scenario_delta_table(snapshots: list) -> dmc.Stack:
    """Compare saved scenario snapshots side-by-side with delta indicators."""
    if len(snapshots) < 2:
        return dmc.Text("Save at least 2 scenarios to compare.", size="sm", c="dimmed")

    # Use first strategy's summaries from each snapshot
    metric_rows = [
        ("RORAC", "mean_through_cycle_rorac", "pct"),
        ("Combined Ratio", "mean_combined_ratio", "pct"),
        ("Cumulative Profit", "mean_cumulative_profit", "money"),
        ("P(Ruin)", "prob_ruin", "pct2"),
        ("VaR(99.5%)", "var_995_cumulative", "money"),
        ("Max Drawdown", "mean_max_drawdown", "money"),
        ("Solvency", "mean_solvency_ratio", "ratio"),
    ]

    higher_better = {"mean_through_cycle_rorac", "mean_cumulative_profit",
                     "var_995_cumulative", "mean_solvency_ratio"}

    header_cells = [dmc.TableTh("Metric")]
    for snap in snapshots:
        header_cells.append(dmc.TableTh(snap.get("name", "?"), style={"textAlign": "right"}))
    header = dmc.TableThead(dmc.TableTr(header_cells))

    body_rows = []
    for label, key, fmt in metric_rows:
        # Get first strategy's value from each snapshot
        vals = []
        for snap in snapshots:
            sums = snap.get("summaries", [])
            if sums:
                vals.append(_safe_num(sums[0].get(key, 0)))
            else:
                vals.append(0.0)

        row_cells = [dmc.TableTd(dmc.Text(label, size="sm", fw=500))]
        for i, v in enumerate(vals):
            # Delta from first snapshot
            delta_el = html.Span()
            if i > 0 and vals[0] != 0:
                diff = v - vals[0]
                is_improvement = (diff > 0) if key in higher_better else (diff < 0)
                if abs(diff) > 1e-6:
                    arrow = "\u25b2" if is_improvement else "\u25bc"
                    d_color = "green" if is_improvement else "red"
                    if fmt in ("pct", "pct2"):
                        d_text = f" {arrow}{abs(diff):.1%}pts"
                    elif fmt == "money":
                        d_text = f" {arrow}{_fmt(abs(diff), 'money')}"
                    else:
                        d_text = f" {arrow}{abs(diff):.2f}"
                    delta_el = dmc.Text(d_text, size="xs", c=d_color, span=True)

            row_cells.append(dmc.TableTd(
                dmc.Group([
                    dmc.Text(_fmt(v, fmt), size="sm", ff="monospace"),
                    delta_el,
                ], gap=4),
                style={"textAlign": "right"},
            ))
        body_rows.append(dmc.TableTr(row_cells))

    return dmc.Stack([
        dmc.Text("SCENARIO COMPARISON", fw=600, size="xs",
                 style={"color": "#6b7280", "letterSpacing": "0.06em"}),
        dmc.Table(
            [header, dmc.TableTbody(body_rows)],
            striped=True, highlightOnHover=True,
            withTableBorder=True, withColumnBorders=True,
        ),
    ], gap="xs")


# ---------------------------------------------------------------------------
# F6: Risk Appetite Assessment
# ---------------------------------------------------------------------------
def risk_appetite_assessment(
    summaries: list, names: list,
    ruin_max: float = 0.02, solvency_min: float = 1.5,
    cr_max: float = 1.05, rorac_min: float = 0.05,
    colors=None,
) -> dmc.Paper:
    """
    Assess each strategy against firm risk appetite thresholds.

    Returns a Paper with pass/fail badges per criterion per strategy.
    """
    n = len(names)
    clrs = _get_colors(colors, n)

    criteria = [
        ("Ruin Prob", "prob_ruin", ruin_max, True),       # True = lower better
        ("Solvency", "mean_solvency_ratio", solvency_min, False),  # False = higher better
        ("Combined Ratio", "mean_combined_ratio", cr_max, True),
        ("RORAC", "mean_through_cycle_rorac", rorac_min, False),
    ]

    strategy_blocks = []
    for i in range(n):
        s = summaries[i]
        badges = []
        passes = 0
        for label, key, limit, lower_is_pass in criteria:
            val = _safe_num(s.get(key, 0))
            if lower_is_pass:
                ok = val <= limit
            else:
                ok = val >= limit
            if ok:
                passes += 1
            color = "green" if ok else "red"
            icon = "PASS" if ok else "FAIL"
            badges.append(dmc.Group([
                dmc.Badge(icon, color=color, size="xs", variant="filled"),
                dmc.Text(f"{label}: {_fmt(val, 'pct' if 'ratio' not in key else 'ratio')}", size="xs"),
                dmc.Text(
                    f"(limit: {limit:.0%})" if key != "mean_solvency_ratio" else f"(limit: {limit:.1f}x)",
                    size="xs", c="dimmed",
                ),
            ], gap=4))

        score_color = "green" if passes == len(criteria) else ("orange" if passes >= 2 else "red")
        strategy_blocks.append(
            dmc.Paper(
                dmc.Stack([
                    dmc.Group([
                        dmc.Text(names[i], fw=600, size="sm", c=clrs[i]),
                        dmc.Badge(f"{passes}/{len(criteria)}", color=score_color,
                                  size="sm", variant="light"),
                    ], justify="space-between"),
                    *badges,
                ], gap="xs"),
                p="sm", radius="sm",
                style={"borderLeft": f"3px solid {clrs[i]}", "borderColor": "#e4e7ec"},
            )
        )

    return dmc.Paper(
        dmc.Stack([
            dmc.Group([
                dmc.Text("RISK APPETITE ASSESSMENT", fw=600, size="xs",
                         style={"color": "#6b7280", "letterSpacing": "0.06em"}),
                dmc.Badge("Board Risk Framework", color="blue", size="xs", variant="light"),
            ], justify="space-between"),
            dmc.Text(
                f"Assessed against: P(Ruin) < {ruin_max:.0%}, Solvency > {solvency_min:.1f}x, "
                f"CR < {cr_max:.0%}, RORAC > {rorac_min:.0%}",
                size="xs", c="dimmed",
            ),
            dmc.SimpleGrid(cols=min(n, 3), spacing="sm", children=strategy_blocks),
        ], gap="sm"),
        p="lg", radius="md", withBorder=True,
    )
