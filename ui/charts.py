"""
Plotly chart builders for all CycleSim exhibits.

Every function returns a plotly.graph_objects.Figure.
Professional financial-grade formatting: hover templates show exact values,
SI-prefix axis labels, regime legends, contextual annotations.

Color scheme uses a 6-strategy palette (Tailwind-aligned):
  Strategy 0 = #2563eb (blue)    Strategy 1 = #dc5c0c (orange)
  Strategy 2 = #059669 (green)   Strategy 3 = #7c3aed (purple)
  Strategy 4 = #db2777 (pink)    Strategy 5 = #0891b2 (cyan)
  Market     = #059669 (emerald) Crisis      = #dc2626 (red)
"""

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Color palette — 6 strategies + market/crisis
# ---------------------------------------------------------------------------
STRATEGY_COLORS = ["#2563eb", "#dc5c0c", "#059669", "#7c3aed", "#db2777", "#0891b2"]
STRATEGY_COLORS_LIGHT = [
    "rgba(37, 99, 235, 0.07)",
    "rgba(220, 92, 12, 0.07)",
    "rgba(5, 150, 105, 0.07)",
    "rgba(124, 58, 237, 0.07)",
    "rgba(219, 39, 119, 0.07)",
    "rgba(8, 145, 178, 0.07)",
]
STRATEGY_COLORS_MED = [
    "rgba(37, 99, 235, 0.18)",
    "rgba(220, 92, 12, 0.18)",
    "rgba(5, 150, 105, 0.18)",
    "rgba(124, 58, 237, 0.18)",
    "rgba(219, 39, 119, 0.18)",
    "rgba(8, 145, 178, 0.18)",
]

# Backward-compat aliases
COLOR_A = STRATEGY_COLORS[0]
COLOR_B = STRATEGY_COLORS[1]
COLOR_A_LIGHT = STRATEGY_COLORS_LIGHT[0]
COLOR_A_MED = STRATEGY_COLORS_MED[0]
COLOR_B_LIGHT = STRATEGY_COLORS_LIGHT[1]
COLOR_B_MED = STRATEGY_COLORS_MED[1]

COLOR_MARKET = "#059669"
COLOR_MARKET_LIGHT = "rgba(5, 150, 105, 0.07)"
COLOR_MARKET_MED = "rgba(5, 150, 105, 0.18)"
COLOR_CRISIS = "#dc2626"
COLOR_GRID = "#e4e7ec"
COLOR_MUTED = "#6b7280"

REGIME_COLORS = {
    0: "#60a5fa",   # soft: blue-400
    1: "#fbbf24",   # firming: amber-400
    2: "#34d399",   # hard: emerald-400
    3: "#f87171",   # crisis: red-400
}
REGIME_NAMES = {0: "Soft", 1: "Firming", 2: "Hard", 3: "Crisis"}

# ---------------------------------------------------------------------------
# Shared Plotly layout
# ---------------------------------------------------------------------------
_FONT = "Inter, system-ui, -apple-system, sans-serif"

_AXIS_STYLE = dict(
    gridcolor="#f3f4f6", gridwidth=1,
    linecolor="#e5e7eb", linewidth=1,
    tickfont=dict(size=11, color="#6b7280"),
    title_font=dict(size=12, color="#374151"),
)

# Register a custom Plotly template that bakes in axis styling.
# Templates never conflict with xaxis_title / yaxis_title kwargs.
_base = pio.templates["plotly_white"]
_cs_template = go.layout.Template(_base)
_cs_template.layout.xaxis = _AXIS_STYLE
_cs_template.layout.yaxis = _AXIS_STYLE
pio.templates["cyclesim"] = _cs_template

LAYOUT_DEFAULTS = dict(
    template="cyclesim",
    height=370,
    font=dict(family=_FONT, size=12, color="#374151"),
    margin=dict(l=56, r=24, t=44, b=44),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(size=11, color="#6b7280"),
        bgcolor="rgba(0,0,0,0)",
    ),
    hovermode="x unified",
    hoverlabel=dict(
        bgcolor="white", font_size=12,
        font_family=_FONT,
        bordercolor="#e5e7eb",
    ),
    plot_bgcolor="#ffffff",
    paper_bgcolor="rgba(0,0,0,0)",
    title_font=dict(size=14, color="#1f2937"),
)


def _years_axis(n_years: int) -> list:
    return list(range(1, n_years + 1))


def _fmt_gbp(val: float) -> str:
    """Human-readable GBP formatting."""
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "\u2014"
    if not np.isfinite(val):
        return "\u2014"
    if abs(val) >= 1e9:
        return f"\u00a3{val/1e9:.1f}bn"
    if abs(val) >= 1e6:
        return f"\u00a3{val/1e6:.0f}m"
    if abs(val) >= 1e3:
        return f"\u00a3{val/1e3:.0f}k"
    return f"\u00a3{val:.0f}"


def _fmt_pct(val: float) -> str:
    try:
        val = float(val)
    except (TypeError, ValueError):
        return "\u2014"
    if not np.isfinite(val):
        return "\u2014"
    return f"{val:.1%}"


def _apply_yformat(fig, yaxis_format: str, yaxis_title: str):
    """Apply y-axis formatting based on content type."""
    if yaxis_format:
        fig.update_layout(yaxis_tickformat=yaxis_format)
    elif any(k in yaxis_title for k in ("GBP", "GWP", "Profit", "Capital")):
        fig.update_layout(yaxis_tickformat=",.3s")


def _get_colors(colors, n):
    """Return n colors from provided list or default palette."""
    c = colors or STRATEGY_COLORS
    return [c[i % len(c)] for i in range(n)]


def _get_colors_light(colors_light, n):
    c = colors_light or STRATEGY_COLORS_LIGHT
    return [c[i % len(c)] for i in range(n)]


def _get_colors_med(colors_med, n):
    c = colors_med or STRATEGY_COLORS_MED
    return [c[i % len(c)] for i in range(n)]


# ---------------------------------------------------------------------------
# Fan chart: percentile bands (single insurer)
# ---------------------------------------------------------------------------
def fan_chart(
    bands: dict,
    yearly_mean: list,
    n_years: int,
    title: str,
    yaxis_title: str,
    color: str = COLOR_A,
    color_light: str = COLOR_A_LIGHT,
    color_med: str = COLOR_A_MED,
    yaxis_format: str = "",
    reference_line: float = None,
    reference_label: str = "",
) -> go.Figure:
    """Fan chart with 5th/25th/50th/75th/95th percentile bands + mean."""
    x = _years_axis(n_years)
    is_pct = yaxis_format and "%" in yaxis_format
    is_money = any(k in yaxis_title for k in ("GBP", "GWP", "Profit", "Capital"))
    fmt = _fmt_pct if is_pct else (_fmt_gbp if is_money else lambda v: f"{v:.2f}")

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=bands["p95"] + bands["p5"][::-1],
        fill="toself", fillcolor=color_light,
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=True, name="5th-95th pctl",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=bands["p75"] + bands["p25"][::-1],
        fill="toself", fillcolor=color_med,
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=True, name="IQR (25th-75th)",
        hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=bands["p50"],
        line=dict(color=color, width=1.5, dash="dot"),
        name="Median",
        hovertemplate="Median: %{customdata}<extra></extra>",
        customdata=[fmt(v) for v in bands["p50"]],
    ))
    fig.add_trace(go.Scatter(
        x=x, y=yearly_mean,
        line=dict(color=color, width=2.5),
        name="Mean",
        hovertemplate="Mean: %{customdata}<extra></extra>",
        customdata=[fmt(v) for v in yearly_mean],
    ))

    if reference_line is not None:
        fig.add_hline(
            y=reference_line, line_dash="dash", line_color=COLOR_MUTED, line_width=1,
            annotation_text=reference_label if reference_label else None,
            annotation_position="bottom right",
            annotation_font_size=10, annotation_font_color=COLOR_MUTED,
        )

    final_val = yearly_mean[-1] if yearly_mean else None
    if final_val is not None:
        fig.add_annotation(
            x=n_years, y=final_val, text=fmt(final_val),
            showarrow=False, xanchor="left", xshift=6,
            font=dict(size=10, color=color, weight="bold"),
        )

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Year", yaxis_title=yaxis_title,
        **LAYOUT_DEFAULTS,
    )
    _apply_yformat(fig, yaxis_format, yaxis_title)
    return fig


# ---------------------------------------------------------------------------
# Comparison fan chart: N insurers overlaid
# ---------------------------------------------------------------------------
def comparison_fan_chart(
    bands_list: list, mean_list: list,
    n_years: int,
    title: str,
    yaxis_title: str,
    names: list,
    colors=None,
    yaxis_format: str = "",
    reference_line: float = None,
) -> go.Figure:
    x = _years_axis(n_years)
    is_pct = yaxis_format and "%" in yaxis_format
    is_money = any(k in yaxis_title for k in ("GBP", "GWP", "Profit", "Capital"))
    fmt = _fmt_pct if is_pct else (_fmt_gbp if is_money else lambda v: f"{v:.2f}")
    n = len(names)
    clrs = _get_colors(colors, n)
    clrs_light = _get_colors_light(None, n)

    fig = go.Figure()

    for i, (bands, mean, name) in enumerate(zip(bands_list, mean_list, names)):
        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=bands["p75"] + bands["p25"][::-1],
            fill="toself", fillcolor=clrs_light[i],
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=mean,
            line=dict(color=clrs[i], width=2.5), name=name,
            hovertemplate=name + ": %{customdata}<extra></extra>",
            customdata=[fmt(v) for v in mean],
        ))

    if reference_line is not None:
        fig.add_hline(y=reference_line, line_dash="dash", line_color=COLOR_MUTED, line_width=1)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        xaxis_title="Year", yaxis_title=yaxis_title,
        **LAYOUT_DEFAULTS,
    )
    _apply_yformat(fig, yaxis_format, yaxis_title)
    return fig


# ---------------------------------------------------------------------------
# Distribution histogram with VaR markers
# ---------------------------------------------------------------------------
def distribution_chart(
    data_list: list,
    title: str, xaxis_title: str,
    names: list,
    colors=None,
    vars_list: list = None,
    xaxis_format: str = "",
) -> go.Figure:
    is_money = "GBP" in xaxis_title
    fmt = _fmt_gbp if is_money else (lambda v: f"{v:.2f}")
    n = len(names)
    clrs = _get_colors(colors, n)

    fig = go.Figure()
    for i, (data, name) in enumerate(zip(data_list, names)):
        fig.add_trace(go.Histogram(
            x=data, name=name, marker_color=clrs[i],
            opacity=0.65, nbinsx=60,
            hovertemplate=f"{name}<br>Bin: %{{x}}<br>Count: %{{y}}<extra></extra>",
        ))
        # VaR marker
        if vars_list and i < len(vars_list) and vars_list[i] is not None:
            fig.add_vline(
                x=vars_list[i], line_dash="dash", line_color=clrs[i], line_width=1.5,
                annotation_text=f"VaR(95%) {fmt(vars_list[i])}",
                annotation_position="top left" if i == 0 else "top right",
                annotation_font=dict(size=10, color=clrs[i]),
            )
        # Mean marker
        if len(data) > 0:
            m = float(np.nanmean(data))
            if np.isfinite(m):
                fig.add_vline(x=m, line_dash="dot", line_color=clrs[i], line_width=1, opacity=0.6)

    fig.update_layout(
        barmode="overlay",
        title=dict(text=title, font=dict(size=14)),
        xaxis_title=xaxis_title,
        yaxis_title="Frequency",
        **LAYOUT_DEFAULTS,
    )
    if xaxis_format:
        fig.update_layout(xaxis_tickformat=xaxis_format)
    elif is_money:
        fig.update_layout(xaxis_tickformat=",.3s")
    return fig


# ---------------------------------------------------------------------------
# Market cycle chart with regime shading
# ---------------------------------------------------------------------------
def market_cycle_chart(market: dict, n_years: int, historical_data=None) -> go.Figure:
    x = _years_axis(n_years)
    lr = market["market_loss_ratio"]
    regime = market["regime"]

    p5 = np.percentile(lr, 5, axis=0)
    p25 = np.percentile(lr, 25, axis=0)
    p75 = np.percentile(lr, 75, axis=0)
    p95 = np.percentile(lr, 95, axis=0)
    mean = lr.mean(axis=0)

    fig = go.Figure()

    regime_mode = np.apply_along_axis(lambda col: np.bincount(col, minlength=4).argmax(), 0, regime)
    for t in range(n_years):
        fig.add_vrect(
            x0=t + 0.5, x1=t + 1.5,
            fillcolor=REGIME_COLORS[regime_mode[t]],
            opacity=0.06, layer="below", line_width=0,
        )

    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=p95.tolist() + p5.tolist()[::-1],
        fill="toself", fillcolor=COLOR_MARKET_LIGHT,
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=True, name="5th-95th pctl", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x + x[::-1], y=p75.tolist() + p25.tolist()[::-1],
        fill="toself", fillcolor=COLOR_MARKET_MED,
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=True, name="IQR (25th-75th)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=mean.tolist(),
        line=dict(color=COLOR_MARKET, width=2.5),
        name="Mean Market LR",
        hovertemplate="Year %{x}<br>Mean LR: %{y:.1%}<extra></extra>",
    ))

    for r_idx, r_name in REGIME_NAMES.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=8, color=REGIME_COLORS[r_idx], symbol="square"),
            name=f"{r_name} regime", showlegend=True,
        ))

    # Historical overlay (F2): scatter Lloyd's actual data points
    if historical_data and n_years <= 30:
        hist_x = list(range(1, min(len(historical_data), n_years) + 1))
        hist_y = [h["market_lr"] for h in historical_data[:min(len(historical_data), n_years)]]
        hist_text = [
            str(h["year"]) if (i % 5 == 0 or i == 0) else ""
            for i, h in enumerate(historical_data[:min(len(historical_data), n_years)])
        ]
        fig.add_trace(go.Scatter(
            x=hist_x, y=hist_y, mode="markers+text",
            marker=dict(symbol="diamond", size=8, color="#374151", opacity=0.7,
                        line=dict(color="white", width=1)),
            text=hist_text, textposition="top center",
            textfont=dict(size=8, color="#6b7280"),
            name="Lloyd's Actual (2001-2024)",
            hovertemplate="Year %{x}<br>Lloyd's LR: %{y:.1%}<extra>Historical</extra>",
        ))

    fig.update_layout(
        title=dict(text="Market Loss Ratio \u2014 AR(2) + Regime Switching", font=dict(size=14)),
        xaxis_title="Year", yaxis_title="Loss Ratio",
        yaxis_tickformat=".0%",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Radar comparison (N strategies, adaptive normalization)
# ---------------------------------------------------------------------------
def radar_chart(summaries: list, names: list, colors=None) -> go.Figure:
    categories = [
        "RORAC", "Underwriting<br>Discipline", "Profit/Risk<br>Ratio",
        "GWP<br>Growth", "Ruin<br>Safety", "Capital<br>Efficiency",
    ]
    n = len(names)
    clrs = _get_colors(colors, n)
    clrs_light = _get_colors_light(None, n)

    def _safe(v, fallback=0.0):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return fallback
        return v

    # Adaptive normalization across all summaries
    rorac_max = max(
        *(abs(_safe(s.get("mean_through_cycle_rorac", 0))) for s in summaries), 0.15
    ) * 1.5
    pr_max = max(
        *(_safe(s.get("profit_to_risk_ratio", 0)) for s in summaries), 2
    ) * 1.3
    ruin_ceil = max(
        *(_safe(s.get("prob_ruin", 0)) for s in summaries), 0.05
    ) * 2

    def norm(val, lo, hi):
        if hi <= lo:
            return 0.5
        return max(0, min(1, (val - lo) / (hi - lo)))

    def build_vals(s):
        return [
            norm(_safe(s.get("mean_through_cycle_rorac", 0)), -rorac_max * 0.2, rorac_max),
            1 - norm(_safe(s.get("mean_combined_ratio", 1)), 0.85, 1.15),
            norm(_safe(s.get("profit_to_risk_ratio", 0)), -pr_max * 0.1, pr_max),
            norm(_safe(s.get("mean_gwp_cagr", 0)), -0.05, 0.15),
            1 - norm(_safe(s.get("prob_ruin", 0)), 0, ruin_ceil),
            norm(_safe(s.get("mean_solvency_ratio", 1)), 0.5, 3),
        ]

    def hover_vals(s):
        return [
            f"RORAC: {_safe(s.get('mean_through_cycle_rorac', 0)):.1%}",
            f"Combined Ratio: {_safe(s.get('mean_combined_ratio', 1)):.1%}",
            f"Profit/Risk: {_safe(s.get('profit_to_risk_ratio', 0)):.2f}",
            f"GWP CAGR: {_safe(s.get('mean_gwp_cagr', 0)):.1%}",
            f"Ruin Prob: {_safe(s.get('prob_ruin', 0)):.2%}",
            f"Solvency: {_safe(s.get('mean_solvency_ratio', 1)):.2f}x",
        ]

    fig = go.Figure()
    for i, (s, name) in enumerate(zip(summaries, names)):
        vals = build_vals(s)
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=categories + [categories[0]],
            fill="toself", fillcolor=clrs_light[i],
            line=dict(color=clrs[i], width=2), name=name,
            text=hover_vals(s) + [hover_vals(s)[0]],
            hovertemplate="%{text}<extra>" + name + "</extra>",
        ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor=COLOR_GRID),
            angularaxis=dict(gridcolor=COLOR_GRID),
        ),
        title=dict(text="Strategy Profile Comparison", font=dict(size=14)),
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k != "hovermode"},
    )
    return fig


# ---------------------------------------------------------------------------
# RORAC scatter / distribution
# ---------------------------------------------------------------------------
def rorac_scatter(rorac_list: list, names: list, colors=None) -> go.Figure:
    n = len(names)
    clrs = _get_colors(colors, n)

    if n == 2 and len(rorac_list) >= 2:
        # Classic A-vs-B scatter
        a = np.array(rorac_list[0], dtype=float)
        b = np.array(rorac_list[1], dtype=float)
        finite_mask = np.isfinite(a) & np.isfinite(b)
        a_fin = a[finite_mask] if finite_mask.any() else np.array([0.0])
        b_fin = b[finite_mask] if finite_mask.any() else np.array([0.0])
        pt_colors = np.where(a > b, clrs[0], clrs[1])

        fig = go.Figure()
        fig.add_trace(go.Scattergl(
            x=a_fin.tolist(), y=b_fin.tolist(),
            mode="markers",
            marker=dict(size=3, color=pt_colors[finite_mask].tolist() if finite_mask.any() else [], opacity=0.5),
            name="Per-path RORAC",
            hovertemplate=f"{names[0]}: %{{x:.1%}}<br>{names[1]}: %{{y:.1%}}<extra></extra>",
        ))
        bound = max(abs(a_fin.min()), abs(a_fin.max()), abs(b_fin.min()), abs(b_fin.max()), 0.01) * 1.05
        fig.add_trace(go.Scatter(
            x=[-bound, bound], y=[-bound, bound],
            line=dict(color=COLOR_MUTED, dash="dash", width=1),
            name="Equal line", hoverinfo="skip",
        ))
        a_wins = float((a_fin > b_fin).mean()) if len(a_fin) > 0 else 0.5
        fig.add_annotation(
            text=f"{names[0]} wins {a_wins:.0%} of paths",
            xref="paper", yref="paper", x=0.02, y=0.98,
            showarrow=False, font=dict(size=11, color=clrs[0] if a_wins > 0.5 else clrs[1]),
            bgcolor="rgba(255,255,255,0.85)", borderpad=4,
        )
        fig.update_layout(
            title=dict(text="Through-Cycle RORAC: Path-by-Path", font=dict(size=14)),
            xaxis_title=f"{names[0]} RORAC", yaxis_title=f"{names[1]} RORAC",
            xaxis_tickformat=".0%", yaxis_tickformat=".0%",
            xaxis_zeroline=True, yaxis_zeroline=True,
            xaxis_zerolinecolor=COLOR_GRID, yaxis_zerolinecolor=COLOR_GRID,
            **LAYOUT_DEFAULTS,
        )
    else:
        # N>2: box plot comparison
        fig = go.Figure()
        for i, (data, name) in enumerate(zip(rorac_list, names)):
            arr = np.array(data, dtype=float)
            arr = arr[np.isfinite(arr)]
            fig.add_trace(go.Box(
                y=arr, name=name, marker_color=clrs[i],
                boxpoints="outliers", jitter=0.3, pointpos=-1.8,
                hovertemplate=name + "<br>RORAC: %{y:.1%}<extra></extra>",
            ))
        fig.update_layout(
            title=dict(text="Through-Cycle RORAC Distribution", font=dict(size=14)),
            yaxis_title="Through-Cycle RORAC",
            yaxis_tickformat=".0%",
            showlegend=False,
            **LAYOUT_DEFAULTS,
        )

    return fig


# ---------------------------------------------------------------------------
# Ruin probability over time
# ---------------------------------------------------------------------------
def ruin_over_time_chart(
    ruin_list: list, n_years: int,
    names: list, colors=None,
) -> go.Figure:
    x = _years_axis(n_years)
    n = len(names)
    clrs = _get_colors(colors, n)
    clrs_light = _get_colors_light(None, n)

    fig = go.Figure()
    for i, (ruin, name) in enumerate(zip(ruin_list, names)):
        fig.add_trace(go.Scatter(
            x=x, y=ruin, line=dict(color=clrs[i], width=2.5), name=name,
            fill="tozeroy", fillcolor=clrs_light[i],
            hovertemplate="Year %{x}: %{y:.2%}<extra>" + name + "</extra>",
        ))

    # Terminal annotation
    parts = " | ".join(f"{names[i]} {ruin_list[i][-1]:.1%}" for i in range(n) if ruin_list[i])
    fig.add_annotation(
        x=n_years, y=max((r[-1] for r in ruin_list if r), default=0),
        text=f"Terminal: {parts}",
        showarrow=True, arrowhead=0, arrowcolor=COLOR_MUTED,
        yshift=20, font=dict(size=10, color=COLOR_MUTED),
    )

    fig.update_layout(
        title=dict(text="Cumulative Probability of Ruin", font=dict(size=14)),
        xaxis_title="Year", yaxis_title="P(Ruin by Year t)",
        yaxis_tickformat=".1%", yaxis_rangemode="tozero",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Attribution waterfall chart
# ---------------------------------------------------------------------------
def attribution_chart(attr_list: list, names: list, colors=None) -> go.Figure:
    categories = ["Underwriting", "Investment", "RI Cost", "Reserve Dev", "Capital Actions", "Total"]
    keys = ["underwriting", "investment", "reinsurance_cost", "reserve_development", "capital_actions", "total"]
    n = len(names)
    clrs = _get_colors(colors, n)

    fig = go.Figure()
    for i, (attr, name) in enumerate(zip(attr_list, names)):
        vals = [attr.get(k, 0) / 1e6 for k in keys]
        fig.add_trace(go.Bar(
            x=categories, y=vals, name=name,
            marker_color=[clrs[i]] * len(vals),
            marker_pattern_shape=[""] * (len(vals) - 1) + ["/"],
            text=[f"\u00a3{v:.0f}m" for v in vals],
            textposition="outside", textfont=dict(size=10),
            hovertemplate="%{x}: \u00a3%{y:.1f}m<extra>" + name + "</extra>",
        ))

    fig.add_hline(y=0, line_color=COLOR_MUTED, line_width=1)
    fig.update_layout(
        title=dict(text="Profit Attribution (Mean per Simulation)", font=dict(size=14)),
        yaxis_title="\u00a3 millions",
        barmode="group",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Single path drill-down (5-row subplot with signal)
# ---------------------------------------------------------------------------
def single_path_chart(
    market: dict, ins_list: list,
    path_idx: int, n_years: int,
    names: list, colors=None,
) -> go.Figure:
    x = _years_axis(n_years)
    n = len(names)
    clrs = _get_colors(colors, n)
    has_signal = "strategy_signal" in ins_list[0]
    n_rows = 5 if has_signal else 4

    subplot_titles = [
        "Market Loss Ratio & Regime",
        "Gross Written Premium",
        "Combined Ratio",
        "Cumulative Profit",
    ]
    row_heights = [0.18, 0.22, 0.22, 0.22]
    if has_signal:
        subplot_titles.append("Strategy Signal")
        row_heights.append(0.16)

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True, vertical_spacing=0.05,
        subplot_titles=subplot_titles,
        row_heights=row_heights,
    )

    # Row 1: Market LR with regime background
    regime = market["regime"][path_idx]
    for t in range(n_years):
        fig.add_vrect(
            x0=t + 0.5, x1=t + 1.5,
            fillcolor=REGIME_COLORS.get(regime[t], COLOR_GRID),
            opacity=0.15, layer="below", line_width=0,
            row=1, col=1,
        )
    mlr = market["market_loss_ratio"][path_idx].tolist()
    fig.add_trace(go.Scatter(
        x=x, y=mlr,
        line=dict(color=COLOR_MARKET, width=2), name="Market LR",
        hovertemplate="Year %{x}: %{y:.1%}<extra>Market LR</extra>",
    ), row=1, col=1)

    for r_idx, r_name in REGIME_NAMES.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=7, color=REGIME_COLORS[r_idx], symbol="square"),
            name=r_name, showlegend=True,
        ), row=1, col=1)

    # Rows 2-4: Insurer metrics
    row_configs = [
        ("gwp", 2, ",.3s", "GWP"),
        ("combined_ratio", 3, ".1%", "CR"),
        ("cumulative_profit", 4, ",.3s", "Cum. Profit"),
    ]
    for key, row, tfmt, hover_label in row_configs:
        for i, (ins, name) in enumerate(zip(ins_list, names)):
            y_data = ins[key][path_idx].tolist()
            if "%" in tfmt:
                ht = f"Year %{{x}}: %{{y:.1%}}<extra>{name}</extra>"
            else:
                ht = f"Year %{{x}}: %{{customdata}}<extra>{name}</extra>"
            fig.add_trace(go.Scatter(
                x=x, y=y_data,
                line=dict(color=clrs[i], width=2), name=name,
                showlegend=(row == 2),
                customdata=[_fmt_gbp(v) for v in y_data] if "%" not in tfmt else None,
                hovertemplate=ht,
            ), row=row, col=1)

    # Reference lines
    fig.add_hline(y=1.0, line_dash="dash", line_color=COLOR_MUTED, line_width=1,
                  row=3, col=1, annotation_text="Breakeven",
                  annotation_font=dict(size=9, color=COLOR_MUTED))
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_MUTED, line_width=1, row=4, col=1)

    # Row 5: Strategy Signal (if available)
    if has_signal:
        for i, (ins, name) in enumerate(zip(ins_list, names)):
            sig = ins["strategy_signal"][path_idx].tolist()
            fig.add_trace(go.Scatter(
                x=x, y=sig,
                line=dict(color=clrs[i], width=2.5), name=f"{name} Signal",
                showlegend=False,
                hovertemplate=f"Year %{{x}}: %{{y:.2f}}<extra>{name} Signal</extra>",
            ), row=5, col=1)
        fig.add_hline(y=0, line_dash="dash", line_color=COLOR_MUTED, line_width=1, row=5, col=1)
        # Green/red shading for grow/shrink zones
        fig.add_hrect(y0=0, y1=2, fillcolor="rgba(52,211,153,0.06)", line_width=0, row=5, col=1)
        fig.add_hrect(y0=-2, y1=0, fillcolor="rgba(248,113,113,0.06)", line_width=0, row=5, col=1)
        fig.update_yaxes(title_text="Signal", row=5, col=1)

    fig.update_layout(
        title=dict(text=f"Path #{path_idx + 1} \u2014 Single Scenario Drill-Down", font=dict(size=14)),
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k not in ("hovermode", "height")},
        hovermode="x",
        height=850 if not has_signal else 1000,
    )
    fig.update_yaxes(tickformat=".0%", row=1, col=1)
    fig.update_yaxes(tickformat=",.3s", row=2, col=1)
    fig.update_yaxes(tickformat=".0%", row=3, col=1)
    fig.update_yaxes(tickformat=",.3s", row=4, col=1)
    fig.update_xaxes(title_text="Year", row=n_rows, col=1)
    return fig


# ---------------------------------------------------------------------------
# Worst paths table data
# ---------------------------------------------------------------------------
def worst_paths_data(results, n_worst: int = 10) -> list:
    """Return list of dicts for the N worst paths (by cumulative profit for first strategy)."""
    cum_first = results.insurers[0]["cumulative_profit"][:, -1]
    worst_idx = np.argsort(cum_first)[:n_worst]

    rows = []
    for rank, idx in enumerate(worst_idx, 1):
        row = {"rank": rank, "path": idx + 1, "strategies": []}
        for j, ins in enumerate(results.insurers):
            max_cr = float(np.nanmax(ins["combined_ratio"][idx]))
            entry = {
                "cum_profit": _fmt_gbp(ins["cumulative_profit"][idx, -1]),
                "terminal_gwp": _fmt_gbp(ins["gwp"][idx, -1]),
                "max_cr": _fmt_pct(max_cr) if np.isfinite(max_cr) else "\u2014",
                "ruined": "Yes" if ins["is_ruined"][idx, -1] else "No",
            }
            row["strategies"].append(entry)
            # Backward compat
            pfx = chr(ord('a') + j)
            row[f"cum_profit_{pfx}"] = entry["cum_profit"]
            row[f"terminal_gwp_{pfx}"] = entry["terminal_gwp"]
            row[f"max_cr_{pfx}"] = entry["max_cr"]
            row[f"ruined_{pfx}"] = entry["ruined"]
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Sensitivity tornado (legacy, kept for compat)
# ---------------------------------------------------------------------------
def sensitivity_tornado(base_results, sensitivity_results: list, param_labels: list,
                        metric_key: str, metric_label: str, name: str) -> go.Figure:
    base_val = base_results.summary_a[metric_key]
    deltas_low = []
    deltas_high = []
    for low_res, high_res in sensitivity_results:
        deltas_low.append(low_res.summary_a[metric_key] - base_val)
        deltas_high.append(high_res.summary_a[metric_key] - base_val)

    swings = [abs(h - l) for l, h in zip(deltas_low, deltas_high)]
    order = np.argsort(swings)[::-1]
    labels = [param_labels[i] for i in order]
    lows = [deltas_low[i] for i in order]
    highs = [deltas_high[i] for i in order]

    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=lows, orientation="h", name="Low", marker_color="#dc2626"))
    fig.add_trace(go.Bar(y=labels, x=highs, orientation="h", name="High", marker_color="#10b981"))
    fig.add_vline(x=0, line_color=COLOR_MUTED, line_width=1)

    fig.update_layout(
        title=dict(text=f"Sensitivity Analysis: {metric_label} ({name})", font=dict(size=14)),
        xaxis_title=f"\u0394 {metric_label}",
        barmode="relative",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Sensitivity tornado from rows (N-strategy)
# ---------------------------------------------------------------------------
def sensitivity_tornado_from_rows(
    rows: list, names: list, colors=None,
) -> go.Figure:
    n = len(names)
    labels = [r["label"] for r in rows]

    fig = make_subplots(rows=1, cols=n, shared_yaxes=True,
                        subplot_titles=list(names),
                        horizontal_spacing=0.08)

    for col_idx in range(n):
        base_vals = [r["base_rorac"][col_idx] for r in rows] if rows else []
        base = base_vals[0] if base_vals else 0
        lows = [r["low_rorac"][col_idx] - base for r in rows]
        highs = [r["high_rorac"][col_idx] - base for r in rows]

        fig.add_trace(go.Bar(
            y=labels, x=lows, orientation="h",
            name="Low", marker_color="#dc2626",
            hovertemplate="%{y}: %{x:+.1%}<extra>Low perturbation</extra>",
            showlegend=(col_idx == 0),
        ), row=1, col=col_idx + 1)
        fig.add_trace(go.Bar(
            y=labels, x=highs, orientation="h",
            name="High", marker_color="#10b981",
            hovertemplate="%{y}: %{x:+.1%}<extra>High perturbation</extra>",
            showlegend=(col_idx == 0),
        ), row=1, col=col_idx + 1)
        fig.add_vline(x=0, line_color=COLOR_MUTED, line_width=1, row=1, col=col_idx + 1)

    fig.update_layout(
        title=dict(text="Sensitivity Analysis: RORAC Impact (\u00b1perturbation)", font=dict(size=14)),
        barmode="relative",
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k != "height"},
        height=max(350, 40 * len(labels) + 100),
    )
    fig.update_xaxes(tickformat="+.1%")
    return fig


# ---------------------------------------------------------------------------
# Efficiency frontier: risk vs return per path
# ---------------------------------------------------------------------------
def efficiency_frontier(summaries: list, names: list, colors=None) -> go.Figure:
    n = len(names)
    clrs = _get_colors(colors, n)

    fig = go.Figure()

    means_dd = []
    means_rorac = []

    for i, (s, name) in enumerate(zip(summaries, names)):
        rorac = np.array(s["through_cycle_rorac_dist"])
        dd = np.array(s["max_drawdown_dist"])

        fig.add_trace(go.Scattergl(
            x=dd, y=rorac, mode="markers",
            marker=dict(size=3, color=clrs[i], opacity=0.3),
            name=name,
            hovertemplate=f"{name}<br>Drawdown: %{{x:,.0f}}<br>RORAC: %{{y:.1%}}<extra></extra>",
        ))

        mean_dd = float(np.nanmean(dd)) if len(dd) > 0 else 0.0
        mean_rorac = float(np.nanmean(rorac)) if len(rorac) > 0 else 0.0
        means_dd.append(mean_dd)
        means_rorac.append(mean_rorac)

        fig.add_trace(go.Scatter(
            x=[mean_dd], y=[mean_rorac],
            mode="markers+text",
            marker=dict(size=14, color=clrs[i], symbol="diamond",
                        line=dict(color="white", width=2)),
            text=[name], textposition="top center",
            textfont=dict(size=11, color=clrs[i]),
            name=f"{name} (mean)", showlegend=False,
            hovertemplate=f"{name} Mean<br>Drawdown: {_fmt_gbp(mean_dd)}<br>RORAC: {mean_rorac:.1%}<extra></extra>",
        ))

    # Connecting line between all means
    if n >= 2:
        fig.add_trace(go.Scatter(
            x=means_dd, y=means_rorac,
            mode="lines", line=dict(color=COLOR_MUTED, dash="dot", width=1.5),
            showlegend=False, hoverinfo="skip",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_MUTED, line_width=1)

    fig.update_layout(
        title=dict(text="Risk-Return Frontier (per path)", font=dict(size=14)),
        xaxis_title="Max Drawdown (GBP)",
        yaxis_title="Through-Cycle RORAC",
        yaxis_tickformat=".0%", xaxis_tickformat=",.3s",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Capital drawdown chart
# ---------------------------------------------------------------------------
def drawdown_chart(
    capital_list: list, n_years: int,
    names: list, colors=None,
) -> go.Figure:
    x = _years_axis(n_years)
    n = len(names)
    clrs = _get_colors(colors, n)
    clrs_light = _get_colors_light(None, n)

    def _mean_drawdown(capital):
        running_max = np.maximum.accumulate(capital, axis=1)
        dd = running_max - capital
        return dd.mean(axis=0)

    fig = go.Figure()
    for i, (capital, name) in enumerate(zip(capital_list, names)):
        dd = _mean_drawdown(capital)
        fig.add_trace(go.Scatter(
            x=x, y=dd.tolist(),
            line=dict(color=clrs[i], width=2.5), name=name,
            fill="tozeroy", fillcolor=clrs_light[i],
            customdata=[_fmt_gbp(v) for v in dd],
            hovertemplate="Year %{x}: %{customdata}<extra>" + name + "</extra>",
        ))

    fig.update_layout(
        title=dict(text="Mean Capital Drawdown Over Time", font=dict(size=14)),
        xaxis_title="Year", yaxis_title="Drawdown (GBP)",
        yaxis_tickformat=",.3s", yaxis_rangemode="tozero",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Regime-conditional performance table data
# ---------------------------------------------------------------------------
def regime_performance_data(
    market: dict, ins_list: list, names: list,
) -> list:
    regime = market["regime"]
    rows = []

    import warnings

    def _safe_regime_mean(arr, mask):
        vals = arr[mask]
        if vals.size == 0:
            return 0.0
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            result = float(np.nanmean(vals))
        return result if np.isfinite(result) else 0.0

    for r_idx, r_name in REGIME_NAMES.items():
        mask = regime == r_idx
        n_obs = mask.sum()
        if n_obs == 0:
            continue

        row = {"regime": r_name, "regime_idx": r_idx, "n_obs": int(n_obs), "strategies": []}
        for j, ins in enumerate(ins_list):
            entry = {
                "cr": _safe_regime_mean(ins["combined_ratio"], mask),
                "rorac": _safe_regime_mean(ins["rorac"], mask),
                "gwp_chg": _safe_regime_mean(ins["gwp_change_pct"], mask),
                "cess": _safe_regime_mean(ins["cession_pct"], mask),
            }
            row["strategies"].append(entry)
            # Backward compat
            pfx = chr(ord('a') + j)
            row[f"cr_{pfx}"] = entry["cr"]
            row[f"rorac_{pfx}"] = entry["rorac"]
            row[f"gwp_chg_{pfx}"] = entry["gwp_chg"]
            row[f"cess_{pfx}"] = entry["cess"]
        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Win probability over time
# ---------------------------------------------------------------------------
def win_probability_chart(
    cum_list: list, n_years: int,
    names: list, colors=None,
) -> go.Figure:
    x = list(range(1, n_years + 1))
    n = len(names)
    clrs = _get_colors(colors, n)

    fig = go.Figure()

    if n <= 1:
        # Single strategy: show P(profit > 0) over time
        cum = cum_list[0]
        p_positive = [float((cum[:, t] > 0).mean()) for t in range(n_years)]
        fig.add_trace(go.Scatter(
            x=x, y=p_positive, mode="lines+markers",
            line=dict(color=clrs[0], width=2.5), marker=dict(size=4),
            name=f"P({names[0]} profitable)",
            hovertemplate="Year %{x}: %{y:.1%}<extra>P(profitable)</extra>",
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color=COLOR_MUTED, line_width=1,
                      annotation_text="50%", annotation_position="bottom right")
        fig.update_layout(
            title=dict(text=f"P({names[0]} Cumulative Profit > 0)", font=dict(size=14)),
        )
    elif n == 2:
        # Classic P(A>B) line
        cum_a, cum_b = cum_list[0], cum_list[1]
        p_a_wins = [(cum_a[:, t] > cum_b[:, t]).mean() for t in range(n_years)]
        fig.add_trace(go.Scatter(
            x=x, y=p_a_wins, mode="lines+markers",
            line=dict(color=clrs[0], width=2.5), marker=dict(size=4),
            name=f"P({names[0]} leads)",
            hovertemplate="Year %{x}: %{y:.1%}<extra>P(" + names[0] + " leads)</extra>",
        ))
        fig.add_hline(y=0.5, line_dash="dash", line_color=COLOR_MUTED, line_width=1,
                      annotation_text="50/50", annotation_position="bottom right")
        terminal = p_a_wins[-1]
        fig.add_annotation(
            x=n_years, y=terminal,
            text=f"Terminal: {terminal:.0%}",
            showarrow=True, arrowhead=2, ax=40, ay=-30,
            font=dict(size=11, color=clrs[0] if terminal > 0.5 else clrs[1]),
        )
        fig.update_layout(
            title=dict(text=f"Win Probability: P({names[0]} Leads Over Time)", font=dict(size=14)),
        )
    else:
        # N>2: P(each is best) stacked area
        for t in range(n_years):
            vals = np.array([cum_list[i][:, t] for i in range(n)])
            best = np.argmax(vals, axis=0)
            for i in range(n):
                p = float((best == i).mean())
                if t == 0:
                    fig.add_trace(go.Scatter(
                        x=[t + 1], y=[p], mode="lines",
                        line=dict(color=clrs[i], width=2), name=names[i],
                        stackgroup="one",
                        hovertemplate="Year %{x}: %{y:.1%}<extra>" + names[i] + "</extra>",
                    ))
                else:
                    fig.data[i].x = tuple(list(fig.data[i].x) + [t + 1])
                    fig.data[i].y = tuple(list(fig.data[i].y) + [p])
        fig.update_layout(
            title=dict(text="P(Best Strategy) Over Time", font=dict(size=14)),
        )

    fig.update_layout(
        xaxis_title="Year", yaxis_title="Probability",
        yaxis_tickformat=".0%", yaxis_range=[0, 1],
        **LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Cycle timing scatter
# ---------------------------------------------------------------------------
def cycle_timing_chart(
    market: dict, ins_list: list,
    names: list, colors=None,
) -> go.Figure:
    n = len(names)
    clrs = _get_colors(colors, n)
    ra = market["rate_adequacy"].ravel()

    # Downsample for performance
    n_total = len(ra)
    max_pts = 10000
    rng = np.random.default_rng(0)

    fig = go.Figure()
    for i, (ins, name) in enumerate(zip(ins_list, names)):
        gwp_chg = ins["gwp_change_pct"].ravel()
        if n_total > max_pts:
            idx = rng.choice(n_total, max_pts, replace=False)
            ra_s, gwp_s = ra[idx], gwp_chg[idx]
        else:
            ra_s, gwp_s = ra, gwp_chg

        fig.add_trace(go.Scattergl(
            x=ra_s, y=gwp_s, mode="markers",
            marker=dict(color=clrs[i], size=3, opacity=0.3),
            name=name,
            hovertemplate="Rate Adequacy: %{x:.2f}<br>GWP Change: %{y:.1%}<extra>" + name + "</extra>",
        ))

    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_MUTED, line_width=1)
    fig.add_vline(x=0, line_dash="dash", line_color=COLOR_MUTED, line_width=1)

    for text, x_pos, y_pos in [
        ("Growing in Hard\n(Disciplined)", 0.5, 0.08),
        ("Growing in Soft\n(Chasing volume)", -0.5, 0.08),
        ("Shrinking in Hard\n(Retreating)", 0.5, -0.08),
        ("Shrinking in Soft\n(De-risking)", -0.5, -0.08),
    ]:
        fig.add_annotation(
            x=x_pos, y=y_pos, text=text, showarrow=False,
            font=dict(size=9, color="#94a3b8"), opacity=0.7,
        )

    fig.update_layout(
        title=dict(text="Cycle Timing: GWP Growth vs Rate Adequacy", font=dict(size=14)),
        xaxis_title="Rate Adequacy (+ = hard, \u2212 = soft)",
        yaxis_title="Annual GWP Change",
        yaxis_tickformat=".0%",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Solvency ratio comparison
# ---------------------------------------------------------------------------
def solvency_comparison_chart(
    solv_list: list, n_years: int,
    names: list, colors=None,
) -> go.Figure:
    x = list(range(1, n_years + 1))
    n = len(names)
    clrs = _get_colors(colors, n)
    clrs_light = _get_colors_light(None, n)

    fig = go.Figure()
    for i, (data, name) in enumerate(zip(solv_list, names)):
        mean = np.nanmean(data, axis=0)
        p25 = np.nanpercentile(data, 25, axis=0)
        p75 = np.nanpercentile(data, 75, axis=0)

        fig.add_trace(go.Scatter(
            x=x + x[::-1],
            y=p75.tolist() + p25.tolist()[::-1],
            fill="toself", fillcolor=clrs_light[i],
            line=dict(color="rgba(0,0,0,0)"),
            showlegend=False, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=x, y=mean.tolist(),
            line=dict(color=clrs[i], width=2.5), name=name,
            hovertemplate="Year %{x}: %{y:.0%}<extra>" + name + "</extra>",
        ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="#dc2626", line_width=1,
                  annotation_text="Min Solvency", annotation_position="bottom right")

    fig.update_layout(
        title=dict(text="Solvency Ratio Over Time", font=dict(size=14)),
        xaxis_title="Year", yaxis_title="Solvency Ratio",
        yaxis_tickformat=".0%",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Cumulative profit buildup
# ---------------------------------------------------------------------------
def cumulative_profit_buildup(
    ins_list: list, n_years: int,
    names: list, colors=None,
) -> go.Figure:
    n = len(names)
    clrs = _get_colors(colors, n)

    fig = make_subplots(
        rows=1, cols=n, shared_yaxes=True,
        subplot_titles=list(names),
        horizontal_spacing=0.06,
    )

    for col, (ins, name) in enumerate(zip(ins_list, names), 1):
        x = list(range(1, n_years + 1))
        uw_cum = np.cumsum(ins["uw_profit"].mean(axis=0))
        inv_cum = np.cumsum(ins["investment_income"].mean(axis=0))
        ri_cum = np.cumsum(-ins["ri_cost"].mean(axis=0))

        fig.add_trace(go.Scatter(
            x=x, y=uw_cum / 1e6, name="UW Profit" if col == 1 else None,
            fill="tozeroy", fillcolor="rgba(52, 211, 153, 0.3)",
            line=dict(color="#34d399", width=1),
            showlegend=(col == 1),
            hovertemplate="Year %{x}: \u00a3%{y:.0f}m<extra>Cumulative UW</extra>",
        ), row=1, col=col)

        fig.add_trace(go.Scatter(
            x=x, y=(uw_cum + inv_cum) / 1e6, name="+ Investment" if col == 1 else None,
            fill="tonexty", fillcolor="rgba(96, 165, 250, 0.3)",
            line=dict(color="#60a5fa", width=1),
            showlegend=(col == 1),
            hovertemplate="Year %{x}: \u00a3%{y:.0f}m<extra>+ Investment</extra>",
        ), row=1, col=col)

        fig.add_trace(go.Scatter(
            x=x, y=(uw_cum + inv_cum + ri_cum) / 1e6, name="\u2212 RI Cost" if col == 1 else None,
            fill="tonexty", fillcolor="rgba(248, 113, 113, 0.15)",
            line=dict(color="#f87171", width=1),
            showlegend=(col == 1),
            hovertemplate="Year %{x}: \u00a3%{y:.0f}m<extra>After RI Cost</extra>",
        ), row=1, col=col)

        total_cum = np.cumsum(ins["total_profit"].mean(axis=0))
        fig.add_trace(go.Scatter(
            x=x, y=total_cum / 1e6, name="Net Total" if col == 1 else None,
            line=dict(color=clrs[col - 1], width=2.5, dash="dot"),
            showlegend=(col == 1),
            hovertemplate="Year %{x}: \u00a3%{y:.0f}m<extra>Net Total</extra>",
        ), row=1, col=col)

    fig.add_hline(y=0, line_color=COLOR_MUTED, line_width=1)
    fig.update_layout(
        title=dict(text="Cumulative Profit Buildup: Where Does the Advantage Come From?", font=dict(size=14)),
        yaxis_title="Cumulative \u00a3 millions",
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k != "height"},
        height=380,
    )
    for c in range(1, n + 1):
        fig.update_xaxes(title_text="Year", row=1, col=c)
    return fig


# ---------------------------------------------------------------------------
# Yearly profit waterfall (single insurer)
# ---------------------------------------------------------------------------
def yearly_profit_waterfall(
    results: dict, n_years: int, name: str, color: str,
) -> go.Figure:
    x = list(range(1, n_years + 1))
    uw = results["uw_profit"].mean(axis=0)
    inv = results["investment_income"].mean(axis=0)
    ri = -results["ri_cost"].mean(axis=0)
    total = results["total_profit"].mean(axis=0)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x, y=uw / 1e6, name="UW Profit", marker_color="#34d399",
        hovertemplate="Year %{x}: \u00a3%{y:.1f}m<extra>UW Profit</extra>",
    ))
    fig.add_trace(go.Bar(
        x=x, y=inv / 1e6, name="Investment", marker_color="#60a5fa",
        hovertemplate="Year %{x}: \u00a3%{y:.1f}m<extra>Investment</extra>",
    ))
    fig.add_trace(go.Bar(
        x=x, y=ri / 1e6, name="RI Cost", marker_color="#f87171",
        hovertemplate="Year %{x}: \u00a3%{y:.1f}m<extra>RI Cost</extra>",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=total / 1e6, mode="lines+markers",
        line=dict(color=color, width=2.5), marker=dict(size=5),
        name="Net Profit",
        hovertemplate="Year %{x}: \u00a3%{y:.1f}m<extra>Net Profit</extra>",
    ))
    fig.add_hline(y=0, line_color=COLOR_MUTED, line_width=1)

    fig.update_layout(
        title=dict(text=f"{name}: Annual Profit Decomposition (Mean)", font=dict(size=14)),
        xaxis_title="Year", yaxis_title="\u00a3 millions",
        barmode="relative",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Capital model preview (single)
# ---------------------------------------------------------------------------
def capital_model_preview(gross_lrs: np.ndarray, net_lrs: np.ndarray = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=gross_lrs, name="Gross LR", nbinsx=80,
        marker_color="#7c3aed", opacity=0.7,
        hovertemplate="LR: %{x:.1%}<br>Count: %{y}<extra>Gross</extra>",
    ))
    if net_lrs is not None:
        fig.add_trace(go.Histogram(
            x=net_lrs, name="Net LR", nbinsx=80,
            marker_color="#06b6d4", opacity=0.5,
            hovertemplate="LR: %{x:.1%}<br>Count: %{y}<extra>Net</extra>",
        ))

    mean_g = float(np.mean(gross_lrs))
    p99 = float(np.percentile(gross_lrs, 99))
    p995 = float(np.percentile(gross_lrs, 99.5))

    fig.add_vline(x=mean_g, line_dash="dash", line_color="#7c3aed", line_width=1.5,
                  annotation_text=f"Mean: {mean_g:.1%}",
                  annotation_position="top right",
                  annotation_font=dict(size=10, color="#7c3aed"))
    fig.add_vline(x=p99, line_dash="dot", line_color="#dc2626", line_width=1,
                  annotation_text=f"1-in-100: {p99:.1%}",
                  annotation_position="top left",
                  annotation_font=dict(size=9, color="#dc2626"))
    fig.add_vline(x=p995, line_dash="dot", line_color="#dc2626", line_width=1,
                  annotation_text=f"1-in-200: {p995:.1%}",
                  annotation_position="top left",
                  annotation_font=dict(size=9, color="#dc2626"))

    fig.update_layout(
        barmode="overlay",
        title=dict(text="Capital Model: Loss Ratio Distribution", font=dict(size=14)),
        xaxis_title="Loss Ratio", yaxis_title="Frequency",
        xaxis_tickformat=".0%",
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k != "height"},
        height=280,
    )
    return fig


# ---------------------------------------------------------------------------
# CR regime heatmap (single insurer)
# ---------------------------------------------------------------------------
def cr_regime_heatmap(
    market: dict, ins_data: dict, n_years: int, name: str, color: str,
) -> go.Figure:
    regime = market["regime"]
    cr = ins_data["combined_ratio"]
    years = list(range(1, n_years + 1))
    regime_labels = [REGIME_NAMES[i].title() for i in sorted(REGIME_NAMES.keys())]

    z = np.full((4, n_years), np.nan)
    for r_idx in range(4):
        for t in range(n_years):
            mask = regime[:, t] == r_idx
            if mask.sum() > 5:
                z[r_idx, t] = np.nanmean(cr[mask, t])

    fig = go.Figure(data=go.Heatmap(
        z=z, x=years, y=regime_labels,
        colorscale=[[0.0, "#10b981"], [0.5, "#fbbf24"], [1.0, "#dc2626"]],
        zmid=1.0,
        colorbar=dict(title="CR", tickformat=".0%"),
        hovertemplate="Year %{x}, %{y}: CR = %{z:.1%}<extra>" + name + "</extra>",
    ))

    fig.update_layout(
        title=dict(text=f"{name}: Combined Ratio by Year \u00d7 Regime", font=dict(size=14)),
        xaxis_title="Year", yaxis_title="Market Regime",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Historical Backtest Charts
# ---------------------------------------------------------------------------
def historical_cr_chart(bt_list, historical_data, names, colors=None):
    n = len(names)
    clrs = _get_colors(colors, n)
    years = bt_list[0].years

    bar_colors = ["#34d399" if h["market_cr"] < 1.0 else "#f87171" for h in historical_data]
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=years, y=[h["market_cr"] for h in historical_data],
        marker_color=bar_colors, opacity=0.45, name="Lloyd's Market CR",
        hovertemplate="Year %{x}<br>Market CR: %{y:.1%}<extra>Lloyd's</extra>",
    ))

    for i, h in enumerate(historical_data):
        r_idx = {"soft": 0, "firming": 1, "hard": 2, "crisis": 3}[h["regime"]]
        fig.add_vrect(
            x0=h["year"] - 0.5, x1=h["year"] + 0.5,
            fillcolor=REGIME_COLORS[r_idx], opacity=0.08, line_width=0, layer="below",
        )

    for i, (bt, name) in enumerate(zip(bt_list, names)):
        fig.add_trace(go.Scatter(
            x=years, y=bt.combined_ratio, mode="lines+markers",
            line=dict(color=clrs[i], width=2.5), marker=dict(size=5),
            name=name, hovertemplate="Year %{x}<br>CR: %{y:.1%}<extra>" + name + "</extra>",
        ))

    fig.add_hline(y=1.0, line_dash="dash", line_color="#9ca3af", line_width=1,
                  annotation_text="Breakeven", annotation_position="top right",
                  annotation_font_size=10, annotation_font_color="#9ca3af")

    year_to_idx = {int(y): i for i, y in enumerate(years)}
    major_events = [h for h in historical_data if h["market_cr"] >= 1.05 or h["rate_change"] >= 0.10]
    for h in major_events:
        idx = year_to_idx.get(h["year"])
        if idx is None or idx >= len(bt_list[0].combined_ratio):
            continue
        y_pos = max(h["market_cr"], bt_list[0].combined_ratio[idx]) + 0.04
        fig.add_annotation(
            x=h["year"], y=y_pos,
            text=h["event"].split(" + ")[0].split("/")[0],
            showarrow=False, font=dict(size=8, color="#6b7280"),
            textangle=-45,
        )

    fig.update_layout(
        title="Historical Lloyd's Market \u2014 Combined Ratio 2001-2024",
        xaxis_title="Year", yaxis_title="Combined Ratio",
        yaxis_tickformat=".0%", barmode="overlay",
        **LAYOUT_DEFAULTS,
    )
    return fig


def historical_cumulative_chart(bt_list, names, colors=None):
    n = len(names)
    clrs = _get_colors(colors, n)
    clrs_light = _get_colors_light(None, n)
    years = bt_list[0].years

    fig = go.Figure()
    for i, (bt, name) in enumerate(zip(bt_list, names)):
        fig.add_trace(go.Scatter(
            x=years, y=bt.cumulative_profit, mode="lines",
            line=dict(color=clrs[i], width=2.5), name=name,
            fill="tozeroy", fillcolor=clrs_light[i],
            hovertemplate="Year %{x}<br>Cumulative: %{y:,.0f}<extra>" + name + "</extra>",
        ))
        fig.add_annotation(
            x=years[-1], y=bt.cumulative_profit[-1],
            text=_fmt_gbp(bt.cumulative_profit[-1]),
            showarrow=True, arrowhead=2, ax=40, ay=-20 + i * 40,
            font=dict(color=clrs[i], size=12, family="JetBrains Mono, monospace"),
        )

    fig.update_layout(
        title="Cumulative Profit \u2014 Historical Replay 2001-2024",
        xaxis_title="Year", yaxis_title="Cumulative Profit (\u00a3)",
        yaxis_tickformat=",.3s",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Counterfactual waterfall (2-strategy only)
# ---------------------------------------------------------------------------
def counterfactual_waterfall(decomposition):
    factors = [
        ("Growth\nTiming", decomposition.growth_timing),
        ("RI\nPurchasing", decomposition.ri_purchasing),
        ("Shrink\nDiscipline", decomposition.shrink_discipline),
        ("Expense\nEfficiency", decomposition.expense_efficiency),
    ]

    labels = [f[0] for f in factors] + ["Total\nGap"]
    values = [f[1] for f in factors] + [decomposition.total_gap]
    measures = ["relative"] * len(factors) + ["total"]

    fig = go.Figure(go.Waterfall(
        x=labels, y=values, measure=measures,
        increasing=dict(marker_color="#34d399"),
        decreasing=dict(marker_color="#f87171"),
        totals=dict(marker_color="#60a5fa"),
        texttemplate="%{y:,.0f}",
        textposition="outside",
        textfont=dict(size=10, family="JetBrains Mono, monospace"),
        connector_line_color="#e5e7eb",
        hovertemplate="%{x}: %{y:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        title="Performance Gap Attribution \u2014 What Drove the Difference?",
        yaxis_title="Cumulative Profit Impact (\u00a3)",
        yaxis_tickformat=",.3s",
        showlegend=False,
        **LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Forward Regime Forecast Charts
# ---------------------------------------------------------------------------
def regime_forecast_chart(forecast, current_regime):
    n_years = forecast.shape[0]
    year0 = np.zeros((1, 4))
    year0[0, current_regime] = 1.0
    full_data = np.vstack([year0, forecast])
    x_labels = ["Now"] + [f"Year {i+1}" for i in range(n_years)]

    def _hex_to_rgba(hex_color, alpha):
        r_val = int(hex_color[1:3], 16)
        g_val = int(hex_color[3:5], 16)
        b_val = int(hex_color[5:7], 16)
        return f"rgba({r_val},{g_val},{b_val},{alpha})"

    fig = go.Figure()
    for r_idx in range(4):
        fig.add_trace(go.Scatter(
            x=x_labels, y=full_data[:, r_idx],
            mode="lines", stackgroup="one",
            groupnorm="percent",
            name=REGIME_NAMES[r_idx],
            line=dict(width=1, color=REGIME_COLORS[r_idx]),
            fillcolor=_hex_to_rgba(REGIME_COLORS[r_idx], 0.8),
            hovertemplate="%{x}: " + REGIME_NAMES[r_idx] + " = %{y:.1f}%<extra></extra>",
        ))

    stationary = full_data[-1]
    dominant = int(np.argmax(stationary))
    fig.add_annotation(
        x=x_labels[-1], y=50, yref="y",
        text=f"Converges to {stationary[dominant]:.0%} {REGIME_NAMES[dominant]}",
        showarrow=False, font=dict(size=10, color="#6b7280"),
        xanchor="right",
    )

    fig.update_layout(
        title=f"Regime Probability Forecast (from {REGIME_NAMES[current_regime]})",
        yaxis_title="Probability", yaxis_ticksuffix="%",
        yaxis_range=[0, 100],
        **LAYOUT_DEFAULTS,
    )
    return fig


def market_clock_chart(transition_matrix, current_regime):
    probs = transition_matrix[current_regime]
    theta_labels = [REGIME_NAMES[i] for i in range(4)]

    fig = go.Figure()
    fig.add_trace(go.Barpolar(
        r=probs, theta=theta_labels,
        width=[80] * 4,
        marker_color=[REGIME_COLORS[i] for i in range(4)],
        marker_line_color=["#1f2937" if i == current_regime else "#e5e7eb" for i in range(4)],
        marker_line_width=[3 if i == current_regime else 1 for i in range(4)],
        opacity=0.85,
        hovertemplate="%{theta}: %{r:.0%}<extra></extra>",
    ))

    import math
    max_prob = max(probs) if max(probs) > 0 else 0.25
    for i, (lbl, prob) in enumerate(zip(theta_labels, probs)):
        angle_deg = i * 90
        angle_rad = math.radians(90 - angle_deg)
        fig.add_annotation(
            x=0.5 + 0.35 * math.cos(angle_rad),
            y=0.5 + 0.35 * math.sin(angle_rad),
            xref="paper", yref="paper",
            text=f"<b>{prob:.0%}</b>",
            showarrow=False,
            font=dict(size=12, color=REGIME_COLORS[i]),
        )

    fig.update_layout(
        title="Market Clock \u2014 Next Year Transitions",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max_prob * 1.3], tickformat=".0%",
                           gridcolor="#e5e7eb", tickfont=dict(size=9, color="#6b7280")),
            angularaxis=dict(tickfont=dict(size=13, color="#374151"), direction="clockwise"),
            bgcolor="rgba(0,0,0,0)",
        ),
        showlegend=False, height=370,
        font=dict(family=_FONT, size=12, color="#374151"),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=60, t=60, b=60),
    )
    fig.add_annotation(
        text=f"\u2b24 {REGIME_NAMES[current_regime]}",
        x=0.5, y=0.5, xref="paper", yref="paper",
        showarrow=False,
        font=dict(size=14, color=REGIME_COLORS[current_regime], family=_FONT),
    )
    return fig


# ---------------------------------------------------------------------------
# Tail Risk Decomposition Chart
# ---------------------------------------------------------------------------
def tail_decomposition_chart(decomp_list, names, colors=None):
    n = len(names)
    clrs = _get_colors(colors, n)

    if not decomp_list or not decomp_list[0] or not decomp_list[0][0].get("has_components", False):
        fig = go.Figure()
        fig.add_annotation(
            text="Component detail not available with imported capital model",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="#6b7280"),
        )
        fig.update_layout(title="Tail Risk Factor Attribution", **LAYOUT_DEFAULTS)
        return fig

    components = [
        ("attritional_pct", "Attritional", "#34d399"),
        ("large_pct", "Large Loss", "#60a5fa"),
        ("cat_pct", "Catastrophe", "#f87171"),
        ("reserve_pct", "Reserves", "#fbbf24"),
        ("ri_pct", "RI Cost", "#a78bfa"),
        ("expense_pct", "Expenses", "#94a3b8"),
    ]

    fig = make_subplots(rows=1, cols=n, subplot_titles=list(names),
                        shared_yaxes=True, horizontal_spacing=0.08)

    for col, (decomp, name) in enumerate(zip(decomp_list, names), 1):
        labels = [d["label"] for d in decomp]
        for key, comp_name, color in components:
            values = [d.get(key, 0) for d in decomp]
            fig.add_trace(go.Bar(
                x=labels, y=values, name=comp_name,
                marker_color=color, showlegend=(col == 1),
                hovertemplate="%{x}: " + comp_name + " = %{y:.0%}<extra>" + name + "</extra>",
            ), row=1, col=col)

    fig.update_layout(
        title="Tail Risk Factor Attribution by Return Period",
        barmode="stack",
        yaxis_tickformat=".0%", yaxis_title="Share of Total Loss",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ---------------------------------------------------------------------------
# Optimizer Charts
# ---------------------------------------------------------------------------
def pareto_frontier_chart(full_opt, summaries, names, colors=None):
    n = len(names)
    clrs = _get_colors(colors, n)
    regimes = ["soft", "firming", "hard", "crisis"]
    titles = [REGIME_NAMES[i] for i in range(4)] + ["Unconditional"]
    results_list = [full_opt.by_regime.get(r) for r in regimes] + [full_opt.unconditional]

    fig = make_subplots(rows=2, cols=3, subplot_titles=titles,
                        horizontal_spacing=0.08, vertical_spacing=0.12)

    strategy_dds = [s.get("mean_max_drawdown", 0) for s in summaries]
    strategy_roracs = [s.get("mean_through_cycle_rorac", 0) for s in summaries]

    for idx, opt_result in enumerate(results_list):
        row = idx // 3 + 1
        col = idx % 3 + 1
        if opt_result is None:
            continue

        dominated = [c for c in opt_result.all_candidates if not c.is_pareto]
        if dominated:
            fig.add_trace(go.Scattergl(
                x=[c.max_drawdown / 1e6 for c in dominated],
                y=[c.rorac * 100 for c in dominated],
                mode="markers", marker=dict(size=4, color="#d1d5db", opacity=0.5),
                showlegend=False, hoverinfo="skip",
            ), row=row, col=col)

        front = sorted(opt_result.pareto_front, key=lambda c: c.max_drawdown)
        if front:
            fig.add_trace(go.Scatter(
                x=[c.max_drawdown / 1e6 for c in front],
                y=[c.rorac * 100 for c in front],
                mode="lines+markers",
                line=dict(color="#059669", width=2),
                marker=dict(size=6, color="#059669"),
                showlegend=(idx == 0), name="Pareto Front",
                hovertemplate="RORAC: %{y:.1f}%<br>Drawdown: \u00a3%{x:.0f}m<extra>Optimal</extra>",
            ), row=row, col=col)

        for i in range(n):
            fig.add_trace(go.Scatter(
                x=[strategy_dds[i] / 1e6], y=[strategy_roracs[i] * 100],
                mode="markers", marker=dict(size=12, color=clrs[i], symbol="diamond"),
                showlegend=(idx == 0), name=names[i],
            ), row=row, col=col)

    fig.update_layout(
        title="Strategy Optimization \u2014 Pareto Frontier by Regime",
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k != "height"},
        height=550,
    )
    for i in range(1, 7):
        fig.update_xaxes(title_text="Max Drawdown (\u00a3m)", row=(i-1)//3+1, col=(i-1)%3+1)
        fig.update_yaxes(title_text="RORAC (%)", row=(i-1)//3+1, col=(i-1)%3+1)
    return fig


# ---------------------------------------------------------------------------
# Strategy DNA radar (current vs optimal, single insurer)
# ---------------------------------------------------------------------------
def strategy_dna_radar(current_params, optimal_params, param_bounds=None):
    if param_bounds is None:
        from cyclesim.optimizer import SEARCH_DIMENSIONS
        param_bounds = SEARCH_DIMENSIONS

    param_names = list(param_bounds.keys())
    short_names = [
        "Growth", "Shrink", "Max Growth", "Cession",
        "Cess Sens", "Expected LR", "Max Shrink", "Adv Sel",
    ]

    def normalize(params_dict):
        vals = []
        for name in param_names:
            lo, hi = param_bounds[name]
            v = params_dict.get(name, (lo + hi) / 2)
            vals.append((v - lo) / (hi - lo) if hi > lo else 0.5)
        return vals

    curr_norm = normalize(current_params)
    opt_norm = normalize(optimal_params)

    theta = short_names + [short_names[0]]
    curr_r = curr_norm + [curr_norm[0]]
    opt_r = opt_norm + [opt_norm[0]]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=curr_r, theta=theta, fill="toself",
        fillcolor="rgba(37, 99, 235, 0.1)", line=dict(color=COLOR_A, dash="dash", width=2),
        name="Current",
    ))
    fig.add_trace(go.Scatterpolar(
        r=opt_r, theta=theta, fill="toself",
        fillcolor="rgba(52, 211, 153, 0.15)", line=dict(color="#059669", width=2),
        name="Optimal",
    ))

    fig.update_layout(
        title="Strategy DNA \u2014 Current vs Optimal",
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 1], tickformat=".0%",
                           gridcolor="#f3f4f6", tickfont=dict(size=9, color="#6b7280")),
            angularaxis=dict(tickfont=dict(size=10, color="#374151")),
            bgcolor="rgba(0,0,0,0)",
        ),
        font=dict(family=_FONT, size=12, color="#374151"),
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=60, r=60, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# F3: Capital Allocation by Risk Component
# ---------------------------------------------------------------------------
def capital_allocation_chart(alloc_list: list, names: list, colors=None) -> go.Figure:
    """Stacked horizontal bar showing Euler-style capital allocation per strategy."""
    n = len(names)
    clrs = _get_colors(colors, n)
    components = [
        ("attritional_pct", "Attritional", "#34d399"),
        ("large_pct", "Large Loss", "#60a5fa"),
        ("cat_pct", "Catastrophe", "#f87171"),
        ("reserve_pct", "Reserves", "#fbbf24"),
        ("ri_pct", "RI Cost", "#a78bfa"),
        ("expense_pct", "Expenses", "#94a3b8"),
    ]

    if not alloc_list or not alloc_list[0]:
        fig = go.Figure()
        fig.add_annotation(
            text="Capital allocation not available (requires parametric loss model)",
            x=0.5, y=0.5, xref="paper", yref="paper",
            showarrow=False, font=dict(size=14, color="#6b7280"),
        )
        fig.update_layout(title="Economic Capital Allocation", **LAYOUT_DEFAULTS)
        return fig

    fig = go.Figure()
    for key, comp_name, color in components:
        vals = [a.get(key, 0) for a in alloc_list]
        fig.add_trace(go.Bar(
            y=names, x=vals, orientation="h", name=comp_name,
            marker_color=color,
            hovertemplate=comp_name + ": %{x:.0%}<extra>%{y}</extra>",
        ))

    fig.update_layout(
        title=dict(text="Economic Capital Allocation by Risk Component", font=dict(size=14)),
        xaxis_title="Share of VaR-Based Economic Capital",
        xaxis_tickformat=".0%",
        barmode="stack",
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k != "height"},
        height=max(250, 80 * n + 100),
    )
    return fig


# ---------------------------------------------------------------------------
# F6: QQ Plot — distribution validation
# ---------------------------------------------------------------------------
def qq_plot_chart(
    gross_lr_list: list, names: list, colors=None,
) -> go.Figure:
    """QQ plot comparing simulated gross LR against LogNormal theoretical fit."""
    n = len(names)
    clrs = _get_colors(colors, n)

    fig = make_subplots(rows=1, cols=n, shared_yaxes=True,
                        subplot_titles=list(names),
                        horizontal_spacing=0.08)

    for col, (data, name) in enumerate(zip(gross_lr_list, names), 1):
        arr = np.array(data, dtype=float)
        arr = arr[np.isfinite(arr) & (arr > 0)]
        if len(arr) < 50:
            continue

        # Fit LogNormal
        log_data = np.log(arr)
        mu_fit = float(np.mean(log_data))
        sigma_fit = float(np.std(log_data))

        # Sorted empirical quantiles
        sorted_data = np.sort(arr)
        n_pts = len(sorted_data)
        probs = (np.arange(1, n_pts + 1) - 0.5) / n_pts

        # Theoretical quantiles
        from scipy.stats import lognorm
        theoretical = lognorm.ppf(probs, s=sigma_fit, scale=np.exp(mu_fit))

        # Downsample for plotting (max 500 points)
        if n_pts > 500:
            idx = np.linspace(0, n_pts - 1, 500, dtype=int)
            sorted_data = sorted_data[idx]
            theoretical = theoretical[idx]

        fig.add_trace(go.Scattergl(
            x=theoretical, y=sorted_data, mode="markers",
            marker=dict(size=3, color=clrs[col - 1], opacity=0.5),
            name=name, showlegend=(col == 1),
            hovertemplate="Theory: %{x:.2f}<br>Actual: %{y:.2f}<extra></extra>",
        ), row=1, col=col)

        # 45-degree reference line
        lo = min(theoretical.min(), sorted_data.min())
        hi = max(theoretical.max(), sorted_data.max())
        fig.add_trace(go.Scatter(
            x=[lo, hi], y=[lo, hi], mode="lines",
            line=dict(color=COLOR_MUTED, dash="dash", width=1),
            showlegend=False, hoverinfo="skip",
        ), row=1, col=col)

        # K-S test
        try:
            from scipy.stats import kstest
            ks_stat, ks_p = kstest(arr, "lognorm", args=(sigma_fit, 0, np.exp(mu_fit)))
            p_color = "#dc2626" if ks_p < 0.05 else "#059669"
            fig.add_annotation(
                text=f"K-S p={ks_p:.3f}",
                xref=f"x{col}" if col > 1 else "x",
                yref=f"y{col}" if col > 1 else "y",
                x=lo + (hi - lo) * 0.05, y=hi - (hi - lo) * 0.05,
                showarrow=False, font=dict(size=10, color=p_color),
                bgcolor="rgba(255,255,255,0.85)", borderpad=3,
            )
        except ImportError:
            pass

    fig.update_layout(
        title=dict(text="Distribution Fit: QQ Plot vs LogNormal", font=dict(size=14)),
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k != "height"},
        height=380,
    )
    for c in range(1, n + 1):
        fig.update_xaxes(title_text="Theoretical", row=1, col=c)
    fig.update_yaxes(title_text="Empirical", row=1, col=1)
    return fig


# ---------------------------------------------------------------------------
# F7: Realized Correlation Matrix Heatmap
# ---------------------------------------------------------------------------
def correlation_heatmap(
    market: dict, ins_data: dict, name: str,
) -> go.Figure:
    """Heatmap of cross-metric correlations across simulated paths."""
    n_paths = ins_data["cumulative_profit"].shape[0]

    # Extract per-path terminal/aggregate values
    metrics = {
        "Cum Profit": ins_data["cumulative_profit"][:, -1],
        "Terminal GWP": ins_data["gwp"][:, -1],
        "Mean CR": np.nanmean(ins_data["combined_ratio"], axis=1),
        "Mean RORAC": np.nanmean(ins_data["rorac"], axis=1),
        "Total RI Cost": ins_data["ri_cost"].sum(axis=1),
        "Mean Cession": np.mean(ins_data["cession_pct"], axis=1),
        "Mean Mkt LR": np.mean(market["market_loss_ratio"], axis=1),
        "Mean Rate Adeq": np.mean(market["rate_adequacy"], axis=1),
    }

    labels = list(metrics.keys())
    n_m = len(labels)
    data_matrix = np.column_stack([metrics[k] for k in labels])

    # Filter out non-finite rows
    valid = np.all(np.isfinite(data_matrix), axis=1)
    if valid.sum() < 10:
        fig = go.Figure()
        fig.add_annotation(text="Insufficient data for correlation", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False)
        fig.update_layout(title=f"{name}: Correlation Matrix", **LAYOUT_DEFAULTS)
        return fig

    corr = np.corrcoef(data_matrix[valid].T)

    # Text annotations
    text_vals = [[f"{corr[i, j]:.2f}" for j in range(n_m)] for i in range(n_m)]

    fig = go.Figure(data=go.Heatmap(
        z=corr, x=labels, y=labels,
        colorscale=[[0, "#2563eb"], [0.5, "#ffffff"], [1, "#dc2626"]],
        zmid=0, zmin=-1, zmax=1,
        text=text_vals, texttemplate="%{text}",
        textfont=dict(size=10),
        colorbar=dict(title="Corr", tickformat=".1f"),
        hovertemplate="Corr(%{x}, %{y}) = %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(text=f"{name}: Cross-Metric Correlation", font=dict(size=14)),
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k != "height"},
        height=420,
        xaxis=dict(tickangle=-45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    return fig


# ---------------------------------------------------------------------------
# F5: Reinsurance Efficient Frontier
# ---------------------------------------------------------------------------
def ri_frontier_chart(sweep_results: list, current_cession: float = None) -> go.Figure:
    """3-panel chart: RORAC, VaR, and Ruin vs cession percentage."""
    if not sweep_results:
        fig = go.Figure()
        fig.add_annotation(text="No sweep data", x=0.5, y=0.5,
                           xref="paper", yref="paper", showarrow=False)
        fig.update_layout(title="RI Efficient Frontier", **LAYOUT_DEFAULTS)
        return fig

    fig = make_subplots(
        rows=1, cols=3, shared_xaxes=True,
        subplot_titles=["RORAC vs Cession", "VaR(99.5%) vs Cession", "Ruin % vs Cession"],
        horizontal_spacing=0.08,
    )

    x = [r["cession_pct"] for r in sweep_results]
    rorac = [r["rorac"] for r in sweep_results]
    var_995 = [r["var_995"] for r in sweep_results]
    ruin = [r["ruin_prob"] for r in sweep_results]

    fig.add_trace(go.Scatter(
        x=x, y=rorac, mode="lines+markers",
        line=dict(color="#2563eb", width=2.5), marker=dict(size=5),
        name="RORAC", hovertemplate="Cession: %{x:.0%}<br>RORAC: %{y:.1%}<extra></extra>",
    ), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=x, y=[v / 1e6 for v in var_995], mode="lines+markers",
        line=dict(color="#dc5c0c", width=2.5), marker=dict(size=5),
        name="VaR(99.5%)", hovertemplate="Cession: %{x:.0%}<br>VaR: \u00a3%{y:.0f}m<extra></extra>",
    ), row=1, col=2)

    fig.add_trace(go.Scatter(
        x=x, y=ruin, mode="lines+markers",
        line=dict(color="#dc2626", width=2.5), marker=dict(size=5),
        name="Ruin %", hovertemplate="Cession: %{x:.0%}<br>Ruin: %{y:.2%}<extra></extra>",
    ), row=1, col=3)

    # Mark current position
    if current_cession is not None:
        for col in range(1, 4):
            fig.add_vline(
                x=current_cession, line_dash="dash", line_color=COLOR_MUTED, line_width=1,
                row=1, col=col,
            )

    fig.update_layout(
        title=dict(text="Reinsurance Efficient Frontier", font=dict(size=14)),
        showlegend=False,
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k != "height"},
        height=350,
    )
    fig.update_xaxes(title_text="Cession %", tickformat=".0%", row=1, col=1)
    fig.update_xaxes(title_text="Cession %", tickformat=".0%", row=1, col=2)
    fig.update_xaxes(title_text="Cession %", tickformat=".0%", row=1, col=3)
    fig.update_yaxes(tickformat=".1%", row=1, col=1)
    fig.update_yaxes(tickformat=".2%", row=1, col=3)
    return fig


# ---------------------------------------------------------------------------
# F2: Seed Stability Strip Chart
# ---------------------------------------------------------------------------
def seed_stability_chart(
    seed_results: list,
    names: list,
    colors=None,
) -> go.Figure:
    """
    Strip plot showing metric variation across multiple random seeds.

    Each dot is one seed's result. Horizontal spread shows how stable
    the metric is across different random number sequences.
    """
    if not seed_results:
        fig = go.Figure()
        fig.update_layout(title="Seed Stability (no data)", **LAYOUT_DEFAULTS)
        return fig

    clrs = colors or STRATEGY_COLORS
    n_strat = len(seed_results[0].get("strategies", []))
    metrics = [
        ("RORAC", "rorac", ".1%"),
        ("VaR(99.5%)", "var_995", ",.0f"),
        ("P(Ruin)", "prob_ruin", ".2%"),
        ("Combined Ratio", "combined_ratio", ".1%"),
    ]

    fig = make_subplots(
        rows=1, cols=len(metrics),
        subplot_titles=[m[0] for m in metrics],
    )

    for col, (label, key, fmt) in enumerate(metrics, 1):
        for i in range(min(n_strat, len(names))):
            vals = [r["strategies"][i][key] for r in seed_results if i < len(r["strategies"])]
            seeds = [r["seed"] for r in seed_results if i < len(r["strategies"])]
            mean_val = np.mean(vals) if vals else 0

            fig.add_trace(go.Scatter(
                x=[names[i]] * len(vals), y=vals,
                mode="markers",
                marker=dict(color=clrs[i % len(clrs)], size=8, opacity=0.7),
                name=names[i] if col == 1 else None,
                showlegend=(col == 1),
                hovertemplate=f"Seed: %{{text}}<br>{label}: %{{y:{fmt}}}<extra>{names[i]}</extra>",
                text=[str(s) for s in seeds],
            ), row=1, col=col)

            # Mean line
            fig.add_trace(go.Scatter(
                x=[names[i]], y=[mean_val],
                mode="markers",
                marker=dict(color=clrs[i % len(clrs)], size=14, symbol="line-ew-open", line_width=3),
                showlegend=False, hoverinfo="skip",
            ), row=1, col=col)

    fig.update_layout(
        title=dict(text="Seed Stability Analysis", font=dict(size=14)),
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k != "height"},
        height=350,
    )
    return fig


# ---------------------------------------------------------------------------
# F4: GPD Tail Fit Chart
# ---------------------------------------------------------------------------
def gpd_fit_chart(
    cumulative_profit_dist: list,
    gpd_result: dict,
    name: str,
    color: str,
) -> go.Figure:
    """
    Empirical loss histogram with GPD curve overlay in the tail region.

    Vertical lines compare empirical vs GPD-extrapolated VaR(99.5%).
    """
    fig = go.Figure()
    data = np.array(cumulative_profit_dist)

    if not gpd_result or not gpd_result.get("fit_successful"):
        fig.add_trace(go.Histogram(
            x=data, nbinsx=80, name="Cumulative Profit",
            marker_color=color, opacity=0.7,
        ))
        fig.update_layout(
            title=f"{name}: Cumulative Profit Distribution (GPD fit unavailable)",
            **LAYOUT_DEFAULTS,
        )
        return fig

    # Empirical histogram
    fig.add_trace(go.Histogram(
        x=data, nbinsx=80, name="Empirical",
        marker_color=color, opacity=0.5, histnorm="probability density",
    ))

    # VaR lines
    var_emp = gpd_result["var_995_empirical"]
    var_gpd = gpd_result["var_995_gpd"]

    fig.add_vline(x=var_emp, line_dash="dash", line_color="#6b7280", line_width=2,
                  annotation_text=f"Emp VaR(99.5%): {var_emp/1e6:.0f}m",
                  annotation_position="top left")
    fig.add_vline(x=var_gpd, line_dash="solid", line_color="#dc2626", line_width=2,
                  annotation_text=f"GPD VaR(99.5%): {var_gpd/1e6:.0f}m",
                  annotation_position="top right")

    shape = gpd_result["shape"]
    n_exc = gpd_result["n_exceedances"]
    fig.add_annotation(
        text=f"GPD shape={shape:.3f}, n_exceedances={n_exc}",
        xref="paper", yref="paper", x=0.98, y=0.95,
        showarrow=False, font=dict(size=10, color="#6b7280"),
        bgcolor="rgba(255,255,255,0.8)", bordercolor="#e5e7eb",
    )

    fig.update_layout(
        title=dict(text=f"{name}: Tail Extrapolation (GPD)", font=dict(size=14)),
        xaxis_title="Cumulative Profit (GBP)",
        yaxis_title="Density",
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k != "height"},
        height=370,
    )
    return fig


# ---------------------------------------------------------------------------
# F5: Model Validation Charts
# ---------------------------------------------------------------------------
def residual_analysis_chart(
    residuals: np.ndarray,
    n_years: int,
) -> go.Figure:
    """
    AR(2) residual diagnostics: time series with bands, ACF, histogram.

    Three panels testing the IID Normal assumption of AR(2) innovations.
    """
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=["Residual Time Series", "Autocorrelation (ACF)", "Residual Distribution"],
    )

    # Use residuals from t=2 onward (first 2 are initialization)
    res = residuals[:, 2:] if residuals.shape[1] > 2 else residuals
    years = list(range(3, n_years + 1))
    mean_res = res.mean(axis=0)
    p5 = np.percentile(res, 5, axis=0)
    p95 = np.percentile(res, 95, axis=0)

    # Panel 1: Residual time series with p5/p95 bands
    fig.add_trace(go.Scatter(
        x=years, y=p95, mode="lines", line=dict(width=0),
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=years, y=p5, mode="lines", line=dict(width=0),
        fill="tonexty", fillcolor="rgba(37, 99, 235, 0.1)",
        showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=years, y=mean_res, mode="lines",
        line=dict(color="#2563eb", width=2), name="Mean Residual",
    ), row=1, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="#9ca3af", row=1, col=1)

    # Panel 2: ACF (autocorrelation lags 1-10)
    all_res = res.flatten()
    all_res = all_res[np.isfinite(all_res)]
    max_lag = min(10, len(all_res) // 4)
    acf_vals = []
    if len(all_res) > max_lag:
        centered = all_res - all_res.mean()
        var = np.sum(centered ** 2)
        for lag in range(1, max_lag + 1):
            if var > 0:
                acf_vals.append(float(np.sum(centered[:-lag] * centered[lag:]) / var))
            else:
                acf_vals.append(0.0)

    lags = list(range(1, len(acf_vals) + 1))
    # Bartlett confidence band: +/- 1.96 / sqrt(N)
    n_obs = len(all_res)
    ci_bound = 1.96 / np.sqrt(max(n_obs, 1))

    fig.add_trace(go.Bar(
        x=lags, y=acf_vals, marker_color="#2563eb", name="ACF",
        showlegend=False,
    ), row=1, col=2)
    fig.add_hline(y=ci_bound, line_dash="dash", line_color="#dc2626", line_width=1, row=1, col=2)
    fig.add_hline(y=-ci_bound, line_dash="dash", line_color="#dc2626", line_width=1, row=1, col=2)
    fig.add_hline(y=0, line_color="#9ca3af", line_width=0.5, row=1, col=2)

    # Panel 3: Histogram vs Normal reference
    fig.add_trace(go.Histogram(
        x=all_res, nbinsx=50, histnorm="probability density",
        marker_color="#2563eb", opacity=0.5, name="Residuals",
        showlegend=False,
    ), row=1, col=3)

    # Normal reference curve
    if len(all_res) > 10:
        mu, sigma = all_res.mean(), all_res.std()
        if sigma > 0:
            x_range = np.linspace(mu - 4 * sigma, mu + 4 * sigma, 100)
            y_norm = np.exp(-0.5 * ((x_range - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))
            fig.add_trace(go.Scatter(
                x=x_range, y=y_norm, mode="lines",
                line=dict(color="#dc2626", width=2, dash="dash"),
                name="N(0, sigma)", showlegend=False,
            ), row=1, col=3)

    fig.update_layout(
        title=dict(text="AR(2) Residual Diagnostics", font=dict(size=14)),
        **{k: v for k, v in LAYOUT_DEFAULTS.items() if k != "height"},
        height=350,
    )
    return fig


def pit_histogram_chart(
    gross_lr_data: np.ndarray,
    name: str,
) -> go.Figure:
    """
    Probability Integral Transform histogram for model validation.

    If the model is correctly specified, PIT values should be uniform(0,1).
    Uses empirical CDF to compute PIT values.
    """
    fig = go.Figure()

    # Flatten across years for more data
    data = gross_lr_data.flatten()
    data = data[np.isfinite(data)]

    if len(data) < 100:
        fig.update_layout(title=f"{name}: PIT (insufficient data)", **LAYOUT_DEFAULTS)
        return fig

    # Empirical CDF → PIT
    sorted_data = np.sort(data)
    n = len(sorted_data)
    # For each observation, its PIT = rank / n
    ranks = np.searchsorted(sorted_data, data, side="right")
    pit_vals = ranks / n

    fig.add_trace(go.Histogram(
        x=pit_vals, nbinsx=20, histnorm="probability",
        marker_color="#2563eb", opacity=0.6, name="PIT",
    ))

    # Uniform reference line
    fig.add_hline(y=0.05, line_dash="dash", line_color="#dc2626", line_width=1.5,
                  annotation_text="Uniform (expected)", annotation_position="top right")

    # Anderson-Darling test
    try:
        from scipy.stats import anderson
        ad_result = anderson(pit_vals, dist="norm")
        stat = ad_result.statistic
        fig.add_annotation(
            text=f"A-D stat: {stat:.3f}",
            xref="paper", yref="paper", x=0.02, y=0.95,
            showarrow=False, font=dict(size=10, color="#6b7280"),
            bgcolor="rgba(255,255,255,0.8)", bordercolor="#e5e7eb",
        )
    except Exception:
        pass

    fig.update_layout(
        title=dict(text=f"{name}: PIT Histogram", font=dict(size=14)),
        xaxis_title="PIT Value",
        yaxis_title="Proportion",
        **LAYOUT_DEFAULTS,
    )
    return fig


def var_backtest_chart(
    results: dict,
    n_years: int,
    name: str,
    color: str,
) -> go.Figure:
    """
    VaR backtest: for each year, count paths breaching predicted VaR(95%).

    Bars show actual breach % per year vs the expected 5% line.
    Kupiec test assesses overall calibration.
    """
    fig = go.Figure()
    cum_profit = results["cumulative_profit"]
    n_paths = cum_profit.shape[0]

    # Compute year-by-year VaR(95%) and breach counts
    breach_pcts = []
    for t in range(n_years):
        # Predicted VaR(95%) from distribution up to year t
        year_data = cum_profit[:, t]
        var_95 = np.percentile(year_data, 5)
        breach_count = (year_data < var_95).sum()
        breach_pcts.append(breach_count / n_paths)

    years = list(range(1, n_years + 1))

    fig.add_trace(go.Bar(
        x=years, y=breach_pcts,
        marker_color=color, opacity=0.7, name="Actual Breach %",
    ))
    fig.add_hline(y=0.05, line_dash="dash", line_color="#dc2626", line_width=2,
                  annotation_text="Expected 5%")

    # Kupiec test (binomial test)
    total_breaches = sum(1 for b in breach_pcts if b > 0.05)
    try:
        from scipy.stats import binomtest
        bt = binomtest(total_breaches, n_years, 0.5)
        fig.add_annotation(
            text=f"Years over 5%: {total_breaches}/{n_years}",
            xref="paper", yref="paper", x=0.98, y=0.95,
            showarrow=False, font=dict(size=10, color="#6b7280"),
            bgcolor="rgba(255,255,255,0.8)", bordercolor="#e5e7eb",
        )
    except Exception:
        pass

    fig.update_layout(
        title=dict(text=f"{name}: VaR(95%) Backtest", font=dict(size=14)),
        xaxis_title="Year",
        yaxis_title="Breach %",
        yaxis_tickformat=".1%",
        **LAYOUT_DEFAULTS,
    )
    return fig
