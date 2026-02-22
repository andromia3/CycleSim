"""
CycleSim — Dash + Mantine entry point.

Run with:
    cd C:/Workspace/projects/CycleSim
    .venv/Scripts/python -m ui.app
"""

import dash
import dash._dash_renderer
dash._dash_renderer._set_react_version("18.2.0")
from dash import html, dcc, callback, Input, Output, State, no_update, ctx
import dash_mantine_components as dmc
import numpy as np

from ui.sidebar import build_sidebar, ALL_INPUT_IDS
from ui.charts import (
    fan_chart, comparison_fan_chart, distribution_chart, market_cycle_chart,
    radar_chart, rorac_scatter, ruin_over_time_chart, attribution_chart,
    single_path_chart, worst_paths_data, efficiency_frontier,
    drawdown_chart, regime_performance_data, win_probability_chart,
    cycle_timing_chart, solvency_comparison_chart,
    sensitivity_tornado_from_rows, cr_regime_heatmap,
    capital_model_preview, yearly_profit_waterfall,
    cumulative_profit_buildup,
    # Pass 26 charts
    historical_cr_chart, historical_cumulative_chart, counterfactual_waterfall,
    regime_forecast_chart, market_clock_chart, tail_decomposition_chart,
    pareto_frontier_chart, strategy_dna_radar,
    # Pass 29 charts
    capital_allocation_chart, qq_plot_chart, correlation_heatmap,
    ri_frontier_chart,
    # Pass 31 charts
    seed_stability_chart, gpd_fit_chart,
    residual_analysis_chart, pit_histogram_chart, var_backtest_chart,
    STRATEGY_COLORS, STRATEGY_COLORS_LIGHT, STRATEGY_COLORS_MED,
    COLOR_A, COLOR_B, COLOR_A_LIGHT, COLOR_A_MED, COLOR_B_LIGHT, COLOR_B_MED,
    COLOR_MARKET, COLOR_MARKET_LIGHT, COLOR_MARKET_MED, COLOR_MUTED,
    LAYOUT_DEFAULTS, REGIME_COLORS, REGIME_NAMES,
)
from ui.exhibits import (
    summary_table, kpi_row, worst_paths_table, drilldown_table,
    regime_performance_table, executive_summary, convergence_indicator,
    strategy_profile_card, return_period_table, stress_scenario_cards,
    # Pass 26 exhibits
    backtest_summary_cards, backtest_year_table, counterfactual_verdict,
    regime_outlook_table, tail_decomposition_table,
    playbook_table, optimizer_gap_cards, optimizer_stats_badge,
    data_provenance_badge,
    # Pass 29 exhibits
    year1_plan_cards, audit_log_table, scenario_delta_table,
    # Pass 31 exhibits
    risk_appetite_assessment,
)
from ui.state import (
    get_or_run, set_capital_model, clear_capital_model, get_capital_model,
    run_sensitivity, build_config, build_optimizer_base_config,
    run_ri_sweep, run_seed_stability,
)
from cyclesim.defaults import (
    SIMULATION_DEFAULTS, REGIMES, MARKET_DEFAULTS, LOSS_DEFAULTS,
    INSURER_PRESETS, STRATEGY_PREFIXES, MAX_STRATEGIES,
)
from cyclesim.io import export_results_to_excel, export_raw_paths_csv, import_capital_model, generate_sample_capital_model
from cyclesim.metrics import compute_return_period_table, compute_tail_decomposition, compute_capital_allocation
from cyclesim.market import compute_regime_forecast, compute_stationary_distribution
from cyclesim.historical import (
    replay_historical, counterfactual_decomposition, LLOYDS_HISTORICAL,
)
from cyclesim.losses import LossParams


def _graph(figure, **kwargs):
    """dcc.Graph wrapper that locks container height to prevent resize loop."""
    import plotly.graph_objects as go
    if figure is None:
        figure = go.Figure()
    h = getattr(getattr(figure, "layout", None), "height", None) or 370
    style = kwargs.pop("style", {})
    style.setdefault("height", f"{h}px")
    return dcc.Graph(figure=figure, style=style, **kwargs)


# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    title="CycleSim",
    external_stylesheets=[dmc.styles.ALL],
    suppress_callback_exceptions=True,
    serve_locally=True,
)



# ---------------------------------------------------------------------------
# Scenario presets — full parameter snapshots
# ---------------------------------------------------------------------------
def _default_insurer(prefix, preset):
    """Extract slider values for an insurer preset."""
    sw = preset.get("signal_weights", {})
    return {
        f"{prefix}_name": preset["name"],
        f"{prefix}_gwp": preset["initial_gwp"],
        f"{prefix}_expense": preset["expense_ratio"],
        f"{prefix}_cession": preset["base_cession_pct"],
        f"{prefix}_inv_return": preset["investment_return"],
        f"{prefix}_expected_lr": preset["expected_lr"],
        f"{prefix}_growth": preset["growth_rate_when_profitable"],
        f"{prefix}_shrink": preset["shrink_rate_when_unprofitable"],
        f"{prefix}_max_growth": preset["max_gwp_growth_pa"],
        f"{prefix}_max_shrink": preset["max_gwp_shrink_pa"],
        f"{prefix}_cess_sens": preset["cession_cycle_sensitivity"],
        f"{prefix}_min_cess": preset["min_cession_pct"],
        f"{prefix}_max_cess": preset["max_cession_pct"],
        f"{prefix}_adv_sel": preset["adverse_selection_sensitivity"],
        f"{prefix}_cap_ratio": preset["capital_ratio"],
        f"{prefix}_coc": preset["cost_of_capital"],
        # Signal weights
        f"{prefix}_sw_own_lr": sw.get("own_lr", 0.35),
        f"{prefix}_sw_market_lr": sw.get("market_lr", 0.20),
        f"{prefix}_sw_rate_adequacy": sw.get("rate_adequacy", 0.20),
        f"{prefix}_sw_rate_change": sw.get("rate_change", 0.15),
        f"{prefix}_sw_capital": sw.get("capital_position", 0.10),
    }

_BASE_MARKET = {
    "ar2_mode": "auto",
    "cycle_period": MARKET_DEFAULTS["cycle_period_years"],
    "min_lr": MARKET_DEFAULTS["min_loss_ratio"],
    "max_lr": MARKET_DEFAULTS["max_loss_ratio"],
    "long_run_lr": MARKET_DEFAULTS["long_run_loss_ratio"],
    "phi_1": 1.1, "phi_2": -0.45, "sigma_epsilon": 0.10,
    "shock_prob": MARKET_DEFAULTS["shock_prob"],
    "shock_magnitude": MARKET_DEFAULTS["shock_magnitude_std"],
}
_BASE_LOSS = {
    "attritional_mean": LOSS_DEFAULTS["attritional_mean"],
    "cycle_sensitivity": LOSS_DEFAULTS["cycle_sensitivity"],
    "tail_sensitivity": LOSS_DEFAULTS["tail_thickness_sensitivity"],
    "large_freq": LOSS_DEFAULTS["large_loss_frequency"],
    "cat_freq": LOSS_DEFAULTS["cat_frequency"],
    "cat_severity": LOSS_DEFAULTS["cat_severity_mean"],
}
_BASE_SIM = {
    "n_paths": SIMULATION_DEFAULTS["n_paths"],
    "n_years": SIMULATION_DEFAULTS["n_years"],
    "seed": SIMULATION_DEFAULTS["random_seed"],
}

_DEFAULT_PRESET = {
    **_BASE_SIM, **_BASE_MARKET, **_BASE_LOSS,
    "n_strategies": 2,
    **{k: v for pfx in STRATEGY_PREFIXES
       for k, v in _default_insurer(pfx, INSURER_PRESETS[pfx]).items()},
}

# ---------------------------------------------------------------------------
# Strategy archetype insurer overrides
# ---------------------------------------------------------------------------
_CONSERVATIVE_B = {**INSURER_PRESETS["a"], "name": "Conservative B"}
_AGGRESSIVE_A = {**INSURER_PRESETS["b"], "name": "Aggressive A"}

# Counter-cyclical: grows in hard markets, shrinks in soft
_COUNTER_CYCLICAL = {
    **INSURER_PRESETS["a"],
    "name": "Counter-Cyclical",
    "growth_rate_when_profitable": 0.12,
    "shrink_rate_when_unprofitable": -0.10,
    "cession_cycle_sensitivity": 0.06,
    "expected_lr": 0.50,
    "signal_weights": {
        "own_lr": 0.15, "market_lr": 0.25,
        "rate_adequacy": 0.30, "rate_change": 0.20, "capital_position": 0.10,
    },
}

# Pro-cyclical: chases momentum, slow to retreat
_PRO_CYCLICAL = {
    **INSURER_PRESETS["b"],
    "name": "Pro-Cyclical",
    "growth_rate_when_profitable": 0.15,
    "shrink_rate_when_unprofitable": -0.02,
    "cession_cycle_sensitivity": -0.03,
    "expected_lr": 0.62,
    "signal_weights": {
        "own_lr": 0.45, "market_lr": 0.10,
        "rate_adequacy": 0.05, "rate_change": 0.30, "capital_position": 0.10,
    },
}

# ---------------------------------------------------------------------------
# 12 Scenario Presets in 3 groups
# ---------------------------------------------------------------------------
SCENARIO_PRESETS = {
    # === Historical Scenarios ===
    "post_911": {
        **_DEFAULT_PRESET,
        "shock_prob": 0.15, "shock_magnitude": 0.16,
        "cycle_period": 6.0, "min_lr": 0.42,
    },
    "katrina_era": {
        **_DEFAULT_PRESET,
        "cat_freq": 0.60, "cat_severity": 0.12,
        "shock_prob": 0.12, "tail_sensitivity": 0.40,
    },
    "soft_2014_2019": {
        **_DEFAULT_PRESET,
        "cycle_period": 10.0, "min_lr": 0.52, "max_lr": 0.68,
        "long_run_lr": 0.58, "shock_prob": 0.04,
    },
    "hard_2021_2024": {
        **_DEFAULT_PRESET,
        "cycle_period": 6.0, "min_lr": 0.44, "max_lr": 0.56,
        "long_run_lr": 0.50, "shock_prob": 0.06,
    },
    # === Strategy Archetypes ===
    "default": _DEFAULT_PRESET,
    "both_conservative": {
        **_DEFAULT_PRESET,
        **_default_insurer("a", INSURER_PRESETS["a"]),
        **_default_insurer("b", _CONSERVATIVE_B),
    },
    "both_aggressive": {
        **_DEFAULT_PRESET,
        **_default_insurer("a", _AGGRESSIVE_A),
        **_default_insurer("b", INSURER_PRESETS["b"]),
    },
    "counter_cyclical": {
        **_DEFAULT_PRESET,
        **_default_insurer("a", _COUNTER_CYCLICAL),
        **_default_insurer("b", _PRO_CYCLICAL),
    },
    # === Stress Tests ===
    "serial_cats": {
        **_DEFAULT_PRESET,
        "cat_freq": 0.80, "cat_severity": 0.12,
        "large_freq": 5.0, "shock_prob": 0.20, "shock_magnitude": 0.18,
    },
    "rate_collapse": {
        **_DEFAULT_PRESET,
        "cycle_period": 15.0, "max_lr": 0.75,
        "long_run_lr": 0.62, "shock_prob": 0.03, "min_lr": 0.55,
    },
    "inflation_shock": {
        **_DEFAULT_PRESET,
        "attritional_mean": 0.60, "cycle_sensitivity": 0.25,
        "tail_sensitivity": 0.50, "large_freq": 4.5,
    },
    "perfect_storm": {
        **_DEFAULT_PRESET,
        "cat_freq": 0.60, "cat_severity": 0.15,
        "shock_prob": 0.18, "shock_magnitude": 0.16,
        "attritional_mean": 0.58, "large_freq": 5.0, "tail_sensitivity": 0.45,
    },
}

# Preset descriptions shown below the selector
PRESET_DESCRIPTIONS = {
    "post_911": "Rapid hardening after crisis. Calibrated to Lloyd's 2002-03: CR dropped from 115% to 91%.",
    "katrina_era": "Mega-cat years bookending a long soft market. Based on 2005 Katrina/2017 HIM pattern.",
    "soft_2014_2019": "Prolonged soft market with capital influx and social inflation. 6 years of deteriorating rates.",
    "hard_2021_2024": "Strong underwriting discipline, peak returns. Lloyd's 2023-24 golden era.",
    "default": "The classic comparison. Disciplined underwriter vs growth-chasing insurer.",
    "both_conservative": "Both strategies use disciplined underwriting. Tests whether universal discipline matters.",
    "both_aggressive": "Both chase growth. Tests the downside of market-wide aggression.",
    "counter_cyclical": "A grows in hard markets and shrinks in soft; B does the opposite. Tests cycle timing value.",
    "serial_cats": "What if mega-cats hit every 18 months? Tests capital resilience under repeated shocks.",
    "rate_collapse": "Floor falls out \u2014 prolonged soft market with no hardening trigger for 15 years.",
    "inflation_shock": "Claims costs surge across all lines. Attritional mean +8pp, tails fatten, large losses increase.",
    "perfect_storm": "Everything goes wrong simultaneously. The scenario your board asks about.",
}


# ---------------------------------------------------------------------------
# Helper functions (must be defined before layout)
# ---------------------------------------------------------------------------
def _placeholder_content():
    """Landing page shown before first simulation run."""
    return dmc.Stack([
        # Hero — minimal, confident
        html.Div([
            html.Div(style={"height": "80px"}),
            dmc.Stack([
                dmc.Text("CycleSim", fw=700, size="xl",
                         style={"fontSize": "2.4rem", "color": "#111827", "letterSpacing": "-0.02em"}),
                dmc.Text(
                    "Monte Carlo strategy simulator for the London Market underwriting cycle.",
                    size="md", c="#6b7280", fw=400,
                    style={"maxWidth": "520px"},
                ),
            ], gap=8, align="center"),
            html.Div(style={"height": "40px"}),
        ], style={"textAlign": "center"}),

        # Three pillars — no emoji, no icons, just substance
        dmc.SimpleGrid(
            cols=3, spacing="lg",
            children=[
                html.Div([
                    dmc.Text("01", size="xs", c="#d1d5db", fw=700,
                             ff="'JetBrains Mono', monospace", mb=4),
                    dmc.Text("Market Engine", fw=600, size="sm", mb=6,
                             style={"color": "#111827"}),
                    dmc.Text(
                        "AR(2) process with Hidden Markov regime switching. "
                        "Four market states, exogenous shocks, calibrated to "
                        "Lloyd\u2019s combined ratios 2001\u20132024.",
                        size="xs", c="#6b7280", lh=1.55,
                    ),
                ], style={"borderTop": "2px solid #2563eb", "paddingTop": "16px"}),
                html.Div([
                    dmc.Text("02", size="xs", c="#d1d5db", fw=700,
                             ff="'JetBrains Mono', monospace", mb=4),
                    dmc.Text("Strategy Model", fw=600, size="sm", mb=6,
                             style={"color": "#111827"}),
                    dmc.Text(
                        "Ten interacting levers per insurer. Multi-signal reactions, "
                        "adverse selection on rapid growth, component-specific "
                        "reinsurance with its own pricing cycle.",
                        size="xs", c="#6b7280", lh=1.55,
                    ),
                ], style={"borderTop": "2px solid #dc5c0c", "paddingTop": "16px"}),
                html.Div([
                    dmc.Text("03", size="xs", c="#d1d5db", fw=700,
                             ff="'JetBrains Mono', monospace", mb=4),
                    dmc.Text("Risk Analytics", fw=600, size="sm", mb=6,
                             style={"color": "#111827"}),
                    dmc.Text(
                        "VaR, TVaR, RORAC, max drawdown, ruin probability. "
                        "Profit attribution, regime-conditional performance, "
                        "sensitivity tornado, path-level drill-down.",
                        size="xs", c="#6b7280", lh=1.55,
                    ),
                ], style={"borderTop": "2px solid #059669", "paddingTop": "16px"}),
            ],
        ),

        html.Div(style={"height": "24px"}),

        # CTA — not a blue alert box
        dmc.Text(
            "Configure parameters in the sidebar, then press Run Simulation.",
            size="sm", c="#6b7280", ta="center",
        ),
        dmc.Text(
            "Defaults are Lloyd\u2019s-calibrated. Two preset strategies loaded: "
            "disciplined underwriter vs aggressive grower.",
            size="xs", c="#adb5bd", ta="center",
        ),
    ], gap=4, mt=0, style={"maxWidth": "800px", "margin": "0 auto"})


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app.layout = dmc.MantineProvider(
    forceColorScheme="light",
    theme={
        "fontFamily": "Inter, system-ui, -apple-system, sans-serif",
        "fontFamilyMonospace": "'JetBrains Mono', 'Fira Code', Consolas, monospace",
        "primaryColor": "blue",
        "defaultRadius": "sm",
    },
    children=dmc.AppShell(
        [
            dmc.AppShellNavbar(
                build_sidebar(),
                p="md",
                style={"overflowY": "auto"},
            ),
            dmc.AppShellMain(
                dmc.Container([
                    dcc.Loading(
                        id="loading-wrapper",
                        type="default",
                        children=html.Div(id="main-content", children=[
                            _placeholder_content(),
                        ]),
                        style={"minHeight": "200px"},
                    ),
                    dcc.Download(id="download-excel"),
                    dcc.Download(id="download-csv-paths"),
                    # Hidden store: incremented on each main sim run so
                    # dependent panels (Strategy Lab) can detect stale data.
                    dcc.Store(id="sim-generation", data=0, storage_type="session"),
                    dcc.Store(id="saved-profiles", data={}, storage_type="local"),
                    dcc.Store(id="audit-log", data=[], storage_type="local"),
                    dcc.Store(id="scenario-snapshots", data=[], storage_type="local"),
                ], fluid=True, p="lg"),
            ),
        ],
        navbar={"width": 360, "breakpoint": "md", "collapsed": {"mobile": True}},
        padding="md",
    ),
)


# ---------------------------------------------------------------------------
# Main callback: run simulation and render all tabs
# ---------------------------------------------------------------------------
@callback(
    Output("main-content", "children"),
    Output("sim-generation", "data"),
    Output("run-btn", "children"),
    Output("audit-log", "data"),
    Input("run-btn", "n_clicks"),
    State("sim-generation", "data"),
    State("audit-log", "data"),
    *[State(id, "value") for id in ALL_INPUT_IDS],
    prevent_initial_call=True,
)
def run_and_render(n_clicks, sim_gen, audit_log, *input_values):
    if not n_clicks:
        return no_update, no_update, no_update, no_update
    next_gen = (sim_gen or 0) + 1

    params = dict(zip(ALL_INPUT_IDS, input_values))
    params["n_paths"] = int(params.get("n_paths") or 1000)
    params["n_years"] = int(params.get("n_years") or 25)
    # Salt seed with generation counter so each Run click produces fresh paths
    params["seed"] = int(params.get("seed") or 42) + next_gen

    import warnings, traceback, time
    print(f"[SIM] gen={next_gen} n_paths={params['n_paths']} seed={params['seed']}", flush=True)
    t0 = time.time()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            results = get_or_run(params)
    except Exception as e:
        traceback.print_exc()
        return dmc.Alert(
            f"Simulation failed: {e}",
            title="Error", color="red", variant="light",
        ), next_gen, "Run Simulation", audit_log or []
    elapsed_sim = time.time() - t0
    print(f"[SIM] done in {elapsed_sim:.2f}s", flush=True)

    # N-strategy convenience lists (clamp to available data)
    n = max(1, min(int(params.get("n_strategies", 2)), len(results.summaries), len(results.insurers)))
    summaries = results.summaries[:n]
    ins_list = results.insurers[:n]
    names = [results.config.insurers[i].name for i in range(n)]
    ny = results.config.n_years

    # Capital model status
    cm_data, cm_info = get_capital_model()
    loss_mode_badge = (
        dmc.Badge("Capital Model Active", color="violet", size="sm", variant="light")
        if cm_data is not None
        else dmc.Badge("Parametric Model", color="gray", size="sm", variant="light")
    )

    # Pre-compute analytics
    wp_rows = worst_paths_data(results, n_worst=10)
    rp_rows = regime_performance_data(results.market, ins_list, names)

    # Return period stress metrics
    try:
        stress_rp_rows = compute_return_period_table(
            results.insurers, results.config.insurers,
        )
    except Exception:
        stress_rp_rows = []

    # Sensitivity analysis (fast: 300 paths per perturbation)
    try:
        import warnings as _w
        with _w.catch_warnings():
            _w.simplefilter("ignore", RuntimeWarning)
            sens_rows = run_sensitivity(params, n_paths_sensitivity=300)
    except Exception:
        sens_rows = []

    # --- Historical backtest (N strategies) ---
    try:
        config = build_config(params)
        loss_params = config.loss_params
        bt_list = [replay_historical(config.insurers[i], loss_params) for i in range(n)]
        cf_decomp = None
        if n == 2:
            cf_decomp = counterfactual_decomposition(
                bt_list[0], bt_list[1], config.insurers[0], config.insurers[1], loss_params,
            )
        has_backtest = True
    except Exception:
        bt_list = []
        cf_decomp = None
        has_backtest = False

    # --- Tail risk decomposition (N strategies) ---
    try:
        decomp_list = [
            compute_tail_decomposition(ins_list[i], results.config.insurers[i])
            for i in range(n)
        ]
        has_tail_decomp = bool(decomp_list and decomp_list[0])
    except Exception:
        decomp_list = []
        has_tail_decomp = False

    # --- Capital allocation (F3) ---
    try:
        cap_alloc_list = [compute_capital_allocation(ins_list[i]) for i in range(n)]
    except Exception:
        cap_alloc_list = [{}] * n

    # --- Forward regime forecast ---
    try:
        trans_matrix = results.market["params"].regime_transition_matrix
        regime_counts = np.bincount(results.market["regime"][:, 0], minlength=4)
        current_regime = int(np.argmax(regime_counts))
        forecast = compute_regime_forecast(trans_matrix, current_regime, n_years=10)
        has_forecast = True
    except Exception:
        forecast = None
        current_regime = 0
        trans_matrix = None
        has_forecast = False

    # --- Seed stability (F2) ---
    seed_stability = None
    try:
        import warnings as _sw
        with _sw.catch_warnings():
            _sw.simplefilter("ignore", RuntimeWarning)
            seed_stability = run_seed_stability(params, n_seeds=5, n_paths_per_seed=300)
    except Exception:
        pass

    # --- Risk appetite params ---
    risk_appetite = {
        "ruin_max": float(params.get("risk_appetite_ruin_max", 0.02)),
        "solvency_min": float(params.get("risk_appetite_solvency_min", 1.5)),
        "cr_max": float(params.get("risk_appetite_cr_max", 1.05)),
        "rorac_min": float(params.get("risk_appetite_rorac_min", 0.05)),
    }

    try:
        content = _build_layout(
            results, summaries, ins_list, names, n, ny,
            cm_data, loss_mode_badge,
            wp_rows, rp_rows, stress_rp_rows, sens_rows,
            bt_list, cf_decomp, has_backtest,
            decomp_list, has_tail_decomp,
            trans_matrix, forecast, current_regime, has_forecast,
            cap_alloc_list=cap_alloc_list,
            audit_log=audit_log,
            run_gen=next_gen, run_seed=params["seed"],
            seed_stability=seed_stability,
            risk_appetite=risk_appetite,
        )
    except Exception as e:
        traceback.print_exc()
        return dmc.Alert(
            f"Simulation failed: {e}",
            title="Error", color="red", variant="light",
        ), next_gen, "Run Simulation", audit_log or []

    # --- Audit log entry (F8) ---
    import datetime
    audit_log = list(audit_log or [])
    entry = {
        "gen": next_gen,
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "n_paths": params["n_paths"],
        "n_years": params["n_years"],
        "seed": params["seed"],
        "n_strategies": n,
        "strategy_names": names,
        "roracs": [s.get("mean_through_cycle_rorac") for s in summaries],
        "combined_ratios": [s.get("mean_combined_ratio") for s in summaries],
        "elapsed": time.time() - t0,
    }
    audit_log.append(entry)
    if len(audit_log) > 20:
        audit_log = audit_log[-20:]

    return content, next_gen, f"Run Simulation (#{next_gen})", audit_log


def _build_layout(
    results, summaries, ins_list, names, n, ny,
    cm_data, loss_mode_badge,
    wp_rows, rp_rows, stress_rp_rows, sens_rows,
    bt_list, cf_decomp, has_backtest,
    decomp_list, has_tail_decomp,
    trans_matrix, forecast, current_regime, has_forecast,
    *, cap_alloc_list=None, audit_log=None,
    run_gen=1, run_seed=42,
    seed_stability=None, risk_appetite=None,
):
    # Regime outlook: build per-strategy regime performance dicts
    regime_perfs = []
    for j in range(n):
        perf = {}
        for r in rp_rows:
            strategies = r.get("strategies", [])
            if j < len(strategies):
                perf[r["regime"].lower()] = {
                    "cr": strategies[j]["cr"],
                    "rorac": strategies[j]["rorac"],
                }
        regime_perfs.append(perf)

    # Dynamic insurer tab headers
    insurer_tabs = [dmc.TabsTab(names[i], value=f"insurer-{i}") for i in range(n)]

    # Dynamic insurer tab panels
    insurer_panels = []
    for i in range(n):
        insurer_panels.append(dmc.TabsPanel(
            _insurer_tab(
                summaries[i], ny, names[i],
                STRATEGY_COLORS[i % len(STRATEGY_COLORS)],
                STRATEGY_COLORS_LIGHT[i % len(STRATEGY_COLORS_LIGHT)],
                STRATEGY_COLORS_MED[i % len(STRATEGY_COLORS_MED)],
                results.config.insurers[i],
                market=results.market, ins_data=ins_list[i],
            ),
            value=f"insurer-{i}", pt="md",
        ))

    # Dynamic drill-down path selector options
    path_options = []
    for i in range(n):
        pfx = chr(ord('a') + i)
        path_options.append({"value": f"best_{pfx}", "label": f"Best ({names[i]})"})
        path_options.append({"value": f"worst_{pfx}", "label": f"Worst ({names[i]})"})
    path_options.append({"value": "median", "label": "Median Path"})
    path_options += [
        {"value": str(i), "label": f"Path #{i+1}"}
        for i in range(min(results.config.n_paths, 20))
    ]

    content = dmc.Stack([
        # KPI row + convergence + export + provenance
        dmc.Group([
            html.Div(kpi_row(summaries, names, results.elapsed_seconds), style={"flex": 1}),
            dmc.Stack([
                dmc.Badge(f"Run #{run_gen}  |  Seed {run_seed}", color="blue", size="sm", variant="outline"),
                data_provenance_badge(has_capital_model=(cm_data is not None)),
                loss_mode_badge,
                convergence_indicator(
                    [s["through_cycle_rorac_dist"] for s in summaries],
                    [s["cumulative_profit_dist"] for s in summaries],
                    results.config.n_paths,
                ),
                dmc.Button(
                    "Export Excel", id="export-btn",
                    variant="light", color="gray", size="xs",
                ),
                dmc.Button(
                    "Export Raw Paths (CSV)", id="export-csv-btn",
                    variant="light", color="gray", size="xs",
                ),
                dmc.Button(
                    "Save Snapshot", id="save-snapshot-btn",
                    variant="light", color="indigo", size="xs",
                ),
            ], gap="xs", align="flex-end"),
        ], align="flex-start"),

        # Tabs
        dmc.Tabs(
            value="overview",
            children=[
                dmc.TabsList([
                    dmc.TabsTab("Overview", value="overview"),
                    dmc.TabsTab("Market", value="market"),
                    *insurer_tabs,
                    dmc.TabsTab("Comparison", value="compare"),
                    dmc.TabsTab("Risk", value="risk"),
                    dmc.TabsTab("Validation", value="validation"),
                    dmc.TabsTab("Drill-Down", value="drilldown"),
                    dmc.TabsTab("Backtest", value="backtest"),
                    dmc.TabsTab("Strategy Lab", value="strategy-lab"),
                    dmc.TabsTab("Methodology", value="methodology"),
                    dmc.TabsTab("Scenarios", value="scenarios"),
                ]),

                # --- OVERVIEW ---
                dmc.TabsPanel(
                    dmc.Stack([
                        executive_summary(
                            summaries, names,
                            results.config.n_paths, ny, results.elapsed_seconds,
                            rp_rows, risk_appetite=risk_appetite,
                        ),
                        # Risk Appetite Assessment (F6)
                        risk_appetite_assessment(
                            summaries, names,
                            ruin_max=risk_appetite["ruin_max"],
                            solvency_min=risk_appetite["solvency_min"],
                            cr_max=risk_appetite["cr_max"],
                            rorac_min=risk_appetite["rorac_min"],
                        ),
                        # Year 1 Business Plan (F4)
                        year1_plan_cards(ins_list, names),
                        dmc.SimpleGrid(
                            cols=2,
                            children=[
                                summary_table(summaries, names),
                                _graph(figure=radar_chart(summaries, names)),
                            ],
                        ),
                        _graph(figure=attribution_chart(
                            [s.get("attribution", {}) for s in summaries], names,
                        )),
                        # Regime-conditional performance breakdown
                        dmc.Paper([
                            dmc.Tooltip(
                                dmc.Text(
                                    "Performance by Market Regime",
                                    fw=600, size="sm", mb="xs",
                                    td="underline", style={"textDecorationStyle": "dotted", "cursor": "help"},
                                ),
                                label="Average combined ratio and RORAC for each strategy during soft, "
                                      "firming, hard, and crisis market regimes.",
                                position="right", multiline=True, w=300,
                            ),
                            dmc.Text(
                                "How does each strategy perform during different phases of the underwriting cycle?",
                                size="xs", c="dimmed", mb="sm",
                            ),
                            regime_performance_table(rp_rows, names),
                        ], p="md", radius="md", withBorder=True),
                        # Sensitivity tornado
                        *(
                            [dmc.Paper([
                                dmc.Text("Parameter Sensitivity", fw=600, size="sm", mb="xs"),
                                dmc.Text(
                                    "Which parameters matter most for RORAC? Each bar shows the RORAC change "
                                    "from a \u00b120-50% perturbation of that parameter.",
                                    size="xs", c="dimmed", mb="sm",
                                ),
                                _graph(figure=sensitivity_tornado_from_rows(sens_rows, names)),
                            ], p="md", radius="md", withBorder=True)]
                            if sens_rows else []
                        ),
                        # Forward Market Outlook
                        *(
                            [dmc.Paper([
                                dmc.Text("Forward Market Outlook", fw=600, size="sm", mb="xs"),
                                dmc.Text(
                                    "Regime probability forecast from the Markov transition matrix. "
                                    "Shows how the market state distribution evolves from the current regime.",
                                    size="xs", c="dimmed", mb="sm",
                                ),
                                dmc.SimpleGrid(
                                    cols=2,
                                    children=[
                                        _graph(figure=market_clock_chart(trans_matrix, current_regime)),
                                        _graph(figure=regime_forecast_chart(forecast, current_regime)),
                                    ],
                                ),
                                regime_outlook_table(forecast, regime_perfs, names),
                            ], p="md", radius="md", withBorder=True)]
                            if has_forecast else []
                        ),
                        # Seed Stability (F2)
                        *(
                            [dmc.Accordion(
                                children=[dmc.AccordionItem(
                                    [
                                        dmc.AccordionControl(dmc.Group([
                                            dmc.Text("Seed Stability", size="sm", fw=500),
                                            dmc.Badge("5 seeds x 300 paths", color="gray",
                                                      size="xs", variant="light"),
                                        ])),
                                        dmc.AccordionPanel(dmc.Stack([
                                            dmc.Text(
                                                "Shows how key metrics vary across different random seeds. "
                                                "Large dispersion indicates the metric is sensitive to seed choice "
                                                "and may need more paths.",
                                                size="xs", c="dimmed",
                                            ),
                                            _graph(figure=seed_stability_chart(
                                                seed_stability, names,
                                            )),
                                        ], gap="xs")),
                                    ],
                                    value="seed-stability",
                                )],
                                variant="contained",
                            )]
                            if seed_stability else []
                        ),
                        # Audit Log (F8)
                        dmc.Accordion(
                            children=[dmc.AccordionItem(
                                [
                                    dmc.AccordionControl(dmc.Text("Run History", size="sm", fw=500)),
                                    dmc.AccordionPanel(audit_log_table(audit_log or [])),
                                ],
                                value="audit",
                            )],
                            variant="contained",
                        ),
                    ], gap="lg"),
                    value="overview", pt="md",
                ),

                # --- MARKET ---
                dmc.TabsPanel(
                    dmc.Stack([
                        _graph(figure=market_cycle_chart(
                            results.market, ny,
                            historical_data=LLOYDS_HISTORICAL if ny <= 30 else None,
                        )),
                        dmc.SimpleGrid(
                            cols=2,
                            children=[
                                _graph(figure=_market_rate_chart(results.market, ny)),
                                _graph(figure=_regime_bar(results.market, ny)),
                            ],
                        ),
                        _market_stats_bar(results.market, ny),
                    ], gap="md"),
                    value="market", pt="md",
                ),

                # --- INSURER TABS (dynamic N) ---
                *insurer_panels,

                # --- COMPARISON ---
                dmc.TabsPanel(
                    dmc.Stack([
                        _graph(figure=comparison_fan_chart(
                            [s["percentile_bands"]["gwp"] for s in summaries],
                            [s["yearly_means"]["gwp"] for s in summaries],
                            ny, "GWP Trajectory", "GWP (GBP)", names,
                        )),
                        _graph(figure=comparison_fan_chart(
                            [s["percentile_bands"]["combined_ratio"] for s in summaries],
                            [s["yearly_means"]["combined_ratio"] for s in summaries],
                            ny, "Combined Ratio Trajectory", "Combined Ratio",
                            names, yaxis_format=".0%", reference_line=1.0,
                        )),
                        _graph(figure=comparison_fan_chart(
                            [s["percentile_bands"]["cumulative_profit"] for s in summaries],
                            [s["yearly_means"]["cumulative_profit"] for s in summaries],
                            ny, "Cumulative Profit Trajectory", "Cumulative Profit (GBP)", names,
                        )),
                        # NWP + Cession comparison
                        dmc.SimpleGrid(
                            cols=2,
                            children=[
                                _graph(figure=comparison_fan_chart(
                                    [s["percentile_bands"]["nwp"] for s in summaries],
                                    [s["yearly_means"]["nwp"] for s in summaries],
                                    ny, "Net Written Premium", "NWP (GBP)", names,
                                )),
                                _graph(figure=comparison_fan_chart(
                                    [s["percentile_bands"]["cession_pct"] for s in summaries],
                                    [s["yearly_means"]["cession_pct"] for s in summaries],
                                    ny, "Cession % Trajectory", "Cession %",
                                    names, yaxis_format=".0%",
                                )),
                            ],
                        ),
                        # Win probability + Solvency
                        dmc.SimpleGrid(
                            cols=2,
                            children=[
                                _graph(figure=win_probability_chart(
                                    [ins["cumulative_profit"] for ins in ins_list],
                                    ny, names,
                                )),
                                _graph(figure=solvency_comparison_chart(
                                    [ins["solvency_ratio"] for ins in ins_list],
                                    ny, names,
                                )),
                            ],
                        ),
                        dmc.SimpleGrid(
                            cols=2,
                            children=[
                                _graph(figure=rorac_scatter(
                                    [s["through_cycle_rorac_dist"] for s in summaries],
                                    names,
                                )),
                                _graph(figure=efficiency_frontier(summaries, names)),
                            ],
                        ),
                        # Profit buildup decomposition
                        _graph(figure=cumulative_profit_buildup(ins_list, ny, names)),
                        # Cycle timing scatter
                        _graph(figure=cycle_timing_chart(results.market, ins_list, names)),
                    ], gap="md"),
                    value="compare", pt="md",
                ),

                # --- RISK ---
                dmc.TabsPanel(
                    dmc.Stack([
                        # Stress scenario cards (1-in-100 and 1-in-200)
                        *(
                            [stress_scenario_cards(stress_rp_rows, names)]
                            if stress_rp_rows else []
                        ),
                        # Return period table
                        *(
                            [dmc.Paper([
                                dmc.Group([
                                    dmc.Text("Return Period Analysis", fw=600, size="sm"),
                                    dmc.Badge("Lloyd's SCR = 1-in-200", color="red",
                                              variant="light", size="xs"),
                                ], justify="space-between", mb="xs"),
                                dmc.Text(
                                    "Tail metrics at standard return periods. "
                                    "1-in-200 is the Lloyd's Solvency Capital Requirement.",
                                    size="xs", c="dimmed", mb="sm",
                                ),
                                return_period_table(stress_rp_rows, names),
                            ], p="md", radius="md", withBorder=True)]
                            if stress_rp_rows else []
                        ),
                        dmc.SimpleGrid(
                            cols=2,
                            children=[
                                _graph(figure=distribution_chart(
                                    [s["cumulative_profit_dist"] for s in summaries],
                                    "Cumulative Profit Distribution", "GBP",
                                    names,
                                    vars_list=[s.get("var_95_cumulative") for s in summaries],
                                )),
                                _graph(figure=distribution_chart(
                                    [s["max_drawdown_dist"] for s in summaries],
                                    "Max Drawdown Distribution", "GBP",
                                    names,
                                )),
                            ],
                        ),
                        dmc.SimpleGrid(
                            cols=2,
                            children=[
                                _graph(figure=ruin_over_time_chart(
                                    [s["ruin_prob_by_year"] for s in summaries],
                                    ny, names,
                                )),
                                _graph(figure=drawdown_chart(
                                    [ins["capital"] for ins in ins_list],
                                    ny, names,
                                )),
                            ],
                        ),
                        _graph(figure=distribution_chart(
                            [s["terminal_capital_dist"] for s in summaries],
                            "Terminal Capital Distribution", "GBP",
                            names,
                        )),
                        # Worst paths table
                        dmc.Paper([
                            dmc.Text("Worst 10 Paths (by Cumulative Profit)", fw=600, size="sm", mb="xs"),
                            worst_paths_table(wp_rows, names),
                        ], p="md", radius="md", withBorder=True),
                        # Tail risk factor attribution
                        *(
                            [dmc.Paper([
                                dmc.Group([
                                    dmc.Text("Tail Risk Factor Attribution", fw=600, size="sm"),
                                    dmc.Badge("Euler Allocation", color="violet",
                                              variant="light", size="xs"),
                                ], justify="space-between", mb="xs"),
                                dmc.Text(
                                    "Decomposition of tail losses by risk factor at standard return periods. "
                                    "Shows which risk factors dominate at the 1-in-200 SCR level.",
                                    size="xs", c="dimmed", mb="sm",
                                ),
                                _graph(figure=tail_decomposition_chart(decomp_list, names)),
                                tail_decomposition_table(decomp_list, names),
                            ], p="md", radius="md", withBorder=True)]
                            if has_tail_decomp else []
                        ),
                        # Capital allocation (F3)
                        dmc.Paper([
                            dmc.Group([
                                dmc.Text("Economic Capital Allocation", fw=600, size="sm"),
                                dmc.Badge("Euler-Style", color="blue",
                                          variant="light", size="xs"),
                            ], justify="space-between", mb="xs"),
                            dmc.Text(
                                "Decomposition of VaR-based economic capital into risk component "
                                "contributions, estimated from tail-path loss composition.",
                                size="xs", c="dimmed", mb="sm",
                            ),
                            _graph(figure=capital_allocation_chart(
                                cap_alloc_list or [], names,
                            )),
                        ], p="md", radius="md", withBorder=True),
                        # GPD tail extrapolation (F4)
                        *(
                            [dmc.Paper([
                                dmc.Group([
                                    dmc.Text("GPD Tail Extrapolation", fw=600, size="sm"),
                                    dmc.Badge("Generalized Pareto", color="violet",
                                              variant="light", size="xs"),
                                ], justify="space-between", mb="xs"),
                                dmc.Text(
                                    "Fits a Generalized Pareto Distribution to tail losses for "
                                    "more reliable VaR extrapolation beyond empirical percentiles.",
                                    size="xs", c="dimmed", mb="sm",
                                ),
                                dmc.SimpleGrid(
                                    cols=min(n, 2),
                                    children=[
                                        _graph(figure=gpd_fit_chart(
                                            summaries[i].get("cumulative_profit_dist", []),
                                            summaries[i].get("gpd_tail"),
                                            names[i],
                                            STRATEGY_COLORS[i % len(STRATEGY_COLORS)],
                                        ))
                                        for i in range(n)
                                        if summaries[i].get("gpd_tail", {}).get("fit_successful")
                                    ] or [dmc.Text(
                                        "GPD fit requires >= 500 paths with sufficient tail data.",
                                        size="sm", c="dimmed", ta="center",
                                        style={"padding": "20px 0"},
                                    )],
                                ),
                            ], p="md", radius="md", withBorder=True)]
                            if any(s.get("gpd_tail", {}).get("fit_successful") for s in summaries)
                            else []
                        ),
                        # Correlation heatmaps
                        dmc.Paper([
                            dmc.Text("Realized Correlation Matrix", fw=600, size="sm", mb="xs"),
                            dmc.Text(
                                "Cross-metric correlations across simulated paths. Identifies "
                                "which risk drivers move together in the Monte Carlo model.",
                                size="xs", c="dimmed", mb="sm",
                            ),
                            dmc.SimpleGrid(
                                cols=min(n, 2),
                                children=[
                                    _graph(figure=correlation_heatmap(
                                        results.market, ins_list[i], names[i],
                                    ))
                                    for i in range(n)
                                ],
                            ),
                        ], p="md", radius="md", withBorder=True),
                    ], gap="md"),
                    value="risk", pt="md",
                ),

                # --- VALIDATION (F5) ---
                dmc.TabsPanel(
                    dmc.Stack([
                        dmc.Paper([
                            dmc.Text("Model Validation", fw=700, size="lg", mb="xs"),
                            dmc.Text(
                                "Formal diagnostic checks on the AR(2) market cycle model and "
                                "loss distribution calibration. Standard actuarial validation suite.",
                                size="sm", c="dimmed",
                            ),
                        ], p="md", radius="md", withBorder=True),
                        # AR(2) Residual diagnostics
                        dmc.Paper([
                            dmc.Group([
                                dmc.Text("AR(2) Residual Diagnostics", fw=600, size="sm"),
                                dmc.Badge("IID Normal Test", color="blue",
                                          variant="light", size="xs"),
                            ], justify="space-between", mb="xs"),
                            dmc.Text(
                                "Residuals from the AR(2) market model should be IID Normal if the "
                                "model is correctly specified. Significant autocorrelation or "
                                "non-normality suggests model misspecification.",
                                size="xs", c="dimmed", mb="sm",
                            ),
                            _graph(figure=residual_analysis_chart(
                                results.market.get("residuals", np.zeros((1, ny))), ny,
                            )),
                        ], p="md", radius="md", withBorder=True),
                        # PIT histograms
                        dmc.Paper([
                            dmc.Group([
                                dmc.Text("Probability Integral Transform (PIT)", fw=600, size="sm"),
                                dmc.Badge("Uniformity Test", color="green",
                                          variant="light", size="xs"),
                            ], justify="space-between", mb="xs"),
                            dmc.Text(
                                "PIT values should be uniform(0,1) if the gross loss ratio model "
                                "is correctly calibrated. Deviations from uniformity indicate "
                                "systematic model bias.",
                                size="xs", c="dimmed", mb="sm",
                            ),
                            dmc.SimpleGrid(
                                cols=min(n, 3),
                                children=[
                                    _graph(figure=pit_histogram_chart(
                                        ins_list[i]["gross_lr"], names[i],
                                    ))
                                    for i in range(n)
                                ],
                            ),
                        ], p="md", radius="md", withBorder=True),
                        # VaR backtest
                        dmc.Paper([
                            dmc.Group([
                                dmc.Text("VaR(95%) Backtest", fw=600, size="sm"),
                                dmc.Badge("Kupiec Test", color="orange",
                                          variant="light", size="xs"),
                            ], justify="space-between", mb="xs"),
                            dmc.Text(
                                "For each year, counts how many paths breach the predicted VaR(95%). "
                                "The actual breach rate should hover around 5%. Persistent "
                                "over/under-prediction indicates VaR model bias.",
                                size="xs", c="dimmed", mb="sm",
                            ),
                            dmc.SimpleGrid(
                                cols=min(n, 2),
                                children=[
                                    _graph(figure=var_backtest_chart(
                                        ins_list[i], ny, names[i],
                                        STRATEGY_COLORS[i % len(STRATEGY_COLORS)],
                                    ))
                                    for i in range(n)
                                ],
                            ),
                        ], p="md", radius="md", withBorder=True),
                        # QQ plot (moved from Risk tab)
                        dmc.Paper([
                            dmc.Text("Distribution Fit Validation", fw=600, size="sm", mb="xs"),
                            dmc.Text(
                                "QQ plot comparing simulated gross loss ratio against a LogNormal "
                                "theoretical fit. K-S test p-value flags significant deviations.",
                                size="xs", c="dimmed", mb="sm",
                            ),
                            _graph(figure=qq_plot_chart(
                                [ins["gross_lr"][:, -1] for ins in ins_list], names,
                            )),
                        ], p="md", radius="md", withBorder=True),
                    ], gap="md"),
                    value="validation", pt="md",
                ),

                # --- DRILL-DOWN ---
                dmc.TabsPanel(
                    dmc.Stack([
                        dmc.Group([
                            dmc.Select(
                                id="path-select",
                                label="Select Path",
                                data=path_options,
                                value="median",
                                size="sm",
                                style={"width": 250},
                            ),
                        ]),
                        _graph(
                            figure=_get_drilldown_figure(results, "median"),
                            id="drilldown-chart",
                        ),
                        # Year-by-year table
                        dmc.Paper([
                            dmc.Text("Year-by-Year Metrics", fw=600, size="sm", mb="xs"),
                            html.Div(
                                id="drilldown-table",
                                children=_get_drilldown_table(results, "median"),
                            ),
                        ], p="md", radius="md", withBorder=True),
                    ], gap="md"),
                    value="drilldown", pt="md",
                ),

                # --- BACKTEST ---
                dmc.TabsPanel(
                    _backtest_tab_content(bt_list, cf_decomp, names)
                    if has_backtest
                    else dmc.Alert(
                        "Historical backtest could not be computed. Check insurer parameters.",
                        title="Backtest Unavailable", color="yellow", variant="light",
                    ),
                    value="backtest", pt="md",
                ),

                # --- STRATEGY LAB ---
                dmc.TabsPanel(
                    dmc.Stack([
                        dmc.Paper([
                            dmc.Text("Strategy Lab", fw=700, size="lg", mb="xs"),
                            dmc.Text(
                                "Searches hundreds of strategy combinations across 8 dimensions using "
                                "Latin Hypercube Sampling, evaluates each with Monte Carlo simulation, "
                                "and identifies the Pareto-optimal risk-return frontier for each market regime.",
                                size="sm", c="dimmed", mb="md",
                            ),
                            dmc.Group([
                                dmc.Stack([
                                    dmc.Text("Candidates to evaluate", size="xs", fw=500),
                                    dmc.Slider(
                                        id="opt-n-candidates", min=50, max=500,
                                        value=200, step=50, size="sm",
                                        marks=[
                                            {"value": 50, "label": "50"},
                                            {"value": 200, "label": "200"},
                                            {"value": 500, "label": "500"},
                                        ],
                                        style={"width": "300px"},
                                    ),
                                ], gap=4),
                                dmc.Button(
                                    "Run Strategy Lab", id="run-optimizer-btn",
                                    variant="filled", color="violet", size="md",
                                    leftSection=None,
                                ),
                            ], align="flex-end", gap="xl"),
                        ], p="md", radius="md", withBorder=True),
                        dcc.Loading(
                            id="optimizer-loading",
                            type="default",
                            children=html.Div(id="optimizer-results", children=[
                                dmc.Text(
                                    "Press 'Run Strategy Lab' to search for optimal strategies. "
                                    "This typically takes 30-90 seconds depending on candidate count.",
                                    size="sm", c="dimmed", ta="center",
                                    style={"padding": "60px 0"},
                                ),
                            ]),
                        ),
                        # RI Efficient Frontier (F5)
                        dmc.Paper([
                            dmc.Group([
                                dmc.Text("Reinsurance Efficient Frontier", fw=600, size="sm"),
                                dmc.Button(
                                    "Run RI Sweep", id="run-ri-frontier-btn",
                                    variant="light", color="teal", size="xs",
                                ),
                            ], justify="space-between", mb="xs"),
                            dmc.Text(
                                "Sweeps cession percentage from 0-50% and plots the risk-return "
                                "trade-off. Helps identify the optimal reinsurance level.",
                                size="xs", c="dimmed", mb="sm",
                            ),
                            dcc.Loading(
                                id="ri-frontier-loading",
                                type="default",
                                children=html.Div(id="ri-frontier-results", children=[
                                    dmc.Text(
                                        "Press 'Run RI Sweep' to evaluate the reinsurance trade-off curve.",
                                        size="sm", c="dimmed", ta="center",
                                        style={"padding": "40px 0"},
                                    ),
                                ]),
                            ),
                        ], p="md", radius="md", withBorder=True),
                    ], gap="md"),
                    value="strategy-lab", pt="md",
                ),

                # --- METHODOLOGY ---
                dmc.TabsPanel(
                    _methodology_tab(results.config),
                    value="methodology", pt="md",
                ),

                # --- SCENARIOS (F1) ---
                dmc.TabsPanel(
                    dmc.Stack([
                        dmc.Paper([
                            dmc.Text("Scenario Comparison", fw=700, size="lg", mb="xs"),
                            dmc.Text(
                                "Save simulation snapshots and compare key metrics across runs. "
                                "Snapshots persist in your browser's localStorage (up to 5).",
                                size="sm", c="dimmed", mb="md",
                            ),
                        ], p="md", radius="md", withBorder=True),
                        html.Div(id="scenario-snapshot-list"),
                        html.Div(id="scenario-delta-area"),
                    ], gap="md"),
                    value="scenarios", pt="md",
                ),
            ],
        ),
    ], gap="lg")

    return content


# ---------------------------------------------------------------------------
# Drill-down path selector callback
# ---------------------------------------------------------------------------
@callback(
    Output("drilldown-chart", "figure"),
    Output("drilldown-table", "children"),
    Input("path-select", "value"),
    State("run-btn", "n_clicks"),
    prevent_initial_call=True,
)
def update_drilldown(path_value, n_clicks):
    if not n_clicks or path_value is None:
        return no_update, no_update
    from ui.state import _cache
    results = _cache.get("results")
    if results is None:
        return no_update, no_update
    return (
        _get_drilldown_figure(results, path_value),
        _get_drilldown_table(results, path_value),
    )


def _resolve_path_idx(results, path_value: str) -> int:
    """Resolve a path selector value to an array index (N-strategy)."""
    cum_first = results.insurers[0]["cumulative_profit"][:, -1]
    n_paths = len(cum_first)

    if path_value.startswith("best_"):
        pfx = path_value[5:]
        idx = ord(pfx) - ord('a')
        if 0 <= idx < len(results.insurers):
            return int(np.argmax(results.insurers[idx]["cumulative_profit"][:, -1]))
    elif path_value.startswith("worst_"):
        pfx = path_value[6:]
        idx = ord(pfx) - ord('a')
        if 0 <= idx < len(results.insurers):
            return int(np.argmin(results.insurers[idx]["cumulative_profit"][:, -1]))
    elif path_value == "median":
        return int(np.argsort(cum_first)[n_paths // 2])
    else:
        try:
            idx = int(path_value)
        except (ValueError, TypeError):
            idx = n_paths // 2
        if idx < 0 or idx >= n_paths:
            idx = n_paths // 2
        return idx

    # Fallback
    return int(np.argsort(cum_first)[n_paths // 2])


def _get_drilldown_figure(results, path_value: str):
    """Build drill-down figure for a selected path."""
    idx = _resolve_path_idx(results, path_value)
    n = len(results.insurers)
    names = [results.config.insurers[i].name for i in range(n)]
    return single_path_chart(
        results.market, list(results.insurers),
        idx, results.config.n_years, names,
    )


def _get_drilldown_table(results, path_value: str):
    """Build drill-down year-by-year table for a selected path."""
    idx = _resolve_path_idx(results, path_value)
    n = len(results.insurers)
    names = [results.config.insurers[i].name for i in range(n)]
    return drilldown_table(
        results.market, list(results.insurers),
        idx, results.config.n_years, names,
    )


def _backtest_tab_content(bt_list, cf_decomp, names):
    """Build the full Backtest tab layout (N strategies)."""
    if not bt_list:
        return dmc.Alert(
            "No backtest data available.",
            title="Backtest Unavailable", color="yellow", variant="light",
        )

    n = len(bt_list)
    children = [
        backtest_summary_cards(bt_list, names),
        _graph(figure=historical_cr_chart(bt_list, LLOYDS_HISTORICAL, names)),
        _graph(figure=historical_cumulative_chart(bt_list, names)),
    ]
    # Counterfactual decomposition only for 2 strategies
    if cf_decomp is not None and n == 2:
        children.extend([
            counterfactual_verdict(cf_decomp, names[0], names[1]),
            _graph(figure=counterfactual_waterfall(cf_decomp)),
        ])
    children.append(
        dmc.Paper([
            dmc.Text("Year-by-Year Backtest Detail", fw=600, size="sm", mb="xs"),
            backtest_year_table(bt_list, LLOYDS_HISTORICAL, names),
        ], p="md", radius="md", withBorder=True),
    )
    return dmc.Stack(children, gap="md")


# ---------------------------------------------------------------------------
# Strategy Lab optimizer callback
# ---------------------------------------------------------------------------
@callback(
    Output("optimizer-results", "children"),
    Input("run-optimizer-btn", "n_clicks"),
    State("opt-n-candidates", "value"),
    *[State(id, "value") for id in ALL_INPUT_IDS],
    prevent_initial_call=True,
)
def run_strategy_lab(n_clicks, n_candidates, *input_values):
    if not n_clicks:
        return no_update

    import warnings, traceback, os

    params = dict(zip(ALL_INPUT_IDS, input_values))
    n_candidates = int(n_candidates or 200)
    n = int(params.get("n_strategies", 2))

    try:
        config = build_config(params)
        base_cfg = build_optimizer_base_config(params)

        from cyclesim.optimizer import run_full_optimization

        # Get current summaries from cache for gap analysis
        from ui.state import _cache
        cached = _cache.get("results")
        names = [config.insurers[i].name for i in range(min(n, len(config.insurers)))]
        summaries = cached.summaries[:n] if cached else []

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            full_opt = run_full_optimization(
                base_config=base_cfg,
                n_candidates=n_candidates,
                n_paths=500,
                n_years=int(params.get("n_years", 25)),
                max_workers=min(os.cpu_count() or 8, 16),
                current_summaries=summaries if summaries else None,
            )

        # Build results layout
        children = []
        children.append(optimizer_stats_badge(full_opt))
        children.append(optimizer_gap_cards(full_opt, names))

        # Pareto frontier chart
        children.append(
            _graph(figure=pareto_frontier_chart(full_opt, summaries, names)),
        )

        # CUO Playbook
        _strat_keys = [
            "growth_rate_when_profitable", "shrink_rate_when_unprofitable",
            "max_gwp_growth_pa", "base_cession_pct",
            "cession_cycle_sensitivity", "expected_lr",
            "max_gwp_shrink_pa", "adverse_selection_sensitivity",
        ]
        current_strategies = [
            {k: getattr(config.insurers[i], k, None) for k in _strat_keys}
            for i in range(min(n, len(config.insurers)))
        ]

        children.append(
            dmc.Paper([
                dmc.Text("CUO Decision Matrix", fw=600, size="sm", mb="xs"),
                dmc.Text(
                    "Optimal strategy parameters for each market regime, "
                    "compared against your current settings.",
                    size="xs", c="dimmed", mb="sm",
                ),
                playbook_table(full_opt, current_strategies, names),
            ], p="md", radius="md", withBorder=True),
        )

        # Strategy DNA radar (compare each strategy vs unconditional optimal)
        if full_opt.unconditional.best_rorac:
            from cyclesim.optimizer import PARAM_BOUNDS, PARAM_NAMES
            radar_children = []
            for i in range(min(n, len(config.insurers))):
                current_i = {k: getattr(config.insurers[i], k, None) for k in _strat_keys}
                radar_children.append(_graph(figure=strategy_dna_radar(
                    current_i,
                    full_opt.unconditional.best_rorac.params,
                    dict(zip(PARAM_NAMES, PARAM_BOUNDS.tolist())),
                )))
            children.append(dmc.SimpleGrid(cols=min(n, 3), children=radar_children))

        return dmc.Stack(children, gap="md")

    except Exception as e:
        traceback.print_exc()
        return dmc.Alert(
            f"Strategy Lab failed: {e}",
            title="Error", color="red", variant="light",
        )


# ---------------------------------------------------------------------------
# Reset Strategy Lab when main simulation is re-run
# ---------------------------------------------------------------------------
@callback(
    Output("optimizer-results", "children", allow_duplicate=True),
    Input("sim-generation", "data"),
    prevent_initial_call=True,
)
def reset_strategy_lab_on_rerun(_gen):
    """Clear stale optimizer results whenever the main simulation re-runs."""
    return dmc.Text(
        "Press 'Run Strategy Lab' to search for optimal strategies. "
        "This typically takes 30-90 seconds depending on candidate count.",
        size="sm", c="dimmed", ta="center",
        style={"padding": "60px 0"},
    )


# ---------------------------------------------------------------------------
# Excel export callback
# ---------------------------------------------------------------------------
@callback(
    Output("download-excel", "data"),
    Input("export-btn", "n_clicks"),
    prevent_initial_call=True,
)
def export_excel(n_clicks):
    if not n_clicks:
        return no_update
    from ui.state import _cache
    results = _cache.get("results")
    if results is None:
        return no_update
    try:
        from io import BytesIO
        buf = BytesIO()
        export_results_to_excel(results, buf)
        buf.seek(0)
        return dcc.send_bytes(buf.getvalue(), "cyclesim_results.xlsx")
    except Exception:
        import traceback
        traceback.print_exc()
        return no_update


# ---------------------------------------------------------------------------
# CSV raw path export callback
# ---------------------------------------------------------------------------
@callback(
    Output("download-csv-paths", "data"),
    Input("export-csv-btn", "n_clicks"),
    prevent_initial_call=True,
)
def export_csv_paths(n_clicks):
    if not n_clicks:
        return no_update
    from ui.state import _cache
    results = _cache.get("results")
    if results is None:
        return no_update
    try:
        from io import BytesIO
        buf = BytesIO()
        export_raw_paths_csv(results, buf, max_paths=1000)
        buf.seek(0)
        return dcc.send_bytes(buf.getvalue(), "cyclesim_raw_paths.csv")
    except Exception:
        import traceback
        traceback.print_exc()
        return no_update


# ---------------------------------------------------------------------------
# Scenario snapshot save callback (F1)
# ---------------------------------------------------------------------------
@callback(
    Output("scenario-snapshots", "data"),
    Input("save-snapshot-btn", "n_clicks"),
    State("scenario-snapshots", "data"),
    State("sim-generation", "data"),
    prevent_initial_call=True,
)
def save_snapshot(n_clicks, snapshots, sim_gen):
    if not n_clicks:
        return no_update
    from ui.state import _cache
    results = _cache.get("results")
    if results is None:
        return no_update
    import datetime
    n = len(results.summaries)
    names = [results.config.insurers[i].name for i in range(n)]
    # Serialize summaries (strip numpy arrays, keep scalars)
    serializable_summaries = []
    for s in results.summaries:
        entry = {}
        for k, v in s.items():
            if isinstance(v, (int, float, str, bool, type(None))):
                entry[k] = v
            elif hasattr(v, 'item'):  # numpy scalar
                entry[k] = float(v)
        serializable_summaries.append(entry)
    snap = {
        "name": f"Run #{sim_gen or 1}",
        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_paths": results.config.n_paths,
        "n_years": results.config.n_years,
        "strategy_names": names,
        "summaries": serializable_summaries,
    }
    snapshots = list(snapshots or [])
    snapshots.append(snap)
    if len(snapshots) > 5:
        snapshots = snapshots[-5:]
    return snapshots


# ---------------------------------------------------------------------------
# Scenario snapshot render + delta callback (F1)
# ---------------------------------------------------------------------------
@callback(
    Output("scenario-snapshot-list", "children"),
    Output("scenario-delta-area", "children"),
    Input("scenario-snapshots", "data"),
    prevent_initial_call=True,
)
def render_scenarios(snapshots):
    snapshots = snapshots or []
    if not snapshots:
        return (
            dmc.Text("No snapshots saved yet. Click 'Save Snapshot' after a run.",
                     size="sm", c="dimmed"),
            html.Div(),
        )
    # Snapshot list table
    header = dmc.TableThead(dmc.TableTr([
        dmc.TableTh("#"), dmc.TableTh("Name"), dmc.TableTh("Time"),
        dmc.TableTh("Paths"), dmc.TableTh("Strategies"),
        dmc.TableTh("Action"),
    ]))
    rows = []
    for i, snap in enumerate(snapshots):
        rows.append(dmc.TableTr([
            dmc.TableTd(str(i + 1)),
            dmc.TableTd(dmc.Text(snap.get("name", "?"), fw=500, size="sm")),
            dmc.TableTd(dmc.Text(snap.get("timestamp", ""), size="xs", ff="monospace")),
            dmc.TableTd(dmc.Text(f"{snap.get('n_paths', ''):,}", size="xs", ff="monospace")),
            dmc.TableTd(dmc.Text(", ".join(snap.get("strategy_names", [])), size="xs")),
            dmc.TableTd(dmc.Button(
                "Delete", id={"type": "delete-snapshot", "index": i},
                variant="subtle", color="red", size="xs", compact=True,
            )),
        ]))
    snap_table = dmc.Table(
        [header, dmc.TableTbody(rows)],
        striped=True, highlightOnHover=True,
        withTableBorder=True,
    )
    # Delta comparison
    delta = scenario_delta_table(snapshots) if len(snapshots) >= 2 else html.Div()
    return snap_table, delta


# ---------------------------------------------------------------------------
# Delete snapshot callback (F1)
# ---------------------------------------------------------------------------
@callback(
    Output("scenario-snapshots", "data", allow_duplicate=True),
    Input({"type": "delete-snapshot", "index": dash.ALL}, "n_clicks"),
    State("scenario-snapshots", "data"),
    prevent_initial_call=True,
)
def delete_snapshot(n_clicks_list, snapshots):
    if not any(n_clicks_list):
        return no_update
    snapshots = list(snapshots or [])
    # Find which button was clicked
    triggered = ctx.triggered_id
    if triggered and isinstance(triggered, dict):
        idx = triggered.get("index")
        if idx is not None and 0 <= idx < len(snapshots):
            snapshots.pop(idx)
    return snapshots


# ---------------------------------------------------------------------------
# RI Efficient Frontier callback (F5)
# ---------------------------------------------------------------------------
@callback(
    Output("ri-frontier-results", "children"),
    Input("run-ri-frontier-btn", "n_clicks"),
    *[State(id, "value") for id in ALL_INPUT_IDS],
    prevent_initial_call=True,
)
def run_ri_frontier(n_clicks, *input_values):
    if not n_clicks:
        return no_update
    import warnings, traceback
    params = dict(zip(ALL_INPUT_IDS, input_values))
    # Get current cession for marker
    current_cession = float(params.get("a_cession", 0.25))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            sweep = run_ri_sweep(params, n_points=11, n_paths_sweep=300)
        return _graph(figure=ri_frontier_chart(sweep, current_cession=current_cession))
    except Exception as e:
        traceback.print_exc()
        return dmc.Alert(
            f"RI frontier sweep failed: {e}",
            title="Error", color="red", variant="light",
        )


# ---------------------------------------------------------------------------
# Reset RI frontier when main simulation re-runs
# ---------------------------------------------------------------------------
@callback(
    Output("ri-frontier-results", "children", allow_duplicate=True),
    Input("sim-generation", "data"),
    prevent_initial_call=True,
)
def reset_ri_frontier_on_rerun(_gen):
    return dmc.Text(
        "Press 'Run RI Sweep' to evaluate the reinsurance trade-off curve.",
        size="sm", c="dimmed", ta="center",
        style={"padding": "40px 0"},
    )


# ---------------------------------------------------------------------------
# Scenario preset / profile load callback — merged (same outputs)
# ---------------------------------------------------------------------------
@callback(
    *[Output(id, "value") for id in ALL_INPUT_IDS],
    Input("scenario-preset", "value"),
    Input("load-profile-btn", "n_clicks"),
    State("load-profile", "value"),
    State("saved-profiles", "data"),
    prevent_initial_call=True,
)
def apply_preset_or_profile(preset_name, load_clicks, profile_name, profiles):
    trigger = ctx.triggered_id
    if trigger == "scenario-preset":
        if not preset_name or preset_name not in SCENARIO_PRESETS:
            return tuple(no_update for _ in ALL_INPUT_IDS)
        preset = SCENARIO_PRESETS[preset_name]
        return tuple(preset.get(id, no_update) for id in ALL_INPUT_IDS)
    elif trigger == "load-profile-btn":
        if not profile_name or not profiles or profile_name not in profiles:
            return tuple(no_update for _ in ALL_INPUT_IDS)
        profile = profiles[profile_name]
        return tuple(profile.get(id, no_update) for id in ALL_INPUT_IDS)
    return tuple(no_update for _ in ALL_INPUT_IDS)


# ---------------------------------------------------------------------------
# Preset description callback
# ---------------------------------------------------------------------------
@callback(
    Output("preset-description", "children"),
    Input("scenario-preset", "value"),
    prevent_initial_call=True,
)
def update_preset_description(preset_name):
    if not preset_name or preset_name not in PRESET_DESCRIPTIONS:
        return dmc.Text("Select a scenario to auto-configure all parameters.",
                        size="xs", c="dimmed")
    return dmc.Text(PRESET_DESCRIPTIONS[preset_name], size="xs", c="dimmed",
                    fs="italic")


# ---------------------------------------------------------------------------
# Save profile callback
# ---------------------------------------------------------------------------
@callback(
    Output("saved-profiles", "data"),
    Output("profile-status", "children"),
    Output("load-profile", "data"),
    Output("profile-name-input", "value"),
    Input("save-profile-btn", "n_clicks"),
    State("profile-name-input", "value"),
    State("saved-profiles", "data"),
    *[State(id, "value") for id in ALL_INPUT_IDS],
    prevent_initial_call=True,
)
def save_profile(n_clicks, name, existing_profiles, *slider_values):
    if not name or not name.strip():
        return (no_update,
                dmc.Text("Enter a name first", size="xs", c="red"),
                no_update, no_update)
    profiles = dict(existing_profiles or {})
    profile = {pid: val for pid, val in zip(ALL_INPUT_IDS, slider_values)}
    profiles[name.strip()] = profile
    options = [{"value": k, "label": k} for k in sorted(profiles.keys())]
    return (profiles,
            dmc.Text(f"Saved '{name.strip()}'", size="xs", c="green"),
            options, "")


# ---------------------------------------------------------------------------
# Delete profile callback
# ---------------------------------------------------------------------------
@callback(
    Output("saved-profiles", "data", allow_duplicate=True),
    Output("profile-status", "children", allow_duplicate=True),
    Output("load-profile", "data", allow_duplicate=True),
    Output("load-profile", "value"),
    Input("delete-profile-btn", "n_clicks"),
    State("load-profile", "value"),
    State("saved-profiles", "data"),
    prevent_initial_call=True,
)
def delete_profile(n_clicks, selected, profiles):
    if not selected or not profiles or selected not in profiles:
        return (no_update,
                dmc.Text("Select a profile first", size="xs", c="red"),
                no_update, no_update)
    profiles = dict(profiles)
    del profiles[selected]
    options = [{"value": k, "label": k} for k in sorted(profiles.keys())]
    return (profiles,
            dmc.Text(f"Deleted '{selected}'", size="xs", c="orange"),
            options, None)


# ---------------------------------------------------------------------------
# Refresh profile dropdown from localStorage on page load
# ---------------------------------------------------------------------------
@callback(
    Output("load-profile", "data", allow_duplicate=True),
    Input("saved-profiles", "modified_timestamp"),
    State("saved-profiles", "data"),
    prevent_initial_call=True,
)
def refresh_profile_dropdown(ts, profiles):
    if not profiles:
        return []
    return [{"value": k, "label": k} for k in sorted(profiles.keys())]


# ---------------------------------------------------------------------------
# Print methodology (clientside — triggers browser print dialog)
# ---------------------------------------------------------------------------
app.clientside_callback(
    "function(n) { if (n) { window.print(); } return window.dash_clientside.no_update; }",
    Output("print-methodology-btn", "n_clicks"),
    Input("print-methodology-btn", "n_clicks"),
    prevent_initial_call=True,
)


# ---------------------------------------------------------------------------
# Capital model upload callback
# ---------------------------------------------------------------------------
@callback(
    Output("cm-status", "children"),
    Input("capital-model-upload", "contents"),
    Input("load-sample-cm", "n_clicks"),
    Input("clear-cm", "n_clicks"),
    State("capital-model-upload", "filename"),
    prevent_initial_call=True,
)
def handle_capital_model(contents, load_sample, clear_clicks, filename):
    trigger = ctx.triggered_id
    if trigger == "clear-cm":
        clear_capital_model()
        return dmc.Text("Cleared \u2014 using parametric model", size="xs", c="dimmed")

    if trigger == "load-sample-cm":
        df = generate_sample_capital_model(n_sims=10000, seed=123)
        from io import BytesIO
        buf = BytesIO()
        df.to_csv(buf, index=False)
        buf.seek(0)
        cm = import_capital_model(buf)
        n = len(cm.gross_loss_ratios)
        mean_glr = float(cm.gross_loss_ratios.mean())
        mean_nlr = float(cm.net_loss_ratios.mean())
        p99 = float(np.percentile(cm.gross_loss_ratios, 99))
        set_capital_model(cm, f"sample_{n}")
        return dmc.Stack([
            dmc.Badge(f"Sample: {n:,} sims loaded", color="violet", size="sm"),
            dmc.Text(f"Mean gross: {mean_glr:.1%} | Net: {mean_nlr:.1%} | 1-in-100: {p99:.1%}",
                     size="xs", c="dimmed"),
            _graph(figure=capital_model_preview(cm.gross_loss_ratios, cm.net_loss_ratios),
                      style={"height": "200px"}, config={"displayModeBar": False}),
        ], gap=2)

    if trigger == "capital-model-upload" and contents:
        import base64
        from io import BytesIO
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        buf = BytesIO(decoded)
        try:
            cm = import_capital_model(buf)
            n = len(cm.gross_loss_ratios)
            mean_glr = float(cm.gross_loss_ratios.mean())
            mean_nlr = float(cm.net_loss_ratios.mean())
            p99 = float(np.percentile(cm.gross_loss_ratios, 99))
            set_capital_model(cm, f"{filename}_{n}")
            return dmc.Stack([
                dmc.Badge(f"{filename}: {n:,} sims loaded", color="green", size="sm"),
                dmc.Text(f"Mean gross: {mean_glr:.1%} | Net: {mean_nlr:.1%} | 1-in-100: {p99:.1%}",
                         size="xs", c="dimmed"),
                _graph(figure=capital_model_preview(cm.gross_loss_ratios, cm.net_loss_ratios),
                          style={"height": "200px"}, config={"displayModeBar": False}),
            ], gap=2)
        except Exception as e:
            return dmc.Alert(str(e), color="red", variant="light", title="Import Error")

    return no_update


# ---------------------------------------------------------------------------
# Toggle AR(2) mode controls
# ---------------------------------------------------------------------------
@callback(
    Output("market-auto-controls", "style"),
    Output("market-direct-controls", "style"),
    Input("ar2_mode", "value"),
)
def toggle_ar2_mode(mode):
    if mode == "direct":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


# ---------------------------------------------------------------------------
# N-strategies visibility callback — show/hide insurer sections C-F
# ---------------------------------------------------------------------------
@callback(
    *[Output(f"insurer-{pfx}-section", "style") for pfx in STRATEGY_PREFIXES[2:]],
    Input("n_strategies", "value"),
)
def toggle_strategy_sections(n_strat):
    n_strat = int(n_strat or 2)
    # STRATEGY_PREFIXES[2:] = ["c", "d", "e", "f"]
    # c visible when n >= 3, d when n >= 4, etc.
    return tuple(
        {"display": "block"} if (i + 3) <= n_strat else {"display": "none"}
        for i in range(len(STRATEGY_PREFIXES) - 2)
    )


# ---------------------------------------------------------------------------
# Slider display value callbacks
# ---------------------------------------------------------------------------
_PCT_SLIDERS = {
    "min_lr", "max_lr", "long_run_lr", "shock_prob",
    "attritional_mean", "discount_rate",
    "risk_appetite_ruin_max", "risk_appetite_cr_max", "risk_appetite_rorac_min",
}
for _pfx in STRATEGY_PREFIXES:
    _PCT_SLIDERS.update({
        f"{_pfx}_expense", f"{_pfx}_cession", f"{_pfx}_inv_return", f"{_pfx}_expected_lr",
        f"{_pfx}_growth", f"{_pfx}_shrink", f"{_pfx}_max_growth", f"{_pfx}_max_shrink",
        f"{_pfx}_min_cess", f"{_pfx}_max_cess", f"{_pfx}_cap_ratio", f"{_pfx}_coc",
        f"{_pfx}_sw_own_lr", f"{_pfx}_sw_market_lr", f"{_pfx}_sw_rate_adequacy",
        f"{_pfx}_sw_rate_change", f"{_pfx}_sw_capital",
        f"{_pfx}_expense_fixed_pct",
    })
_INT_SLIDERS = {"n_paths", "n_years", "n_strategies"}
_GWP_SLIDERS = {f"{_pfx}_gwp" for _pfx in STRATEGY_PREFIXES}

_SLIDER_IDS = [
    sid for sid in ALL_INPUT_IDS
    if sid not in {"ar2_mode", "seed"} and not sid.endswith("_name")
]


def _fmt_slider(val, sid):
    if val is None:
        return ""
    if sid in _INT_SLIDERS:
        return f"{int(val):,}"
    if sid in _GWP_SLIDERS:
        return f"{val / 1e6:.0f}m"
    if sid in _PCT_SLIDERS:
        return f"{val:.0%}"
    return f"{val:.3g}"


for _sid in _SLIDER_IDS:
    _sid_captured = _sid

    @callback(
        Output(f"{_sid}-display", "children"),
        Input(_sid, "value"),
        prevent_initial_call=True,
    )
    def _update_display(val, sid=_sid_captured):
        return _fmt_slider(val, sid)


# ---------------------------------------------------------------------------
# Methodology tab
# ---------------------------------------------------------------------------
def _methodology_tab(config):
    """Chartered actuary-grade technical documentation with model architecture."""

    # -- helpers --
    def _sec(title, body_items, accent=""):
        cls = "methodology-section"
        if accent:
            cls += f" accent-{accent}"
        return dmc.Paper(
            html.Div(
                dmc.Stack([dmc.Text(title, fw=700, size="lg"), *body_items], gap="xs"),
                className=cls,
            ),
            p="lg", radius="md", withBorder=True,
        )

    def _p(text):
        return dmc.Text(text, size="sm", style={"lineHeight": 1.7})

    def _f(label, expr):
        return dmc.Group([
            dmc.Text(label, size="sm", fw=600, style={"minWidth": 180}),
            html.Div(expr, className="formula-block"),
        ], gap="sm", style={"flexWrap": "wrap"})

    def _div(label):
        return dmc.Divider(label=label, labelPosition="left", my="xs")

    def _node(title, sub):
        return html.Div([
            html.Div(title, className="node-title"),
            html.Div(sub, className="node-sub"),
        ], className="methodology-node")

    def _arrow():
        return html.Div("\u2192", className="methodology-arrow")

    def _down_arrow():
        return html.Div("\u2193", className="methodology-return-arrow")

    # -- Architecture diagram --
    flow = html.Div([
        _node("Random Seed", "Reproducibility"),
        _arrow(),
        _node("Market Cycle", "AR(2) + HMM Regime"),
        _arrow(),
        _node("Loss Model", "Attritional + Large + Cat"),
        _arrow(),
        _node("Strategy Signal", "5-Input Weighted Blend"),
        html.Div(className="methodology-row-break"),
        _down_arrow(),
        html.Div(className="methodology-row-break"),
        _node("Risk Metrics", "RORAC, VaR, Ruin"),
        _arrow(),
        _node("Capital Mgmt", "Injection / Dividend"),
        _arrow(),
        _node("P&L", "UW + Investment \u2212 RI Cost"),
        _arrow(),
        _node("GWP & RI", "Growth / Cession Decisions"),
    ], className="methodology-flow")

    # -- Insurer config table --
    ins_rows = []
    for i, ins in enumerate(config.insurers):
        sw = ins.signal_weights
        ins_rows.append(html.Tr([
            html.Td(ins.name, style={"fontWeight": 600}),
            html.Td(f"\u00a3{ins.initial_gwp/1e6:.0f}m"),
            html.Td(f"{ins.expense_ratio:.0%}"),
            html.Td(f"{ins.base_cession_pct:.0%}"),
            html.Td(f"{ins.expected_lr:.0%}"),
            html.Td(f"{ins.growth_rate_when_profitable:+.0%} / {ins.shrink_rate_when_unprofitable:+.0%}"),
            html.Td(f"{sw.get('own_lr',0):.0%} / {sw.get('market_lr',0):.0%} / {sw.get('rate_adequacy',0):.0%} / {sw.get('rate_change',0):.0%} / {sw.get('capital_position',0):.0%}"),
        ]))

    mp = config.market_params

    return dmc.Stack([
        # Title
        dmc.Text("Technical Methodology", fw=700, size="xl"),
        _p(
            "This document describes the mathematical framework, calibration sources, "
            "and computational methods used in CycleSim. All parameters are calibrated "
            "to Lloyd\u2019s of London market data unless a user-uploaded capital model is active."
        ),
        dmc.Alert(
            "This is a strategic planning tool, not a reserving or pricing model. "
            "Results are illustrative and should not be used as the sole basis for "
            "regulatory capital calculations or individual risk pricing.",
            title="Intended Use", color="blue", variant="light",
        ),

        # Architecture diagram
        dmc.Paper(
            dmc.Stack([
                dmc.Text("Model Architecture", fw=700, size="lg"),
                _p("Each simulation year proceeds through the following pipeline. "
                   "All operations are vectorized across N paths; only the year loop is sequential."),
                flow,
            ], gap="sm"),
            p="lg", radius="md", withBorder=True,
        ),

        # =====================================================================
        # 1. MODEL OVERVIEW
        # =====================================================================
        _sec("1. Model Overview", [
            _p(
                "CycleSim is a Monte Carlo simulation engine that models the interaction "
                "between insurance market cycles and insurer strategy decisions. The engine "
                "generates N stochastic market paths over T years, then simulates 1\u20136 insurer "
                "strategies simultaneously against each path."
            ),
            _p(
                "The market cycle is modelled as a second-order autoregressive process AR(2) "
                "with a Hidden Markov regime-switching overlay (4 states: soft, firming, hard, "
                "crisis). Loss ratios are sampled from parametric distributions (LogNormal "
                "attritional + Compound Poisson large + Compound Poisson catastrophe) or from "
                "an imported capital model. Each insurer reacts to market signals via a "
                "configurable 5-input weighted signal function."
            ),
            _div("Key Assumptions"),
            dmc.List([
                dmc.ListItem("Paths are independent \u2014 no systemic contagion between simulated insurers"),
                dmc.ListItem("Market cycle is common to all insurers \u2014 they compete in the same market"),
                dmc.ListItem("Year-over-year state dependency: each year\u2019s decisions depend on prior year outcomes"),
                dmc.ListItem("All monetary values in GBP \u2014 no FX risk modelled"),
                dmc.ListItem("Investment returns are deterministic (no interest rate model)"),
                dmc.ListItem("Adverse selection is a simplified linear penalty, not a full selection model"),
            ], size="sm", spacing="xs"),
        ]),

        # =====================================================================
        # 2. MARKET CYCLE
        # =====================================================================
        _sec("2. Market Cycle Generation", [
            _p(
                "The market loss ratio follows an AR(2) process with regime-switching dynamics. "
                "This captures both the cyclical mean-reversion observed in insurance markets "
                "(Venezian 1985, Cummins & Outreville 1987) and the abrupt regime transitions "
                "(soft \u2192 hard \u2192 crisis) documented by Wang et al. (2010)."
            ),

            _div("AR(2) Process"),
            _f("Core equation", "y(t) = \u03c6\u2080 + \u03c6\u2081\u00b7y(t\u22121) + \u03c6\u2082\u00b7y(t\u22122) + \u03c3_vol(regime)\u00b7\u03b5(t),  \u03b5 ~ N(0,1)"),
            _f("Market LR mapping", "market_lr(t) = \u03bc + y_norm(t) \u00d7 (max_lr \u2212 min_lr) / 2"),
            _f("Bound enforcement", "market_lr(t) = clip(market_lr(t), min_lr, max_lr)"),

            _div("Stationarity & Oscillation Conditions"),
            _f("Stationarity", "\u03c6\u2081 + \u03c6\u2082 < 1,  \u03c6\u2082 \u2212 \u03c6\u2081 < 1,  |\u03c6\u2082| < 1"),
            _f("Oscillatory", "\u03c6\u2081\u00b2 + 4\u03c6\u2082 < 0  (complex conjugate roots)"),
            _f("Implied period", "T = 2\u03c0 / arccos(\u03c6\u2081 / (2\u221a(\u2212\u03c6\u2082)))"),

            _div("Auto-Calibration (from cycle period)"),
            _f("\u03c6\u2082 from period", "\u03c6\u2082 = \u22120.5 \u00d7 (\u03c9/\u03c0)^0.5,  clipped to [\u22120.75, \u22120.15],  \u03c9 = 2\u03c0/T"),
            _f("\u03c6\u2081 from period", "\u03c6\u2081 = 2\u221a(\u2212\u03c6\u2082) \u00d7 cos(\u03c9),  enforced \u03c6\u2081+\u03c6\u2082 < 0.99"),
            _f("Variance target", "\u03c3\u00b2 = (max\u2212min)\u00b2/4 \u00d7 (1\u2212\u03c6\u2082)((1+\u03c6\u2082)\u00b2 \u2212 \u03c6\u2081\u00b2)"),
            _p(
                "In \u2018auto\u2019 mode the user sets cycle period, min/max LR, and the engine solves "
                "for \u03c6\u2081, \u03c6\u2082, \u03c3. In \u2018direct\u2019 mode the user specifies \u03c6\u2081, \u03c6\u2082, \u03c3 explicitly."
            ),

            _div("Regime-Switching (Hidden Markov Model)"),
            _p(
                "A 4-state Hidden Markov Model overlays the AR(2) process. States: "
                "Soft (0), Firming (1), Hard (2), Crisis (3). Transitions follow a "
                "4\u00d74 Markov matrix calibrated to Lloyd\u2019s historical regime durations."
            ),
            _f("Transition", "regime(t) ~ Categorical(P[regime(t\u22121), :])"),
            _p("Default transition matrix P:"),
            html.Table([
                html.Thead(html.Tr([html.Th("From \u2193 / To \u2192")] + [html.Th(r) for r in ["Soft", "Firming", "Hard", "Crisis"]])),
                html.Tbody([
                    html.Tr([html.Td("Soft", style={"fontWeight": 600}), html.Td("0.75"), html.Td("0.18"), html.Td("0.05"), html.Td("0.02")]),
                    html.Tr([html.Td("Firming", style={"fontWeight": 600}), html.Td("0.10"), html.Td("0.55"), html.Td("0.30"), html.Td("0.05")]),
                    html.Tr([html.Td("Hard", style={"fontWeight": 600}), html.Td("0.05"), html.Td("0.10"), html.Td("0.75"), html.Td("0.10")]),
                    html.Tr([html.Td("Crisis", style={"fontWeight": 600}), html.Td("0.05"), html.Td("0.25"), html.Td("0.55"), html.Td("0.15")]),
                ]),
            ], style={"fontSize": "13px", "borderCollapse": "collapse", "width": "100%",
                       "border": "1px solid #e2e8f0", "marginBottom": "8px"}),

            _div("Regime Multipliers"),
            html.Table([
                html.Thead(html.Tr([html.Th("Regime"), html.Th("LR Multiplier"), html.Th("Volatility Multiplier"), html.Th("Interpretation")])),
                html.Tbody([
                    html.Tr([html.Td("Soft"), html.Td("1.08 (+8%)"), html.Td("1.0"), html.Td("Deteriorating underwriting standards")]),
                    html.Tr([html.Td("Firming"), html.Td("1.00"), html.Td("0.8"), html.Td("Neutral, lower volatility")]),
                    html.Tr([html.Td("Hard"), html.Td("0.93 (\u22127%)"), html.Td("0.7"), html.Td("Strong discipline, tight pricing")]),
                    html.Tr([html.Td("Crisis"), html.Td("1.25 (+25%)"), html.Td("2.5"), html.Td("Market dislocation, extreme volatility")]),
                ]),
            ], style={"fontSize": "13px", "borderCollapse": "collapse", "width": "100%",
                       "border": "1px solid #e2e8f0", "marginBottom": "8px"}),

            _div("Expected Regime Dwell Times"),
            _p("The expected duration in each regime before transitioning, "
               "derived from the diagonal of the transition matrix: E[dwell] = 1 / (1 - P[i,i])."),
            html.Table([
                html.Thead(html.Tr([html.Th("Regime"), html.Th("P(stay)"), html.Th("E[dwell] (years)"), html.Th("Lloyd's Observed")])),
                html.Tbody([
                    html.Tr([html.Td("Soft"), html.Td("0.75"), html.Td("4.0"), html.Td("3-5 years (2014-2018)")]),
                    html.Tr([html.Td("Firming"), html.Td("0.55"), html.Td("2.2"), html.Td("1-3 years (transitional)")]),
                    html.Tr([html.Td("Hard"), html.Td("0.75"), html.Td("4.0"), html.Td("3-5 years (2019-2023)")]),
                    html.Tr([html.Td("Crisis"), html.Td("0.15"), html.Td("1.2"), html.Td("1-2 years (event-driven)")]),
                ]),
            ], style={"fontSize": "13px", "borderCollapse": "collapse", "width": "100%",
                       "border": "1px solid #e2e8f0", "marginBottom": "8px"}),
            _p("The short crisis dwell time (1.2 years) reflects that crisis states are "
               "event-driven and self-correcting. The symmetric 4-year soft/hard dwell "
               "times match the observed Lloyd's cycle half-period of ~4 years."),

            _div("Exogenous Shocks"),
            _f("Shock probability", "P(shock) = shock_prob per year (default 8%)"),
            _f("Shock impact", "If shock: 50% chance regime flips to Crisis; LR += shock_magnitude \u00d7 N(0,1)"),
            _f("Default magnitude", "\u03c3_shock = 0.12 (12pp standard deviation)"),

            _div("Rate Adequacy & Rate Change"),
            _f("Rate adequacy", "ra(t) = (long_run_lr \u2212 market_lr(t)) / (max_lr \u2212 min_lr) \u00d7 2"),
            _f("Rate change", "rc(t) = soft_rate + (1 \u2212 (lr\u2212min)/(max\u2212min)) \u00d7 (hard_rate \u2212 soft_rate)"),
            _p("Rate adequacy is positive in hard markets (good for writing) and negative in soft markets."),
        ], accent=""),

        # =====================================================================
        # 3. LOSS MODEL
        # =====================================================================
        _sec("3. Loss Model (Parametric)", [
            _p(
                "When no capital model is uploaded, losses are generated from three independent "
                "stochastic components: attritional, large, and catastrophe. All three are "
                "cycle-sensitive \u2014 soft markets produce higher means and fatter tails."
            ),

            _div("Attritional Losses (LogNormal)"),
            _f("Cycle-adjusted mean", "adj_mean = attritional_mean \u00d7 (1 \u2212 cycle_sensitivity \u00d7 rate_adequacy)"),
            _f("Bounds", "adj_mean = clip(adj_mean, 0.20, 0.85)"),
            _f("Tail thickness", "adj_cv = attritional_cv \u00d7 (1 + tail_sensitivity \u00d7 max(\u2212rate_adequacy, 0))"),
            _p("In soft markets (negative rate adequacy), the coefficient of variation increases, "
               "producing fatter tails and more volatile attritional outcomes."),
            _f("LogNormal params", "\u03c3_ln = \u221a(log(1 + CV\u00b2)),  \u03bc_ln = log(mean) \u2212 \u03c3_ln\u00b2/2"),
            _f("Sample", "attritional ~ LogNormal(\u03bc_ln, \u03c3_ln)"),

            _div("Large Losses (Compound Poisson-Pareto)"),
            _f("Freq adjustment", "adj_freq = large_loss_freq \u00d7 (1 + 0.15 \u00d7 max(\u2212rate_adequacy, 0))"),
            _f("Frequency", "N ~ Poisson(adj_freq)"),
            _f("Severity", "X = (Pareto(\u03b1) + 1) \u00d7 x_min,  capped at 25% of GWP"),
            _f("Defaults", "\u03b1 = 1.8,  x_min = 0.01 (1% of GWP),  cap = 0.25"),
            _f("Total", "large_lr = \u03a3 X_i  for i = 1..N"),

            _div("Catastrophe Losses (Compound Poisson-LogNormal)"),
            _f("Frequency", "N_cat ~ Poisson(cat_frequency)  (default 0.30 \u2248 1 every 3 years)"),
            _f("Severity params", "\u03c3_cat = \u221a(log(1 + CV_cat\u00b2)),  \u03bc_cat = log(mean_cat) \u2212 \u03c3_cat\u00b2/2"),
            _f("Severity", "severity ~ LogNormal(\u03bc_cat, \u03c3_cat)  (default mean 8% of GWP, CV 1.5)"),
            _f("Total", "cat_lr = \u03a3 severity_i  for i = 1..N_cat"),

            _div("Gross Loss Ratio"),
            _f("Gross LR", "gross_lr(t) = attritional(t) + large(t) + cat(t)"),

            _div("Net Loss Ratio (Component-Specific Reinsurance)"),
            _p("Different RI effectiveness by component reflects the structure of typical "
               "Lloyd\u2019s reinsurance programmes:"),
            _f("Attritional (QS-like)", "net_att = att \u00d7 (1 \u2212 cession% \u00d7 0.90)"),
            _f("Large (XL-like)", "net_large = large \u00d7 (1 \u2212 cession% \u00d7 0.75)"),
            _f("Cat (Cat XL)", "net_cat = cat \u00d7 (1 \u2212 cession% \u00d7 0.95)"),
            _f("Net LR", "net_lr = net_att + net_large + net_cat"),

            _div("Capital Model Override"),
            _p(
                "When a capital model is uploaded (CSV/Excel with gross_lr and net_lr columns), "
                "loss ratios are sampled directly from the empirical distribution with "
                "quantile-dependent cycle adjustment:"
            ),
            _f("Below median", "cycle_factor = 1 \u2212 cycle_sensitivity \u00d7 1.0 \u00d7 rate_adequacy  (attritional-dominated)"),
            _f("50th\u201390th pctile", "cycle_factor = 1 \u2212 cycle_sensitivity \u00d7 0.7 \u00d7 rate_adequacy  (large-loss territory)"),
            _f("Above 90th pctile", "cycle_factor = 1 \u2212 cycle_sensitivity \u00d7 0.3 \u00d7 rate_adequacy  (cat/extreme)"),
        ], accent="green"),

        # =====================================================================
        # 4. RESERVE DEVELOPMENT
        # =====================================================================
        _sec("4. Prior-Year Reserve Development", [
            _p(
                "Prior-year reserves develop each year with a random component plus a "
                "systematic penalty for underwriting years written in soft markets. The penalty "
                "is lagged by 3 years, reflecting the typical emergence period for adverse development."
            ),
            _f("Base development", "dev ~ N(0, 0.02) per year  (2pp std dev)"),
            _f("Soft market penalty", "dev += reserve_dev_soft_penalty \u00d7 max(\u2212rate_adequacy_{t\u22123}, 0)"),
            _f("Default penalty", "0.03 (3pp per unit of soft-market underwriting, lagged 3 years)"),
            _p(
                "Interpretation: a year written at rate_adequacy = \u22121.0 (deep soft market) "
                "will develop 3pp adversely when that cohort matures 3 years later, in addition "
                "to the random (\u00b12pp) base development."
            ),
        ], accent="amber"),

        # =====================================================================
        # 5. STRATEGY SIGNAL
        # =====================================================================
        _sec("5. Strategy Signal Model", [
            _p(
                "Each insurer reacts to a weighted blend of 5 market signals. The blended "
                "signal determines whether the insurer grows or shrinks its book. Positive "
                "signal = \u201cthings are good\u201d = grow; negative = \u201cthings are bad\u201d = shrink."
            ),

            _div("Five Signal Components"),
            _f("1. Own LR", "own_signal = (expected_lr \u2212 own_lr) / expected_lr"),
            _p("Below expected LR = positive (profitable). Own results are the strongest signal for most insurers."),
            _f("2. Market LR", "market_signal = (0.55 \u2212 market_lr) / 0.55"),
            _p("Normalized to long-run ~55%. Below long-run mean = hard market = positive."),
            _f("3. Rate Adequacy", "rate_signal = rate_adequacy  (direct from market model)"),
            _p("Positive in hard markets. Measures how far current rates are from equilibrium."),
            _f("4. Rate Change", "rate_chg_signal = rate_change / 0.10"),
            _p("Normalized so +10% rate increase \u2192 signal \u2248 1.0. Captures hardening momentum."),
            _f("5. Capital Position", "capital_signal = (solvency_ratio \u2212 1.5) / 1.5"),
            _p("Centered at 1.5\u00d7 solvency. High solvency = room to grow."),

            _div("Signal Blending"),
            _f("Blended signal", "S = w_own\u00b7own + w_mkt\u00b7market + w_ra\u00b7rate_adeq + w_rc\u00b7rate_chg + w_cap\u00b7capital"),
            _p("Default weights:"),
            html.Table([
                html.Thead(html.Tr([html.Th("Component"), html.Th("Weight"), html.Th("Rationale")])),
                html.Tbody([
                    html.Tr([html.Td("Own LR"), html.Td("35%"), html.Td("Primary driver: own profitability")]),
                    html.Tr([html.Td("Market LR"), html.Td("20%"), html.Td("Market-wide conditions")]),
                    html.Tr([html.Td("Rate Adequacy"), html.Td("20%"), html.Td("Rate environment relative to equilibrium")]),
                    html.Tr([html.Td("Rate Change"), html.Td("15%"), html.Td("Hardening/softening momentum")]),
                    html.Tr([html.Td("Capital Position"), html.Td("10%"), html.Td("Balance sheet capacity")]),
                ]),
            ], style={"fontSize": "13px", "borderCollapse": "collapse", "width": "100%",
                       "border": "1px solid #e2e8f0", "marginBottom": "8px"}),
            _p("Weights are automatically normalised to sum to 1.0 at runtime. "
               "If user-specified weights sum to e.g. 0.80, each is divided by 0.80 "
               "before blending."),
        ], accent="purple"),

        # =====================================================================
        # 6. GWP & RI DYNAMICS
        # =====================================================================
        _sec("6. GWP & Reinsurance Dynamics", [
            _div("GWP Change"),
            _f("If signal > 0", "\u0394 = growth_rate \u00d7 min(|S| \u00d7 2, 1)"),
            _f("If signal \u2264 0", "\u0394 = shrink_rate \u00d7 min(|S| \u00d7 2, 1)"),
            _f("Bounds", "\u0394 = clip(\u0394, max_shrink_pa, max_growth_pa)"),
            _f("New GWP", "GWP(t) = GWP(t\u22121) \u00d7 (1 + \u0394)"),
            _p("The \u00d72 scaling means a signal magnitude of 0.5 produces the full growth/shrink rate."),

            _div("Dynamic Cession %"),
            _f("Cession", "cess = base + cess_sens \u00d7 max(\u2212ra, 0) \u2212 cess_sens \u00d7 0.5 \u00d7 max(ra, 0)"),
            _f("Bounds", "cess = clip(cess, min_cession, max_cession)"),
            _p("Positive cess_sens = defensive (buy more RI in soft markets). "
               "Negative = aggressive (buy less RI when market softens)."),
            dmc.Alert(
                "The cession formula is intentionally asymmetric: soft-market sensitivity is "
                "2\u00d7 stronger than hard-market reduction (0.5\u00d7 factor). This reflects observed "
                "Lloyd\u2019s behaviour \u2014 insurers increase RI buying faster in deteriorating markets "
                "than they reduce it in improving markets (PwC/Strategy& cession analysis).",
                title="Asymmetry Note", color="blue", variant="light",
            ),

            _div("Net Written Premium"),
            _f("NWP", "NWP(t) = GWP(t) \u00d7 (1 \u2212 cession%(t))"),
        ], accent=""),

        # =====================================================================
        # 7. ADVERSE SELECTION & EXPENSES
        # =====================================================================
        _sec("7. Adverse Selection & Dynamic Expenses", [
            _div("Adverse Selection Penalty"),
            _p("When an insurer grows faster than the market average, it attracts marginal "
               "(worse quality) risks. The penalty is multiplicative on gross loss ratio."),
            _f("Excess growth", "excess = max(insurer_gwp_change \u2212 market_avg_gwp_change, 0)"),
            _f("Penalty", "adv_sel = 1 + sensitivity \u00d7 excess  (applied to gross LR)"),
            _f("Default", "sensitivity = 0.10: 10% excess growth \u2192 1% worse LR"),

            _div("Dynamic Expense Ratio"),
            _p("Rapid growth or shrinkage increases the expense ratio due to hiring costs, "
               "system buildout, or fixed-cost spreading over a smaller base."),
            _f("If growth > 3%", "expense += expense_growth_penalty \u00d7 (gwp_change / 0.10)"),
            _f("If shrink > 3%", "expense += expense_shrink_penalty \u00d7 (|gwp_change| / 0.10)"),
            _f("If |change| < 3%", "expense += expense_stability_bonus  (negative = bonus)"),
            _f("Defaults", "+2pp per 10% growth,  +1pp per 10% shrinkage,  \u22121pp stability bonus"),
        ], accent="amber"),

        # =====================================================================
        # 8. RI PRICING
        # =====================================================================
        _sec("8. Reinsurance Pricing", [
            _p("Reinsurance has its own pricing cycle, lagging the primary market by approximately "
               "1 year (Guy Carpenter Global Property Cat ROL Index)."),
            _f("RI cost rate", "ri_rate = ri_cost_base + ri_cost_cycle_sens \u00d7 rate_adequacy_lagged"),
            _f("Bounds", "ri_rate = clip(ri_rate, 0.15, 0.55)"),
            _f("RI cost", "ri_cost = ceded_premium \u00d7 ri_rate = GWP \u00d7 cession% \u00d7 ri_rate"),
            _f("Lag", "rate_adequacy_lagged = rate_adequacy(t \u2212 ri_cost_lag_years)"),
            _f("Defaults", "base = 30% of ceded premium,  cycle_sens = \u00b110pp,  lag = 1 year"),
            _p("In a hard RI market, RI cost can reach 55% of ceded premium. In soft RI markets, "
               "as low as 15%."),
        ], accent="teal"),

        # =====================================================================
        # 9. P&L CALCULATION
        # =====================================================================
        _sec("9. Profit & Loss Calculation", [
            _p("The annual P&L for each insurer on each path:"),
            _f("Net claims", "net_claims = NWP \u00d7 net_lr"),
            _f("Reserve impact", "reserve_impact = NWP \u00d7 reserve_development"),
            _f("Expenses", "expenses = NWP \u00d7 dynamic_expense_ratio"),
            _f("UW profit", "uw_profit = NWP \u2212 net_claims \u2212 reserve_impact \u2212 expenses \u2212 ri_cost"),
            _div("Investment Income"),
            _f("Float base", "float = NWP \u00d7 0.5 + capital  (50% of premium held as float)"),
            _f("Investment income", "inv_income = max(float, 0) \u00d7 investment_return"),
            _div("Total Profit & Combined Ratio"),
            _f("Total profit", "total_profit = uw_profit + inv_income"),
            _f("Combined ratio", "CR = (net_claims + reserve_impact + expenses + ri_cost) / NWP"),
            _p("CR < 100% indicates underwriting profit before investment income."),
        ], accent="green"),

        # =====================================================================
        # 10. CAPITAL MANAGEMENT
        # =====================================================================
        _sec("10. Capital Management", [
            _f("Economic capital", "econ_cap = max(NWP \u00d7 capital_ratio, 1.0)"),
            _f("Solvency ratio", "solvency = available_capital / economic_capital"),
            _div("Capital Update"),
            _f("After profit", "capital = capital + total_profit"),
            _div("Injection (Recapitalisation)"),
            _f("Trigger", "if solvency < capital_injection_trigger  (default 1.20\u00d7)"),
            _f("Amount", "injection = economic_capital \u00d7 0.25  (25% of required capital)"),
            _div("Dividend Extraction"),
            _f("Trigger", "if solvency > dividend_extraction_trigger  (default 2.00\u00d7)"),
            _f("Amount", "dividend = max(0, (capital \u2212 econ_cap \u00d7 1.5) \u00d7 0.5)"),
            _p("Extracts 50% of excess capital above 1.5\u00d7 required."),
            _div("Ruin"),
            _f("Condition", "capital < ruin_threshold  (default 0)"),
            _f("Effect", "GWP set to 0, insurer ceases operations on that path"),
            _div("RORAC (per year per path)"),
            _f("RORAC", "rorac(t) = total_profit(t) / economic_capital(t)"),

            _div("VaR-Based Economic Capital (Distribution-Derived)"),
            _p(
                "In addition to the ratio-based capital requirement, CycleSim computes a "
                "distribution-derived capital figure from the simulation output:"
            ),
            _f("VaR-based capital", "econ_cap_VaR = max(0, \u2212VaR_{99.5%}(annual_profit))"),
            _p(
                "This represents the capital needed to absorb the 1-in-200 worst annual loss. "
                "It is reported alongside the ratio-based figure for comparison. The ratio-based "
                "capital is used for solvency calculations during the simulation; the VaR-based "
                "figure is a diagnostic cross-check."
            ),
            dmc.Alert(
                "The ratio-based capital requirement (NWP \u00d7 capital_ratio) is a simplification. "
                "A full Solvency II internal model would derive the SCR from the 1-in-200 "
                "loss distribution. The VaR-based capital metric provides this cross-check. "
                "If VaR-based capital significantly exceeds ratio-based capital, the capital_ratio "
                "parameter may be too low for the risk profile.",
                title="Capital Adequacy Note", color="blue", variant="light",
            ),
        ], accent="red"),

        # =====================================================================
        # 11. KEY METRICS
        # =====================================================================
        _sec("11. Key Metrics & Definitions", [
            _f("Through-cycle RORAC", "mean(annual_profit across all years) / mean(economic_capital)"),
            _f("VaR 95%", "5th percentile of terminal cumulative profit"),
            _f("VaR 99.5%", "0.5th percentile of terminal cumulative profit  (Lloyd\u2019s SCR standard)"),
            _f("TVaR 99.5%", "E[profit | profit < VaR_{99.5%}]  (expected shortfall / conditional tail expectation)"),
            _f("VaR-based capital", "max(0, \u2212VaR_{99.5%}(annual profit))  \u2014 distribution-derived SCR cross-check"),
            _f("Ruin probability", "P(capital < 0 at any year in [1, T])"),
            _f("Max drawdown", "max(running_max(capital) \u2212 capital_t) per path"),
            _f("Sharpe-like ratio", "mean(annual RORAC) / std(annual RORAC)"),

            _div("Return Period Table"),
            _p("Standard actuarial return periods aligned to Lloyd\u2019s SCR framework:"),
            html.Table([
                html.Thead(html.Tr([html.Th("Return Period"), html.Th("Percentile"), html.Th("Interpretation")])),
                html.Tbody([
                    html.Tr([html.Td("1-in-10"), html.Td("10th"), html.Td("Moderate adverse year")]),
                    html.Tr([html.Td("1-in-50"), html.Td("2nd"), html.Td("Significant loss event")]),
                    html.Tr([html.Td("1-in-100"), html.Td("1st"), html.Td("Severe loss year")]),
                    html.Tr([html.Td("1-in-200"), html.Td("0.5th"), html.Td("Lloyd\u2019s SCR / Solvency II standard")]),
                ]),
            ], style={"fontSize": "13px", "borderCollapse": "collapse", "width": "100%",
                       "border": "1px solid #e2e8f0", "marginBottom": "8px"}),

            _div("Tail Decomposition"),
            _p(
                "Euler-style marginal attribution: for the worst paths at each return period, "
                "decompose total loss into component contributions (attritional, large, cat, "
                "reserve development, RI cost, expenses). Each component\u2019s contribution is "
                "expressed as a percentage of the total loss at that return period."
            ),

            _div("Profit Attribution"),
            _f("Components", "UW profit + investment income \u2212 RI cost \u2212 reserve dev + net capital actions"),
            _p("Capital actions = dividends extracted \u2212 capital injected."),
        ], accent=""),

        # =====================================================================
        # 12. OPTIMISER, BACKTEST, FORECAST
        # =====================================================================
        _sec("12. Strategy Optimiser (Strategy Lab)", [
            _p(
                "Latin Hypercube Sampling across 8 strategy dimensions. Each candidate is "
                "evaluated with a reduced Monte Carlo simulation. Non-dominated sorting "
                "extracts the Pareto frontier on RORAC vs ruin probability vs max drawdown."
            ),
            _f("Search space", "8 dimensions \u00d7 N candidates (Latin Hypercube)"),
            _f("Evaluation", "Each candidate: M-path Monte Carlo (reduced from main simulation)"),
            _f("Pareto extraction", "Non-dominated sort on (RORAC\u2191, ruin\u2193, drawdown\u2193)"),
            _f("Regime-conditional", "Force starting regime \u2192 optimise per market state"),
            _f("Gap analysis", "Distance from current strategy to nearest Pareto-optimal point"),
        ], accent=""),

        _sec("13. Historical Backtest (2001\u20132024)", [
            _p(
                "Deterministic replay of strategies against Lloyd\u2019s market data 2001\u20132024. "
                "No Monte Carlo \u2014 a single fixed path through every historical market dislocation."
            ),
            _f("Data source", "Lloyd\u2019s Annual Reports 2001\u20132024 (market LR, CR, rate change, regime)"),
            _f("Key events", "9/11, Katrina, GFC, Tohoku/Thai floods, HIM, COVID, Ukraine"),
            _div("Counterfactual Decomposition (2-strategy only)"),
            _p(
                "Sequential factor-swap methodology: run intermediate backtests swapping "
                "one strategy dimension at a time (growth \u2192 RI \u2192 shrink \u2192 expenses) to "
                "isolate each factor\u2019s contribution to the cumulative performance gap."
            ),
        ], accent=""),

        _sec("14. Forward Regime Forecast", [
            _f("Method", "P(regime at t+k) = e_current @ P^k  (matrix power of transition matrix)"),
            _p(
                "Starting from the dominant regime in year 1, compute forward probabilities "
                "by raising the transition matrix to successive powers. Converges to the "
                "stationary distribution (left eigenvector of P corresponding to eigenvalue 1)."
            ),
            _f("Expected RORAC", "\u03a3 P(regime_i, year_k) \u00d7 RORAC_i  for each forecast year k"),
        ], accent=""),

        # =====================================================================
        # 13. CURRENT CONFIGURATION
        # =====================================================================
        _sec("15. Current Simulation Configuration", [
            _div("AR(2) Parameters"),
            _f("\u03c6\u2081", f"{mp.phi_1:.6f}"),
            _f("\u03c6\u2082", f"{mp.phi_2:.6f}"),
            _f("\u03c3_\u03b5", f"{mp.sigma_epsilon:.6f}"),
            _f("Implied period", f"{mp.implied_cycle_period:.1f} years"),
            _f("Stationary", "Yes" if mp.is_stationary else "NO — WARNING"),
            _f("Oscillatory", "Yes" if mp.is_oscillatory else "No (monotonic)"),
            _f("Long-run LR", f"{mp.long_run_loss_ratio:.3f}"),
            _f("LR range", f"[{mp.min_loss_ratio:.3f}, {mp.max_loss_ratio:.3f}]"),
            _f("Shock prob", f"{mp.shock_prob:.0%} p.a."),
            _f("Shock magnitude", f"\u03c3 = {mp.shock_magnitude_std:.3f}"),

            _div("Simulation"),
            _f("Paths", f"{config.n_paths:,}"),
            _f("Years", f"{config.n_years}"),
            _f("Seed", f"{config.random_seed}"),

            _div("Insurer Strategies"),
            html.Table([
                html.Thead(html.Tr([
                    html.Th("Name"), html.Th("GWP"), html.Th("Expense"), html.Th("Cession"),
                    html.Th("Expected LR"), html.Th("Growth/Shrink"),
                    html.Th("Signal Wts (own/mkt/ra/rc/cap)"),
                ])),
                html.Tbody(ins_rows),
            ], style={"fontSize": "12px", "borderCollapse": "collapse", "width": "100%",
                       "border": "1px solid #e2e8f0", "overflowX": "auto"}),
        ], accent="teal"),

        # =====================================================================
        # 14. CALIBRATION SOURCES
        # =====================================================================
        _sec("16. Calibration & Data Sources", [
            _p("Default parameters are calibrated to the following sources:"),
            dmc.List([
                dmc.ListItem("Market cycle period: Lloyd\u2019s market loss ratios 1993\u20132024 (~8-year cycle, observed peaks 2001, 2005, 2011, 2017, 2020)"),
                dmc.ListItem("Loss ratio bounds: Lloyd\u2019s aggregate attritional LR range [0.47, 0.62]"),
                dmc.ListItem("Regime transition matrix: Lloyd\u2019s annual report regime classification 2001\u20132024"),
                dmc.ListItem("Cat frequency/severity: Lloyd\u2019s Realistic Disaster Scenarios & Guy Carpenter ROL Index"),
                dmc.ListItem("Expense ratios: Lloyd\u2019s average 34\u201338% of GWP (2015\u20132024)"),
                dmc.ListItem("Cession rates: PwC/Strategy& \u201cDiscipline Delivers Results\u201d \u2014 Lloyd\u2019s average 22\u201325%"),
                dmc.ListItem("Capital ratios: Solvency II SCR coverage targets, Lloyd\u2019s 40\u201350% of NWP"),
                dmc.ListItem("Investment returns: Bank of England base rate + 150bps, net of investment management fees"),
                dmc.ListItem("Adverse selection: actuarial judgement \u2014 10% excess growth producing ~1% LR deterioration"),
            ], size="sm", spacing="xs"),
            dmc.Alert(
                "Default parameters are illustrative and calibrated to Lloyd\u2019s aggregate data. "
                "For insurer-specific analysis, upload your own capital model or adjust parameters "
                "to match your book of business.",
                title="Important", color="yellow", variant="light",
            ),
        ], accent=""),

        # =====================================================================
        # 15. ASSUMPTIONS & LIMITATIONS
        # =====================================================================
        _sec("17. Assumptions & Limitations", [
            _p("Users should be aware of the following model limitations when interpreting results:"),
            dmc.List([
                dmc.ListItem("Paths are independent \u2014 no systemic contagion or correlation between simulated insurers"),
                dmc.ListItem("No interest rate risk model \u2014 investment returns are deterministic per year. No discounting of liabilities: cash flows are undiscounted, meaning the time value of money is only partially captured through investment income on float. A formal Solvency II technical provisions calculation would require discounting at the risk-free yield curve"),
                dmc.ListItem("No explicit claims inflation model \u2014 claims inflation is captured only indirectly through cycle sensitivity (soft markets = higher LR) and tail thickness (soft markets = fatter tails). In high-inflation environments (e.g., 2022-2024), this may understate loss ratio deterioration. A secular inflation trend overlay would improve realism for long-tail classes"),
                dmc.ListItem("Single currency (GBP) \u2014 no foreign exchange risk"),
                dmc.ListItem("Parametric loss model is simplified \u2014 real books of business have hundreds of classes with varying loss characteristics"),
                dmc.ListItem("Capital model import overrides parametric losses but strategy logic (growth, expenses, RI) still applies parametrically"),
                dmc.ListItem("Regime transition matrix is stationary \u2014 no structural breaks, learning, or adaptation over time"),
                dmc.ListItem("No reinsurer default risk \u2014 all RI recoveries are assumed collectable"),
                dmc.ListItem("Adverse selection is a linear approximation \u2014 real selection effects are non-linear and class-specific"),
                dmc.ListItem("Expense dynamics are stylised \u2014 real expense behaviour depends on organisational structure, not just growth rate"),
                dmc.ListItem("No tax modelling \u2014 all returns are pre-tax"),
                dmc.ListItem("Float calculation assumes 50% of NWP is held as investable float \u2014 actual float depends on payment patterns, class mix, and claims settlement speed. Long-tail classes (liability) hold more float than short-tail (property)"),
            ], size="sm", spacing="xs"),
        ], accent="red"),

        # =====================================================================
        # 16. GLOSSARY
        # =====================================================================
        _sec("18. Glossary of Terms", [
            *[dmc.Group([
                html.Span(term, className="glossary-term", style={"fontSize": "13px", "minWidth": "180px"}),
                html.Span(defn, className="glossary-def", style={"fontSize": "13px"}),
            ], gap="sm", style={"flexWrap": "wrap"})
            for term, defn in [
                ("GWP", "Gross Written Premium \u2014 total premium income before reinsurance cessions"),
                ("NWP", "Net Written Premium \u2014 GWP minus ceded premium (GWP \u00d7 (1 \u2212 cession%))"),
                ("Loss Ratio (LR)", "Incurred claims / earned premium. <100% = underwriting profit on claims alone"),
                ("Combined Ratio (CR)", "Loss ratio + expense ratio + RI cost ratio. <100% = underwriting profit"),
                ("Expense Ratio", "Operating expenses / NWP. Includes acquisition, administration, and management costs"),
                ("Cession %", "Fraction of GWP ceded to reinsurers. Higher = more risk transfer, less net retention"),
                ("RORAC", "Return on Risk-Adjusted Capital \u2014 profit / economic capital. The primary profitability metric"),
                ("VaR", "Value at Risk \u2014 the loss threshold at a given confidence level (e.g., 99.5%)"),
                ("TVaR / ES", "Tail Value at Risk / Expected Shortfall \u2014 expected loss given that loss exceeds VaR"),
                ("Solvency Ratio", "Available capital / required economic capital. >1.0 = solvent, >1.5 = well-capitalised"),
                ("Economic Capital", "Capital required to support the risk profile (NWP \u00d7 capital_ratio)"),
                ("Rate Adequacy", "How far current market rates are from long-run equilibrium. Positive = hard (adequate), negative = soft"),
                ("Ruin", "Capital falls below the ruin threshold (default 0). Insurer ceases operations on that path"),
                ("Float", "Premium held by the insurer between collection and claims payment, available for investment"),
                ("Reserve Development", "Change in prior-year loss reserves. Adverse = reserves were insufficient; favourable = reserves were excessive"),
                ("Return Period", "1-in-N = the loss severity expected to be exceeded once every N years (e.g., 1-in-200 = 0.5th percentile)"),
            ]],
        ], accent=""),

        # =====================================================================
        # 17. REFERENCES
        # =====================================================================
        _sec("19. References", [
            dmc.List([
                dmc.ListItem("Venezian, E.C. (1985) \u201cRatemaking methods and profit cycles in property and liability insurance.\u201d Journal of Risk and Insurance, 52(3), 477\u2013500."),
                dmc.ListItem("Cummins, J.D. & Outreville, J.F. (1987) \u201cAn international analysis of underwriting cycles in property-liability insurance.\u201d Journal of Risk and Insurance, 54(2), 246\u2013262."),
                dmc.ListItem("Wang, S.S., Major, J.A., Pan, C.H. & Leong, J.W. (2010) \u201cUS property-casualty: underwriting cycle modeling and risk benchmarks.\u201d Variance, 4(2), 165\u2013188."),
                dmc.ListItem("Boyer, M.M., Jacquier, E. & Van Norden, S. (2012) \u201cAre underwriting cycles real and forecastable?\u201d Journal of Risk and Insurance, 79(4), 995\u20131015."),
                dmc.ListItem("Lloyd\u2019s of London Annual Reports (2001\u20132024). Market-level combined ratios, loss ratios, expense ratios, and rate change data."),
                dmc.ListItem("PwC/Strategy& (2019) \u201cDiscipline Delivers Results: Performance and resilience in Lloyd\u2019s.\u201d Cession rate analysis."),
                dmc.ListItem("Guy Carpenter (2024) Global Property Catastrophe Rate-on-Line Index. Reinsurance pricing cycle data."),
                dmc.ListItem("Boyer, M.M. & Jacquier, E. (2012) Cycle parameter bounds and stationarity conditions for AR(2) insurance cycle models."),
            ], size="sm", spacing="xs"),
        ], accent=""),

        # =====================================================================
        # 20. CORRELATION STRUCTURE
        # =====================================================================
        _sec("20. Correlation & Dependency Structure", [
            _p(
                "Understanding which components are correlated and which are independent "
                "is critical for interpreting tail risk results."
            ),
            _div("Independent Components"),
            dmc.List([
                dmc.ListItem("Attritional, large, and catastrophe losses are sampled independently within each year"),
                dmc.ListItem("Reserve development noise is independent of current-year losses"),
                dmc.ListItem("Investment returns are deterministic (no correlation with underwriting)"),
            ], size="sm", spacing="xs"),
            _div("Correlated Through Market Cycle"),
            dmc.List([
                dmc.ListItem(
                    "All three loss components share the same rate_adequacy input \u2014 soft markets "
                    "simultaneously increase attritional mean, fatten tails (via CV), and raise "
                    "large-loss frequency. This creates implicit positive correlation in soft markets."
                ),
                dmc.ListItem(
                    "Adverse selection penalty applies uniformly to all gross loss components, "
                    "creating correlation between growth decisions and loss experience."
                ),
                dmc.ListItem(
                    "Reserve development is correlated with past market conditions via the lagged "
                    "soft-market penalty \u2014 years written in soft markets develop adversely 3 years later."
                ),
                dmc.ListItem(
                    "RI cost is correlated with the primary cycle via the lagged rate adequacy input."
                ),
            ], size="sm", spacing="xs"),
            _div("Cross-Insurer Correlation"),
            _p(
                "All insurers share the same market paths (market LR, rate adequacy, regime). "
                "The only source of differentiation is their strategy parameters (signal weights, "
                "growth/shrink rates, cession sensitivity). Loss draws are independent per insurer "
                "\u2014 no contagion or shared catastrophe events between simulated insurers."
            ),
            dmc.Alert(
                "The independent loss component assumption means that in extreme scenarios, "
                "the model may underestimate joint tail risk (e.g., a large loss triggering "
                "a catastrophe event in the same year). The cycle sensitivity partially "
                "compensates by making all components worse simultaneously in soft markets.",
                title="Limitation", color="yellow", variant="light",
            ),
        ], accent="purple"),

        # =====================================================================
        # 21. MODEL VALIDATION
        # =====================================================================
        _sec("21. Model Validation", [
            _p(
                "Model validation compares simulated output against Lloyd\u2019s historical data "
                "(2001\u20132024). The Backtest tab provides a deterministic replay; this section "
                "describes the statistical validation framework."
            ),
            _div("Market Cycle Validation"),
            dmc.List([
                dmc.ListItem(
                    "AR(2) implied cycle period should match observed Lloyd\u2019s cycle length "
                    "of 6\u201310 years. Current calibration: "
                    f"{mp.implied_cycle_period:.1f} years."
                ),
                dmc.ListItem(
                    "Simulated loss ratio range should bracket historical range. "
                    f"Model: [{mp.min_loss_ratio:.2f}, {mp.max_loss_ratio:.2f}]. "
                    "Historical Lloyd\u2019s (2001\u20132024): [0.48, 0.78]."
                ),
                dmc.ListItem(
                    "Regime distribution should approximate historical frequencies. "
                    "Lloyd\u2019s 2001\u20132024: ~29% soft, ~21% firming, ~33% hard, ~17% crisis."
                ),
            ], size="sm", spacing="xs"),
            _div("Loss Model Validation"),
            dmc.List([
                dmc.ListItem(
                    "Mean attritional LR at equilibrium (rate_adequacy \u2248 0) should approximate "
                    "Lloyd\u2019s long-run attritional of ~52%. Model default: "
                    f"{config.loss_params.attritional_mean:.0%}."
                ),
                dmc.ListItem(
                    "Cat frequency of ~0.30/year implies 1 cat event every ~3 years, consistent "
                    "with Lloyd\u2019s experiencing major events in 2001, 2005, 2011, 2017, 2020 "
                    "(5 events in 24 years = 0.21/year; model is slightly conservative)."
                ),
                dmc.ListItem(
                    "Combined ratio distribution: model should produce CR > 100% in ~30\u201340% "
                    "of years, matching Lloyd\u2019s historical frequency of underwriting losses."
                ),
            ], size="sm", spacing="xs"),
            _div("Recommended Validation Checks"),
            dmc.List([
                dmc.ListItem(
                    "Run the Historical Backtest tab and compare simulated strategy P&L "
                    "against known Lloyd\u2019s results for 2001\u20132024."
                ),
                dmc.ListItem(
                    "Check the Overview tab\u2019s loss ratio fan chart: the median should track "
                    "close to the long-run mean, and the 5th/95th percentiles should bracket "
                    "all but the most extreme historical years."
                ),
                dmc.ListItem(
                    "Run with 10,000+ paths and compare the regime distribution bar chart "
                    "against the stationary distribution. Expected steady-state: "
                    "soft ~35%, firming ~25%, hard ~30%, crisis ~10%."
                ),
                dmc.ListItem(
                    "For tail metrics (VaR, 1-in-200), use 50,000+ paths to ensure "
                    "the 0.5th percentile is estimated from at least 250 observations."
                ),
            ], size="sm", spacing="xs"),
        ], accent="green"),

        # =====================================================================
        # 22. CONVERGENCE
        # =====================================================================
        _sec("22. Convergence Analysis", [
            _p(
                "Monte Carlo estimates are subject to sampling error. The convergence "
                "indicator on the Overview tab reports the coefficient of variation (CV) "
                "of the RORAC estimate."
            ),
            _div("Convergence Metric"),
            _f("Standard error", "SE = std(RORAC across paths) / \u221a(n_paths)"),
            _f("Coefficient of variation", "CV = SE / |mean(RORAC)|"),
            _div("Convergence Thresholds"),
            html.Table([
                html.Thead(html.Tr([html.Th("CV"), html.Th("Status"), html.Th("Interpretation")])),
                html.Tbody([
                    html.Tr([html.Td("< 3%"), html.Td("Converged (green)"), html.Td("Mean RORAC reliable to \u00b11\u20132pp")]),
                    html.Tr([html.Td("3\u201310%"), html.Td("Adequate (yellow)"), html.Td("Directional results OK, tail metrics unreliable")]),
                    html.Tr([html.Td("> 10%"), html.Td("Unstable (red)"), html.Td("Increase path count before drawing conclusions")]),
                ]),
            ], style={"fontSize": "13px", "borderCollapse": "collapse", "width": "100%",
                       "border": "1px solid #e2e8f0", "marginBottom": "8px"}),
            _div("Recommended Path Counts"),
            html.Table([
                html.Thead(html.Tr([html.Th("Use Case"), html.Th("Paths"), html.Th("Rationale")])),
                html.Tbody([
                    html.Tr([html.Td("Interactive exploration"), html.Td("1,000"), html.Td("Fast feedback, directional only")]),
                    html.Tr([html.Td("Strategy comparison"), html.Td("5,000"), html.Td("RORAC differences reliable to ~1pp")]),
                    html.Tr([html.Td("Board reporting"), html.Td("10,000"), html.Td("Smooth distributions, stable tail metrics")]),
                    html.Tr([html.Td("1-in-200 / SCR"), html.Td("50,000+"), html.Td("At least 250 obs in the 0.5th percentile tail")]),
                    html.Tr([html.Td("Full sensitivity analysis"), html.Td("1,000 per run"), html.Td("Parallelised; total compute scales linearly")]),
                ]),
            ], style={"fontSize": "13px", "borderCollapse": "collapse", "width": "100%",
                       "border": "1px solid #e2e8f0", "marginBottom": "8px"}),
            _p(
                "All random number generation uses seeded NumPy default_rng for full reproducibility. "
                "Changing the seed produces a different realisation of the same stochastic process."
            ),
        ], accent=""),

        # =====================================================================
        # 23. REVERSE STRESS TESTING
        # =====================================================================
        _sec("23. Reverse Stress Testing", [
            _p(
                "Reverse stress testing identifies the conditions under which the business model "
                "fails. Unlike forward stress tests (which apply a fixed shock and measure the "
                "impact), reverse stress tests start from a defined failure outcome and work "
                "backwards to find what scenarios produce it."
            ),
            _div("Failure Definition"),
            _f("Ruin", "Capital < 0 at any point during the projection"),
            _f("Regulatory breach", "Solvency ratio < 1.0 (capital injection required)"),
            _f("Commercial failure", "Through-cycle RORAC < cost of capital"),

            _div("Approach: One-at-a-Time Parameter Search"),
            _p(
                "For each key parameter, binary search for the threshold value that produces "
                "ruin probability > 5%. This identifies the margin between current calibration "
                "and the break point."
            ),
            dmc.List([
                dmc.ListItem("Use the Sensitivity tab to perturb parameters one at a time"),
                dmc.ListItem("Observe which parameters have the largest RORAC swing (tornado chart)"),
                dmc.ListItem("For the most sensitive parameters, manually adjust until ruin probability exceeds your threshold"),
                dmc.ListItem("The gap between the current value and the break value is the margin of safety"),
            ], size="sm", spacing="xs"),

            _div("Key Parameters to Stress"),
            html.Table([
                html.Thead(html.Tr([html.Th("Parameter"), html.Th("Direction"), html.Th("Failure Mechanism")])),
                html.Tbody([
                    html.Tr([html.Td("Cat frequency"), html.Td("\u2191 Increase"), html.Td("Exhaust RI and capital via repeated cat events")]),
                    html.Tr([html.Td("Shock probability"), html.Td("\u2191 Increase"), html.Td("More frequent crisis regimes, volatile LR")]),
                    html.Tr([html.Td("Max LR (soft)"), html.Td("\u2191 Increase"), html.Td("Deeper soft market losses")]),
                    html.Tr([html.Td("Growth rate"), html.Td("\u2191 Increase"), html.Td("Adverse selection + expense penalty at scale")]),
                    html.Tr([html.Td("Cession %"), html.Td("\u2193 Decrease"), html.Td("More net retention, less risk transfer")]),
                    html.Tr([html.Td("Capital ratio"), html.Td("\u2193 Decrease"), html.Td("Thinner capital buffer, earlier ruin")]),
                    html.Tr([html.Td("Cycle sensitivity"), html.Td("\u2191 Increase"), html.Td("Larger LR swings through cycle")]),
                ]),
            ], style={"fontSize": "13px", "borderCollapse": "collapse", "width": "100%",
                       "border": "1px solid #e2e8f0", "marginBottom": "8px"}),

            _div("Joint Stress Scenarios"),
            _p(
                "The sensitivity tab performs one-at-a-time analysis. For joint stresses "
                "(e.g., cat frequency + soft market + reduced cession simultaneously), "
                "adjust multiple parameters in the sidebar and observe the combined effect "
                "on ruin probability and VaR in the Overview tab."
            ),
            dmc.Alert(
                "Lloyd's and the PRA require reverse stress testing as part of the ORSA "
                "(Own Risk and Solvency Assessment). This tool supports the analysis by "
                "providing rapid parameter exploration, but the formal reverse stress test "
                "report should document the scenarios, their plausibility, and management actions.",
                title="Regulatory Context", color="blue", variant="light",
            ),
        ], accent="red"),

        # =====================================================================
        # 24. MODEL GOVERNANCE
        # =====================================================================
        _sec("24. Model Governance & Change Record", [
            _p(
                "This section documents the model version history and material changes. "
                "All changes are version-controlled in git with full audit trail."
            ),
            _div("Version History"),
            html.Table([
                html.Thead(html.Tr([html.Th("Version"), html.Th("Date"), html.Th("Change Description")])),
                html.Tbody([
                    html.Tr([html.Td("Pass 1-10"), html.Td("2025 Q3"), html.Td("Core engine: AR(2) market cycle, parametric loss model, 2-insurer comparison")]),
                    html.Tr([html.Td("Pass 11-15"), html.Td("2025 Q3"), html.Td("Added regime-switching HMM, historical backtest (2001-2024), strategy optimiser")]),
                    html.Tr([html.Td("Pass 16-20"), html.Td("2025 Q4"), html.Td("Capital model import, tail decomposition, regime forecast, data provenance")]),
                    html.Tr([html.Td("Pass 21-25"), html.Td("2025 Q4"), html.Td("Strategy DNA fingerprint, reserve development, RI pricing cycle, convergence diagnostics")]),
                    html.Tr([html.Td("Pass 26"), html.Td("2026 Q1"), html.Td("Dynamic expenses, adverse selection, 5-signal strategy model, component-specific RI")]),
                    html.Tr([html.Td("Pass 27"), html.Td("2026 Q1"), html.Td("N-strategy refactoring (2-6 insurers), list-based architecture, backward compat")]),
                    html.Tr([html.Td("Pass 28"), html.Td("2026 Q1"), html.Td("12 grouped presets, save/load profiles, chartered actuary documentation, signal weight normalisation, VaR-based capital, reverse stress test framework, regime dwell time validation, CSV path export")]),
                ]),
            ], style={"fontSize": "13px", "borderCollapse": "collapse", "width": "100%",
                       "border": "1px solid #e2e8f0", "marginBottom": "8px"}),

            _div("Model Risk Classification"),
            dmc.List([
                dmc.ListItem("Model type: Strategic planning / scenario analysis tool"),
                dmc.ListItem("Risk tier: Medium (not used for regulatory capital or pricing)"),
                dmc.ListItem("Validation frequency: Annual review recommended"),
                dmc.ListItem("Independent review: Required before use in board-level decisions"),
                dmc.ListItem("Change control: All parameter and logic changes logged in version control"),
            ], size="sm", spacing="xs"),

            _div("Peer Review Checklist"),
            dmc.List([
                dmc.ListItem("Verify AR(2) stationarity conditions hold for chosen parameters"),
                dmc.ListItem("Confirm regime dwell times are plausible against historical cycle data"),
                dmc.ListItem("Check convergence indicator is green (CV < 3%) for key metrics"),
                dmc.ListItem("Run historical backtest and compare against known Lloyd's results"),
                dmc.ListItem("Review sensitivity tornado chart for unexpected dominant parameters"),
                dmc.ListItem("Validate VaR-based capital against ratio-based capital (should be same order of magnitude)"),
                dmc.ListItem("Confirm assumptions and limitations section is complete and current"),
            ], size="sm", spacing="xs"),
        ], accent="teal"),

        # =====================================================================
        # COMPUTATIONAL
        # =====================================================================
        _sec("25. Computational Notes", [
            _f("Vectorisation", "NumPy array operations \u2014 no per-path Python loops"),
            _f("Parallelism", "ProcessPoolExecutor for sensitivity analysis & strategy optimiser"),
            _f("This run", f"{config.n_paths:,} paths \u00d7 {config.n_years} years \u00d7 {len(config.insurers)} strategies"),
            _f("Performance", "~0.09s for 1,000 \u00d7 25 \u00d7 2 insurers (Ryzen 9 9950X)"),
        ], accent=""),

        # =====================================================================
        # PDF EXPORT
        # =====================================================================
        dmc.Paper(
            dmc.Stack([
                dmc.Text("Export", fw=700, size="lg"),
                _p("Use your browser\u2019s print function (Ctrl+P / Cmd+P) to save this "
                   "methodology as PDF. Select \u201cSave as PDF\u201d as the destination."),
                dmc.Button(
                    "Print Methodology (Ctrl+P)",
                    id="print-methodology-btn",
                    variant="light", color="gray", size="sm",
                ),
            ], gap="xs"),
            p="lg", radius="md", withBorder=True,
        ),

    ], gap="lg")


# ---------------------------------------------------------------------------
# Helper chart builders
# ---------------------------------------------------------------------------
def _market_rate_chart(market: dict, n_years: int):
    """Market rate change fan chart with professional hover."""
    import plotly.graph_objects as go
    x = list(range(1, n_years + 1))
    rc = market["market_rate_change"]
    p25 = np.percentile(rc, 25, axis=0)
    p75 = np.percentile(rc, 75, axis=0)
    mean = rc.mean(axis=0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x + x[::-1],
        y=p75.tolist() + p25.tolist()[::-1],
        fill="toself", fillcolor=COLOR_MARKET_LIGHT,
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=True, name="IQR (25th-75th)", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x, y=mean.tolist(),
        line=dict(color=COLOR_MARKET, width=2.5), name="Mean Rate Change",
        hovertemplate="Year %{x}: %{y:.1%}<extra>Mean Rate \u0394</extra>",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color=COLOR_MUTED, line_width=1)
    fig.update_layout(
        title=dict(text="Market Rate Change", font=dict(size=14)),
        xaxis_title="Year", yaxis_title="Annual Rate Change",
        yaxis_tickformat=".0%", **LAYOUT_DEFAULTS,
    )
    return fig


def _regime_bar(market: dict, n_years: int):
    """Regime distribution by year with tooltips."""
    import plotly.graph_objects as go
    regime = market["regime"]
    n_paths = regime.shape[0]
    x = list(range(1, n_years + 1))

    fig = go.Figure()
    for r_idx, r_name in enumerate(REGIMES):
        counts = (regime == r_idx).sum(axis=0) / n_paths
        fig.add_trace(go.Bar(
            x=x, y=counts.tolist(), name=r_name.title(),
            marker_color=REGIME_COLORS[r_idx],
            hovertemplate="Year %{x}: %{y:.0%}<extra>" + r_name.title() + "</extra>",
        ))

    fig.update_layout(
        barmode="stack",
        title=dict(text="Regime Distribution by Year", font=dict(size=14)),
        xaxis_title="Year", yaxis_title="Proportion",
        yaxis_tickformat=".0%",
        **LAYOUT_DEFAULTS,
    )
    return fig


def _market_stats_bar(market: dict, n_years: int):
    """Summary statistics about the market simulation."""
    lr = market["market_loss_ratio"]
    regime = market["regime"]
    n_paths = regime.shape[0]

    # Calculate key stats
    mean_lr = float(lr.mean())
    vol_lr = float(lr.std())
    crisis_pct = float((regime == 3).any(axis=1).mean())
    shock_pct = float(market["shock_mask"].any(axis=1).mean())

    return dmc.SimpleGrid(
        cols=4, spacing="md",
        children=[
            dmc.Paper(
                dmc.Stack([
                    dmc.Text("Mean Market LR", size="xs", c="dimmed", fw=600),
                    dmc.Text(f"{mean_lr:.1%}", size="lg", fw=700, ff="monospace"),
                ], gap=2, align="center"),
                p="sm", radius="md", withBorder=True, style={"textAlign": "center"},
            ),
            dmc.Paper(
                dmc.Stack([
                    dmc.Text("LR Volatility", size="xs", c="dimmed", fw=600),
                    dmc.Text(f"{vol_lr:.1%}", size="lg", fw=700, ff="monospace"),
                ], gap=2, align="center"),
                p="sm", radius="md", withBorder=True, style={"textAlign": "center"},
            ),
            dmc.Paper(
                dmc.Stack([
                    dmc.Text("Paths with Crisis", size="xs", c="dimmed", fw=600),
                    dmc.Text(f"{crisis_pct:.0%}", size="lg", fw=700, ff="monospace", c="red"),
                ], gap=2, align="center"),
                p="sm", radius="md", withBorder=True, style={"textAlign": "center"},
            ),
            dmc.Paper(
                dmc.Stack([
                    dmc.Text("Paths with Shock", size="xs", c="dimmed", fw=600),
                    dmc.Text(f"{shock_pct:.0%}", size="lg", fw=700, ff="monospace"),
                ], gap=2, align="center"),
                p="sm", radius="md", withBorder=True, style={"textAlign": "center"},
            ),
        ],
    )


def _insurer_tab(summary: dict, n_years: int, name: str, color, color_light, color_med,
                 insurer_params=None, market=None, ins_data=None):
    """Build a full insurer detail tab with all operational fan charts."""
    bands = summary["percentile_bands"]
    means = summary["yearly_means"]

    children = []

    # Strategy profile card at top (if params available)
    if insurer_params is not None:
        children.append(strategy_profile_card(
            name=name,
            growth=insurer_params.growth_rate_when_profitable,
            shrink=insurer_params.shrink_rate_when_unprofitable,
            max_growth=insurer_params.max_gwp_growth_pa,
            max_shrink=insurer_params.max_gwp_shrink_pa,
            expected_lr=insurer_params.expected_lr,
            cess_sens=insurer_params.cession_cycle_sensitivity,
            base_cession=insurer_params.base_cession_pct,
            adv_sel=insurer_params.adverse_selection_sensitivity,
            cap_ratio=insurer_params.capital_ratio,
        ))

    children.extend([
        # Row 1: GWP + Combined Ratio (the headline metrics)
        dmc.SimpleGrid(
            cols=2,
            children=[
                _graph(figure=fan_chart(
                    bands["gwp"], means["gwp"], n_years,
                    f"{name}: Gross Written Premium", "GWP (GBP)",
                    color, color_light, color_med,
                )),
                _graph(figure=fan_chart(
                    bands["combined_ratio"], means["combined_ratio"], n_years,
                    f"{name}: Combined Ratio", "Combined Ratio",
                    color, color_light, color_med,
                    yaxis_format=".0%", reference_line=1.0, reference_label="Breakeven",
                )),
            ],
        ),
        # Row 2: Cumulative Profit + Capital
        dmc.SimpleGrid(
            cols=2,
            children=[
                _graph(figure=fan_chart(
                    bands["cumulative_profit"], means["cumulative_profit"], n_years,
                    f"{name}: Cumulative Profit", "Cumulative Profit (GBP)",
                    color, color_light, color_med,
                )),
                _graph(figure=fan_chart(
                    bands["capital"], means["capital"], n_years,
                    f"{name}: Capital Position", "Capital (GBP)",
                    color, color_light, color_med,
                    reference_line=0, reference_label="Ruin",
                )),
            ],
        ),
        # Row 3: Cession % + Net LR (the RI strategy story)
        dmc.SimpleGrid(
            cols=2,
            children=[
                _graph(figure=fan_chart(
                    bands["cession_pct"], means["cession_pct"], n_years,
                    f"{name}: Cession %", "Cession %",
                    color, color_light, color_med,
                    yaxis_format=".0%",
                )),
                _graph(figure=fan_chart(
                    bands["net_lr"], means["net_lr"], n_years,
                    f"{name}: Net Loss Ratio", "Net Loss Ratio",
                    color, color_light, color_med,
                    yaxis_format=".0%",
                )),
            ],
        ),
        # Row 4: RORAC + Expense Ratio
        dmc.SimpleGrid(
            cols=2,
            children=[
                _graph(figure=fan_chart(
                    bands["rorac"], means["rorac"], n_years,
                    f"{name}: Annual RORAC", "RORAC",
                    color, color_light, color_med,
                    yaxis_format=".0%",
                )),
                _graph(figure=fan_chart(
                    bands["expense_ratio"], means["expense_ratio"], n_years,
                    f"{name}: Expense Ratio", "Expense Ratio",
                    color, color_light, color_med,
                    yaxis_format=".0%",
                )),
            ],
        ),
    ])

    # Profit decomposition waterfall
    if ins_data is not None:
        children.append(
            _graph(figure=yearly_profit_waterfall(
                ins_data, n_years, name, color,
            ))
        )

    # CR regime heatmap (if market data available)
    if market is not None and ins_data is not None:
        children.append(
            _graph(figure=cr_regime_heatmap(
                market, ins_data, n_years, name, color,
            ))
        )

    return dmc.Stack(children, gap="md")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    debug = os.environ.get("CYCLESIM_DEBUG", "0") == "1"
    app.run(
        host="0.0.0.0",
        debug=debug,
        port=8050,
        # Enable Dash Dev Tools (callback graph + component inspector)
        # without hot-reload which can cause issues on Windows
        dev_tools_ui=True,
        dev_tools_props_check=True,
        dev_tools_hot_reload=False,
    )
