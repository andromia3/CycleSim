"""
Dash Mantine sidebar with all parameter controls.

Returns a layout component and provides functions to extract
parameter values from callback inputs.

Every slider has a tooltip explaining the parameter and its calibration source.
"""

import dash_mantine_components as dmc
from dash import html, dcc

from cyclesim.defaults import (
    MARKET_DEFAULTS, LOSS_DEFAULTS, SIMULATION_DEFAULTS,
    INSURER_PRESETS, STRATEGY_PREFIXES, MAX_STRATEGIES,
)


def _slider(id: str, label: str, min: float, max: float, value: float,
            step: float = None, marks: list = None, format_pct: bool = False,
            tooltip: str = ""):
    """Create a labeled slider with optional tooltip."""
    if step is None:
        step = (max - min) / 100

    label_children = [dmc.Text(label, size="sm", fw=500)]
    if tooltip:
        label_children = [
            dmc.Tooltip(
                dmc.Text(label, size="sm", fw=500, td="underline", style={"textDecorationStyle": "dotted", "cursor": "help"}),
                label=tooltip,
                multiline=True, w=280, withArrow=True,
                position="top-start",
            )
        ]

    return dmc.Stack([
        dmc.Group([
            *label_children,
            dmc.Text(
                f"{value:.0%}" if format_pct else f"{value:.3g}",
                id=f"{id}-display", size="sm", c="dimmed",
            ),
        ], justify="space-between"),
        dmc.Slider(
            id=id, min=min, max=max, value=value, step=step,
            size="sm",
            marks=marks or [],
            styles={"markLabel": {"fontSize": 10}},
        ),
    ], gap=2)


def _section(title: str, children: list, initially_open: bool = True):
    """Collapsible accordion section."""
    return dmc.AccordionItem(
        value=title.lower().replace(" ", "-"),
        children=[
            dmc.AccordionControl(
                dmc.Text(title, fw=600, size="sm"),
            ),
            dmc.AccordionPanel(
                dmc.Stack(children, gap="sm"),
            ),
        ],
    )


def _insurer_section(prefix: str, preset: dict, title: str):
    """Build an insurer parameter section with tooltips."""
    sw = preset.get("signal_weights", {})
    return _section(title, [
        dmc.TextInput(
            id=f"{prefix}_name", label="Name",
            value=preset["name"], size="sm",
        ),
        _slider(f"{prefix}_gwp", "Initial GWP", 100e6, 2000e6,
                preset["initial_gwp"], step=50e6,
                tooltip="Gross written premium at year 0. Mid-size Lloyd\u2019s syndicate: ~\u00a3500m."),
        _slider(f"{prefix}_expense", "Expense Ratio", 0.28, 0.45,
                preset["expense_ratio"], format_pct=True,
                tooltip="Baseline expense ratio before dynamic adjustments. Lloyd\u2019s 2023 average: 34.4%."),
        _slider(f"{prefix}_cession", "Base Cession %", 0.10, 0.40,
                preset["base_cession_pct"], format_pct=True,
                tooltip="Fraction of GWP ceded to reinsurers at cycle equilibrium. Lloyd\u2019s average: 22-25% (PwC/Strategy&)."),
        _slider(f"{prefix}_inv_return", "Investment Return", 0.01, 0.08,
                preset["investment_return"], format_pct=True,
                tooltip="Annual return on float + capital. UK gilt + credit spread, net of fees."),
        _slider(f"{prefix}_expected_lr", "Expected Loss Ratio", 0.40, 0.70,
                preset["expected_lr"], format_pct=True,
                tooltip="The loss ratio the insurer considers \u2018adequate\u2019. Below this = profitable signal. Lower = more conservative."),

        dmc.Divider(label="Signal Weights", labelPosition="center", size="xs"),
        dmc.Text(
            "How the strategy weighs 5 market signals to decide grow/shrink. Weights are normalized internally.",
            size="xs", c="dimmed",
        ),
        _slider(f"{prefix}_sw_own_lr", "Own Loss Ratio", 0.0, 1.0,
                sw.get("own_lr", 0.35), step=0.05,
                tooltip="Weight on own loss ratio vs expected. Higher = more reactive to own underwriting performance."),
        _slider(f"{prefix}_sw_market_lr", "Market Loss Ratio", 0.0, 1.0,
                sw.get("market_lr", 0.20), step=0.05,
                tooltip="Weight on market-wide loss ratio. Higher = more responsive to industry-wide profitability signals."),
        _slider(f"{prefix}_sw_rate_adequacy", "Rate Adequacy", 0.0, 1.0,
                sw.get("rate_adequacy", 0.20), step=0.05,
                tooltip="Weight on the rate adequacy index (current rate vs long-run equilibrium). Higher = more cycle-aware."),
        _slider(f"{prefix}_sw_rate_change", "Rate Change Momentum", 0.0, 1.0,
                sw.get("rate_change", 0.15), step=0.05,
                tooltip="Weight on year-over-year rate change direction. Higher = more momentum-driven."),
        _slider(f"{prefix}_sw_capital", "Capital Position", 0.0, 1.0,
                sw.get("capital_position", 0.10), step=0.05,
                tooltip="Weight on own solvency ratio relative to target. Higher = more capital-constrained decision-making."),

        dmc.Divider(label="Reaction Rules", labelPosition="center", size="xs"),
        _slider(f"{prefix}_growth", "Growth When Profitable", 0.0, 0.25,
                preset["growth_rate_when_profitable"], format_pct=True,
                tooltip="GWP growth rate when the blended strategy signal is positive. Higher = more aggressive."),
        _slider(f"{prefix}_shrink", "Shrink When Unprofitable", -0.30, 0.0,
                preset["shrink_rate_when_unprofitable"], format_pct=True,
                tooltip="GWP shrinkage rate when signal is negative. More negative = faster de-risking."),
        _slider(f"{prefix}_max_growth", "Max Growth p.a.", 0.0, 0.30,
                preset["max_gwp_growth_pa"], format_pct=True,
                tooltip="Hard cap on annual GWP growth. Rapid growth attracts adverse selection penalties."),
        _slider(f"{prefix}_max_shrink", "Max Shrink p.a.", -0.40, 0.0,
                preset["max_gwp_shrink_pa"], format_pct=True,
                tooltip="Hard cap on annual GWP shrinkage. Fixed costs mean shrinkage increases expense ratio."),

        dmc.Divider(label="Reinsurance Dynamics", labelPosition="center", size="xs"),
        _slider(f"{prefix}_cess_sens", "Cession Cycle Sensitivity", -0.10, 0.10,
                preset["cession_cycle_sensitivity"],
                tooltip="How cession % reacts to the cycle. Positive = buy more RI in soft markets (defensive). Negative = buy less (aggressive)."),
        _slider(f"{prefix}_min_cess", "Min Cession %", 0.05, 0.30,
                preset["min_cession_pct"], format_pct=True,
                tooltip="Floor on cession %. Even aggressive insurers maintain some RI program."),
        _slider(f"{prefix}_max_cess", "Max Cession %", 0.20, 0.50,
                preset["max_cession_pct"], format_pct=True,
                tooltip="Ceiling on cession %. Capacity constraints limit how much can be ceded."),

        dmc.Divider(label="Advanced", labelPosition="center", size="xs"),
        _slider(f"{prefix}_adv_sel", "Adverse Selection Sensitivity", 0.0, 0.30,
                preset["adverse_selection_sensitivity"],
                tooltip="Multiplicative LR penalty per unit of excess growth vs market. Growing 10% faster than market with 0.10 sensitivity = 1% worse LR."),
        _slider(f"{prefix}_cap_ratio", "Capital Ratio (% of NWP)", 0.30, 0.60,
                preset["capital_ratio"], format_pct=True,
                tooltip="Required economic capital as fraction of NWP. Lloyd\u2019s typical range: 40-50%."),
        _slider(f"{prefix}_coc", "Cost of Capital", 0.06, 0.15,
                preset["cost_of_capital"], format_pct=True,
                tooltip="Hurdle rate for RORAC calculation. Lloyd\u2019s literature: 8-12% post-tax, ~10% pre-tax."),

        dmc.Divider(label="Expense Model", labelPosition="center", size="xs"),
        _slider(f"{prefix}_expense_fixed_pct", "Fixed Expense %", 0.0, 1.0,
                preset.get("expense_fixed_pct", 0.60), format_pct=True,
                tooltip="Fraction of expenses that are fixed (don't scale with premium volume). "
                        "Higher = more operating leverage. Shrinking the book concentrates fixed costs. "
                        "0% = fully proportional (legacy model). 60% = typical Lloyd's syndicate."),
    ], initially_open=(prefix == "a"))


def build_sidebar() -> dmc.Stack:
    """Build the full sidebar layout."""
    # Build insurer sections for all 6 possible strategies
    insurer_sections = []
    for i, prefix in enumerate(STRATEGY_PREFIXES):
        preset = INSURER_PRESETS[prefix]
        title = f"Insurer {prefix.upper()}"
        section = _insurer_section(prefix, preset, title)
        if i < 2:
            # A and B are always visible
            insurer_sections.append(section)
        else:
            # C-F are wrapped in a togglable div, hidden by default
            insurer_sections.append(
                html.Div(
                    section,
                    id=f"insurer-{prefix}-section",
                    style={"display": "none"},
                )
            )

    return dmc.Stack([
        # Header
        dmc.Stack([
            dmc.Text("CycleSim", fw=700, size="lg", lh=1,
                     style={"color": "white", "letterSpacing": "-0.01em"}),
            dmc.Text("Market Cycle Strategy Simulator", size="xs", lh=1,
                     style={"color": "rgba(255,255,255,0.45)"}),
        ], gap=4, mt=4, mb=4),

        dmc.Divider(),

        # Scenario presets (grouped)
        dmc.Select(
            id="scenario-preset",
            label="Scenario Preset",
            placeholder="Choose a scenario...",
            data=[
                {"group": "Historical Scenarios", "items": [
                    {"value": "post_911", "label": "Post-9/11 Hard Market"},
                    {"value": "katrina_era", "label": "Katrina/HIM Cat Stress"},
                    {"value": "soft_2014_2019", "label": "Soft Market Grind (2014-19)"},
                    {"value": "hard_2021_2024", "label": "Hard Market Discipline (2021-24)"},
                ]},
                {"group": "Strategy Archetypes", "items": [
                    {"value": "default", "label": "Disciplined vs Aggressive"},
                    {"value": "both_conservative", "label": "Two Conservative Players"},
                    {"value": "both_aggressive", "label": "Two Aggressive Players"},
                    {"value": "counter_cyclical", "label": "Counter-Cyclical vs Pro-Cyclical"},
                ]},
                {"group": "Stress Tests", "items": [
                    {"value": "serial_cats", "label": "Serial Catastrophes"},
                    {"value": "rate_collapse", "label": "Rate Collapse"},
                    {"value": "inflation_shock", "label": "Inflation / Social Inflation"},
                    {"value": "perfect_storm", "label": "Perfect Storm"},
                ]},
            ],
            size="sm", clearable=True, searchable=True,
        ),
        html.Div(id="preset-description", children=[
            dmc.Text("Select a scenario to auto-configure all parameters.",
                     size="xs", c="dimmed"),
        ]),

        dmc.Divider(label="My Profiles", labelPosition="center", size="xs"),

        # Save current parameters as a named profile
        dmc.Group([
            dmc.TextInput(id="profile-name-input", placeholder="Profile name...",
                          size="sm", style={"flex": 1}),
            dmc.Button("Save", id="save-profile-btn", size="sm",
                       variant="light", color="green"),
        ], gap="xs"),

        # Load / delete saved profiles
        dmc.Select(id="load-profile", placeholder="No saved profiles",
                   data=[], size="sm", clearable=True),
        dmc.Group([
            dmc.Button("Load", id="load-profile-btn", size="sm",
                       variant="light", color="blue", style={"flex": 1}),
            dmc.Button("Delete", id="delete-profile-btn", size="sm",
                       variant="light", color="red", style={"flex": 1}),
        ], gap="xs"),
        html.Div(id="profile-status"),

        # Run button
        dmc.Button(
            "Run Simulation", id="run-btn",
            fullWidth=True, size="md",
            variant="filled", color="blue",
        ),

        dmc.Accordion(
            multiple=True,
            value=["simulation", "insurer-a"],
            children=[
                # Simulation settings
                _section("Simulation", [
                    _slider("n_paths", "Number of Paths", 100, 500000,
                            SIMULATION_DEFAULTS["n_paths"], step=100,
                            tooltip="Monte Carlo simulation paths. More paths = smoother distributions. 1,000 for interactive, 10,000+ for production quality. >100k paths uses significant memory."),
                    _slider("n_years", "Horizon (Years)", 5, 100,
                            SIMULATION_DEFAULTS["n_years"], step=1,
                            tooltip="Projection horizon. 25 years covers ~3 full market cycles at the default 8-year period. 50+ for long-tail analysis."),
                    dmc.NumberInput(
                        id="seed", label="Random Seed",
                        value=SIMULATION_DEFAULTS["random_seed"],
                        min=1, max=999999, size="sm",
                    ),
                    _slider("discount_rate", "Discount Rate", 0.0, 0.10,
                            SIMULATION_DEFAULTS["discount_rate"], step=0.005,
                            format_pct=True,
                            tooltip="Annual discount rate for present value calculations. 0% = undiscounted. 3-5% typical for regulatory/economic capital."),
                    dmc.Divider(label="Strategies", labelPosition="center", size="xs"),
                    _slider("n_strategies", "Number of Strategies", 2, MAX_STRATEGIES,
                            2, step=1,
                            tooltip="Compare 2\u20136 insurer strategies side-by-side. Each strategy has its own parameter section below."),
                ]),

                # Risk Appetite Framework (F6)
                _section("Risk Appetite", [
                    dmc.Text(
                        "Define firm risk appetite thresholds. Strategies are assessed pass/fail against these limits.",
                        size="xs", c="dimmed",
                    ),
                    _slider("risk_appetite_ruin_max", "Max Ruin Probability", 0.0, 0.10,
                            SIMULATION_DEFAULTS["risk_appetite_ruin_max"], step=0.005,
                            format_pct=True,
                            tooltip="Maximum acceptable probability of ruin over the projection horizon."),
                    _slider("risk_appetite_solvency_min", "Min Solvency Ratio", 1.0, 2.5,
                            SIMULATION_DEFAULTS["risk_appetite_solvency_min"], step=0.1,
                            tooltip="Minimum acceptable mean solvency ratio (available capital / economic capital)."),
                    _slider("risk_appetite_cr_max", "Max Combined Ratio", 0.90, 1.15,
                            SIMULATION_DEFAULTS["risk_appetite_cr_max"], step=0.01,
                            format_pct=True,
                            tooltip="Maximum acceptable mean combined ratio. Above 100% = underwriting loss."),
                    _slider("risk_appetite_rorac_min", "Min RORAC", 0.0, 0.15,
                            SIMULATION_DEFAULTS["risk_appetite_rorac_min"], step=0.005,
                            format_pct=True,
                            tooltip="Minimum acceptable through-cycle RORAC. Typically at or above cost of capital."),
                ]),

                # Market cycle
                _section("Market Cycle", [
                    dmc.SegmentedControl(
                        id="ar2_mode", value="auto",
                        data=[
                            {"value": "auto", "label": "User-Friendly"},
                            {"value": "direct", "label": "Direct AR(2)"},
                        ],
                        size="xs", fullWidth=True,
                    ),
                    dmc.Text(
                        "User-friendly mode calibrates AR(2) coefficients automatically from cycle period and loss ratio bounds.",
                        size="xs", c="dimmed",
                    ),
                    # User-friendly controls
                    html.Div(id="market-auto-controls", children=[
                        _slider("cycle_period", "Cycle Period (years)", 4, 15,
                                MARKET_DEFAULTS["cycle_period_years"], step=0.5,
                                tooltip="Target cycle period in years. Lloyd\u2019s observed ~8yr peak-to-peak (2009\u20132017 soft, 2017\u20132023 hard)."),
                        _slider("min_lr", "Min Loss Ratio (Hard)", 0.35, 0.55,
                                MARKET_DEFAULTS["min_loss_ratio"], format_pct=True,
                                tooltip="Attritional loss ratio in a hard market. Lloyd\u2019s 2023-2024: 47-48%."),
                        _slider("max_lr", "Max Loss Ratio (Soft)", 0.55, 0.80,
                                MARKET_DEFAULTS["max_loss_ratio"], format_pct=True,
                                tooltip="Attritional loss ratio at the soft market peak. Lloyd\u2019s 2017-2019: 57-62%."),
                        _slider("long_run_lr", "Long-Run Loss Ratio", 0.45, 0.65,
                                MARKET_DEFAULTS["long_run_loss_ratio"], format_pct=True,
                                tooltip="Equilibrium loss ratio the AR(2) process mean-reverts to. Lloyd\u2019s 15-year average: ~54.5%."),
                    ]),
                    # Direct AR(2) controls
                    html.Div(id="market-direct-controls", style={"display": "none"}, children=[
                        _slider("phi_1", "\u03c6\u2081 (persistence)", 0.5, 1.5, 1.1, step=0.01,
                                tooltip="AR(2) persistence parameter. Higher = slower cycle transitions. Oscillation requires \u03c6\u00b2\u2081 + 4\u03c6\u2082 < 0."),
                        _slider("phi_2", "\u03c6\u2082 (oscillation)", -0.8, -0.1, -0.45, step=0.01,
                                tooltip="AR(2) oscillation parameter. Must be negative for cyclical behavior. More negative = faster oscillation."),
                        _slider("sigma_epsilon", "\u03c3 (innovation)", 0.02, 0.20, 0.10, step=0.005,
                                tooltip="Standard deviation of AR(2) innovation term. Higher = noisier cycles with more extreme moves."),
                    ]),
                    _slider("shock_prob", "Shock Probability", 0.0, 0.25,
                            MARKET_DEFAULTS["shock_prob"], format_pct=True,
                            tooltip="Annual probability of a market dislocation. ~8% = one major shock per 12 years historically (9/11, GFC, COVID)."),
                    _slider("shock_magnitude", "Shock Magnitude (pp)", 0.05, 0.25,
                            MARKET_DEFAULTS["shock_magnitude_std"],
                            tooltip="Standard deviation of shock innovation in percentage points. 12pp = typical catastrophe-induced rate hardening."),
                ], initially_open=False),

                # Loss model
                _section("Loss Model", [
                    # Capital model upload
                    dmc.Paper([
                        dmc.Text("Capital Model Import", size="sm", fw=600, mb=4),
                        dmc.Text(
                            "Upload your own 10,000 loss ratio sims (CSV/Excel) to replace the parametric model.",
                            size="xs", c="dimmed", mb="xs",
                        ),
                        dcc.Upload(
                            id="capital-model-upload",
                            children=dmc.Button(
                                "Upload Capital Model", variant="light",
                                color="violet", size="xs", fullWidth=True,
                            ),
                            accept=".csv,.xlsx,.xls",
                        ),
                        dmc.Group([
                            dmc.Button(
                                "Load Sample (10k)", id="load-sample-cm",
                                variant="subtle", color="gray", size="xs",
                            ),
                            dmc.Button(
                                "Clear", id="clear-cm",
                                variant="subtle", color="red", size="xs",
                            ),
                        ], gap="xs", mt=4),
                        html.Div(id="cm-status", style={"marginTop": 4}),
                    ], p="xs", radius="sm", withBorder=True,
                       style={"borderColor": "#e9ecef", "backgroundColor": "#f8f9fa"}),

                    dmc.Divider(label="Parametric Loss Model", labelPosition="center", size="xs"),
                    dmc.Text(
                        "These parameters are used when no capital model is uploaded.",
                        size="xs", c="dimmed",
                    ),
                    _slider("attritional_mean", "Attritional Mean LR", 0.40, 0.65,
                            LOSS_DEFAULTS["attritional_mean"], format_pct=True,
                            tooltip="Mean attritional loss ratio at cycle equilibrium. This is the base before cycle adjustment."),
                    _slider("cycle_sensitivity", "Cycle Sensitivity", 0.05, 0.30,
                            LOSS_DEFAULTS["cycle_sensitivity"],
                            tooltip="How much the attritional LR shifts per unit of rate adequacy change. Higher = losses more cycle-dependent."),
                    _slider("tail_sensitivity", "Tail Thickness Sensitivity", 0.0, 0.6,
                            LOSS_DEFAULTS["tail_thickness_sensitivity"],
                            tooltip="In soft markets, loss ratio volatility increases (fatter tails from deteriorating risk selection). 0.3 = CV increases 30% per unit toward soft extreme."),
                    _slider("large_freq", "Large Loss Frequency", 0.5, 8.0,
                            LOSS_DEFAULTS["large_loss_frequency"], step=0.5,
                            tooltip="Expected large losses per year. Modeled as compound Poisson-Pareto. Frequency increases in soft markets."),
                    _slider("cat_freq", "Cat Frequency", 0.05, 1.0,
                            LOSS_DEFAULTS["cat_frequency"], step=0.05,
                            tooltip="Expected cat events per year (~0.3 = one every 3 years). Independent of underwriting cycle."),
                    _slider("cat_severity", "Cat Severity (mean)", 0.02, 0.20,
                            LOSS_DEFAULTS["cat_severity_mean"],
                            tooltip="Mean cat loss as fraction of GWP. 8% mean with high CV produces realistic tail behavior."),
                ], initially_open=False),

                # All insurer sections (A-F)
                *insurer_sections,
            ],
        ),
    ], gap="md", p="md", style={"width": "100%"})


# ---------------------------------------------------------------------------
# Parameter IDs for callback wiring
# ---------------------------------------------------------------------------
# Per-insurer parameter suffixes (17 core + 5 signal weights = 22 per insurer)
_INSURER_PARAM_SUFFIXES = [
    "_name", "_gwp", "_expense", "_cession", "_inv_return",
    "_expected_lr", "_growth", "_shrink", "_max_growth", "_max_shrink",
    "_cess_sens", "_min_cess", "_max_cess",
    "_adv_sel", "_cap_ratio", "_coc",
    # Signal weights
    "_sw_own_lr", "_sw_market_lr", "_sw_rate_adequacy", "_sw_rate_change", "_sw_capital",
    # Expense model (F8)
    "_expense_fixed_pct",
]

# Build the full list: simulation/market/loss params + n_strategies + all insurer params
ALL_INPUT_IDS = [
    "n_paths", "n_years", "seed", "ar2_mode",
    "cycle_period", "min_lr", "max_lr", "long_run_lr",
    "phi_1", "phi_2", "sigma_epsilon",
    "shock_prob", "shock_magnitude",
    "attritional_mean", "cycle_sensitivity", "tail_sensitivity",
    "large_freq", "cat_freq", "cat_severity",
    "discount_rate",
    "risk_appetite_ruin_max", "risk_appetite_solvency_min",
    "risk_appetite_cr_max", "risk_appetite_rorac_min",
    "n_strategies",
]

# Add all 6 prefixes x 22 params
for _pfx in STRATEGY_PREFIXES:
    for _sfx in _INSURER_PARAM_SUFFIXES:
        ALL_INPUT_IDS.append(f"{_pfx}{_sfx}")
