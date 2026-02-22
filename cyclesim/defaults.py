"""
Lloyd's-calibrated default parameters for all CycleSim modules.

Sources:
- Lloyd's Annual Reports 2015-2024 (combined ratios, expense ratios, cession rates)
- Venezian (1985), Cummins & Outreville (1987) — AR(2) cycle models
- Wang, Major, Pan & Leong (2010) — regime-switching underwriting cycle model
- PwC/Strategy& "Discipline Delivers Results" — reinsurance cession analysis
- Guy Carpenter Global Property Cat ROL Index — reinsurance pricing cycles
- Boyer, Jacquier & Van Norden (2012) — cycle parameter bounds
"""

# ---------------------------------------------------------------------------
# Market Cycle (AR(2) + Hidden Markov Regime Overlay)
# ---------------------------------------------------------------------------
MARKET_DEFAULTS = {
    # --- AR(2) user-friendly calibration targets ---
    "cycle_period_years": 8.0,          # 6-10yr observed; 8yr central (Lloyd's 2009-2017-2023)
    "min_loss_ratio": 0.47,             # Hard market attritional (Lloyd's 2023-2024)
    "max_loss_ratio": 0.62,             # Soft market attritional (Lloyd's 2017-2019)
    "long_run_loss_ratio": 0.545,       # Long-run mean (Lloyd's 15-year average)

    # --- Direct AR(2) overrides (None = auto-calibrate from above) ---
    "phi_1": None,
    "phi_2": None,
    "sigma_epsilon": None,

    # --- Regime-switching overlay ---
    # Transition matrix [from_state, to_state]: soft, firming, hard, crisis
    "regime_transition_matrix": [
        [0.75, 0.18, 0.05, 0.02],   # soft    → soft/firming/hard/crisis
        [0.10, 0.55, 0.30, 0.05],   # firming → ...
        [0.05, 0.10, 0.75, 0.10],   # hard    → ...
        [0.05, 0.25, 0.55, 0.15],   # crisis  → ... (crisis is short-lived)
    ],
    # Regime loss ratio multipliers (applied to AR(2) base)
    "regime_lr_multipliers": {
        "soft": 1.08,       # +8% above AR(2) prediction
        "firming": 1.00,    # neutral
        "hard": 0.93,       # -7% below AR(2) prediction
        "crisis": 1.25,     # +25% — catastrophe / dislocation year
    },
    # Regime volatility multipliers on epsilon
    "regime_vol_multipliers": {
        "soft": 1.0,
        "firming": 0.8,
        "hard": 0.7,
        "crisis": 2.5,
    },

    # --- Shock parameters ---
    "shock_prob": 0.08,                 # ~8% p.a. — 9/11, GFC, COVID, Ukraine
    "shock_magnitude_std": 0.12,        # 12pp std dev of shock innovation

    # --- Observable mapping ---
    "market_expense_ratio": 0.36,       # Lloyd's 2023: 34.4%; 36% with acquisition in growth
    "hard_market_rate_change_pa": 0.09, # +9% midpoint of +6 to +12%
    "soft_market_rate_change_pa": -0.05,# -5% midpoint of -3 to -7%
    "max_rate_change_pa": 0.20,         # Cap: no single year > +20%
    "min_rate_change_pa": -0.12,        # Floor: no single year < -12%
}

# ---------------------------------------------------------------------------
# Loss Ratio Model (Parametric)
# ---------------------------------------------------------------------------
LOSS_DEFAULTS = {
    # --- Attritional ---
    "attritional_mean": 0.52,           # Mean at equilibrium
    "attritional_cv": 0.08,             # Coefficient of variation
    "cycle_sensitivity": 0.15,          # LR shift per unit rate adequacy change

    # --- Cycle-dependent tail thickness ---
    # In soft markets, tails get fatter (risk quality deteriorates)
    "tail_thickness_sensitivity": 0.3,  # CV increases by 30% per unit toward soft extreme

    # --- Large losses ---
    "large_loss_frequency": 3.0,        # Expected per year
    "large_loss_pareto_alpha": 1.8,     # Shape parameter
    "large_loss_pareto_xmin": 0.01,     # Min loss as fraction of GWP
    "large_loss_cap": 0.25,             # Cap single large loss at 25% of GWP

    # --- Catastrophe ---
    "cat_frequency": 0.30,              # Expected per year (~1 every 3 years)
    "cat_severity_mean": 0.08,          # Mean cat loss as fraction of GWP
    "cat_severity_cv": 1.5,             # High CV

    # --- Prior-year reserve development ---
    "reserve_dev_mean": 0.0,            # Mean development (0 = no bias at equilibrium)
    "reserve_dev_std": 0.02,            # 2pp std dev per year
    "reserve_dev_cycle_lag": 3,         # Development manifests 3 years after writing
    "reserve_dev_soft_penalty": 0.03,   # Soft-market years develop 3pp adversely

    # --- Reinsurance effectiveness by component ---
    "ri_effectiveness_attritional": 0.90,   # QS-like: recovers most attritional
    "ri_effectiveness_large": 0.75,         # XL-like: less effective for large
    "ri_effectiveness_cat": 0.95,           # Cat XL: highly effective for cat
}

# ---------------------------------------------------------------------------
# Insurer Strategy
# ---------------------------------------------------------------------------
INSURER_DEFAULTS = {
    "name": "Base Insurer",
    "initial_gwp": 500_000_000,         # 500m — mid-size Lloyd's syndicate
    "expense_ratio": 0.36,              # 36% baseline
    "base_cession_pct": 0.23,           # 23% — Lloyd's average (PwC/Strategy&)
    "investment_return": 0.035,         # 3.5% — gilt + spread, net of fees
    "capital_ratio": 0.45,              # 45% of NWP as economic capital
    "cost_of_capital": 0.10,            # 10% pre-tax hurdle

    # --- Strategy signals ---
    # Insurer reacts to a weighted blend of 5 signals
    "signal_weights": {
        "own_lr": 0.35,                # Weight on own loss ratio vs expected
        "market_lr": 0.20,             # Weight on market loss ratio
        "rate_adequacy": 0.20,         # Weight on rate adequacy index
        "rate_change": 0.15,           # Weight on market rate change momentum
        "capital_position": 0.10,      # Weight on own solvency ratio
    },

    # --- Growth/shrink reactions ---
    "expected_lr": 0.55,                # The LR they consider "adequate"
    "growth_rate_when_profitable": 0.08,
    "shrink_rate_when_unprofitable": -0.05,
    "max_gwp_growth_pa": 0.15,
    "max_gwp_shrink_pa": -0.20,

    # --- Adverse selection ---
    # Growth faster than market attracts worse risks
    "adverse_selection_sensitivity": 0.10,  # 10% LR penalty per 10% excess growth

    # --- Expense ratio dynamics ---
    # Rapid growth or shrinkage increases expense ratio
    "expense_growth_penalty": 0.02,     # +2pp expense ratio per 10% rapid growth
    "expense_shrink_penalty": 0.01,     # +1pp expense ratio per 10% shrinkage
    "expense_stability_bonus": -0.01,   # -1pp if GWP change < 3%
    "expense_fixed_pct": 0.60,         # 60% of expenses are fixed (operating leverage)
    "expense_base_gwp": None,          # Reference GWP for expense calibration (None = initial_gwp)

    # --- Reinsurance dynamics ---
    "cession_cycle_sensitivity": 0.03,  # Buy 3pp more RI per unit toward soft
    "min_cession_pct": 0.15,
    "max_cession_pct": 0.35,

    # --- Reinsurance pricing ---
    # RI cost as fraction of ceded premium (varies with cycle)
    "ri_cost_base": 0.30,              # 30% of ceded premium as RI cost at equilibrium
    "ri_cost_cycle_sensitivity": 0.10, # RI cost increases 10pp per unit of hardening
    "ri_cost_lag_years": 1,            # RI market lags primary by ~1 year

    # --- Capital management ---
    "capital_injection_trigger": 1.20,
    "dividend_extraction_trigger": 2.00,
    "ruin_threshold": 0,
}

# ---------------------------------------------------------------------------
# Preset Insurer Profiles
# ---------------------------------------------------------------------------
INSURER_A_PRESET = {
    **INSURER_DEFAULTS,
    "name": "Disciplined Underwriter",
    "growth_rate_when_profitable": 0.08,
    "shrink_rate_when_unprofitable": -0.08,
    "max_gwp_growth_pa": 0.12,
    "max_gwp_shrink_pa": -0.15,
    "cession_cycle_sensitivity": 0.04,      # Buys more RI in soft markets (defensive)
    "expected_lr": 0.53,                     # Conservative threshold
    "signal_weights": {
        "own_lr": 0.25,
        "market_lr": 0.20,
        "rate_adequacy": 0.25,              # More market-aware
        "rate_change": 0.20,                # Watches rate momentum closely
        "capital_position": 0.10,
    },
}

INSURER_B_PRESET = {
    **INSURER_DEFAULTS,
    "name": "Aggressive Grower",
    "growth_rate_when_profitable": 0.15,
    "shrink_rate_when_unprofitable": -0.03,
    "max_gwp_growth_pa": 0.25,
    "max_gwp_shrink_pa": -0.10,
    "cession_cycle_sensitivity": -0.02,     # Buys LESS RI in soft markets (aggressive)
    "expected_lr": 0.60,                     # Higher tolerance
    "signal_weights": {
        "own_lr": 0.50,                     # Mostly reacts to own results
        "market_lr": 0.10,
        "rate_adequacy": 0.10,
        "rate_change": 0.20,                # Chases rate momentum
        "capital_position": 0.10,
    },
}

INSURER_C_PRESET = {
    **INSURER_DEFAULTS,
    "name": "Strategy C",
}

INSURER_D_PRESET = {
    **INSURER_DEFAULTS,
    "name": "Strategy D",
}

INSURER_E_PRESET = {
    **INSURER_DEFAULTS,
    "name": "Strategy E",
}

INSURER_F_PRESET = {
    **INSURER_DEFAULTS,
    "name": "Strategy F",
}

# All presets in order, indexed by prefix letter
INSURER_PRESETS = {
    "a": INSURER_A_PRESET,
    "b": INSURER_B_PRESET,
    "c": INSURER_C_PRESET,
    "d": INSURER_D_PRESET,
    "e": INSURER_E_PRESET,
    "f": INSURER_F_PRESET,
}

# ---------------------------------------------------------------------------
# Strategy display
# ---------------------------------------------------------------------------
STRATEGY_PREFIXES = ["a", "b", "c", "d", "e", "f"]
MAX_STRATEGIES = 6
STRATEGY_COLORS = ["#2563eb", "#dc5c0c", "#059669", "#7c3aed", "#db2777", "#0891b2"]
SIGNAL_WEIGHT_DEFAULTS = {
    "own_lr": 0.35,
    "market_lr": 0.20,
    "rate_adequacy": 0.20,
    "rate_change": 0.15,
    "capital": 0.10,
}

# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
SIMULATION_DEFAULTS = {
    "n_paths": 5000,
    "n_years": 25,
    "random_seed": 42,
    "discount_rate": 0.0,
    # Risk appetite framework
    "risk_appetite_ruin_max": 0.02,
    "risk_appetite_solvency_min": 1.5,
    "risk_appetite_cr_max": 1.05,
    "risk_appetite_rorac_min": 0.05,
}

# ---------------------------------------------------------------------------
# Regime labels (ordered)
# ---------------------------------------------------------------------------
REGIMES = ["soft", "firming", "hard", "crisis"]
