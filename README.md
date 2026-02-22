# CycleSim

Monte Carlo strategy simulator for insurance underwriting cycles. Compare how different underwriting strategies perform through full market cycles — hard markets, soft markets, crises, and everything in between.

Built for actuaries, underwriters, and insurance strategists who want to stress-test portfolio decisions before committing capital.

## Why this exists

Insurance markets are cyclical. Premiums swing between soft markets (low rates, loose underwriting) and hard markets (high rates, tight standards), driven by capital flows, loss events, and competitive dynamics. A strategy that looks brilliant in a hard market can destroy capital in a soft one.

CycleSim lets you test that before it happens. Define two strategies — say a disciplined underwriter vs an aggressive grower — and simulate them through 5,000 independent 25-year market paths. See where one wins, where the other goes bust, and what the expected risk-adjusted return looks like across the full cycle.

## Features

### Market cycle engine
- **AR(2) process** for base cycle dynamics (calibrated to 8-year Lloyd's cycle period)
- **4-state hidden Markov regime switching** — soft, firming, hard, crisis — with configurable transition matrix
- Regime-dependent volatility multipliers and loss ratio shocks
- Exogenous shock events (~8% annual probability: 9/11, GFC, COVID-type scenarios)
- Rate adequacy index tracking through-cycle pricing pressure

### Strategy model
Each insurer strategy reacts to a **weighted blend of 5 signals**:

| Signal | What it measures |
|--------|-----------------|
| Own loss ratio | How the insurer's own book performed vs expectations |
| Market loss ratio | How the overall market performed |
| Rate adequacy | Whether current pricing covers expected losses |
| Rate change momentum | Whether the market is hardening or softening |
| Capital position | Current solvency ratio relative to target |

The signal blend drives growth/shrink decisions, subject to configurable guardrails (max +15% growth, max -20% shrink per annum).

### Real-world dynamics
- **Adverse selection** — growing faster than the market attracts worse risks, penalizing the loss ratio
- **Operating leverage** — 60% of expenses are fixed costs; shrinking the book concentrates them over fewer premiums
- **Reinsurance cycle lag** — RI market hardens 1 year after primary, with its own cost dynamics
- **Dynamic expense ratio** — rapid growth or shrinkage both increase expenses; stability gets a bonus
- **Capital management** — automatic injection below 1.2x solvency, dividend extraction above 2.0x

### Risk metrics
- **RORAC** (Return on Risk-Adjusted Capital) — through-cycle mean
- **VaR / TVaR** at 95% and 99.5% — with bootstrap confidence intervals (B=500)
- **GPD tail extrapolation** — fits Generalized Pareto Distribution to loss tail for more reliable extreme quantiles
- **Ruin probability** — percentage of paths where capital hits zero
- **Present value** — optional discounting of future cash flows
- **Risk appetite framework** — define firm-wide thresholds and get pass/fail assessments

### Capital model import
Upload your own capital model output (CSV/Excel with 10,000+ simulated loss ratios from ReMetrica, Igloo, Tyche, etc.) to replace the parametric loss model with your actual internal model data. CycleSim uses your gross and net loss ratio distributions directly, preserving all correlation structure from your capital model.

### Historical backtesting
Replay strategies against actual Lloyd's of London combined ratios from 2001-2024. See how each strategy would have performed through the 9/11 aftermath, Katrina, the GFC, COVID, and the 2023-2024 hard market. Includes counterfactual decomposition showing what drove performance differences.

### Strategy optimizer
Latin Hypercube sampler explores the parameter space across 16 CPU cores to find Pareto-optimal strategies — maximizing RORAC while minimizing ruin probability. Visualized as an interactive efficient frontier.

### Model validation
- AR(2) residual diagnostics (time series, ACF, normality test)
- Probability Integral Transform histogram (model calibration check)
- VaR backtest (Kupiec test for breach frequency)
- Seed stability analysis (metric dispersion across random seeds)

## Interface

10 interactive tabs built with Dash + Mantine + Plotly:

| Tab | What's there |
|-----|-------------|
| **Overview** | Executive summary, KPI cards, risk appetite pass/fail, year-1 business plan, audit log |
| **Market** | Cycle chart with regime bands, rate change, regime distribution, market clock |
| **Comparison** | Side-by-side fan charts (GWP, combined ratio, profit, capital), RORAC scatter, win probability |
| **Risk** | VaR/TVaR distributions, GPD tail fit, ruin probability over time, drawdown, efficiency frontier |
| **Sensitivity** | One-at-a-time tornado charts for each parameter's impact on RORAC |
| **Validation** | AR(2) residuals, PIT histogram, VaR backtest, QQ plots |
| **Drill-Down** | Single-path scenario explorer with all metrics year-by-year |
| **Insurer tabs** | Per-strategy deep dive: profit waterfall, cession dynamics, regime heatmap, correlation matrix |
| **Strategy Lab** | Optimizer with Pareto frontier, RI efficient frontier, strategy DNA radar |
| **History** | Lloyd's 2001-2024 backtest replay with counterfactual decomposition |

40+ interactive Plotly charts, all with professional hover templates, SI-prefix formatting, and regime colour-coding.

## Quick start

```bash
git clone https://github.com/andromia3/CycleSim.git
cd CycleSim

python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # Linux/Mac

pip install -r requirements.txt
python -m ui.app
```

Open **http://localhost:8050** in your browser. Click "Run Simulation" to generate 5,000 paths.

## Default parameters

All defaults are calibrated to publicly available Lloyd's of London data:

| Parameter | Default | Source |
|-----------|---------|--------|
| Cycle period | 8 years | Lloyd's 2009-2017-2023 observed peaks |
| Long-run loss ratio | 54.5% | Lloyd's 15-year average |
| Loss ratio range | 47-62% | Hard (2023-2024) to soft (2017-2019) |
| Expense ratio | 36% | Lloyd's 2023 + acquisition costs |
| Cession rate | 23% | Lloyd's average (PwC/Strategy&) |
| Investment return | 3.5% | Gilt + spread, net of fees |
| Capital ratio | 45% of NWP | Mid-size Lloyd's syndicate |
| Crisis probability | ~2% per year | Regime transition matrix |
| Shock probability | ~8% per year | 9/11, GFC, COVID, Ukraine frequency |

Two preset strategies ship out of the box:

**Disciplined Underwriter** — conservative growth caps (12%), watches rate adequacy closely, buys more reinsurance in soft markets, lower loss ratio tolerance (53%).

**Aggressive Grower** — high growth caps (25%), reacts mainly to own results, buys less reinsurance in soft markets, higher loss ratio tolerance (60%).

## Running tests

```bash
# Unit tests (32 tests, ~8s)
python -m pytest tests/ -q

# Full render tests (74 tests — validates every chart function)
python tests/full_render_test.py
```

## Technical details

### Simulation engine

The core engine is fully vectorized with NumPy — no per-path loops. 5,000 paths x 25 years x 2 strategies runs in under 1 second. The optimizer parallelizes across all available CPU cores using `ProcessPoolExecutor`.

### Architecture

```
cyclesim/                   Pure NumPy engine (no UI dependencies)
  market.py                 AR(2) + HMM market cycle generator
  losses.py                 3-component loss model (attritional + large + cat)
  insurer.py                Strategy reaction model (5-signal blend)
  simulator.py              Monte Carlo orchestrator
  metrics.py                Summary statistics, bootstrap CIs, GPD tail fitting
  optimizer.py              Latin Hypercube Pareto optimizer
  historical.py             Lloyd's 2001-2024 backtest data and replay
  io.py                     Capital model CSV/Excel import, Excel export
  defaults.py               All default parameters with calibration sources

ui/                         Dash + Mantine web layer
  app.py                    Main application entry point, all tab layouts
  charts.py                 40+ Plotly chart builder functions
  exhibits.py               Summary tables, KPI cards, executive summary
  sidebar.py                150+ parameter sliders with tooltips
  state.py                  Simulation cache, sensitivity analysis, seed stability
  assets/custom.css         Mantine theme overrides
```

### Key dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >= 1.26 | Vectorized simulation engine |
| scipy | >= 1.11 | GPD fitting, statistical tests |
| pandas | >= 2.0 | Capital model import, Excel export |
| numba | >= 0.59 | JIT compilation for hot paths |
| dash | >= 2.15 | Web application framework |
| dash-mantine-components | >= 0.14 | UI component library |
| plotly | >= 5.18 | Interactive charting |
| openpyxl | >= 3.1 | Excel workbook export |

## Academic references

The market cycle model draws on established actuarial literature:

- Venezian, E.C. (1985). "Ratemaking Methods and Profit Cycles in Property and Liability Insurance." *Journal of Risk and Insurance*.
- Cummins, J.D. & Outreville, J.F. (1987). "An International Analysis of Underwriting Cycles in Property-Liability Insurance." *Journal of Risk and Insurance*.
- Wang, S., Major, J., Pan, C. & Leong, J. (2010). "US Property-Casualty: Underwriting Cycle Modeling and Risk Benchmarks." *Variance*.
- Boyer, M.M., Jacquier, E. & Van Norden, S. (2012). "Are Underwriting Cycles Real and Forecastable?" *Journal of Risk and Insurance*.

## Disclaimer

This tool uses synthetic parameters calibrated to publicly available Lloyd's of London aggregate data. It is not based on proprietary or confidential data. Results are illustrative and should not be used as the sole basis for underwriting, capital, or strategic decisions.

## License

MIT — free for any use, including commercial. See [LICENSE](LICENSE).
