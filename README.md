<div align="center">

# CycleSim

**Monte Carlo Strategy Simulator for Insurance Underwriting Cycles**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-3776ab?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Dash 2.15+](https://img.shields.io/badge/dash-2.15%2B-00b4d8?logo=plotly&logoColor=white)](https://dash.plotly.com)
[![NumPy](https://img.shields.io/badge/numpy-%E2%89%A51.26-013243?logo=numpy&logoColor=white)](https://numpy.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Stress-test underwriting strategies through **5,000 simulated market cycles** before committing capital.
Compare disciplined underwriters against aggressive growers across hard markets, soft markets, crises, and everything in between.

[Getting Started](#getting-started) &bull; [How It Works](#how-it-works) &bull; [Features](#features) &bull; [Configuration](#configuration) &bull; [Architecture](#architecture) &bull; [References](#references)

</div>

---

## The Problem

Insurance markets are cyclical. Premiums swing between soft markets (low rates, loose underwriting) and hard markets (high rates, tight standards), driven by capital flows, loss events, and competitive dynamics.

A strategy that looks brilliant in a hard market can destroy capital in a soft one. But you can't wait 25 years to find out.

**CycleSim lets you test that before it happens.** Define two strategies, simulate them through thousands of independent market paths, and see where one wins, where the other goes bust, and what the risk-adjusted return looks like across the full cycle.

## Getting Started

```bash
git clone https://github.com/andromia3/CycleSim.git
cd CycleSim

python -m venv .venv
.venv/Scripts/activate        # Windows
# source .venv/bin/activate   # Linux / macOS

pip install -r requirements.txt
python -m ui.app
```

Open **http://localhost:8050** &mdash; click **Run Simulation** to generate 5,000 paths.

<details>
<summary><strong>Requirements</strong></summary>

| Package | Version | Role |
|---------|---------|------|
| `numpy` | &ge; 1.26 | Vectorized simulation engine |
| `scipy` | &ge; 1.11 | GPD fitting, statistical tests |
| `pandas` | &ge; 2.0 | Capital model import, Excel export |
| `numba` | &ge; 0.59 | JIT compilation for hot paths |
| `dash` | &ge; 2.15 | Web application framework |
| `dash-mantine-components` | &ge; 0.14 | UI component library |
| `plotly` | &ge; 5.18 | Interactive charting |
| `openpyxl` | &ge; 3.1 | Excel workbook export |

</details>

---

## How It Works

CycleSim chains three models together &mdash; each year of each simulated path flows through all three:

```
                    +-----------------+
                    |  Market Engine  |
                    |  AR(2) + HMM   |
                    +--------+--------+
                             |
                     rate adequacy,
                   regime, rate change
                             |
                    +--------v--------+
                    |   Loss Model    |
                    |  3-component    |
                    +--------+--------+
                             |
                   gross & net loss
                    ratios per path
                             |
                    +--------v--------+
                    | Strategy Model  |
                    |  5-signal blend |
                    +-----------------+
                             |
                     GWP, expenses,
                   profit, capital, RI
```

<details>
<summary><strong>1. Market Cycle Engine</strong></summary>

The market is driven by an **AR(2) autoregressive process** calibrated to an 8-year cycle period (matching observed Lloyd's peaks in 2009, 2017, 2023), overlaid with a **4-state Hidden Markov regime model**:

| Regime | Vol Multiplier | LR Shock | Typical Duration |
|--------|:-:|:-:|:-:|
| Soft | 1.0x | +8% | 3-4 years |
| Firming | 0.8x | neutral | 1-2 years |
| Hard | 0.7x | -7% | 3-4 years |
| Crisis | 2.5x | +25% | 1 year |

**Transition matrix** (rows = from, columns = to):

```
         Soft   Firm   Hard   Crisis
Soft   [ 0.75   0.18   0.05   0.02 ]
Firm   [ 0.10   0.55   0.30   0.05 ]
Hard   [ 0.05   0.10   0.75   0.10 ]
Crisis [ 0.05   0.25   0.55   0.15 ]
```

On top of the regime model, **exogenous shocks** hit with ~8% annual probability (9/11, GFC, COVID-type events), injecting a 12pp standard deviation innovation into the loss ratio.

**Rate adequacy** tracks cumulative pricing pressure: when rates are inadequate (< 1.0), the market is underpricing risk.

</details>

<details>
<summary><strong>2. Loss Model</strong></summary>

Three independent loss components are sampled per path per year:

| Component | Distribution | Key Parameters |
|-----------|-------------|----------------|
| **Attritional** | LogNormal | mean 52%, CV 8%, cycle-sensitive |
| **Large losses** | Poisson(3) &times; Pareto(&alpha;=1.8) | capped at 25% of GWP per event |
| **Catastrophe** | Poisson(0.3) &times; LogNormal | mean 8% of GWP, CV 1.5 |

The attritional component is **cycle-sensitive**: in soft markets the mean shifts upward and the tail thickens (CV increases by up to 30%). This captures the real-world deterioration of risk quality when underwriting standards loosen.

**Prior-year reserve development** is modeled with a 3-year lag: years written in soft markets develop adversely by 3pp on average.

**Reinsurance** effectiveness varies by component:

| Component | RI Recovery Rate |
|-----------|:---:|
| Attritional | 90% (QS-like) |
| Large loss | 75% (XL-like) |
| Catastrophe | 95% (Cat XL) |

**Capital model import**: Upload your own internal model output (CSV/Excel with 10,000+ simulated loss ratios from ReMetrica, Igloo, Tyche, etc.) to replace the parametric model. CycleSim uses your gross and net distributions directly, preserving your correlation structure.

</details>

<details>
<summary><strong>3. Strategy Model</strong></summary>

Each insurer strategy reacts to a **weighted blend of 5 market signals**:

```
Signal Score = w_1 * f(own_lr)
             + w_2 * f(market_lr)
             + w_3 * f(rate_adequacy)
             + w_4 * f(rate_change)
             + w_5 * f(capital_position)
```

| Signal | What It Measures | Typical Weight |
|--------|-----------------|:-:|
| Own loss ratio | Book performance vs expectations | 25-50% |
| Market loss ratio | Industry-wide performance | 10-20% |
| Rate adequacy | Cumulative pricing pressure | 10-25% |
| Rate change momentum | Market hardening/softening | 15-20% |
| Capital position | Solvency ratio vs target | 10% |

The composite signal drives growth/shrink decisions, subject to guardrails:

| Guardrail | Disciplined | Aggressive |
|-----------|:-:|:-:|
| Max annual growth | +12% | +25% |
| Max annual shrink | -15% | -10% |
| Loss ratio tolerance | 53% | 60% |

</details>

<details>
<summary><strong>4. Real-World Dynamics</strong></summary>

Beyond the core models, CycleSim captures dynamics that make insurance strategy non-trivial:

| Dynamic | Mechanism | Impact |
|---------|-----------|--------|
| **Adverse selection** | Growing faster than market attracts worse risks | +10% LR penalty per 10% excess growth |
| **Operating leverage** | 60% of expenses are fixed costs | Shrinking concentrates fixed costs over fewer premiums |
| **Reinsurance cycle lag** | RI market hardens 1 year after primary | RI cost increases 10pp per unit of market hardening |
| **Dynamic expenses** | Rapid growth/shrinkage increases expenses | +2pp per 10% rapid growth, +1pp per 10% shrinkage |
| **Expense stability bonus** | Stable books run leaner | -1pp when GWP change < 3% |
| **Capital management** | Automatic injection/extraction | Inject below 1.2x solvency, dividend above 2.0x |

</details>

---

## Features

### Risk Metrics

| Metric | Method | Detail |
|--------|--------|--------|
| **RORAC** | Through-cycle mean | Return on Risk-Adjusted Capital |
| **VaR** (95%, 99.5%) | Empirical + Bootstrap CI | B=500, 90% confidence intervals |
| **TVaR** (95%, 99.5%) | Empirical + Bootstrap CI | Tail conditional expectation |
| **GPD Tail** | Generalized Pareto fit | Parametric extrapolation beyond empirical tail |
| **Ruin Probability** | Path counting | % of paths where capital hits zero |
| **Present Value** | Configurable discount rate | Optional NPV of future cash flows |
| **Risk Appetite** | Threshold framework | Pass/fail against firm-defined limits |

<details>
<summary><strong>GPD tail extrapolation</strong></summary>

Empirical VaR at 99.5% from 5,000 paths relies on only ~25 tail observations. CycleSim fits a **Generalized Pareto Distribution** to losses exceeding the 90th percentile, enabling more reliable extreme quantile estimation:

```
VaR(p) = u + (sigma / xi) * ((n / n_u * (1-p))^(-xi) - 1)
```

where `u` is the threshold, `sigma` is the scale, `xi` is the shape parameter, and `n_u` is the number of exceedances. The fit is validated visually in the Risk tab with an empirical vs GPD overlay chart.

</details>

<details>
<summary><strong>Bootstrap confidence intervals</strong></summary>

All tail metrics include 90% bootstrap confidence intervals (B=500 resamples). Path indices are resampled (not just terminal values) to preserve the RORAC computation structure. This runs in ~50ms on 5,000 paths.

</details>

### Historical Backtesting

Replay strategies against **actual Lloyd's of London combined ratios from 2001-2024**:

| Year | Event | Market CR |
|------|-------|:-:|
| 2001 | 9/11 + Enron | 115% |
| 2005 | Katrina / Rita / Wilma | 107% |
| 2008 | GFC + Hurricane Ike | 97% |
| 2011 | Tohoku / Thai floods | 114% |
| 2017 | HIM hurricanes | 115% |
| 2020 | COVID-19 | 110% |
| 2023 | Hard market peak | 84% |
| 2024 | Continued hardening | 86% |

Includes **counterfactual decomposition**: sequential factor-swap methodology isolating what drove performance differences between strategies (growth decisions, reinsurance, expense management, adverse selection).

### Strategy Optimizer

**Latin Hypercube sampling** across 16 CPU cores explores the parameter space to find **Pareto-optimal strategies** &mdash; maximizing RORAC while minimizing ruin probability. Results are visualized as an interactive efficient frontier with strategy DNA radar charts.

### Model Validation

| Test | What It Checks |
|------|---------------|
| AR(2) residual diagnostics | Time series, ACF (lags 1-10), normality |
| PIT histogram | Probability Integral Transform &mdash; should be uniform if model is well-calibrated |
| VaR backtest | Kupiec test for breach frequency against predicted VaR(95%) |
| Seed stability | Metric dispersion across 5 random seeds (RORAC, VaR, ruin probability) |
| QQ plot | Quantile-quantile comparison against theoretical distribution |

---

## Interface

10 interactive tabs built with [Dash](https://dash.plotly.com) + [Mantine](https://www.dash-mantine-components.com) + [Plotly](https://plotly.com/python/):

<table>
<tr>
<td width="200"><strong>Overview</strong></td>
<td>Executive summary with risk-return trade-off analysis, KPI cards, risk appetite pass/fail badges, year-1 business plan projection, simulation audit log</td>
</tr>
<tr>
<td><strong>Market</strong></td>
<td>Cycle chart with regime colour bands, rate change trajectory, regime distribution pie, market clock phase diagram</td>
</tr>
<tr>
<td><strong>Comparison</strong></td>
<td>Side-by-side fan charts (GWP, combined ratio, profit, capital) with percentile bands, RORAC scatter, win probability analysis</td>
</tr>
<tr>
<td><strong>Risk</strong></td>
<td>VaR/TVaR distributions with bootstrap CIs, GPD tail fit overlay, ruin probability over time, maximum drawdown, efficiency frontier</td>
</tr>
<tr>
<td><strong>Sensitivity</strong></td>
<td>One-at-a-time tornado charts showing each parameter's marginal impact on RORAC</td>
</tr>
<tr>
<td><strong>Validation</strong></td>
<td>AR(2) residual diagnostics, PIT histograms, VaR backtest (Kupiec), QQ plots</td>
</tr>
<tr>
<td><strong>Drill-Down</strong></td>
<td>Single-path scenario explorer &mdash; select any simulated path and inspect all metrics year-by-year</td>
</tr>
<tr>
<td><strong>Insurer&nbsp;Tabs</strong></td>
<td>Per-strategy deep dive: profit waterfall, cession dynamics, regime heatmap, correlation matrix</td>
</tr>
<tr>
<td><strong>Strategy&nbsp;Lab</strong></td>
<td>Pareto optimizer with efficient frontier, RI efficient frontier, strategy DNA radar chart</td>
</tr>
<tr>
<td><strong>History</strong></td>
<td>Lloyd's 2001-2024 backtest replay with counterfactual decomposition chart</td>
</tr>
</table>

> **40+ interactive Plotly charts** with professional hover templates, SI-prefix formatting, and regime colour-coding throughout.

---

## Configuration

### Default Parameters

All defaults are calibrated to publicly available **Lloyd's of London** data:

<details>
<summary><strong>Market parameters</strong></summary>

| Parameter | Default | Calibration Source |
|-----------|:-------:|-------------------|
| Cycle period | 8 years | Lloyd's observed peaks: 2009, 2017, 2023 |
| Long-run loss ratio | 54.5% | Lloyd's 15-year average |
| Hard market LR | 47% | Lloyd's 2023-2024 |
| Soft market LR | 62% | Lloyd's 2017-2019 |
| Market expense ratio | 36% | Lloyd's 2023 + acquisition costs |
| Hard market rate change | +9% p.a. | Midpoint of +6% to +12% |
| Soft market rate change | -5% p.a. | Midpoint of -3% to -7% |
| Crisis probability | ~2% p.a. | Regime transition matrix |
| Shock probability | ~8% p.a. | 9/11, GFC, COVID, Ukraine frequency |

</details>

<details>
<summary><strong>Insurer parameters</strong></summary>

| Parameter | Default | Source |
|-----------|:-------:|-------|
| Initial GWP | &pound;500M | Mid-size Lloyd's syndicate |
| Expense ratio | 36% | Lloyd's 2023 |
| Base cession rate | 23% | Lloyd's average (PwC/Strategy&) |
| Investment return | 3.5% | Gilt + spread, net of fees |
| Capital ratio | 45% of NWP | Mid-size syndicate typical |
| Cost of capital | 10% | Pre-tax hurdle rate |
| Fixed expense share | 60% | Operating leverage calibration |

</details>

<details>
<summary><strong>Loss model parameters</strong></summary>

| Parameter | Default | Note |
|-----------|:-------:|------|
| Attritional mean | 52% | At equilibrium |
| Attritional CV | 8% | Process volatility |
| Cycle sensitivity | 0.15 | LR shift per unit rate adequacy change |
| Large loss frequency | 3.0 p.a. | Expected count |
| Large loss Pareto &alpha; | 1.8 | Shape parameter |
| Cat frequency | 0.3 p.a. | ~1 every 3 years |
| Cat severity mean | 8% of GWP | Mean cat loss |
| Reserve dev lag | 3 years | Development manifests after writing |
| Soft market reserve penalty | +3pp | Adverse development from soft years |

</details>

### Preset Strategies

Two preset strategies ship out of the box for immediate comparison:

<table>
<tr>
<th width="50%">Disciplined Underwriter</th>
<th width="50%">Aggressive Grower</th>
</tr>
<tr>
<td>

- Conservative growth caps (+12%)
- Watches rate adequacy closely (25% weight)
- Buys more RI in soft markets
- Lower loss ratio tolerance (53%)
- Quick to shrink when unprofitable (-8%)

</td>
<td>

- High growth caps (+25%)
- Reacts mainly to own results (50% weight)
- Buys less RI in soft markets
- Higher loss ratio tolerance (60%)
- Slow to shrink (-3%)

</td>
</tr>
</table>

Up to **6 strategies** can be compared simultaneously, each with 150+ configurable parameters via the sidebar.

---

## Architecture

```
cyclesim/                     Pure NumPy engine (no UI dependencies)
  market.py ................ AR(2) + HMM market cycle generator
  losses.py ................ 3-component loss model (attritional + large + cat)
  insurer.py ............... Strategy reaction model (5-signal blend)
  simulator.py ............. Monte Carlo orchestrator
  metrics.py ............... Summary stats, bootstrap CIs, GPD tail fitting
  optimizer.py ............. Latin Hypercube Pareto optimizer
  historical.py ............ Lloyd's 2001-2024 backtest data & replay
  io.py .................... Capital model CSV/Excel import, Excel export
  defaults.py .............. All default parameters with calibration sources

ui/                           Dash + Mantine web layer
  app.py ................... Main application, all tab layouts (3,100 lines)
  charts.py ................ 40+ Plotly chart builder functions
  exhibits.py .............. Summary tables, KPI cards, executive summary
  sidebar.py ............... 150+ parameter sliders with tooltips
  state.py ................. Simulation cache, sensitivity, seed stability
  assets/custom.css ........ Mantine theme overrides

tests/                        Test suite
  test_*.py ................ 32 unit tests (pytest)
  full_render_test.py ...... 74 render tests (validates every chart function)
```

> **~13,000 lines** across 24 Python files. The core engine (`cyclesim/`) has zero UI dependencies and can be used as a standalone library.

### Performance

| Operation | Time | Method |
|-----------|:----:|--------|
| Full simulation (5,000 paths &times; 25 years &times; 2 strategies) | < 1s | Fully vectorized NumPy, no per-path loops |
| Bootstrap CIs (B=500) | ~50ms | Path-index resampling |
| Sensitivity analysis (10 parameters) | ~5s | Parallel `ProcessPoolExecutor` |
| Strategy optimization (500 candidates) | ~30s | Latin Hypercube + all CPU cores |

---

## Running Tests

```bash
# Unit tests (32 tests, ~8s)
python -m pytest tests/ -q

# Full render tests (74 tests - validates every chart function)
python tests/full_render_test.py
```

---

## Mathematical Details

<details>
<summary><strong>AR(2) cycle dynamics</strong></summary>

The base market cycle follows a second-order autoregressive process:

```
y(t) = phi_0 + phi_1 * y(t-1) + phi_2 * y(t-2) + epsilon(t)
```

where `y(t)` is the detrended loss ratio deviation. The AR(2) coefficients are auto-calibrated from the user-specified cycle period using:

```
phi_1 = 2 * r * cos(2*pi / T)
phi_2 = -r^2
```

where `T` is the cycle period (default 8 years) and `r` is chosen to place the roots near the unit circle for persistent oscillation. The innovation `epsilon(t)` is scaled by the current regime's volatility multiplier.

</details>

<details>
<summary><strong>Adverse selection model</strong></summary>

When an insurer grows faster than the market, it attracts marginal risks that other insurers are shedding:

```
LR_penalty = adverse_selection_sensitivity * max(0, own_growth - market_growth)
```

This creates a natural brake on aggressive growth: the faster you grow relative to the market, the worse your risk quality becomes. Calibrated at 10% LR penalty per 10% excess growth.

</details>

<details>
<summary><strong>Operating leverage</strong></summary>

The fixed/variable expense split creates operating leverage:

```
fixed_cost    = expense_ratio * fixed_pct * base_gwp    (absolute amount)
variable_cost = expense_ratio * (1 - fixed_pct) * gwp(t)
expense_ratio(t) = (fixed_cost + variable_cost) / nwp(t)
```

When the book shrinks, fixed costs are spread over fewer premiums, amplifying the expense ratio. When the book grows, fixed costs are diluted. This captures the real operational constraint that you can't fire half your actuaries because premiums dropped 15%.

</details>

---

## References

The market cycle model draws on established actuarial literature:

- Venezian, E.C. (1985). "Ratemaking Methods and Profit Cycles in Property and Liability Insurance." *Journal of Risk and Insurance*.
- Cummins, J.D. & Outreville, J.F. (1987). "An International Analysis of Underwriting Cycles in Property-Liability Insurance." *Journal of Risk and Insurance*.
- Wang, S., Major, J., Pan, C. & Leong, J. (2010). "US Property-Casualty: Underwriting Cycle Modeling and Risk Benchmarks." *Variance*.
- Boyer, M.M., Jacquier, E. & Van Norden, S. (2012). "Are Underwriting Cycles Real and Forecastable?" *Journal of Risk and Insurance*.

---

## Disclaimer

This tool uses synthetic parameters calibrated to publicly available Lloyd's of London aggregate data. It is not based on proprietary or confidential data. Results are illustrative and should not be used as the sole basis for underwriting, capital, or strategic decisions.

## License

[MIT](LICENSE) &mdash; free for any use, including commercial.
