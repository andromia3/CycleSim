# CycleSim

Monte Carlo strategy simulator for insurance underwriting cycles. Compare how different underwriting strategies perform through market cycles — hard markets, soft markets, crises, and everything in between.

Built for actuaries, underwriters, and insurance strategists who want to stress-test portfolio decisions before committing capital.

## What it does

- Simulates 5,000+ paths through realistic market cycles (AR(2) + hidden Markov regime switching, calibrated to Lloyd's of London data)
- Compares up to 6 underwriting strategies side-by-side across identical market conditions
- Models real-world dynamics: adverse selection on rapid growth, operating leverage, reinsurance cycle lag, multi-signal strategy reactions
- Full risk metrics: RORAC, VaR, TVaR, ruin probability, combined ratio, with bootstrap confidence intervals and GPD tail extrapolation
- Import your own capital model output (10,000+ sims from ReMetrica, Igloo, Tyche, etc.) to ground results in your data
- Historical backtesting against Lloyd's 2001-2024 combined ratios
- Interactive — drag a slider, watch 5,000 paths recalculate

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate    # Windows
# source .venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
python -m ui.app
```

Open http://localhost:8050 in your browser.

## Project structure

```
cyclesim/           Core simulation engine
  market.py         AR(2) + HMM regime-switching market cycle
  losses.py         3-component loss model (attritional, large, cat)
  insurer.py        Strategy model (5 signals, adverse selection, operating leverage)
  simulator.py      Monte Carlo orchestrator
  metrics.py        Risk-adjusted metrics, bootstrap CIs, GPD tail fitting
  optimizer.py      Latin Hypercube strategy optimizer with Pareto frontier
  historical.py     Lloyd's 2001-2024 backtest replay
  io.py             Capital model import/export
  defaults.py       Lloyd's-calibrated default parameters

ui/                 Dash + Mantine web interface
  app.py            Main application (10 tabs, 40+ charts)
  charts.py         Plotly chart builders
  exhibits.py       Summary tables, executive summary, risk appetite
  sidebar.py        Parameter controls
  state.py          Simulation state management

tests/              Test suite (32 unit + 74 render tests)
```

## Requirements

- Python 3.11+
- numpy, scipy, pandas, numba, dash, dash-mantine-components, plotly, openpyxl

## Disclaimer

This tool uses synthetic Lloyd's-calibrated parameters for illustration. It is not based on proprietary data and should not be used as the sole basis for underwriting or capital decisions.

## License

MIT
