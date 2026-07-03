#Quantitative Multi-Asset Trading Framework

> Institutional-grade backtesting, portfolio optimization, and risk analytics for a multi-asset universe spanning equities, fixed income, commodities, FX, and crypto.

---

## Architecture

```
Historical Data → Cleaning → Feature Engineering → Alpha Signals
    → Portfolio Optimization → Risk Management → Backtesting
        → Performance Analytics → Interactive Dashboard
```

## Asset Universe

| Asset Class    | Ticker | Instrument         |
|:---------------|:-------|:-------------------|
| Equity         | SPY    | S&P 500 ETF        |
| Tech Equity    | QQQ    | Nasdaq 100 ETF     |
| Fixed Income   | TLT    | 20+ Year Treasury  |
| Commodity      | GLD    | Gold ETF           |
| Commodity      | USO    | Oil ETF            |
| FX             | EURUSD | EUR/USD            |
| Crypto         | BTC    | Bitcoin             |

**Period**: 2012 – 2026 (14+ years daily data)

---

## Features

### Alpha Strategies
- **MA Crossover** — SMA20 vs SMA100
- **Mean Reversion** — RSI oversold/overbought
- **Time-Series Momentum** — top/bottom quintile returns
- **Volatility Breakout** — ATR-based breakout detection
- **Cross-Sectional Momentum** — rank-based asset selection

### Portfolio Optimization
- Equal Weight (1/N)
- Risk Parity (inverse-vol)
- Mean-Variance (Markowitz)
- Maximum Sharpe Ratio
- Minimum Variance

### Risk Management
- Volatility targeting (10% annualized)
- Per-position stop-loss (−5%) and take-profit (+10%)
- Portfolio drawdown control (−15% trigger)
- Max 25% per-asset exposure
- 1.2× leverage constraint
- Transaction costs (10 bps) + slippage (5 bps)

### Performance Metrics
Sharpe, Sortino, Calmar, Information Ratio, VaR, CVaR, Max Drawdown, Win Rate, Profit Factor, Expectancy, Turnover

### Premium Interactive Dashboard
A Figma-inspired, institutional-grade Streamlit application featuring:
- **Glassmorphism UI** with deep navy/charcoal styling and JetBrains Mono typography
- **Interactive Configuration** (sidebar) for parameters, risk limits, and costs
- **Real-time Analytics** across 5 tabs (Overview, Performance, Risk, Trades, Optimization)
- **Publication-Quality Plotly Charts**: Equity curve, drawdown, rolling Sharpe, rolling volatility, monthly return heatmap, correlation matrix, asset allocation, efficient frontier, risk contribution, trade history

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run backtest from CLI
python -c "from src.backtester import Backtester; bt = Backtester(); result = bt.run()"

# 3. Launch interactive dashboard
streamlit run app.py
```

---

## Project Structure

```
QUANT/
├── data/                  # Cached CSV data
├── figures/               # Generated charts
├── src/
│   ├── loader.py          # Data fetching & cleaning
│   ├── indicators.py      # Technical indicators & risk metrics
│   ├── strategy.py        # 5 alpha signal generators
│   ├── optimizer.py       # Portfolio optimization (5 methods)
│   ├── risk.py            # Risk management layer
│   ├── execution.py       # Trade execution with costs
│   ├── backtester.py      # Event-driven backtesting engine
│   ├── analytics.py       # Performance metrics & matplotlib charts
│   └── dashboard.py       # Plotly chart components
├── tests/
│   └── test_smoke.py      # Smoke tests
├── app.py                 # Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Tech Stack

Python • Pandas • NumPy • SciPy • yfinance • Matplotlib • Seaborn • Plotly • Streamlit

