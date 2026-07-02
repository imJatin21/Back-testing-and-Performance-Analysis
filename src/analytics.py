"""
analytics.py — Performance metrics and visualization.

Computes institutional-grade metrics (Sharpe, Sortino, Calmar, VaR, CVaR,
win rate, profit factor, etc.) and generates publication-quality charts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

# Style
plt.rcParams.update({
    "figure.facecolor": "#0e1117",
    "axes.facecolor": "#0e1117",
    "axes.edgecolor": "#333",
    "axes.labelcolor": "#ccc",
    "text.color": "#ccc",
    "xtick.color": "#999",
    "ytick.color": "#999",
    "grid.color": "#222",
    "grid.alpha": 0.5,
    "font.family": "sans-serif",
    "font.size": 10,
})

ACCENT = "#00d4aa"
RED    = "#ff4b6e"
GOLD   = "#ffd700"
BLUE   = "#4da6ff"
COLORS = ["#00d4aa", "#ff4b6e", "#ffd700", "#4da6ff",
          "#c084fc", "#fb923c", "#38bdf8"]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Metrics
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Analytics:
    """Compute performance, risk, and trading metrics."""

    def __init__(self, portfolio_value: pd.Series,
                 daily_returns: pd.Series,
                 benchmark_value: pd.Series = None,
                 trade_log: pd.DataFrame = None,
                 weights_history: pd.DataFrame = None,
                 rf_annual: float = 0.0):
        self.pv = portfolio_value
        self.ret = daily_returns
        self.bench = benchmark_value
        self.trades = trade_log
        self.weights = weights_history
        self.rf = rf_annual

    # ── Return Metrics ────────────────────────────────────────────────────

    def total_return(self) -> float:
        return self.pv.iloc[-1] / self.pv.iloc[0] - 1

    def cagr(self) -> float:
        n_years = (self.pv.index[-1] - self.pv.index[0]).days / 365.25
        return (self.pv.iloc[-1] / self.pv.iloc[0]) ** (1 / n_years) - 1 if n_years > 0 else 0

    def annual_return(self) -> float:
        return self.ret.mean() * 252

    def monthly_returns(self) -> pd.Series:
        return self.pv.resample("ME").last().pct_change()

    # ── Risk Metrics ──────────────────────────────────────────────────────

    def annual_volatility(self) -> float:
        return self.ret.std() * np.sqrt(252)

    def max_drawdown(self) -> float:
        peak = self.pv.expanding().max()
        dd = (self.pv - peak) / peak
        return dd.min()

    def drawdown_series(self) -> pd.Series:
        peak = self.pv.expanding().max()
        return (self.pv - peak) / peak

    def var_95(self) -> float:
        return np.percentile(self.ret.dropna(), 5)

    def cvar_95(self) -> float:
        var = self.var_95()
        return self.ret[self.ret <= var].mean()

    # ── Risk-Adjusted Metrics ─────────────────────────────────────────────

    def sharpe_ratio(self) -> float:
        rf_daily = (1 + self.rf) ** (1/252) - 1
        excess = self.ret - rf_daily
        return excess.mean() / excess.std() * np.sqrt(252) if excess.std() > 0 else 0

    def sortino_ratio(self) -> float:
        rf_daily = (1 + self.rf) ** (1/252) - 1
        excess = self.ret - rf_daily
        downside = excess[excess < 0].std()
        return excess.mean() / downside * np.sqrt(252) if downside > 0 else 0

    def calmar_ratio(self) -> float:
        mdd = abs(self.max_drawdown())
        return self.cagr() / mdd if mdd > 0 else 0

    def information_ratio(self) -> float:
        if self.bench is None:
            return np.nan
        bench_ret = self.bench.pct_change().fillna(0)
        active = self.ret - bench_ret
        te = active.std() * np.sqrt(252)
        return active.mean() * 252 / te if te > 0 else 0

    # ── Trading Metrics ───────────────────────────────────────────────────

    def win_rate(self) -> float:
        wins = (self.ret > 0).sum()
        total = (self.ret != 0).sum()
        return wins / total if total > 0 else 0

    def profit_factor(self) -> float:
        gains = self.ret[self.ret > 0].sum()
        losses = abs(self.ret[self.ret < 0].sum())
        return gains / losses if losses > 0 else np.inf

    def avg_gain(self) -> float:
        g = self.ret[self.ret > 0]
        return g.mean() if len(g) > 0 else 0

    def avg_loss(self) -> float:
        l = self.ret[self.ret < 0]
        return l.mean() if len(l) > 0 else 0

    def expectancy(self) -> float:
        wr = self.win_rate()
        return wr * self.avg_gain() + (1 - wr) * self.avg_loss()

    # ── Portfolio Metrics ─────────────────────────────────────────────────

    def total_trades(self) -> int:
        return len(self.trades) if self.trades is not None else 0

    def turnover(self) -> float:
        if self.weights is None:
            return 0
        return self.weights.diff().abs().sum(axis=1).mean() * 252

    def rolling_sharpe(self, window: int = 60) -> pd.Series:
        rf_daily = (1 + self.rf) ** (1/252) - 1
        excess = self.ret - rf_daily
        return (excess.rolling(window).mean() /
                excess.rolling(window).std()) * np.sqrt(252)

    def rolling_volatility(self, window: int = 60) -> pd.Series:
        return self.ret.rolling(window).std() * np.sqrt(252)

    # ── Summary Report ────────────────────────────────────────────────────

    def summary(self) -> dict:
        """Return all metrics as a dict."""
        return {
            "Total Return":       f"{self.total_return():.2%}",
            "CAGR":               f"{self.cagr():.2%}",
            "Annual Volatility":  f"{self.annual_volatility():.2%}",
            "Sharpe Ratio":       f"{self.sharpe_ratio():.2f}",
            "Sortino Ratio":      f"{self.sortino_ratio():.2f}",
            "Calmar Ratio":       f"{self.calmar_ratio():.2f}",
            "Max Drawdown":       f"{self.max_drawdown():.2%}",
            "VaR (95%)":          f"{self.var_95():.4f}",
            "CVaR (95%)":         f"{self.cvar_95():.4f}",
            "Win Rate":           f"{self.win_rate():.2%}",
            "Profit Factor":      f"{self.profit_factor():.2f}",
            "Expectancy":         f"{self.expectancy():.6f}",
            "Information Ratio":  f"{self.information_ratio():.2f}",
            "Annual Turnover":    f"{self.turnover():.2f}",
            "Total Trades":       f"{self.total_trades():,}",
        }

    def print_report(self):
        print("\n" + "=" * 50)
        print("  PERFORMANCE REPORT")
        print("=" * 50)
        for k, v in self.summary().items():
            print(f"  {k:22s} : {v}")
        print("=" * 50)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Visualization
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class Visualizer:
    """Generate publication-quality charts for backtest results."""

    def __init__(self, result, analytics: Analytics,
                 fig_dir: str = None):
        self.result = result
        self.an = analytics
        if fig_dir is None:
            fig_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "figures")
        self.fig_dir = fig_dir
        os.makedirs(self.fig_dir, exist_ok=True)

    def _save(self, fig, name):
        path = os.path.join(self.fig_dir, f"{name}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        print(f"  → Saved {path}")

    # 1. Equity Curve
    def plot_equity_curve(self):
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(self.an.pv.index, self.an.pv / 1e6, color=ACCENT, lw=1.5,
                label="Strategy")
        if self.an.bench is not None:
            ax.plot(self.an.bench.index, self.an.bench / 1e6, color="#555",
                    lw=1, ls="--", label="Benchmark (EW)")
        ax.set_title("Portfolio Equity Curve", fontsize=14, fontweight="bold")
        ax.set_ylabel("Portfolio Value ($M)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("$%.2fM"))
        ax.legend(frameon=False)
        ax.grid(True)
        self._save(fig, "equity_curve")

    # 2. Drawdown
    def plot_drawdown(self):
        dd = self.an.drawdown_series()
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.fill_between(dd.index, dd, 0, color=RED, alpha=0.4)
        ax.plot(dd.index, dd, color=RED, lw=0.8)
        ax.set_title("Underwater (Drawdown) Chart", fontsize=14, fontweight="bold")
        ax.set_ylabel("Drawdown")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(True)
        self._save(fig, "drawdown")

    # 3. Rolling Sharpe
    def plot_rolling_sharpe(self):
        rs = self.an.rolling_sharpe(60)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(rs.index, rs, color=GOLD, lw=1)
        ax.axhline(0, color="#555", lw=0.8, ls="--")
        ax.set_title("Rolling Sharpe Ratio (60-day)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Sharpe")
        ax.grid(True)
        self._save(fig, "rolling_sharpe")

    # 4. Rolling Volatility
    def plot_rolling_volatility(self):
        rv = self.an.rolling_volatility(60)
        fig, ax = plt.subplots(figsize=(14, 4))
        ax.plot(rv.index, rv, color=BLUE, lw=1)
        ax.set_title("Rolling Volatility (60-day, annualized)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Volatility")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(True)
        self._save(fig, "rolling_volatility")

    # 5. Monthly Returns Heatmap
    def plot_monthly_heatmap(self):
        mr = self.an.monthly_returns()
        mr = mr.dropna()
        pivot = pd.DataFrame({
            "Year": mr.index.year,
            "Month": mr.index.month,
            "Return": mr.values,
        })
        pivot = pivot.pivot_table(index="Year", columns="Month",
                                   values="Return", aggfunc="sum")
        pivot.columns = ["Jan","Feb","Mar","Apr","May","Jun",
                         "Jul","Aug","Sep","Oct","Nov","Dec"][:len(pivot.columns)]

        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(pivot, annot=True, fmt=".1%", center=0,
                    cmap="RdYlGn", linewidths=0.5, ax=ax,
                    cbar_kws={"format": mticker.PercentFormatter(1.0)})
        ax.set_title("Monthly Returns Heatmap", fontsize=14, fontweight="bold")
        self._save(fig, "monthly_heatmap")

    # 6. Correlation Matrix
    def plot_correlation(self):
        if self.result.close_matrix is None:
            return
        corr = self.result.close_matrix.pct_change().corr()
        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm",
                    vmin=-1, vmax=1, center=0, square=True,
                    linewidths=0.5, ax=ax)
        ax.set_title("Asset Correlation Matrix", fontsize=14, fontweight="bold")
        self._save(fig, "correlation_matrix")

    # 7. Asset Allocation Over Time
    def plot_allocation(self):
        if self.result.weights_history is None:
            return
        w = self.result.weights_history
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.stackplot(w.index, *[w[c] for c in w.columns],
                     labels=w.columns, colors=COLORS[:len(w.columns)],
                     alpha=0.8)
        ax.set_title("Asset Allocation Over Time", fontsize=14, fontweight="bold")
        ax.set_ylabel("Weight")
        ax.legend(loc="upper left", frameon=False, fontsize=8, ncol=len(w.columns))
        ax.set_ylim(0, 1.3)
        ax.grid(True)
        self._save(fig, "allocation")

    # 8. Risk Contribution Pie
    def plot_risk_contribution(self):
        if self.result.weights_history is None or self.result.close_matrix is None:
            return
        w = self.result.weights_history.iloc[-1].values
        ret = self.result.close_matrix.pct_change().dropna()
        cov = ret.cov().values * 252
        port_vol = np.sqrt(w @ cov @ w)
        marginal = cov @ w
        risk_contrib = w * marginal / port_vol if port_vol > 0 else w

        fig, ax = plt.subplots(figsize=(8, 8))
        rc = np.abs(risk_contrib)
        if rc.sum() > 0:
            ax.pie(rc, labels=self.result.close_matrix.columns,
                   autopct="%1.1f%%", colors=COLORS[:len(rc)],
                   textprops={"color": "white"})
        ax.set_title("Risk Contribution (Latest)", fontsize=14, fontweight="bold")
        self._save(fig, "risk_contribution")

    # 9. Efficient Frontier (requires optimizer)
    def plot_efficient_frontier(self):
        from src.optimizer import PortfolioOptimizer
        ret = self.result.close_matrix.pct_change().dropna().iloc[-252:]
        opt = PortfolioOptimizer(ret)
        frontier = opt.efficient_frontier(50)
        if frontier.empty:
            return

        fig, ax = plt.subplots(figsize=(10, 7))
        sc = ax.scatter(frontier["volatility"], frontier["return"],
                        c=frontier["sharpe"], cmap="viridis", s=20)
        plt.colorbar(sc, ax=ax, label="Sharpe Ratio")

        # Mark key portfolios
        for method, marker, color in [
            ("max_sharpe", "*", GOLD),
            ("min_variance", "D", BLUE),
        ]:
            w = getattr(opt, method)()
            r = opt.portfolio_return(w)
            v = opt.portfolio_volatility(w)
            ax.scatter(v, r, marker=marker, s=200, c=color,
                       edgecolors="white", zorder=5, label=method.replace("_", " ").title())

        ax.set_xlabel("Annualized Volatility")
        ax.set_ylabel("Annualized Return")
        ax.set_title("Efficient Frontier", fontsize=14, fontweight="bold")
        ax.legend(frameon=False)
        ax.grid(True)
        self._save(fig, "efficient_frontier")

    # ── Generate All ──────────────────────────────────────────────────────

    def generate_all(self):
        """Generate all charts."""
        print("\n=== Generating Visualizations ===")
        self.plot_equity_curve()
        self.plot_drawdown()
        self.plot_rolling_sharpe()
        self.plot_rolling_volatility()
        self.plot_monthly_heatmap()
        self.plot_correlation()
        self.plot_allocation()
        self.plot_risk_contribution()
        self.plot_efficient_frontier()
        print(f"  [OK] All charts saved to {self.fig_dir}/")
