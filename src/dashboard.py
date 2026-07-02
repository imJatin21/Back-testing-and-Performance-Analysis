"""
dashboard.py — Streamlit dashboard components.

Provides page-rendering functions for the interactive dashboard.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# ─── Color Palette ────────────────────────────────────────────────────────────

BG       = "#0B0F19"
CARD_BG  = "#131929"
ACCENT   = "#00D4AA"
RED      = "#ff4b6e"
GOLD     = "#ffd700"
BLUE     = "#4da6ff"
PURPLE   = "#c084fc"
ORANGE   = "#fb923c"
COLORS   = [ACCENT, RED, GOLD, BLUE, PURPLE, ORANGE, "#38bdf8"]

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(family="Inter, system-ui, sans-serif", color="#8892a4"),
    margin=dict(l=50, r=30, t=60, b=40),
    hovermode="x unified",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chart Builders (return plotly Figure objects)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def chart_equity_curve(portfolio_value, benchmark_value=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=portfolio_value.index, y=portfolio_value / 1e6,
        name="Strategy", line=dict(color=ACCENT, width=2),
        fill="tozeroy", fillcolor="rgba(0,212,170,0.08)",
    ))
    if benchmark_value is not None:
        fig.add_trace(go.Scatter(
            x=benchmark_value.index, y=benchmark_value / 1e6,
            name="Benchmark (EW)", line=dict(color="#555", width=1, dash="dash"),
        ))
    fig.update_layout(
        title="Portfolio Equity Curve",
        yaxis_title="Value ($M)",
        yaxis_tickprefix="$", yaxis_ticksuffix="M",
        **LAYOUT_DEFAULTS,
    )
    return fig


def chart_drawdown(drawdown_series):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown_series.index, y=drawdown_series,
        fill="tozeroy", fillcolor="rgba(255,75,110,0.3)",
        line=dict(color=RED, width=1),
        name="Drawdown",
    ))
    fig.update_layout(
        title="Underwater (Drawdown) Chart",
        yaxis_title="Drawdown",
        yaxis_tickformat=".1%",
        **LAYOUT_DEFAULTS,
    )
    return fig


def chart_rolling_sharpe(rolling_sharpe):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index, y=rolling_sharpe,
        line=dict(color=GOLD, width=1.5), name="Rolling Sharpe",
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#555")
    fig.add_hline(y=1, line_dash="dot", line_color=ACCENT, opacity=0.5,
                  annotation_text="Sharpe=1")
    fig.update_layout(
        title="Rolling Sharpe Ratio (60-day)",
        yaxis_title="Sharpe Ratio",
        **LAYOUT_DEFAULTS,
    )
    return fig


def chart_rolling_volatility(rolling_vol):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=rolling_vol.index, y=rolling_vol,
        line=dict(color=BLUE, width=1.5), name="Volatility",
        fill="tozeroy", fillcolor="rgba(77,166,255,0.08)",
    ))
    fig.update_layout(
        title="Rolling Volatility (60-day, annualized)",
        yaxis_title="Volatility",
        yaxis_tickformat=".1%",
        **LAYOUT_DEFAULTS,
    )
    return fig


def chart_monthly_heatmap(monthly_returns):
    mr = monthly_returns.dropna()
    df = pd.DataFrame({
        "Year": mr.index.year,
        "Month": mr.index.month,
        "Return": mr.values,
    })
    month_names = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]
    pivot = df.pivot_table(index="Year", columns="Month",
                           values="Return", aggfunc="sum")
    pivot.columns = [month_names[c-1] for c in pivot.columns]

    fig = go.Figure(data=go.Heatmap(
        z=pivot.values * 100,
        x=pivot.columns,
        y=pivot.index.astype(str),
        text=[[f"{v:.1f}%" if not np.isnan(v) else "" for v in row]
              for row in pivot.values * 100],
        texttemplate="%{text}",
        colorscale="RdYlGn",
        zmid=0,
        colorbar=dict(title="%"),
    ))
    fig.update_layout(
        title="Monthly Returns Heatmap",
        yaxis=dict(autorange="reversed"),
        **LAYOUT_DEFAULTS,
    )
    return fig


def chart_correlation(close_matrix):
    corr = close_matrix.pct_change().corr()
    fig = go.Figure(data=go.Heatmap(
        z=corr.values,
        x=corr.columns,
        y=corr.index,
        text=[[f"{v:.2f}" for v in row] for row in corr.values],
        texttemplate="%{text}",
        colorscale="RdBu_r",
        zmid=0, zmin=-1, zmax=1,
    ))
    fig.update_layout(
        title="Asset Correlation Matrix",
        height=500,
        **LAYOUT_DEFAULTS,
    )
    return fig


def chart_allocation(weights_history):
    fig = go.Figure()
    for i, col in enumerate(weights_history.columns):
        fig.add_trace(go.Scatter(
            x=weights_history.index, y=weights_history[col],
            stackgroup="one", name=col,
            line=dict(width=0.5, color=COLORS[i % len(COLORS)]),
        ))
    fig.update_layout(
        title="Asset Allocation Over Time",
        yaxis_title="Weight",
        **LAYOUT_DEFAULTS,
    )
    return fig


def chart_efficient_frontier(optimizer):
    frontier = optimizer.efficient_frontier(50)
    if frontier.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=frontier["volatility"], y=frontier["return"],
        mode="markers",
        marker=dict(color=frontier["sharpe"], colorscale="Viridis",
                    size=8, colorbar=dict(title="Sharpe")),
        name="Frontier",
    ))

    # Mark max Sharpe & min variance
    for method, label, color, symbol in [
        ("max_sharpe", "Max Sharpe", GOLD, "star"),
        ("min_variance", "Min Variance", BLUE, "diamond"),
    ]:
        w = getattr(optimizer, method)()
        r = optimizer.portfolio_return(w)
        v = optimizer.portfolio_volatility(w)
        fig.add_trace(go.Scatter(
            x=[v], y=[r], mode="markers+text",
            marker=dict(color=color, size=16, symbol=symbol,
                        line=dict(color="white", width=1)),
            text=[label], textposition="top center",
            name=label,
        ))

    fig.update_layout(
        title="Efficient Frontier",
        xaxis_title="Annualized Volatility",
        yaxis_title="Annualized Return",
        xaxis_tickformat=".1%", yaxis_tickformat=".1%",
        **LAYOUT_DEFAULTS,
    )
    return fig


def chart_risk_contribution(weights, cov_matrix, asset_names):
    w = np.array(weights)
    cov = np.array(cov_matrix)
    port_vol = np.sqrt(w @ cov @ w)
    if port_vol > 0:
        marginal = cov @ w
        rc = w * marginal / port_vol
    else:
        rc = w

    fig = go.Figure(data=go.Bar(
        x=asset_names, y=np.abs(rc),
        marker_color=COLORS[:len(asset_names)],
    ))
    fig.update_layout(
        title="Risk Contribution by Asset",
        yaxis_title="Risk Contribution",
        **LAYOUT_DEFAULTS,
    )
    return fig


def chart_trade_history(trade_log):
    if trade_log is None or trade_log.empty:
        return go.Figure()
    buys = trade_log[trade_log["direction"] == "BUY"]
    sells = trade_log[trade_log["direction"] == "SELL"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=buys["date"], y=buys["notional"],
        mode="markers", name="BUY",
        marker=dict(color=ACCENT, size=6, symbol="triangle-up"),
    ))
    fig.add_trace(go.Scatter(
        x=sells["date"], y=sells["notional"],
        mode="markers", name="SELL",
        marker=dict(color=RED, size=6, symbol="triangle-down"),
    ))
    fig.update_layout(
        title="Trade History",
        yaxis_title="Notional ($)",
        yaxis_tickprefix="$",
        **LAYOUT_DEFAULTS,
    )
    return fig


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Metric Cards (return styled HTML)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def metric_card(label: str, value: str, delta: str = None,
                good: bool = True) -> str:
    """Return an HTML metric card for Streamlit markdown."""
    delta_color = ACCENT if good else RED
    delta_html = (f'<span style="color:{delta_color};font-size:0.85rem">'
                  f'{delta}</span>') if delta else ""
    return f"""
    <div style="background:{CARD_BG};padding:18px 22px;border-radius:12px;
                border:1px solid #2a2e3e;min-width:160px;">
        <div style="color:#888;font-size:0.8rem;margin-bottom:4px;
                    text-transform:uppercase;letter-spacing:0.05em">{label}</div>
        <div style="color:white;font-size:1.6rem;font-weight:700">{value}</div>
        {delta_html}
    </div>
    """
