"""
app.py — Premium Streamlit Dashboard for Quantitative Trading Framework.

Figma-inspired glassmorphism design with professional financial terminal aesthetic.
Run:  streamlit run app.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import numpy as np
import pandas as pd

from src.loader import DataLoader, ASSET_CONFIG
from src.indicators import compute_all_assets
from src.strategy import generate_signals
from src.optimizer import PortfolioOptimizer
from src.backtester import Backtester, BacktestConfig
from src.analytics import Analytics
from src.dashboard import (
    chart_equity_curve, chart_drawdown, chart_rolling_sharpe,
    chart_rolling_volatility, chart_monthly_heatmap, chart_correlation,
    chart_allocation, chart_efficient_frontier, chart_risk_contribution,
    chart_trade_history,
    BG, CARD_BG, ACCENT, RED, GOLD, BLUE,
)

# ─── Page Config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="QTS | Quantitative Trading System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium CSS ──────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');

    /* ── Base ────────────────────────────────────────────── */
    .stApp {
        background: #0B0F19;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    .block-container {
        padding-top: 1.5rem !important;
        max-width: 1400px;
    }

    /* ── Sidebar ─────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #080C14 0%, #0D1120 100%);
        border-right: 1px solid rgba(255,255,255,0.04);
    }
    section[data-testid="stSidebar"] .stMarkdown h2 {
        color: #fff;
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding: 4px 0;
    }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stTextInput label {
        color: #8892a4 !important;
        font-size: 0.78rem !important;
        font-weight: 500 !important;
        letter-spacing: 0.02em;
    }

    /* ── Logo Header ─────────────────────────────────────── */
    .logo-container {
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 20px 0 24px 0;
    }
    .logo-icon {
        width: 44px;
        height: 44px;
        background: linear-gradient(135deg, #00D4AA 0%, #00A885 100%);
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.3rem;
        font-weight: 800;
        color: #0B0F19;
        box-shadow: 0 4px 20px rgba(0, 212, 170, 0.25);
    }
    .logo-text {
        display: flex;
        flex-direction: column;
    }
    .logo-text .name {
        color: #fff;
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    .logo-text .sub {
        color: #4a5568;
        font-size: 0.7rem;
        font-weight: 500;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }

    /* ── Hero Section ────────────────────────────────────── */
    .hero-section {
        background: linear-gradient(135deg, #0D1120 0%, #131929 50%, #0F1525 100%);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 20px;
        padding: 36px 40px;
        margin-bottom: 28px;
        position: relative;
        overflow: hidden;
    }
    .hero-section::before {
        content: '';
        position: absolute;
        top: -2px;
        left: 30%;
        right: 30%;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00D4AA, transparent);
        border-radius: 2px;
    }
    .hero-section h1 {
        color: #fff;
        font-size: 1.7rem;
        font-weight: 800;
        margin: 0 0 6px 0;
        letter-spacing: -0.02em;
    }
    .hero-section p {
        color: #5a6577;
        font-size: 0.9rem;
        margin: 0;
        font-weight: 400;
    }
    .hero-section .accent {
        color: #00D4AA;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 212, 170, 0.08);
        border: 1px solid rgba(0, 212, 170, 0.15);
        color: #00D4AA;
        font-size: 0.7rem;
        font-weight: 600;
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 14px;
        letter-spacing: 0.03em;
        text-transform: uppercase;
    }

    /* ── Metric Cards ────────────────────────────────────── */
    .metric-card {
        background: linear-gradient(145deg, #131929 0%, #0F1422 100%);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 16px;
        padding: 22px 24px;
        position: relative;
        overflow: hidden;
        transition: all 0.3s ease;
    }
    .metric-card:hover {
        border-color: rgba(0, 212, 170, 0.15);
        transform: translateY(-1px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }
    .metric-card .label {
        color: #5a6577;
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 8px;
    }
    .metric-card .value {
        color: #fff;
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.55rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .metric-card .value.positive { color: #00D4AA; }
    .metric-card .value.negative { color: #FF4B6E; }
    .metric-card .value.neutral  { color: #FFD700; }
    .metric-card .sub-value {
        color: #4a5568;
        font-size: 0.72rem;
        font-weight: 500;
        margin-top: 6px;
    }
    .metric-card .glow {
        position: absolute;
        top: 0;
        right: 0;
        width: 80px;
        height: 80px;
        border-radius: 50%;
        filter: blur(40px);
        opacity: 0.08;
    }
    .glow-teal   { background: #00D4AA; }
    .glow-red    { background: #FF4B6E; }
    .glow-gold   { background: #FFD700; }
    .glow-blue   { background: #4DA6FF; }

    /* ── Tabs ────────────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(11, 15, 25, 0.6);
        backdrop-filter: blur(10px);
        padding: 4px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.04);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px;
        color: #5a6577;
        padding: 10px 22px;
        font-weight: 500;
        font-size: 0.85rem;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #8892a4;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #1a2235 0%, #151d2e 100%) !important;
        color: #fff !important;
        font-weight: 600;
        box-shadow: 0 2px 12px rgba(0,0,0,0.2);
    }

    /* ── Section Headers ─────────────────────────────────── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin: 28px 0 18px 0;
    }
    .section-header .icon {
        width: 32px;
        height: 32px;
        background: rgba(0, 212, 170, 0.08);
        border: 1px solid rgba(0, 212, 170, 0.12);
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.9rem;
    }
    .section-header h3 {
        color: #fff;
        font-size: 1rem;
        font-weight: 600;
        margin: 0;
        letter-spacing: -0.01em;
    }
    .section-header .desc {
        color: #4a5568;
        font-size: 0.78rem;
        font-weight: 400;
        margin-left: auto;
    }

    /* ── Chart Container ─────────────────────────────────── */
    .chart-container {
        background: linear-gradient(145deg, #111827 0%, #0D1120 100%);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 16px;
        padding: 6px;
        margin-bottom: 20px;
    }

    /* ── Data Table ───────────────────────────────────────── */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }

    /* ── Streamlit Overrides ──────────────────────────────── */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, #131929 0%, #0F1422 100%);
        border: 1px solid rgba(255,255,255,0.05);
        padding: 18px 20px;
        border-radius: 14px;
    }
    div[data-testid="stMetric"] label {
        color: #5a6577 !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        color: #fff !important;
    }

    /* ── Landing Cards ───────────────────────────────────── */
    .asset-card {
        background: linear-gradient(145deg, #131929 0%, #0F1422 100%);
        border: 1px solid rgba(255,255,255,0.05);
        padding: 20px;
        border-radius: 14px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .asset-card:hover {
        border-color: rgba(0, 212, 170, 0.2);
        box-shadow: 0 4px 24px rgba(0, 212, 170, 0.06);
    }
    .asset-card .ticker {
        color: #00D4AA;
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.15rem;
    }
    .asset-card .name {
        color: #8892a4;
        font-size: 0.72rem;
        margin-top: 4px;
    }
    .asset-card .class-tag {
        display: inline-block;
        color: #4a5568;
        font-size: 0.6rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        margin-top: 8px;
        background: rgba(255,255,255,0.03);
        padding: 3px 10px;
        border-radius: 6px;
    }
    
    /* ── Button ───────────────────────────────────────────── */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #00D4AA 0%, #00A885 100%) !important;
        color: #0B0F19 !important;
        font-weight: 700 !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 12px 24px !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.02em;
        box-shadow: 0 4px 20px rgba(0, 212, 170, 0.2);
        transition: all 0.3s ease;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 28px rgba(0, 212, 170, 0.35) !important;
        transform: translateY(-1px);
    }

    /* ── Expander ─────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 10px !important;
        color: #8892a4 !important;
        font-weight: 500 !important;
        font-size: 0.82rem !important;
    }

    /* ── Divider ──────────────────────────────────────────── */
    .premium-divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, rgba(255,255,255,0.06), transparent);
        margin: 24px 0;
    }

    /* ── Status Bar ───────────────────────────────────────── */
    .status-bar {
        display: flex;
        align-items: center;
        gap: 20px;
        padding: 10px 20px;
        background: rgba(0, 212, 170, 0.04);
        border: 1px solid rgba(0, 212, 170, 0.08);
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .status-item {
        display: flex;
        align-items: center;
        gap: 6px;
        color: #5a6577;
        font-size: 0.72rem;
        font-weight: 500;
    }
    .status-dot {
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #00D4AA;
        box-shadow: 0 0 8px rgba(0, 212, 170, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ─────────────────────────────────────────────────────────────────

st.sidebar.markdown("""
<div class="logo-container">
    <div class="logo-icon">Q</div>
    <div class="logo-text">
        <span class="name">QuantTrader</span>
        <span class="sub">Multi-Asset System</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## Configuration")

with st.sidebar.expander("Date Range", expanded=True):
    start_date = st.text_input("Start Date", "2012-01-01")
    end_date = st.text_input("End Date", "2026-01-01")

with st.sidebar.expander("Portfolio", expanded=True):
    initial_capital = st.number_input("Initial Capital ($)", value=1_000_000,
                                       step=100_000, format="%d")
    optimizer_method = st.selectbox("Optimization Method", [
        "max_sharpe", "min_variance", "equal_weight",
        "risk_parity", "mean_variance"
    ], format_func=lambda x: x.replace("_", " ").title())
    rebalance_freq = st.slider("Rebalance Frequency (days)", 5, 63, 21)

with st.sidebar.expander("Risk Controls", expanded=False):
    vol_target = st.slider("Vol Target (%)", 5, 30, 10) / 100
    stop_loss = st.slider("Stop Loss (%)", 1, 20, 5) / -100
    take_profit = st.slider("Take Profit (%)", 5, 50, 10) / 100
    max_drawdown = st.slider("Max Drawdown Trigger (%)", 5, 30, 15) / -100
    max_weight = st.slider("Max Weight per Asset (%)", 10, 50, 25) / 100
    max_leverage = st.slider("Max Leverage", 1.0, 2.0, 1.2, 0.1)

with st.sidebar.expander("Transaction Costs", expanded=False):
    transaction_cost = st.slider("Transaction Cost (bps)", 0, 50, 10) / 10000
    slippage = st.slider("Slippage (bps)", 0, 20, 5) / 10000

st.sidebar.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

run_button = st.sidebar.button("Run Backtest", use_container_width=True,
                                type="primary")


# ─── Metric Card Builder ─────────────────────────────────────────────────────

def render_metric(label, value, color_class="neutral", sub_text="", glow="glow-teal"):
    return f"""
    <div class="metric-card">
        <div class="glow {glow}"></div>
        <div class="label">{label}</div>
        <div class="value {color_class}">{value}</div>
        {f'<div class="sub-value">{sub_text}</div>' if sub_text else ''}
    </div>
    """


# ─── Hero Header ─────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-section">
    <div class="hero-badge">Institutional Grade</div>
    <h1>Multi-Asset <span class="accent">Trading</span> Framework</h1>
    <p>Portfolio optimization &bull; Risk analytics &bull; Backtesting engine &bull; 7 asset classes</p>
</div>
""", unsafe_allow_html=True)


# ─── Run Backtest ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def run_backtest(config_dict):
    cfg = BacktestConfig(**config_dict)
    bt = Backtester(config=cfg)
    return bt.run()


if run_button or "result" in st.session_state:
    if run_button:
        config_dict = dict(
            initial_capital=initial_capital,
            rebalance_freq=rebalance_freq,
            lookback=252,
            optimizer_method=optimizer_method,
            vol_target=vol_target,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_drawdown=max_drawdown,
            max_weight=max_weight,
            max_leverage=max_leverage,
            transaction_cost=transaction_cost,
            slippage=slippage,
            start_date=start_date,
            end_date=end_date,
        )
        with st.spinner("Running backtest..."):
            result = run_backtest(config_dict)
        st.session_state["result"] = result
        st.session_state["config_dict"] = config_dict
    else:
        result = st.session_state["result"]

    # Build analytics
    an = Analytics(
        portfolio_value=result.portfolio_value,
        daily_returns=result.daily_returns,
        benchmark_value=result.benchmark_value,
        trade_log=result.trade_log,
        weights_history=result.weights_history,
    )

    # Status bar
    n_days = len(result.portfolio_value)
    st.markdown(f"""
    <div class="status-bar">
        <div class="status-item"><div class="status-dot"></div> Backtest Complete</div>
        <div class="status-item">{n_days:,} trading days</div>
        <div class="status-item">7 assets</div>
        <div class="status-item">{result.portfolio_value.index[0].strftime('%b %Y')} - {result.portfolio_value.index[-1].strftime('%b %Y')}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Tabs ──────────────────────────────────────────────────────────────

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "Performance", "Risk",
        "Trades", "Optimization"
    ])

    # ─── Tab 1: Overview ──────────────────────────────────────────────────

    with tab1:
        total_ret = an.total_return()
        cagr = an.cagr()
        sharpe = an.sharpe_ratio()
        mdd = an.max_drawdown()
        win = an.win_rate()

        cols = st.columns(5)
        cards = [
            ("Total Return", f"{total_ret:+.1%}",
             "positive" if total_ret >= 0 else "negative",
             "Since inception", "glow-teal" if total_ret >= 0 else "glow-red"),
            ("CAGR", f"{cagr:+.1%}",
             "positive" if cagr >= 0 else "negative",
             "Annualized", "glow-teal" if cagr >= 0 else "glow-red"),
            ("Sharpe Ratio", f"{sharpe:.2f}",
             "positive" if sharpe >= 1 else "neutral",
             "Risk-adjusted", "glow-gold"),
            ("Max Drawdown", f"{mdd:.1%}",
             "negative", "Peak to trough", "glow-red"),
            ("Win Rate", f"{win:.1%}",
             "positive" if win >= 0.5 else "neutral",
             "Daily", "glow-blue"),
        ]
        for col, (label, value, color, sub, glow) in zip(cols, cards):
            with col:
                st.markdown(render_metric(label, value, color, sub, glow),
                            unsafe_allow_html=True)

        st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

        # Equity Curve
        st.markdown("""
        <div class="section-header">
            <div class="icon">📈</div>
            <h3>Equity Curve</h3>
            <span class="desc">Strategy vs Equal-Weight Benchmark</span>
        </div>
        """, unsafe_allow_html=True)

        st.plotly_chart(chart_equity_curve(result.portfolio_value,
                                            result.benchmark_value),
                        use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="section-header">
                <div class="icon">📊</div>
                <h3>Asset Allocation</h3>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(chart_allocation(result.weights_history),
                            use_container_width=True)
        with c2:
            st.markdown("""
            <div class="section-header">
                <div class="icon">🔗</div>
                <h3>Correlation Matrix</h3>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(chart_correlation(result.close_matrix),
                            use_container_width=True)

    # ─── Tab 2: Performance ───────────────────────────────────────────────

    with tab2:
        cols = st.columns(4)
        perf = [
            ("Sortino Ratio", f"{an.sortino_ratio():.2f}", "neutral", "glow-gold"),
            ("Calmar Ratio", f"{an.calmar_ratio():.2f}", "neutral", "glow-gold"),
            ("Info Ratio", f"{an.information_ratio():.2f}", "neutral", "glow-blue"),
            ("Profit Factor", f"{an.profit_factor():.2f}", "neutral", "glow-teal"),
        ]
        for col, (label, value, color, glow) in zip(cols, perf):
            with col:
                st.markdown(render_metric(label, value, color, "", glow),
                            unsafe_allow_html=True)

        st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="section-header">
            <div class="icon">📅</div>
            <h3>Monthly Returns</h3>
            <span class="desc">Year-over-Year Breakdown</span>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(chart_monthly_heatmap(an.monthly_returns()),
                        use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="section-header">
                <div class="icon">⚡</div>
                <h3>Rolling Sharpe</h3>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(chart_rolling_sharpe(an.rolling_sharpe()),
                            use_container_width=True)
        with c2:
            st.markdown("""
            <div class="section-header">
                <div class="icon">📉</div>
                <h3>Drawdown</h3>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(chart_drawdown(an.drawdown_series()),
                            use_container_width=True)

    # ─── Tab 3: Risk ─────────────────────────────────────────────────────

    with tab3:
        cols = st.columns(4)
        risk = [
            ("Volatility", f"{an.annual_volatility():.1%}", "neutral", "glow-blue"),
            ("VaR (95%)", f"{an.var_95():.4f}", "negative", "glow-red"),
            ("CVaR (95%)", f"{an.cvar_95():.4f}", "negative", "glow-red"),
            ("Max Drawdown", f"{an.max_drawdown():.1%}", "negative", "glow-red"),
        ]
        for col, (label, value, color, glow) in zip(cols, risk):
            with col:
                st.markdown(render_metric(label, value, color, "", glow),
                            unsafe_allow_html=True)

        st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="section-header">
            <div class="icon">📊</div>
            <h3>Rolling Volatility</h3>
            <span class="desc">60-day annualized</span>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(chart_rolling_volatility(an.rolling_volatility()),
                        use_container_width=True)

        if result.weights_history is not None and result.close_matrix is not None:
            st.markdown("""
            <div class="section-header">
                <div class="icon">🎯</div>
                <h3>Risk Contribution</h3>
                <span class="desc">Per-asset contribution to total portfolio risk</span>
            </div>
            """, unsafe_allow_html=True)
            latest_w = result.weights_history.iloc[-1].values
            ret = result.close_matrix.pct_change().dropna()
            cov = ret.cov().values * 252
            st.plotly_chart(
                chart_risk_contribution(latest_w, cov,
                                        list(result.close_matrix.columns)),
                use_container_width=True,
            )

    # ─── Tab 4: Trades ────────────────────────────────────────────────────

    with tab4:
        total_cost = result.trade_log["cost"].sum() if not result.trade_log.empty else 0
        cols = st.columns(3)
        trade_cards = [
            ("Total Trades", f"{an.total_trades():,}", "neutral", "glow-teal"),
            ("Annual Turnover", f"{an.turnover():.2f}", "neutral", "glow-blue"),
            ("Total Costs", f"${total_cost:,.0f}", "negative", "glow-red"),
        ]
        for col, (label, value, color, glow) in zip(cols, trade_cards):
            with col:
                st.markdown(render_metric(label, value, color, "", glow),
                            unsafe_allow_html=True)

        st.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)

        st.markdown("""
        <div class="section-header">
            <div class="icon">💹</div>
            <h3>Trade History</h3>
            <span class="desc">Buy & Sell Executions</span>
        </div>
        """, unsafe_allow_html=True)
        st.plotly_chart(chart_trade_history(result.trade_log),
                        use_container_width=True)

        if not result.trade_log.empty:
            st.markdown("""
            <div class="section-header">
                <div class="icon">📋</div>
                <h3>Trade Log</h3>
                <span class="desc">Last 100 trades</span>
            </div>
            """, unsafe_allow_html=True)
            st.dataframe(
                result.trade_log.tail(100).style.format({
                    "weight_change": "{:.4f}",
                    "notional": "${:,.2f}",
                    "cost": "${:,.2f}",
                }),
                use_container_width=True,
                height=400,
            )

    # ─── Tab 5: Optimization ─────────────────────────────────────────────

    with tab5:
        st.markdown("""
        <div class="section-header">
            <div class="icon">🎯</div>
            <h3>Efficient Frontier</h3>
            <span class="desc">Risk-Return Tradeoff Space</span>
        </div>
        """, unsafe_allow_html=True)

        ret = result.close_matrix.pct_change().dropna().iloc[-252:]
        opt = PortfolioOptimizer(ret, max_weight=max_weight)

        st.plotly_chart(chart_efficient_frontier(opt),
                        use_container_width=True)

        st.markdown("""
        <div class="section-header">
            <div class="icon">📊</div>
            <h3>Optimization Comparison</h3>
            <span class="desc">All 5 portfolio construction methods</span>
        </div>
        """, unsafe_allow_html=True)

        results_all = opt.run_all()
        comparison = pd.DataFrame(results_all).T
        st.dataframe(comparison[["annual_return", "annual_volatility",
                                  "sharpe_ratio"]].style.format({
            "annual_return": "{:.2%}",
            "annual_volatility": "{:.2%}",
            "sharpe_ratio": "{:.2f}",
        }), use_container_width=True)

        # Weight comparison bar chart
        st.markdown("""
        <div class="section-header">
            <div class="icon">📐</div>
            <h3>Weight Distribution</h3>
            <span class="desc">Across optimization methods</span>
        </div>
        """, unsafe_allow_html=True)

        weight_df = pd.DataFrame(
            {k: v["weights"] for k, v in results_all.items()}
        )
        import plotly.graph_objects as go
        colors_list = [ACCENT, RED, GOLD, BLUE, "#c084fc"]
        fig = go.Figure()
        for i, method in enumerate(weight_df.columns):
            fig.add_trace(go.Bar(
                name=method, x=weight_df.index, y=weight_df[method],
                marker_color=colors_list[i % len(colors_list)],
                marker_line=dict(width=0),
            ))
        fig.update_layout(
            barmode="group",
            template="plotly_dark",
            paper_bgcolor="#0B0F19", plot_bgcolor="#0B0F19",
            font=dict(family="Inter, system-ui, sans-serif", color="#8892a4"),
            margin=dict(l=50, r=30, t=30, b=40),
            legend=dict(orientation="h", y=1.08),
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Report Download ───────────────────────────────────────────────────
    st.sidebar.markdown('<div class="premium-divider"></div>', unsafe_allow_html=True)
    report_text = "QUANTITATIVE TRADING SYSTEM - PERFORMANCE REPORT\n"
    report_text += "=" * 50 + "\n\n"
    for k, v in an.summary().items():
        report_text += f"{k:25s}: {v}\n"
    st.sidebar.download_button(
        "Download Report",
        data=report_text,
        file_name="backtest_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

else:
    # ── Landing Page ──────────────────────────────────────────────────────

    st.markdown("""
    <div style="text-align:center; padding:50px 20px 30px;">
        <div style="display:inline-flex;align-items:center;gap:6px;
                    background:rgba(0,212,170,0.06);border:1px solid rgba(0,212,170,0.12);
                    color:#00D4AA;font-size:0.68rem;font-weight:600;padding:4px 14px;
                    border-radius:20px;letter-spacing:0.04em;text-transform:uppercase;
                    margin-bottom:20px;">
            Ready to Deploy
        </div>
        <h2 style="color:white;font-size:1.6rem;font-weight:800;margin:0 0 8px;
                   letter-spacing:-0.02em;">
            Configure & <span style="color:#00D4AA">Backtest</span>
        </h2>
        <p style="color:#4a5568;font-size:0.9rem;max-width:460px;margin:0 auto 36px;
                  line-height:1.6;">
            Set your strategy parameters in the sidebar, then click
            <strong style="color:#00D4AA">Run Backtest</strong> to analyze performance
            across the multi-asset universe.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Asset cards
    asset_cols = st.columns(len(ASSET_CONFIG))
    for col, (key, cfg) in zip(asset_cols, ASSET_CONFIG.items()):
        with col:
            st.markdown(f"""
            <div class="asset-card">
                <div class="ticker">{key}</div>
                <div class="name">{cfg['name']}</div>
                <div class="class-tag">{cfg['class']}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; padding:40px 20px 20px;">
        <div style="display:flex; justify-content:center; gap:40px; flex-wrap:wrap;">
            <div style="text-align:center;">
                <div style="color:#00D4AA;font-family:'JetBrains Mono',monospace;
                            font-size:1.5rem;font-weight:700;">5</div>
                <div style="color:#4a5568;font-size:0.7rem;font-weight:500;
                            text-transform:uppercase;letter-spacing:0.06em;margin-top:4px;">
                    Alpha Strategies</div>
            </div>
            <div style="text-align:center;">
                <div style="color:#FFD700;font-family:'JetBrains Mono',monospace;
                            font-size:1.5rem;font-weight:700;">5</div>
                <div style="color:#4a5568;font-size:0.7rem;font-weight:500;
                            text-transform:uppercase;letter-spacing:0.06em;margin-top:4px;">
                    Optimization Methods</div>
            </div>
            <div style="text-align:center;">
                <div style="color:#4DA6FF;font-family:'JetBrains Mono',monospace;
                            font-size:1.5rem;font-weight:700;">14+</div>
                <div style="color:#4a5568;font-size:0.7rem;font-weight:500;
                            text-transform:uppercase;letter-spacing:0.06em;margin-top:4px;">
                    Years of Data</div>
            </div>
            <div style="text-align:center;">
                <div style="color:#FF4B6E;font-family:'JetBrains Mono',monospace;
                            font-size:1.5rem;font-weight:700;">15+</div>
                <div style="color:#4a5568;font-size:0.7rem;font-weight:500;
                            text-transform:uppercase;letter-spacing:0.06em;margin-top:4px;">
                    Risk Metrics</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
