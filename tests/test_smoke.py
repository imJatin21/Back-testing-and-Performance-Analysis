"""
test_smoke.py — Smoke tests for all modules.

Runs a mini-backtest on synthetic data to verify the pipeline works end-to-end.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import pytest


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """Generate 500 days of synthetic OHLCV data for 3 assets."""
    np.random.seed(42)
    n_days = 500
    dates = pd.bdate_range("2020-01-01", periods=n_days)

    data = {}
    for asset in ["ASSET_A", "ASSET_B", "ASSET_C"]:
        price = 100 * np.exp(np.cumsum(np.random.randn(n_days) * 0.01))
        df = pd.DataFrame({
            "Open":      price * (1 + np.random.randn(n_days) * 0.002),
            "High":      price * (1 + np.abs(np.random.randn(n_days) * 0.005)),
            "Low":       price * (1 - np.abs(np.random.randn(n_days) * 0.005)),
            "Close":     price,
            "Adj Close": price,
            "Volume":    np.random.randint(1e6, 1e7, n_days),
        }, index=dates)
        df.index.name = "Date"
        data[asset] = df

    return data


@pytest.fixture
def close_matrix(synthetic_data):
    return pd.DataFrame({k: v["Close"] for k, v in synthetic_data.items()})


@pytest.fixture
def returns_matrix(close_matrix):
    return close_matrix.pct_change().dropna()


# ─── Tests ────────────────────────────────────────────────────────────────────

class TestIndicators:
    def test_compute_all(self, synthetic_data):
        from src.indicators import Indicators
        ind = Indicators(benchmark_close=synthetic_data["ASSET_A"]["Close"])
        result = ind.compute_all(synthetic_data["ASSET_A"])
        assert "rsi" in result.columns
        assert "sma20" in result.columns
        assert "atr_14" in result.columns
        assert len(result) == len(synthetic_data["ASSET_A"])

    def test_compute_all_assets(self, synthetic_data):
        from src.indicators import compute_all_assets
        result = compute_all_assets(synthetic_data, benchmark_key="ASSET_A")
        assert set(result.keys()) == set(synthetic_data.keys())


class TestStrategy:
    def test_generate_signals(self, synthetic_data):
        from src.indicators import compute_all_assets
        from src.strategy import generate_signals
        enriched = compute_all_assets(synthetic_data, benchmark_key="ASSET_A")
        signals = generate_signals(enriched)
        for asset_key, sig_df in signals.items():
            assert "composite_signal" in sig_df.columns
            assert set(sig_df["composite_signal"].unique()).issubset({-1, 0, 1})


class TestOptimizer:
    def test_all_methods(self, returns_matrix):
        from src.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer(returns_matrix, max_weight=0.5)
        results = opt.run_all()
        assert len(results) == 5
        for name, summary in results.items():
            w = list(summary["weights"].values())
            assert abs(sum(w) - 1.0) < 0.05, f"{name}: weights don't sum to ~1"
            assert np.isfinite(summary["sharpe_ratio"])

    def test_efficient_frontier(self, returns_matrix):
        from src.optimizer import PortfolioOptimizer
        opt = PortfolioOptimizer(returns_matrix, max_weight=0.5)
        frontier = opt.efficient_frontier(20)
        assert len(frontier) > 0
        assert "return" in frontier.columns
        assert "volatility" in frontier.columns


class TestRisk:
    def test_risk_manager(self):
        from src.risk import RiskManager
        rm = RiskManager()
        assert rm.vol_target_scalar(0.20) == pytest.approx(0.5)
        assert rm.vol_target_scalar(0.05) == pytest.approx(1.2)  # capped at leverage

    def test_clip_weights(self):
        from src.risk import RiskManager
        rm = RiskManager(max_weight=0.25)
        w = np.array([0.5, 0.3, 0.2])
        clipped = rm.clip_weights(w)
        assert np.all(clipped <= 0.25 + 1e-10)


class TestExecution:
    def test_execute_rebalance(self):
        from src.execution import ExecutionEngine
        ee = ExecutionEngine(transaction_cost=0.001, slippage=0.0005)
        current = np.array([0.5, 0.3, 0.2])
        target = np.array([0.33, 0.33, 0.34])
        new_w, cost = ee.execute_rebalance(
            "2020-01-01", current, target, 1_000_000, ["A", "B", "C"]
        )
        assert np.allclose(new_w, target)
        assert cost > 0
        assert len(ee.get_trade_log()) > 0


class TestAnalytics:
    def test_metrics(self):
        from src.analytics import Analytics
        np.random.seed(42)
        n = 500
        dates = pd.bdate_range("2020-01-01", periods=n)
        pv = pd.Series(1e6 * np.exp(np.cumsum(np.random.randn(n) * 0.005)),
                       index=dates)
        ret = pv.pct_change().fillna(0)
        an = Analytics(pv, ret)
        summary = an.summary()
        assert "Sharpe Ratio" in summary
        assert "Max Drawdown" in summary
        assert np.isfinite(an.sharpe_ratio())
        assert an.max_drawdown() < 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
