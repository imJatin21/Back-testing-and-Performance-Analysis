"""
backtester.py — Event-driven backtesting engine.

Orchestrates: DataLoader → Indicators → Strategy → Optimizer →
              RiskManager → ExecutionEngine → Analytics.
"""

import numpy as np
import pandas as pd

from src.loader import DataLoader
from src.indicators import Indicators, compute_all_assets
from src.strategy import CombinedStrategy, generate_signals
from src.optimizer import PortfolioOptimizer
from src.risk import RiskManager
from src.execution import ExecutionEngine


class BacktestConfig:
    """Configuration parameters for the backtester."""

    def __init__(self, **kwargs):
        self.initial_capital: float    = kwargs.get("initial_capital", 1_000_000)
        self.rebalance_freq: int       = kwargs.get("rebalance_freq", 21)  # ~monthly
        self.lookback: int             = kwargs.get("lookback", 252)       # 1-year window
        self.optimizer_method: str     = kwargs.get("optimizer_method", "max_sharpe")
        self.vol_target: float         = kwargs.get("vol_target", 0.10)
        self.stop_loss: float          = kwargs.get("stop_loss", -0.05)
        self.take_profit: float        = kwargs.get("take_profit", 0.10)
        self.max_drawdown: float       = kwargs.get("max_drawdown", -0.15)
        self.max_weight: float         = kwargs.get("max_weight", 0.25)
        self.max_leverage: float       = kwargs.get("max_leverage", 1.20)
        self.transaction_cost: float   = kwargs.get("transaction_cost", 0.001)
        self.slippage: float           = kwargs.get("slippage", 0.0005)
        self.start_date: str           = kwargs.get("start_date", None)
        self.end_date: str             = kwargs.get("end_date", None)


class BacktestResult:
    """Container for backtest results."""

    def __init__(self):
        self.portfolio_value: pd.Series = None
        self.daily_returns: pd.Series = None
        self.weights_history: pd.DataFrame = None
        self.trade_log: pd.DataFrame = None
        self.signals: dict = None
        self.data: dict = None
        self.close_matrix: pd.DataFrame = None
        self.config: BacktestConfig = None
        self.benchmark_value: pd.Series = None


class Backtester:
    """
    Full-pipeline backtester.

    Usage
    -----
    >>> bt = Backtester()
    >>> result = bt.run()
    """

    def __init__(self, config: BacktestConfig = None, data_dir: str = None):
        self.config = config or BacktestConfig()
        self.data_dir = data_dir

    def run(self, data: dict = None, close_matrix: pd.DataFrame = None) -> BacktestResult:
        """
        Execute the full backtest pipeline.

        Parameters
        ----------
        data : pre-loaded & cleaned data dict (optional; will fetch if None).
        close_matrix : pre-built close matrix (optional).
        """
        cfg = self.config
        result = BacktestResult()
        result.config = cfg

        # ── 1. Load Data ──────────────────────────────────────────────────
        if data is None:
            print("\n=== Step 1: Loading Data ===")
            loader = DataLoader(data_dir=self.data_dir)
            raw = loader.fetch_all()
            data = loader.clean(raw)
            close_matrix = loader.get_close_matrix(data)

        # ── 2. Compute Indicators ─────────────────────────────────────────
        print("\n=== Step 2: Computing Indicators ===")
        enriched = compute_all_assets(data)

        # ── 3. Generate Signals ───────────────────────────────────────────
        print("\n=== Step 3: Generating Signals ===")
        signals = generate_signals(enriched)

        # ── 4. Set up components ──────────────────────────────────────────
        risk_mgr = RiskManager(
            vol_target=cfg.vol_target,
            stop_loss=cfg.stop_loss,
            take_profit=cfg.take_profit,
            max_drawdown=cfg.max_drawdown,
            max_weight=cfg.max_weight,
            max_leverage=cfg.max_leverage,
        )
        exec_engine = ExecutionEngine(
            transaction_cost=cfg.transaction_cost,
            slippage=cfg.slippage,
        )

        # ── 5. Build return matrix ────────────────────────────────────────
        asset_names = list(close_matrix.columns)
        returns_matrix = close_matrix.pct_change().dropna()

        # Apply date filters
        if cfg.start_date:
            returns_matrix = returns_matrix[returns_matrix.index >= cfg.start_date]
        if cfg.end_date:
            returns_matrix = returns_matrix[returns_matrix.index <= cfg.end_date]

        # Build signal matrix aligned to returns
        signal_matrix = pd.DataFrame(index=returns_matrix.index,
                                     columns=asset_names, data=0.0)
        for asset in asset_names:
            if asset in signals:
                sig = signals[asset]["composite_signal"]
                common = signal_matrix.index.intersection(sig.index)
                signal_matrix.loc[common, asset] = sig.loc[common]

        # ── 6. Event Loop ─────────────────────────────────────────────────
        print("\n=== Step 4: Running Backtest ===")
        n_days = len(returns_matrix)
        dates = returns_matrix.index

        portfolio_value = np.zeros(n_days)
        portfolio_value[0] = cfg.initial_capital
        weights_history = np.zeros((n_days, len(asset_names)))
        current_weights = np.zeros(len(asset_names))

        for t in range(1, n_days):
            # Daily P&L from yesterday's weights
            daily_ret = returns_matrix.iloc[t].values
            weighted_return = np.dot(current_weights, daily_ret)
            portfolio_value[t] = portfolio_value[t-1] * (1 + weighted_return)

            # Rebalance?
            if t >= cfg.lookback and t % cfg.rebalance_freq == 0:
                # Lookback returns for optimization
                lb_returns = returns_matrix.iloc[t-cfg.lookback:t]

                # Get current signals
                current_signals = signal_matrix.iloc[t].values

                # Optimize weights
                optimizer = PortfolioOptimizer(
                    lb_returns, max_weight=cfg.max_weight
                )
                method = getattr(optimizer, cfg.optimizer_method, optimizer.max_sharpe)
                raw_weights = method()

                # Apply signal filter: zero out assets with negative signals
                target_weights = raw_weights.copy()
                target_weights[current_signals < 0] = 0
                # Re-normalize if any weight was zeroed
                if target_weights.sum() > 0:
                    target_weights = target_weights / target_weights.sum()

                # Apply risk management
                pv_series = pd.Series(portfolio_value[:t+1], index=dates[:t+1])
                realized_vol = returns_matrix.iloc[t-60:t].values @ current_weights
                port_vol = pd.Series(realized_vol).std() * np.sqrt(252) if t > 60 else 0.10

                target_weights = risk_mgr.adjust_weights(
                    target_weights, port_vol, pv_series, t
                )

                # Execute rebalance
                current_weights, cost = exec_engine.execute_rebalance(
                    dates[t], current_weights, target_weights,
                    portfolio_value[t], asset_names
                )
                portfolio_value[t] -= cost

            weights_history[t] = current_weights

        # ── 7. Package results ────────────────────────────────────────────
        result.portfolio_value = pd.Series(portfolio_value, index=dates,
                                           name="Portfolio")
        result.daily_returns = result.portfolio_value.pct_change().fillna(0)
        result.weights_history = pd.DataFrame(weights_history, index=dates,
                                              columns=asset_names)
        result.trade_log = exec_engine.get_trade_log()
        result.signals = signals
        result.data = enriched
        result.close_matrix = close_matrix.loc[dates]

        # Benchmark: equal-weight buy-and-hold
        bench_returns = returns_matrix.mean(axis=1)
        bench_value = cfg.initial_capital * (1 + bench_returns).cumprod()
        result.benchmark_value = bench_value
        result.benchmark_value.name = "Benchmark (EW)"

        total_cost = exec_engine.total_costs()
        n_trades = len(result.trade_log)
        final_val = portfolio_value[-1]
        total_ret = (final_val / cfg.initial_capital - 1)

        print(f"\n{'-'*50}")
        print(f"  Final Value   : ${final_val:>14,.2f}")
        print(f"  Total Return  : {total_ret:>+13.2%}")
        print(f"  Total Trades  : {n_trades:>14,}")
        print(f"  Total Costs   : ${total_cost:>14,.2f}")
        print(f"{'-'*50}")

        return result


if __name__ == "__main__":
    bt = Backtester()
    result = bt.run()
