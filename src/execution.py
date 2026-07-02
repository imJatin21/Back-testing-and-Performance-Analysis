"""
execution.py — Trade execution engine with transaction costs and slippage.

Records all trades in a structured log for analytics.
"""

import numpy as np
import pandas as pd


class ExecutionEngine:
    """Simulates trade execution with realistic friction."""

    def __init__(self, transaction_cost: float = 0.001,
                 slippage: float = 0.0005):
        """
        Parameters
        ----------
        transaction_cost : round-trip cost as fraction (default 10 bps).
        slippage : slippage as fraction (default 5 bps).
        """
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.trade_log: list[dict] = []

    def execute_rebalance(self, date, current_weights: np.ndarray,
                          target_weights: np.ndarray,
                          portfolio_value: float,
                          asset_names: list[str]) -> tuple[np.ndarray, float]:
        """
        Execute a rebalance from current_weights → target_weights.

        Returns
        -------
        new_weights : np.ndarray  — post-execution weights
        total_cost  : float       — dollar cost of trading
        """
        turnover = np.abs(target_weights - current_weights)
        trade_value = turnover * portfolio_value

        # Total friction = transaction cost + slippage on traded notional
        cost_rate = self.transaction_cost + self.slippage
        total_cost = trade_value.sum() * cost_rate

        # Log individual trades
        for i, asset in enumerate(asset_names):
            delta = target_weights[i] - current_weights[i]
            if abs(delta) > 1e-6:
                self.trade_log.append({
                    "date": date,
                    "asset": asset,
                    "direction": "BUY" if delta > 0 else "SELL",
                    "weight_change": round(delta, 6),
                    "notional": round(abs(delta) * portfolio_value, 2),
                    "cost": round(abs(delta) * portfolio_value * cost_rate, 2),
                })

        return target_weights.copy(), total_cost

    def get_trade_log(self) -> pd.DataFrame:
        """Return trade log as DataFrame."""
        if not self.trade_log:
            return pd.DataFrame(columns=["date", "asset", "direction",
                                          "weight_change", "notional", "cost"])
        return pd.DataFrame(self.trade_log)

    def total_costs(self) -> float:
        """Return total dollar trading costs."""
        return sum(t["cost"] for t in self.trade_log)

    def turnover(self) -> float:
        """Return total turnover (sum of absolute weight changes)."""
        return sum(abs(t["weight_change"]) for t in self.trade_log)
