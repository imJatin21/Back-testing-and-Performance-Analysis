"""
risk.py — Risk management layer.

Implements position sizing (volatility targeting), stop-loss, take-profit,
maximum drawdown control, exposure limits, and leverage constraints.
"""

import numpy as np
import pandas as pd


class RiskManager:
    """Applies institutional risk controls to portfolio positions."""

    def __init__(
        self,
        vol_target: float = 0.10,       # 10 % annualized vol target
        stop_loss: float = -0.05,        # −5 % per-position stop
        take_profit: float = 0.10,       # +10 % per-position take-profit
        max_drawdown: float = -0.15,     # −15 % portfolio drawdown → de-risk
        max_weight: float = 0.25,        # 25 % per-asset cap
        max_leverage: float = 1.20,      # 1.2× gross exposure cap
    ):
        self.vol_target = vol_target
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.max_drawdown = max_drawdown
        self.max_weight = max_weight
        self.max_leverage = max_leverage

    # ── Position Sizing (Vol Targeting) ───────────────────────────────────

    def vol_target_scalar(self, realized_vol: float) -> float:
        """Scale factor to achieve target volatility.

        If portfolio annualized vol is 20 % and target is 10 %,
        scalar = 0.5, halving position sizes.
        """
        if realized_vol <= 0 or np.isnan(realized_vol):
            return 1.0
        return min(self.vol_target / realized_vol, self.max_leverage)

    # ── Per-Position P&L Controls ─────────────────────────────────────────

    def apply_stop_take(self, position_pnl: pd.Series,
                        signal: pd.Series) -> pd.Series:
        """Zero out signal where cumulative P&L breaches stop-loss or
        take-profit thresholds."""
        adjusted = signal.copy()
        cum_pnl = position_pnl.cumsum()

        # Simple trailing: reset tracking after flat periods
        running_pnl = pd.Series(0.0, index=signal.index)
        entry_val = 0.0
        in_trade = False

        for i in range(len(signal)):
            if signal.iloc[i] != 0 and not in_trade:
                in_trade = True
                entry_val = cum_pnl.iloc[i]
            elif signal.iloc[i] == 0:
                in_trade = False
                running_pnl.iloc[i] = 0.0
                continue

            if in_trade:
                running_pnl.iloc[i] = cum_pnl.iloc[i] - entry_val
                if running_pnl.iloc[i] <= self.stop_loss:
                    adjusted.iloc[i] = 0  # stop-loss hit
                    in_trade = False
                elif running_pnl.iloc[i] >= self.take_profit:
                    adjusted.iloc[i] = 0  # take-profit hit
                    in_trade = False

        return adjusted

    # ── Portfolio-Level Drawdown Control ──────────────────────────────────

    def drawdown_scalar(self, portfolio_value: pd.Series) -> pd.Series:
        """Return a scalar series: 1.0 normally, 0.5 when drawdown
        exceeds threshold."""
        peak = portfolio_value.expanding().max()
        drawdown = (portfolio_value - peak) / peak
        scalar = pd.Series(1.0, index=portfolio_value.index)
        scalar[drawdown < self.max_drawdown] = 0.5
        return scalar

    # ── Weight Clipping ───────────────────────────────────────────────────

    def clip_weights(self, weights: np.ndarray) -> np.ndarray:
        """Enforce per-asset cap and re-normalize."""
        w = np.clip(weights, -self.max_weight, self.max_weight)
        # Ensure gross exposure ≤ max_leverage
        gross = np.abs(w).sum()
        if gross > self.max_leverage:
            w = w * (self.max_leverage / gross)
        return w

    # ── Composite Risk Adjustment ─────────────────────────────────────────

    def adjust_weights(self, raw_weights: np.ndarray,
                       realized_vol: float,
                       portfolio_value: pd.Series,
                       current_idx: int) -> np.ndarray:
        """Apply all risk filters to raw weights at a given timestep."""
        # 1. Vol-target scaling
        vol_scalar = self.vol_target_scalar(realized_vol)

        # 2. Drawdown scaling
        dd_scalar = 1.0
        if current_idx > 0:
            dd_series = self.drawdown_scalar(portfolio_value[:current_idx+1])
            dd_scalar = dd_series.iloc[-1]

        scaled = raw_weights * vol_scalar * dd_scalar

        # 3. Weight clipping
        return self.clip_weights(scaled)
