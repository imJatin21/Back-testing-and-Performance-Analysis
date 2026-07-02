"""
strategy.py — Alpha signal generation.

Five independent trading strategies, each producing a signal Series
with values in {-1, 0, +1}. A CombinedStrategy aggregates them via
equal-weighted voting.
"""

import numpy as np
import pandas as pd


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Individual Strategies
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class MACrossover:
    """Strategy 1 — Moving Average Crossover.

    +1 when SMA20 > SMA100, −1 when SMA20 < SMA100.
    """
    name = "MA Crossover"

    @staticmethod
    def generate(df: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=df.index, name="ma_cross")
        signal[df["sma20"] > df["sma100"]] = 1
        signal[df["sma20"] < df["sma100"]] = -1
        return signal


class MeanReversion:
    """Strategy 2 — RSI Mean Reversion.

    +1 when RSI < 30 (oversold), −1 when RSI > 70 (overbought).
    """
    name = "Mean Reversion"

    @staticmethod
    def generate(df: pd.DataFrame) -> pd.Series:
        signal = pd.Series(0, index=df.index, name="mean_rev")
        signal[df["rsi"] < 30] = 1
        signal[df["rsi"] > 70] = -1
        return signal


class TimeSeriesMomentum:
    """Strategy 3 — Time-Series Momentum.

    +1 if 20-day return is in the top 20 % of its own expanding history,
    −1 if in the bottom 20 %.
    """
    name = "TS Momentum"

    @staticmethod
    def generate(df: pd.DataFrame) -> pd.Series:
        ret = df["ret_20d"]
        q80 = ret.expanding().quantile(0.80)
        q20 = ret.expanding().quantile(0.20)
        signal = pd.Series(0, index=df.index, name="ts_mom")
        signal[ret > q80] = 1
        signal[ret < q20] = -1
        return signal


class VolatilityBreakout:
    """Strategy 4 — Volatility Breakout (ATR-based).

    +1 when close breaks above previous close + 1.5 × ATR,
    −1 when close breaks below previous close − 1.5 × ATR.
    """
    name = "Vol Breakout"

    @staticmethod
    def generate(df: pd.DataFrame) -> pd.Series:
        upper = df["Close"].shift(1) + 1.5 * df["atr_14"]
        lower = df["Close"].shift(1) - 1.5 * df["atr_14"]
        signal = pd.Series(0, index=df.index, name="vol_break")
        signal[df["Close"] > upper] = 1
        signal[df["Close"] < lower] = -1
        return signal


class CrossSectionalMomentum:
    """Strategy 5 — Cross-Sectional Momentum.

    Rank all assets by 20-day return each day.
    +1 for top 2, −1 for bottom 2, 0 for middle.
    """
    name = "XS Momentum"

    @staticmethod
    def generate_all(data: dict[str, pd.DataFrame]) -> dict[str, pd.Series]:
        # Build return matrix
        ret_matrix = pd.DataFrame(
            {k: df["ret_20d"] for k, df in data.items()}
        )
        ranks = ret_matrix.rank(axis=1, ascending=True)
        n = ranks.shape[1]

        signals = {}
        for asset in ranks.columns:
            sig = pd.Series(0, index=ranks.index, name="xs_mom")
            sig[ranks[asset] >= n - 1] = 1         # top 2
            sig[ranks[asset] <= 2] = -1             # bottom 2
            signals[asset] = sig
        return signals


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Combined Signal
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ALL_STRATEGIES = [MACrossover, MeanReversion, TimeSeriesMomentum, VolatilityBreakout]


class CombinedStrategy:
    """Aggregate multiple alpha signals into a composite signal.

    For each asset on each day the composite signal is the sign of the
    mean of individual strategy signals. A small threshold (0.1) avoids
    trading on marginal consensus.
    """

    def __init__(self, strategies=None, threshold: float = 0.1):
        self.strategies = strategies or ALL_STRATEGIES
        self.threshold = threshold

    def generate(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """
        Returns
        -------
        dict  asset_key → DataFrame with columns:
              [strategy1, strategy2, …, composite_signal]
        """
        # Cross-sectional momentum signals
        xs_signals = CrossSectionalMomentum.generate_all(data)

        result = {}
        for asset_key, df in data.items():
            signals = pd.DataFrame(index=df.index)

            # Per-asset strategies
            for strat in self.strategies:
                signals[strat.name] = strat.generate(df)

            # Cross-sectional
            if asset_key in xs_signals:
                signals["XS Momentum"] = xs_signals[asset_key]

            # Composite: sign of average signal (with threshold)
            avg = signals.mean(axis=1)
            composite = pd.Series(0, index=df.index, name="composite_signal")
            composite[avg > self.threshold] = 1
            composite[avg < -self.threshold] = -1
            signals["composite_signal"] = composite

            result[asset_key] = signals

        return result


def generate_signals(data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Convenience wrapper — returns signal DataFrames for all assets."""
    cs = CombinedStrategy()
    signals = cs.generate(data)
    for key, sig_df in signals.items():
        pct_long  = (sig_df["composite_signal"] == 1).mean() * 100
        pct_short = (sig_df["composite_signal"] == -1).mean() * 100
        print(f"  [OK] {key:8s} | Long {pct_long:5.1f}% | Short {pct_short:5.1f}% | "
              f"Flat {100-pct_long-pct_short:5.1f}%")
    return signals
