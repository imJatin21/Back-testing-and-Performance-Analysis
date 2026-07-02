"""
indicators.py — Technical indicators, risk metrics, and regime detection.

Computes per-asset features: returns, trend, momentum, volatility,
risk metrics, and market regime labels.
"""

import numpy as np
import pandas as pd


class Indicators:
    """Compute all technical and risk features for a single asset DataFrame."""

    def __init__(self, benchmark_close: pd.Series = None):
        """
        Parameters
        ----------
        benchmark_close : pd.Series, optional
            Benchmark close prices (e.g. SPY) for beta calculation.
        """
        self.benchmark_close = benchmark_close

    # ── Returns ───────────────────────────────────────────────────────────

    @staticmethod
    def daily_return(close: pd.Series) -> pd.Series:
        return close.pct_change()

    @staticmethod
    def log_return(close: pd.Series) -> pd.Series:
        return np.log(close / close.shift(1))

    @staticmethod
    def rolling_return(close: pd.Series, window: int = 20) -> pd.Series:
        return close.pct_change(window)

    # ── Trend ─────────────────────────────────────────────────────────────

    @staticmethod
    def sma(close: pd.Series, window: int) -> pd.Series:
        return close.rolling(window).mean()

    @staticmethod
    def ema(close: pd.Series, span: int = 20) -> pd.Series:
        return close.ewm(span=span, adjust=False).mean()

    @staticmethod
    def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
        """Returns (macd_line, signal_line, histogram)."""
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    # ── Momentum ──────────────────────────────────────────────────────────

    @staticmethod
    def rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def momentum(close: pd.Series, period: int = 10) -> pd.Series:
        return close - close.shift(period)

    @staticmethod
    def rate_of_change(close: pd.Series, period: int = 10) -> pd.Series:
        return close.pct_change(period) * 100

    # ── Volatility ────────────────────────────────────────────────────────

    @staticmethod
    def rolling_std(close: pd.Series, window: int = 20) -> pd.Series:
        return close.pct_change().rolling(window).std()

    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series,
            period: int = 14) -> pd.Series:
        tr1 = high - low
        tr2 = (high - close.shift(1)).abs()
        tr3 = (low - close.shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def historical_vol(close: pd.Series, window: int = 60) -> pd.Series:
        return close.pct_change().rolling(window).std() * np.sqrt(252)

    # ── Risk Metrics ──────────────────────────────────────────────────────

    def rolling_beta(self, close: pd.Series, window: int = 60) -> pd.Series:
        if self.benchmark_close is None:
            return pd.Series(np.nan, index=close.index, name="beta")
        asset_ret = close.pct_change()
        bench_ret = self.benchmark_close.pct_change()
        cov = asset_ret.rolling(window).cov(bench_ret)
        var = bench_ret.rolling(window).var()
        return cov / var

    @staticmethod
    def var_historical(returns: pd.Series, confidence: float = 0.95,
                       window: int = 252) -> pd.Series:
        return returns.rolling(window).quantile(1 - confidence)

    @staticmethod
    def cvar_historical(returns: pd.Series, confidence: float = 0.95,
                        window: int = 252) -> pd.Series:
        q = returns.rolling(window).quantile(1 - confidence)
        # For each window, CVaR = mean of returns <= VaR
        def _cvar(x):
            threshold = np.percentile(x, (1 - confidence) * 100)
            tail = x[x <= threshold]
            return tail.mean() if len(tail) > 0 else np.nan
        return returns.rolling(window).apply(_cvar, raw=True)

    @staticmethod
    def rolling_sharpe(returns: pd.Series, window: int = 60,
                       rf_annual: float = 0.0) -> pd.Series:
        rf_daily = (1 + rf_annual) ** (1/252) - 1
        excess = returns - rf_daily
        return (excess.rolling(window).mean() /
                excess.rolling(window).std()) * np.sqrt(252)

    # ── Market Regime ─────────────────────────────────────────────────────

    @staticmethod
    def regime_trend(close: pd.Series) -> pd.Series:
        """Bull if close > SMA200, else Bear."""
        sma200 = close.rolling(200).mean()
        regime = pd.Series("Bear", index=close.index, name="trend_regime")
        regime[close > sma200] = "Bull"
        return regime

    @staticmethod
    def regime_volatility(close: pd.Series, window: int = 60) -> pd.Series:
        """High Vol if realized vol > median, else Low Vol."""
        vol = close.pct_change().rolling(window).std() * np.sqrt(252)
        median_vol = vol.expanding().median()
        regime = pd.Series("Low Vol", index=close.index, name="vol_regime")
        regime[vol > median_vol] = "High Vol"
        return regime

    # ── Full Feature Builder ──────────────────────────────────────────────

    def compute_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all indicators for one asset.

        Parameters
        ----------
        df : DataFrame with columns Open, High, Low, Close, Volume

        Returns
        -------
        DataFrame with original + indicator columns appended.
        """
        out = df.copy()
        c = out["Close"]
        h = out["High"]
        l = out["Low"]

        # Returns
        out["daily_return"]    = self.daily_return(c)
        out["log_return"]      = self.log_return(c)
        out["ret_5d"]          = self.rolling_return(c, 5)
        out["ret_20d"]         = self.rolling_return(c, 20)
        out["ret_60d"]         = self.rolling_return(c, 60)

        # Trend
        out["sma20"]           = self.sma(c, 20)
        out["sma50"]           = self.sma(c, 50)
        out["sma100"]          = self.sma(c, 100)
        out["sma200"]          = self.sma(c, 200)
        out["ema20"]           = self.ema(c, 20)
        macd_l, macd_s, macd_h = self.macd(c)
        out["macd"]            = macd_l
        out["macd_signal"]     = macd_s
        out["macd_hist"]       = macd_h

        # Momentum
        out["rsi"]             = self.rsi(c)
        out["momentum"]        = self.momentum(c)
        out["roc"]             = self.rate_of_change(c)

        # Volatility
        out["rolling_std_20"]  = self.rolling_std(c, 20)
        out["atr_14"]          = self.atr(h, l, c, 14)
        out["hist_vol_60"]     = self.historical_vol(c, 60)

        # Risk
        returns = out["daily_return"]
        out["rolling_beta"]    = self.rolling_beta(c, 60)
        out["var_95"]          = self.var_historical(returns, 0.95)
        out["cvar_95"]         = self.cvar_historical(returns, 0.95)
        out["rolling_sharpe"]  = self.rolling_sharpe(returns, 60)

        # Regime
        out["trend_regime"]    = self.regime_trend(c)
        out["vol_regime"]      = self.regime_volatility(c)

        return out


def compute_all_assets(data: dict[str, pd.DataFrame],
                       benchmark_key: str = "SPY") -> dict[str, pd.DataFrame]:
    """Compute indicators for every asset in the universe."""
    bench_close = data[benchmark_key]["Close"] if benchmark_key in data else None
    ind = Indicators(benchmark_close=bench_close)
    result = {}
    for key, df in data.items():
        result[key] = ind.compute_all(df)
        print(f"  [OK] Indicators computed for {key} ({len(result[key].columns)} cols)")
    return result
