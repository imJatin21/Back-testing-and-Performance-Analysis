"""
loader.py — Data fetching, caching, and cleaning for multi-asset universe.

Supports: Equities (SPY, QQQ), Bonds (TLT), Commodities (GLD, USO),
          FX (EURUSD=X), Crypto (BTC-USD).
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime


# ─── Asset Universe ──────────────────────────────────────────────────────────

ASSET_CONFIG = {
    "SPY":     {"ticker": "SPY",      "name": "S&P 500 ETF",     "class": "Equity"},
    "QQQ":     {"ticker": "QQQ",      "name": "Nasdaq 100 ETF",  "class": "Equity"},
    "TLT":     {"ticker": "TLT",      "name": "20+ Year Treasury","class": "Fixed Income"},
    "GLD":     {"ticker": "GLD",      "name": "Gold ETF",        "class": "Commodity"},
    "USO":     {"ticker": "USO",      "name": "Oil ETF",         "class": "Commodity"},
    "EURUSD":  {"ticker": "EURUSD=X", "name": "EUR/USD",         "class": "FX"},
    "BTC":     {"ticker": "BTC-USD",  "name": "Bitcoin",         "class": "Crypto"},
}

DEFAULT_START = "2012-01-01"
DEFAULT_END   = datetime.today().strftime("%Y-%m-%d")


class DataLoader:
    """Fetches, caches, and cleans multi-asset OHLCV data."""

    def __init__(self, data_dir: str = None,
                 start: str = DEFAULT_START, end: str = DEFAULT_END):
        if data_dir is None:
            data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        self.data_dir = data_dir
        self.start = start
        self.end = end
        os.makedirs(self.data_dir, exist_ok=True)

    # ── Public API ────────────────────────────────────────────────────────

    def fetch_all(self, force: bool = False) -> dict[str, pd.DataFrame]:
        """Fetch all assets. Uses cached CSVs unless *force* is True."""
        data = {}
        for asset_key, cfg in ASSET_CONFIG.items():
            csv_path = os.path.join(self.data_dir, f"{asset_key}.csv")

            if not force and os.path.exists(csv_path):
                df = pd.read_csv(csv_path, index_col="Date", parse_dates=True)
            else:
                df = self._download(cfg["ticker"], asset_key)
                df.to_csv(csv_path)

            data[asset_key] = df
            print(f"  [OK] {asset_key:8s} | {len(df):>5,} rows | "
                  f"{df.index.min().date()} -> {df.index.max().date()}")

        return data

    def load_cached(self) -> dict[str, pd.DataFrame]:
        """Load previously cached data (no network calls)."""
        data = {}
        for asset_key in ASSET_CONFIG:
            csv_path = os.path.join(self.data_dir, f"{asset_key}.csv")
            if not os.path.exists(csv_path):
                raise FileNotFoundError(
                    f"No cached data for {asset_key}. Run fetch_all() first.")
            data[asset_key] = pd.read_csv(csv_path, index_col="Date",
                                          parse_dates=True)
        return data

    def clean(self, data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        """Clean & align all asset DataFrames to a common date index."""
        cleaned = {}
        for key, df in data.items():
            df = df.copy()
            # Keep standard OHLCV columns
            cols_keep = [c for c in ["Open", "High", "Low", "Close",
                                      "Adj Close", "Volume"] if c in df.columns]
            df = df[cols_keep]
            # Forward-fill then back-fill small gaps
            df = df.ffill().bfill()
            # Drop rows where Close is still NaN
            df = df.dropna(subset=["Close"])
            cleaned[key] = df

        # Align to common date range (intersection)
        common_idx = cleaned[list(cleaned.keys())[0]].index
        for df in cleaned.values():
            common_idx = common_idx.intersection(df.index)

        for key in cleaned:
            cleaned[key] = cleaned[key].loc[common_idx]

        return cleaned

    def get_close_matrix(self, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Return a single DataFrame of adjusted close prices (assets as cols)."""
        close_col = "Adj Close"
        frames = {}
        for key, df in data.items():
            col = close_col if close_col in df.columns else "Close"
            frames[key] = df[col]
        return pd.DataFrame(frames)

    # ── Private helpers ───────────────────────────────────────────────────

    def _download(self, ticker: str, label: str) -> pd.DataFrame:
        """Download OHLCV data from Yahoo Finance."""
        print(f"  >> Downloading {label} ({ticker}) ...")
        df = yf.download(ticker, start=self.start, end=self.end,
                         auto_adjust=False, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")

        # yfinance may return MultiIndex columns for single ticker
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)

        df.index.name = "Date"
        return df


# ─── Convenience ─────────────────────────────────────────────────────────────

def load_and_clean(data_dir: str = None, force: bool = False):
    """One-liner: fetch → clean → return (data_dict, close_matrix)."""
    loader = DataLoader(data_dir=data_dir)
    raw = loader.fetch_all(force=force)
    data = loader.clean(raw)
    closes = loader.get_close_matrix(data)
    return data, closes


if __name__ == "__main__":
    data, closes = load_and_clean()
    print(f"\n{'-'*50}")
    print(f"Universe : {list(closes.columns)}")
    print(f"Period   : {closes.index[0].date()} -> {closes.index[-1].date()}")
    print(f"Obs      : {len(closes):,}")
    print(closes.tail())
