"""
optimizer.py — Portfolio construction & optimization.

Methods: Equal Weight, Risk Parity, Mean-Variance (Markowitz),
         Maximum Sharpe, Minimum Variance.
Also computes the Efficient Frontier for visualization.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


class PortfolioOptimizer:
    """Portfolio weight optimization for N assets."""

    def __init__(self, returns: pd.DataFrame,
                 max_weight: float = 0.25, min_weight: float = 0.0,
                 risk_free_rate: float = 0.0):
        """
        Parameters
        ----------
        returns : DataFrame of daily returns (assets as columns).
        max_weight : maximum weight per asset (default 25%).
        min_weight : minimum weight per asset (default 0 = long-only).
        risk_free_rate : annualized risk-free rate.
        """
        self.returns = returns.dropna()
        self.n_assets = returns.shape[1]
        self.asset_names = list(returns.columns)
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.rf = risk_free_rate

        # Annualized stats
        self.mean_returns = self.returns.mean() * 252
        self.cov_matrix = self.returns.cov() * 252

    # ── Weight Bounds & Constraints ───────────────────────────────────────

    def _bounds(self):
        return [(self.min_weight, self.max_weight)] * self.n_assets

    def _constraints(self):
        return {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}

    def _init_weights(self):
        return np.ones(self.n_assets) / self.n_assets

    # ── Portfolio Metrics ─────────────────────────────────────────────────

    def portfolio_return(self, weights):
        return np.dot(weights, self.mean_returns)

    def portfolio_volatility(self, weights):
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

    def portfolio_sharpe(self, weights):
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        return (ret - self.rf) / vol if vol > 0 else 0.0

    # ── 1. Equal Weight ───────────────────────────────────────────────────

    def equal_weight(self) -> np.ndarray:
        return np.ones(self.n_assets) / self.n_assets

    # ── 2. Risk Parity ────────────────────────────────────────────────────

    def risk_parity(self) -> np.ndarray:
        """Inverse-volatility weighting, normalized."""
        vols = self.returns.std() * np.sqrt(252)
        inv_vol = 1.0 / vols
        weights = inv_vol / inv_vol.sum()
        clipped = np.clip(weights.values, self.min_weight, self.max_weight)
        # Re-normalize after clipping to ensure weights sum to 1
        return clipped / clipped.sum() if clipped.sum() > 0 else clipped

    # ── 3. Mean-Variance (Markowitz) ──────────────────────────────────────

    def mean_variance(self, target_return: float = None) -> np.ndarray:
        """Minimize variance for a given target return.
        If target_return is None, uses midpoint of asset returns."""
        if target_return is None:
            target_return = self.mean_returns.mean()

        constraints = [
            self._constraints(),
            {"type": "eq",
             "fun": lambda w: self.portfolio_return(w) - target_return}
        ]
        result = minimize(
            lambda w: self.portfolio_volatility(w),
            self._init_weights(),
            method="SLSQP",
            bounds=self._bounds(),
            constraints=constraints,
        )
        return result.x if result.success else self.equal_weight()

    # ── 4. Maximum Sharpe ─────────────────────────────────────────────────

    def max_sharpe(self) -> np.ndarray:
        """Maximize Sharpe ratio."""
        result = minimize(
            lambda w: -self.portfolio_sharpe(w),
            self._init_weights(),
            method="SLSQP",
            bounds=self._bounds(),
            constraints=self._constraints(),
        )
        return result.x if result.success else self.equal_weight()

    # ── 5. Minimum Variance ───────────────────────────────────────────────

    def min_variance(self) -> np.ndarray:
        """Minimize portfolio variance (no return constraint)."""
        result = minimize(
            lambda w: self.portfolio_volatility(w),
            self._init_weights(),
            method="SLSQP",
            bounds=self._bounds(),
            constraints=self._constraints(),
        )
        return result.x if result.success else self.equal_weight()

    # ── Efficient Frontier ────────────────────────────────────────────────

    def efficient_frontier(self, n_points: int = 50) -> pd.DataFrame:
        """Compute efficient frontier: (return, volatility, sharpe) points."""
        min_ret = self.mean_returns.min()
        max_ret = self.mean_returns.max()
        # Ensure we have a meaningful spread
        if abs(max_ret - min_ret) < 1e-8:
            max_ret = min_ret + 0.01
        target_returns = np.linspace(min_ret, max_ret, n_points)

        # Adjust max_weight if needed to ensure feasibility
        eff_max = max(self.max_weight, 1.0 / self.n_assets)
        eff_bounds = [(self.min_weight, eff_max)] * self.n_assets

        frontier = []
        for target in target_returns:
            constraints = [
                self._constraints(),
                {"type": "eq",
                 "fun": lambda w, t=target: self.portfolio_return(w) - t}
            ]
            result = minimize(
                lambda w: self.portfolio_volatility(w),
                self._init_weights(),
                method="SLSQP",
                bounds=eff_bounds,
                constraints=constraints,
                options={"ftol": 1e-12, "maxiter": 500},
            )
            if result.success:
                vol = self.portfolio_volatility(result.x)
                sharpe = (target - self.rf) / vol if vol > 0 else 0
                frontier.append({
                    "return": target,
                    "volatility": vol,
                    "sharpe": sharpe,
                })

        return pd.DataFrame(frontier)

    # ── Summary ───────────────────────────────────────────────────────────

    def summarize(self, weights: np.ndarray, label: str = "") -> dict:
        """Return a summary dict for given weights."""
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        sharpe = self.portfolio_sharpe(weights)
        return {
            "label": label,
            "weights": dict(zip(self.asset_names, np.round(weights, 4))),
            "annual_return": round(ret, 4),
            "annual_volatility": round(vol, 4),
            "sharpe_ratio": round(sharpe, 4),
        }

    def run_all(self) -> dict[str, dict]:
        """Run all 5 optimization methods and return summaries."""
        methods = {
            "Equal Weight":     self.equal_weight,
            "Risk Parity":      self.risk_parity,
            "Mean-Variance":    self.mean_variance,
            "Max Sharpe":       self.max_sharpe,
            "Min Variance":     self.min_variance,
        }
        results = {}
        for name, method in methods.items():
            w = method()
            results[name] = self.summarize(w, name)
            print(f"  [OK] {name:18s} | Ret {results[name]['annual_return']:+.2%} | "
                  f"Vol {results[name]['annual_volatility']:.2%} | "
                  f"Sharpe {results[name]['sharpe_ratio']:.2f}")
        return results
