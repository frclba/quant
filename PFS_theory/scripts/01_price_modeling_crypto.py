#!/usr/bin/env python3
"""
Price Modeling (Crypto) — Script version

Replicates the Lesson 1 notebook in a non-notebook script:
- Simulate Brownian motion and Geometric Brownian Motion (GBM)
- Compute returns and cumulative returns
- Rolling statistics, risk/return scatter, correlation heatmap
- Fetch real crypto market data (fallback via yfinance) and plot

Outputs figures to outputs/01_price_modeling/
"""

from __future__ import annotations

import os
import sys
import math
import warnings
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# Global plotting setup (no inline imports, per user rule)
plt.style.use("dark_background")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # yfinance is optional; script will still run with simulations


def ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def simulate_brownian(num_steps: int, mu: float, sigma: float, start_price: float, seed: int = 42) -> pd.Series:
    """Return a price path using arithmetic Brownian motion.

    p(t) = p0 + mu * t + sigma * w(t)
    """
    np.random.seed(seed)
    white_noise = np.random.normal(0.0, 1.0, num_steps)
    t_index = np.arange(0, num_steps)
    prices = start_price + mu * t_index + sigma * white_noise
    return pd.Series(prices, name="brownian")


def simulate_gbm(num_steps: int, mu: float, sigma: float, start_price: float, seed: int = 7) -> pd.Series:
    """Return a price path using Geometric Brownian Motion (GBM).

    p(t) = p0 * exp((mu - sigma^2/2) * t + sigma * w(t))
    """
    np.random.seed(seed)
    white_noise = np.random.normal(0.0, 1.0, num_steps)
    t_index = np.arange(0, num_steps)
    prices = start_price * np.exp((mu - (sigma ** 2) / 2.0) * t_index + sigma * white_noise)
    return pd.Series(prices, name="gbm")


def compute_log_returns(prices: pd.Series) -> pd.Series:
    return np.log(prices).diff().dropna()


def cumulative_return_percent(simple_returns: pd.Series) -> pd.Series:
    # simple_returns = exp(log_returns) - 1
    compounded = np.cumprod(1.0 + simple_returns.values) - 1.0
    return 100.0 * pd.Series(compounded, index=simple_returns.index)


def plot_series(series: pd.Series, title: str, ylabel: str, save_path: str) -> None:
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(series.values)
    ax.set_title(title)
    ax.set_xlabel("time [steps]")
    ax.set_ylabel(ylabel)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_returns_bars(log_returns: pd.Series, title: str, save_path: str) -> None:
    simple_returns = np.exp(log_returns) - 1.0
    up = simple_returns.clip(lower=0.0)
    down = simple_returns.clip(upper=0.0)

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.bar(range(len(up)), 100.0 * down, color="red", edgecolor="black")
    ax.bar(range(len(up)), 100.0 * up, color="blue", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel("time [steps]")
    ax.set_ylabel("Return [%]")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_cumulative_returns(log_returns: pd.Series, title: str, save_path: str) -> None:
    simple_returns = np.exp(log_returns) - 1.0
    cum_pct = cumulative_return_percent(simple_returns)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.plot(cum_pct.values, color="yellow")
    ax.bar(range(len(simple_returns)), 100.0 * simple_returns, color="gray", alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("time [steps]")
    ax.set_ylabel("Cumulative return [%]")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_rolling_stats(prices: pd.Series, returns_pct: pd.Series, save_path: str, title: str = "Rolling stats") -> None:
    fig = plt.subplots(figsize=(20, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(prices.values, c="m", label="Price")
    ax1.plot(pd.Series(prices).rolling(window=6).mean().values, c="b", label="Rolling mean")
    ax2 = ax1.twinx()
    ax2.plot(pd.Series(returns_pct).rolling(window=5).std().values, c="g", label="Rolling std")
    ax1.set_title(title)
    ax1.legend(loc="upper left")
    ax2.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_risk_return_scatter(returns_df: pd.DataFrame, save_path: str, title: str = "Risk/Return (Simulated)") -> None:
    means = returns_df.mean()
    risks = returns_df.std()
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(risks, means)
    for i, col in enumerate(returns_df.columns):
        ax.annotate(col, (risks[i], means[i]))
    ax.set_xlabel("Risk (std %)")
    ax.set_ylabel("Return (mean %)")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, linewidths=0.5, ax=ax, cmap="viridis")
    plt.yticks(rotation=0)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def fetch_crypto_closes(symbols: List[str], start: str, end: str) -> pd.DataFrame:
    if yf is None:
        return pd.DataFrame()
    data: Dict[str, pd.Series] = {}
    for sym in symbols:
        try:
            dfc = yf.download(sym, start=start, end=end, interval="1d")
            if len(dfc) > 0:
                data[sym] = dfc["Close"].rename(sym)
        except Exception:
            continue
    return pd.DataFrame(data).dropna()


def main() -> None:
    output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs", "01_price_modeling")
    ensure_dir(output_dir)

    # 1) Simulations: Brownian and GBM
    brown = simulate_brownian(num_steps=60, mu=0.2, sigma=0.9, start_price=12.0, seed=42)
    gbm = simulate_gbm(num_steps=60, mu=0.02, sigma=0.11, start_price=12.0, seed=7)

    plot_series(brown, "Price modeled with Brownian motion", "Price [$]", os.path.join(output_dir, "brownian.png"))
    plot_series(gbm, "Price modeled with Geometric Brownian Motion", "Price [$]", os.path.join(output_dir, "gbm.png"))

    # 2) Returns and cumulative returns
    log_ret_brown = compute_log_returns(brown)
    plot_returns_bars(log_ret_brown, "Returns (Brownian)", os.path.join(output_dir, "returns_brownian.png"))
    plot_cumulative_returns(log_ret_brown, "Cumulative return (Brownian)", os.path.join(output_dir, "cumret_brownian.png"))

    # 3) Rolling stats example
    returns_pct_brown = (np.exp(log_ret_brown) - 1.0) * 100.0
    plot_rolling_stats(brown, returns_pct_brown, os.path.join(output_dir, "rolling_stats_brownian.png"), title="Rolling stats (Brownian)")

    # 4) Simulated multi-asset risk/return and correlation
    def _gbm(p0: float, mu_: float, sig_: float, steps: int, seed_: int) -> pd.Series:
        return simulate_gbm(num_steps=steps, mu=mu_, sigma=sig_, start_price=p0, seed=seed_)

    po = [12, 12, 12, 12, 12]
    mu = [0.003, 0.005, 0.0033, 0.0041, 0.006]
    sigma = [0.08, 0.01, 0.02, 0.05, 0.022]
    steps = 60

    sim_prices = pd.concat([
        _gbm(po[i], mu[i], sigma[i], steps, seed_=123 + i).rename(f"p{i+1}") for i in range(len(po))
    ], axis=1)

    sim_returns_pct = 100.0 * sim_prices.diff() / sim_prices.iloc[0]
    plot_risk_return_scatter(sim_returns_pct.dropna(), os.path.join(output_dir, "risk_return_sim.png"))
    plot_correlation_heatmap(sim_returns_pct.dropna(), os.path.join(output_dir, "corr_sim.png"), "Correlation (Simulated Returns)")

    # 5) Real crypto data (fallback via yfinance)
    symbols = ["BTC-USD", "ETH-USD", "ADA-USD"]
    start = (pd.Timestamp.utcnow() - pd.Timedelta(days=180)).strftime("%Y-%m-%d")
    end = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    market = fetch_crypto_closes(symbols, start, end)

    if len(market.columns) >= 2:
        rets = market.pct_change().dropna()

        fig, ax = plt.subplots(figsize=(10, 5))
        market.plot(ax=ax, title="Crypto Close Prices [USD]")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price [USD]")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "market_prices.png"))
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        (100.0 * rets).plot(ax=ax, title="Daily Returns [%]")
        ax.set_xlabel("Date")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "market_returns.png"))
        plt.close(fig)

        plot_correlation_heatmap(rets, os.path.join(output_dir, "market_corr.png"), "Correlation (Market Returns)")
    else:
        print("No market data available (check internet or yfinance installed).")

    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()


