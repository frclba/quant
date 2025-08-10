# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all,tags
#     formats: ipynb,py:percent
#     notebook_metadata_filter: -all,kernelspec,jupytext
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
# ---

# %% [markdown]
# # Notebook 4: Strategy Backtesting for Crypto
#
# This notebook sets up a reproducible, git-friendly workflow using Jupytext. It focuses on building a clean backtesting scaffold for simple strategies on crypto pairs. The paired `.py` file will be the source of truth for editing; `.ipynb` is kept for execution and visualization.
#

# %% [markdown]
# ## Workflow: Edit as .py, Execute as .ipynb
#
# - The file is paired via `.jupytext.toml` in this directory using `ipynb,py:percent`.
# - You can open and edit the paired `.py` file directly in Cursor for best refactoring support.
# - The notebook mirrors the `.py` content upon save/refresh.
#

# %%
# %% [markdown]
# Environment & Imports

# %%
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional, Protocol, Tuple
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # matplotlib may not be present in minimal envs
    plt = None

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

# Resolve a stable base directory both in notebooks (no __file__) and scripts
try:
    BASE_DIR = Path(__file__).resolve().parent
except NameError:
    BASE_DIR = Path.cwd()

DATA_DIR = (BASE_DIR / ".." / "data").resolve()


# %%
# %% [markdown]
# Data Loading Helpers

# %%
def read_price_csv(path: str, parse_dates: Optional[List[str]] = None) -> pd.DataFrame:
    if parse_dates is None:
        parse_dates = ["timestamp", "date", "time"]
    existing = [c for c in parse_dates if c in open(path).readline().lower()]
    df = pd.read_csv(path, parse_dates=existing or None)
    # Normalize time column name if present
    for c in ["timestamp", "date", "time"]:
        if c in df.columns:
            df.rename(columns={c: "timestamp"}, inplace=True)
            break
    # Ensure sorted by time if timestamp exists
    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)
        df.set_index("timestamp", inplace=True)
    return df


def ensure_price_columns(df: pd.DataFrame) -> pd.DataFrame:
    # Common minimal columns for OHLCV
    required = ["close"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe columns: {list(df.columns)}")
    return df



# %%
# %% [markdown]
# Indicators

# %%
def compute_sma(series: pd.Series, window: int) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be > 0")
    return series.rolling(window=window, min_periods=window).mean()


def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    if window <= 0:
        raise ValueError("window must be > 0")
    delta = series.diff()
    gain = (delta.clip(lower=0)).rolling(window=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window=window).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi



# %%
# %% [markdown]
# Strategy Interface & Example Strategies

# %%
class Strategy(Protocol):
    def __call__(self, prices: pd.DataFrame) -> pd.Series:
        ...


def sma_cross_strategy(short_window: int, long_window: int) -> Strategy:
    if short_window >= long_window:
        raise ValueError("short_window must be < long_window")

    def _strategy(prices: pd.DataFrame) -> pd.Series:
        close = ensure_price_columns(prices).loc[:, "close"]
        sma_s = compute_sma(close, short_window)
        sma_l = compute_sma(close, long_window)
        signal = (sma_s > sma_l).astype(int) - (sma_s < sma_l).astype(int)
        return signal.fillna(0).astype(int)

    return _strategy


def rsi_reversion_strategy(lower: int = 30, upper: int = 70, window: int = 14) -> Strategy:
    def _strategy(prices: pd.DataFrame) -> pd.Series:
        close = ensure_price_columns(prices).loc[:, "close"]
        rsi = compute_rsi(close, window=window)
        longs = (rsi < lower).astype(int)
        shorts = (rsi > upper).astype(int)
        return (longs - shorts).fillna(0).astype(int)

    return _strategy



# %%
# %% [markdown]
# Vectorized Backtester

# %%
@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.Series
    trades: pd.DataFrame
    stats: dict


def run_backtest(prices: pd.DataFrame, strategy: Strategy, fee_bps: float = 5.0) -> BacktestResult:
    prices = ensure_price_columns(prices)
    close = prices["close"].astype(float)
    positions = strategy(prices).shift(1).fillna(0).astype(int)

    # Compute log returns
    log_returns = np.log(close / close.shift(1)).fillna(0.0)

    # Apply trading costs when position changes
    position_change = positions.diff().abs().fillna(0)
    fee = (fee_bps / 10000.0) * position_change

    strategy_returns = positions * log_returns - fee
    equity_curve = strategy_returns.cumsum().apply(np.exp)

    # Trade summary
    trades = pd.DataFrame({
        "position": positions,
        "position_change": position_change,
        "log_return": log_returns,
        "strategy_log_return": strategy_returns,
        "equity_curve": equity_curve,
    })

    # Basic stats
    total_return = float(np.exp(strategy_returns.sum()) - 1.0)
    ann_factor = 365 if close.index.freq is None else (365 if close.index.freq.n == 1 and close.index.freq.name in ["D", "day"] else 252)
    ann_return = float(strategy_returns.mean() * ann_factor)
    ann_vol = float(strategy_returns.std(ddof=0) * np.sqrt(ann_factor))
    sharpe = float(ann_return / ann_vol) if ann_vol > 0 else float("nan")

    stats = {
        "total_return": total_return,
        "annualized_return": ann_return,
        "annualized_vol": ann_vol,
        "sharpe": sharpe,
        "fee_bps": fee_bps,
    }

    return BacktestResult(
        equity_curve=equity_curve,
        returns=strategy_returns,
        positions=positions,
        trades=trades,
        stats=stats,
    )



# %%
# %% [markdown]
# Example Usage (replace with your data path)

# %%
# Example; set your CSV path here or drop a small sample into ../data
example_csv = os.path.join(DATA_DIR, "BTCUSDT_1d_sample.csv")
if os.path.exists(example_csv):
    df_prices = read_price_csv(example_csv)
    strat = sma_cross_strategy(short_window=10, long_window=50)
    result = run_backtest(df_prices, strat, fee_bps=5)
    display(pd.Series(result.stats))
    if plt is not None:
        ax = result.equity_curve.plot(title="Strategy Equity Curve")
        ax.set_ylabel("Equity (exp of cum log returns)")
else:
    print(f"No sample data found at: {example_csv}\nPlace a CSV with a 'close' column and optional 'timestamp' index.")


# %% [markdown]
# ## Notes
#
# - Keep `.ipynb` out of heavy diffs; the paired `.py` will be cleaner to review.
# - You can create additional strategies following the `Strategy` protocol and pass them to `run_backtest`.
# - Consider adding position sizing, risk management, and portfolio-level aggregation in a later iteration.
#
