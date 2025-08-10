## Notebook coverage matrix (old_fga_quant → crypto system)

This document guarantees 100% coverage of the original notebooks’ content, adapted from stocks/MT5 to crypto with an exchange (Mercado Bitcoin).

### Key adaptations
- Asset class: crypto spot (24/7). Use time-based rolling windows; no market sessions.
- Data source: exchange REST/WebSocket (historical + live), not Yahoo/MT5.
- Execution: authenticated REST client with guardrails; no MT5 socket.

### Aula 1 — Price modeling, returns, volatility, Sharpe, correlation
- Covered by
  - `indicators/ta_indicators.py`: return calculations (log/simple), rolling mean/std/vol.
  - `stats/portfolio.py`: Sharpe, cumulative returns, correlation/covariance.
  - `notebooks/crypto_stochastic_modeling.ipynb`:
    - Brownian and Geometric Brownian Motion simulators for educational parity.
    - Risk/return diagram for crypto assets (scatter of vol vs mean return).
- Deliverables/DoD
  - Functions for GBM and Brownian simulation with seeded randomness.
  - Parity plots: cumulative returns, volatility, and Sharpe for top N crypto symbols.

### Aula 2 — Modern Portfolio Theory (Markowitz)
- Covered by
  - `stats/portfolio.py`: returns, covariance (plus shrinkage), correlation heatmap.
  - `portfolio/optimizers.py`: efficient frontier, min-variance, max-Sharpe; capital market line chart.
  - `portfolio/allocators.py`: constraints (min/max weight), turnover caps; rebalancing simulator.
  - `notebooks/portfolio_crypto_frontier.ipynb`.
- Deliverables/DoD
  - Frontier scatter and CML on a crypto universe; min-risk and max-Sharpe portfolios produced.
  - Heatmaps of correlation/covariance and summary tables of weights/metrics.

### Aula 3 — Technical indicators and frequency response
- Covered by
  - `indicators/ta_indicators.py`: SMA, EMA, RSI, ATR, MACD, Bollinger, VWAP.
  - `notebooks/indicators_crypto_demo.ipynb`: candlestick plots (mplfinance), indicator overlays.
  - `notebooks/mm_frequency_response.ipynb`: frequency-domain analysis of moving averages (FIR view).
- Deliverables/DoD
  - Indicator functions unit-tested on toy inputs.
  - Candlestick and overlay plots for selected crypto symbols.
  - Frequency response chart of SMA/EMA filters.

### Aula 4 — Algo trading: crossover, MACD, Bollinger
- Covered by
  - `strategies/crossover.py`, `strategies/macd.py`, `strategies/bollinger.py`: pure signal + sizing.
  - `backtest/engine.py`, `backtest/fees.py`, `backtest/slippage.py`, `backtest/metrics.py`.
  - `notebooks/strategy_cross_macd_bb_crypto.ipynb`: signals, positions, trades, and PnL.
- Deliverables/DoD
  - Vectorized backtests with fees/slippage; trade logs and equity curves.
  - Metrics: Sharpe, max drawdown, hit rate, payoff, turnover; parity with notebook-style outputs.

### Aula 5 — Long & Short (Pairs), cointegration, residuals
- Covered by
  - `stats/cointegration.py`: Engle–Granger (OLS + ADF) and pairwise `coint` scanning; intersection of methods; half-life estimation.
  - `strategies/pairs.py`: residual z-score entry/exit; hedge ratio sizing; stop-outs.
  - `notebooks/pairs_crypto_cointegration.ipynb`.
- Deliverables/DoD
  - Ranked pairs list by p-value and correlation; plot residuals with z-scores; enter/exit markers.
  - Backtest of pairs strategy on liquid majors (e.g., BTC/ETH) with risk controls.

### Aula Extra 1 — EDA (Exploratory Data Analysis)
- Covered by
  - `notebooks/crypto_eda.ipynb`: missingness, distributions, autocorrelation/partial autocorrelation, volatility clustering.
- Deliverables/DoD
  - Basic EDA report with plots and observations for top symbols; data quality summary.

### Aula Extra 3 — K-Means and portfolio for swing
- Covered by
  - `features/pipelines.py`: feature matrix (std, mean returns, turnover, liquidity); KMeans clustering.
  - `portfolio/allocators.py`: per-cluster selection (best Sharpe per cluster) as universe filter.
  - `notebooks/universe_selection_kmeans.ipynb`.
- Deliverables/DoD
  - Cluster scatter by risk/return; chosen representatives; comparison vs naive universe.

### Aula Extra 5 — Random Forest for algo trading
- Covered by
  - `features/pipelines.py`: label creation (buy/stop) based on crossover events and realized returns; features (cross, ATR, volume, RSI, volatility).
  - `features/registry.py`: model train/eval/persist with scikit-learn (DT/RF); metrics (accuracy, precision/recall, ROC-AUC).
  - `strategies/crossover.py`: gating by ML signal.
  - `notebooks/ml_gating_for_crossover.ipynb`.
- Deliverables/DoD
  - Train/test split with leakage controls (time split); saved model; backtest uplift vs baseline reported.

### Aula 8 — Python–MT5 socket (replaced)
- Covered by
  - `execution/client_mb.py` (REST, HMAC signing), `execution/live_loop.py` (paper + live modes), `execution/order_manager.py` (idempotency + reconciliation), `execution/risk.py` (guardrails).
- Deliverables/DoD
  - Paper trading run logs; later, small live run with strict limits; audit trail of orders/fills.

### Cross-cutting deliverables
- Data ingestion via `data/adapters/mercado_bitcoin.py` (REST/WS) with rate limits, reconnect, QA, and Parquet storage.
- Quality reports via `data/quality.py` and `scripts/quality_report.py` (gaps/dups, monotonic timestamps).
- Configuration in `configs/default.yaml`; logging via `utils/logging.py` (structured fields).
- Testing: unit (indicators, PnL, cointegration), integration (adapter/backfill/WS), property (PnL conservation, monotonic ts), record/replay for WS.

### Acceptance summary (coverage = 100%)
- Every topic and artifact from the first five notebooks (and the relevant extras) has a direct crypto counterpart.
- All MT5/socket portions are replaced with exchange client + live loop.
- For each notebook, there is a named notebook and corresponding code module(s), plus explicit DoD items.


