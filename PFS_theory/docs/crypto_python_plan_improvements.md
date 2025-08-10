## Improvements and crypto mapping for the Python plan

### Opportunities to improve `docs/crypto_python_plan.md`

- Data model
  - Add datasets: funding rates, mark/index prices, borrow rates, open interest, and L2 order book snapshots (for realistic slippage).
  - Include currency context (BRL vs USD/USDT pairs on Mercado Bitcoin) and FX normalization if needed.
  - Add symbol metadata: tick/lot sizes, fee tiers (taker/maker), min notional; version and date them.
  - Data QA: add explicit expectations (no timestamp gaps, monotonic ts, dedup), with validation checks (Great Expectations-style).

- Exchange integration
  - WebSocket resilience: heartbeat, jittered backoff, replay-from-last-ts on reconnect; persistent raw stream logs for replay tests.
  - Rate limits: rolling counters and adaptive throttling; per-endpoint budgets.
  - Idempotency: client order IDs with deterministic naming; reconciliation job to resolve orphan orders.

- Indicators and stats
  - Rolling windows by time (e.g., 14D, 4H) for 24/7 markets; avoid session assumptions.
  - Portfolio math: shrinkage covariance (Ledoit–Wolf), robust covariance, and correlation cleaning.
  - Risk models: volatility targeting, drawdown targeting, and regime filters.

- Smart portfolios (stock notebooks → crypto)
  - Efficient frontier/Markowitz: integrate PyPortfolioOpt (weights, constraints, turnover), practical constraints (max/min weights, leverage caps).
  - HRP: Hierarchical Risk Parity for stable allocations in volatile baskets.
  - Rebalancing: periodic plus drift thresholds; include fees/slippage; schedule with liquidity awareness.
  - Universe: rules for inclusion/exclusion (top N by volume/age), survivorship-bias controls, de-list handling.

- Pairs trading (Aula 5 adaptation)
  - Use statsmodels ADF/coint and add Johansen (vecm.coint_johansen) for robustness.
  - Hedge ratio estimation and half-life of mean reversion; dynamic z-score thresholds with volatility scaling.
  - Prefer spot pairs with a shared quote (BRL/USDT); be cautious with stablecoins and illiquid alts.

- Strategies
  - Crossover/MACD/BB: volatility scaling for sizing; ATR-based stops; filter by liquidity (volume/turnover).
  - Portfolio overlay: combine single-asset signals under allocator (vol-target + concentration caps).

- Backtesting
  - L2-based slippage and partial-fill simulation; participation-rate slippage alternative.
  - Walk-forward evaluation; time-split CV for ML gates; leakage checks.
  - Handle symbol migrations/redenominations; metadata changes mid-backtest.

- Execution and risk
  - Risk checks: max per-asset and aggregate exposure, daily loss cut, trade count limits, circuit breakers.
  - Order types: support post-only (if available), IOC/FOK simulation; retry ladders with price protection bands.
  - Monitoring: live metrics (latency, fill ratio, rejects, cancel/replace stats), plus alerting.

- Testing and ops
  - Record/replay framework for WS streams; deterministic end-to-end tests.
  - Chaos tests: disconnects, rate-limit bursts, clock skew, partial API outages.
  - Config: pydantic-settings; env-var secrets; separate paper/live profiles; incident runbooks.

- Repository layout additions
  - `portfolio/optimizers.py` and `portfolio/allocators.py` for frontier/HRP and allocation logic.
  - `features/` for ML pipelines and a simple model registry (joblib).
  - `quality/` for data validation checks and QA reports.

### Mapping the first 5 notebooks in `old_fga_quant/` to crypto

- Aula 1 (stochastic modeling, returns, Sharpe, correlation)
  - Crypto: use log returns on spot, rolling stats on 24/7 time windows, emphasize volatility targeting; correlation heatmaps for a crypto basket.

- Aula 2 (Markowitz portfolios)
  - Crypto: efficient frontier with shrinkage covariance; HRP baseline; weight/turnover constraints; weekly/monthly rebalance with drift triggers; compare Sharpe/drawdown vs equal-weight.

- Aula 3–4 (indicators and simple algos)
  - Crypto: migrate SMA/EMA/MACD/RSI/BB and crossover strategies with ATR stops and liquidity filters; walk-forward tuning; include MB fees/slippage.

- Aula 5 (pairs, cointegration)
  - Crypto: focus on liquid majors (BTC/ETH/USDT/BRL); Engle–Granger + Johansen; compute half-life; conservative mean-reversion with strict stop-outs and spread z-scoring; watch for structural breaks.

### Concrete additions to the Python plan

- Dependencies
  - PyPortfolioOpt, statsmodels vecm (Johansen), optional mlfinlab (HRP), scikit-learn pipelines.

- Extended datasets
  - `funding.parquet`, `index_mark.parquet`, `open_interest.parquet`, `orderbook_l2.parquet` with clear schemas.

- New research notebooks
  - `portfolio_crypto_frontier.ipynb` (Markowitz + HRP on top-N symbols)
  - `pairs_crypto_cointegration.ipynb` (ADF/Johansen, z-score entries, half-life)
  - `execution_sizing.ipynb` (volatility targeting, position caps)

- Backtester tasks
  - Portfolio rebalancing engine with turnover and cost accounting.
  - Pairs engine with hedge ratio and rolling spread/z-scoring.

- Live safeguards
  - Kill-switch on daily drawdown; order protection bands; idempotent client IDs and reconciliation job.

### Minimal next steps
- Add datasets and schema for funding/index/open-interest and L2; update adapter interfaces.
- Introduce PyPortfolioOpt + HRP into `portfolio/` and a rebalance simulator in `backtest/`.
- Create the two research notebooks (frontier + pairs) on a crypto universe (top symbols on Mercado Bitcoin).
- Implement volatility targeting and simple risk caps in the live loop.


