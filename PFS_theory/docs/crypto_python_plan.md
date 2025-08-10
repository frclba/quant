old## Python Crypto Quant System Plan (from zero)

### 1) Objective
- Build a practical, reliable crypto quant system (research → backtest → live trading) using Python only.
- Reuse core ideas from the stock notebooks (indicators, cointegration, ML gating) and apply them to crypto markets.

### 2) Guiding principles
- Keep data, research, backtesting, and live trading aligned via shared schemas (Parquet).
- Start simple, validate parity with notebooks, then iterate.
- Favor clarity over premature optimization; add performance only where needed.

### 3) Key differences: stocks → crypto
- Markets run 24/7/365; no daily close. Use rolling windows by time not sessions.
- Fees, tick sizes, and lot sizes differ per symbol.
- Volatility and liquidity vary widely; slippage models are important.
- Exchanges enforce strict rate limits; WebSocket streams are preferred for live data.

### 4) Tech stack (Python)
- Core: Python 3.11+, pandas or Polars (Py), NumPy, SciPy.
- Stats: statsmodels (ADF, cointegration, OLS), scikit-learn (KMeans/DT/RF), pandas-ta or TA-Lib.
- IO: httpx (REST), websockets (streaming), hmac/hashlib for signing, tenacity for retries.
- Storage: pyarrow + Parquet; optionally duckdb for ad-hoc queries.
- Infra: pydantic (schemas), loguru/structlog (logging), typer or click (CLI), pyyaml (config).
- Testing: pytest, hypothesis (property tests), freezegun (time control).

### 5) Repository layout (proposed)
```
crypto_quant/
  data/
    adapters/
      mercado_bitcoin.py
    schemas.py
    writer.py
    reader.py
  indicators/
    ta_indicators.py
  stats/
    cointegration.py
    portfolio.py
  strategies/
    crossover.py
    macd.py
    bollinger.py
    pairs.py
  backtest/
    engine.py
    slippage.py
    fees.py
    metrics.py
  execution/
    client_mb.py
    order_manager.py
    risk.py
    live_loop.py
  utils/
    time.py
    config.py
    logging.py
notebooks/
  research_*.ipynb
data/
  parquet/
configs/
  default.yaml
scripts/
  ingest_mb.py
  backtest.py
  trade_live.py
```

### 6) Data model and storage
- Format: Parquet (columnar, fast, language-agnostic), partitioned by `dataset`/`symbol`/`date`.
- Datasets:
  - ohlcv: ts, open, high, low, close, volume, symbol, exchange.
  - trades: ts, price, size, side, trade_id, symbol, exchange.
  - orderbook (optional): ts, bids/asks snapshots or deltas.
  - orders/fills (private): ts, order_id, side, type, price, size, status, fee, symbol.
- Conventions:
  - Timestamps in UTC, ISO 8601 or epoch ns.
  - Numeric types as float64/decimal where needed; store fees and sizes precisely.
  - Include metadata files (JSON/YAML) for symbol filters, tick/lot sizes, fees.

### 7) Exchange integration (Mercado Bitcoin)
- Public data collector:
  - Historical REST: pull OHLCV/trades where available; paginate; respect rate limits; backfill gaps.
  - Live WebSocket: subscribe to trades/order book (if offered); auto-reconnect; heartbeat/time sync; write stream to Parquet in rolling files.
  - Data quality: deduplicate, enforce monotonic timestamps, validate schema.
- Private trading client:
  - HMAC signing for authenticated REST; clock skew handling; idempotent request IDs; backoff/retry on transient errors.
  - Core ops: get balances; place/cancel orders; query open orders and fills; translate to internal types.
  - Safety: dry‑run mode; min notional checks; partial fill handling; consistent rounding to tick/lot.

### 8) Indicators and stats (parity with notebooks)
- Indicators: SMA/EMA, RSI, ATR, MACD, Bollinger, VWAP using pandas/pandas-ta.
- Portfolio basics: returns, rolling stats, covariance/correlation matrices.
- Cointegration: Engle–Granger with statsmodels (ADF + OLS); utility to score pair lists.
- ML gating: KMeans + DecisionTree/RandomForest from scikit-learn.

### 9) Strategies (initial set)
- Trend-following crossover (long-only and long/flat).
- MACD signal follow.
- Bollinger mean reversion (band touch/break and re-entry logic).
- Pairs trading on cointegrated crypto pairs (e.g., BTC/ETH, large-cap majors) with spread z-score entry/exit.

### 10) Backtesting engine
- Vectorized first (fast to implement), event-driven later if needed.
- Inputs: Parquet OHLCV/trades; fees and slippage model; position sizing rules.
- Features: trade logs, PnL, equity curve, exposure, Sharpe, max drawdown, hit rate, payoff, turnover.
- Reproducibility: deterministic configs; fixed seeds; snapshot configs with results.

### 11) Live trading loop
- Components: signal generator → risk checks → order manager → execution client.
- Order manager: idempotency, replace vs cancel/new, stale order cleanup, reconciliation with exchange state.
- Risk controls: max position per asset, max daily loss, max leverage (if applicable), circuit breaker.
- Observability: structured logs, compact dashboards (latency, fills, slippage, rejects), alerting on anomalies.

### 12) Testing strategy
- Unit tests: indicators, stats, slippage/fees, order translations.
- Integration tests: adapter against sandbox or recorded responses; backtester on synthetic datasets with known answers.
- Property tests: invariants (e.g., PnL accounting, no negative balances, monotonic timestamps).
- End‑to‑end dry‑run: live loop in paper mode with fake orders, checking state transitions and logs.

### 13) Security and secrets
- Store API keys in environment variables or secret manager; never commit secrets.
- redact secrets in logs; limit scopes where possible.

### 14) Phased plan with deliverables and exit criteria
1) Week 1: Project bootstrap
   - Deliverables: repo structure, configs, logging, CLI scaffolds; Parquet writer/reader; basic schemas.
   - Exit: sample dataset round-tripped (write/read) with validation.

2) Week 2–3: Mercado Bitcoin data adapter
   - Deliverables: historical OHLCV/trades ingestor; WebSocket live streamer with reconnect; rate-limit handling.
   - Exit: N days of data for top symbols in Parquet; simple quality report (missing data, duplicates).

3) Week 4–5: Indicators, stats, and parity notebooks
   - Deliverables: indicator library; portfolio stats; cointegration scorer; research notebooks reproducing examples on crypto.
   - Exit: charts and metrics matching expectations; unit tests for indicators/stats.

4) Week 6: Backtester v1 (vectorized)
   - Deliverables: run crossover/MACD/BB strategies with fees and simple slippage; metrics and trade logs.
   - Exit: validated PnL curves on sample periods; config snapshots saved.

5) Week 7: Strategy expansion + ML gates
   - Deliverables: pairs trading prototype; KMeans/DT/RF gating for crossover; model persistence.
   - Exit: backtests with ML gates outperform baseline on OOS segments (defined thresholds).

6) Week 8: Live trading (paper → small real)
   - Deliverables: live loop with risk checks, order manager, reconciliation; paper trading first, then small-size real.
   - Exit: stable 1–2 weeks in paper; then controlled real run with tight risk limits.

### 15) Acceptance criteria (per capability)
- Ingestion: <0.1% duplicate trades; rate-limit safe; auto-recovery on disconnect.
- Indicators: unit-tested parity with known formulas; rolling windows correct across day boundaries.
- Backtester: fee/slippage applied; PnL bookkeeping reconciles to trade logs; reproducible results.
- Live trading: no orphan orders; idempotent submissions; configurable guardrails; clear, timestamped audit logs.

### 16) Immediate next actions
- Initialize project with the proposed layout.
- Implement Parquet writer/reader and basic schemas.
- Build Mercado Bitcoin historical OHLCV ingestor for 1–2 symbols and validate output.


