## Python Crypto Quant System — Implementation Blueprint

This is a beginner-friendly, software‑engineering plan to take the idea to production in clear steps.

### 1) One‑page overview
- Goal: Research → Backtest → Live trade crypto on Mercado Bitcoin using Python only.
- Scope v1: Ingest market data, compute indicators, run backtests for simple strategies, and place guarded live orders.
- Exchange: Mercado Bitcoin (historical via REST, live via WebSocket; authenticated REST for orders).
- Data format: Parquet (fast, consistent, shareable across tools).

### 2) Minimal feature checklist (v1)
- Data ingestion: historical candles + live trades/order book stream; write Parquet; basic data quality checks.
- Indicators/statistics: SMA/EMA, RSI, ATR, MACD, Bollinger, returns, rolling mean/std, covariance/correlation.
- Strategies: Moving‑average crossover, MACD follow, Bollinger reversion; optional pairs (BTC/ETH) after parity.
- Backtester: Vectorized backtests with fees/slippage, trade logs, PnL/equity metrics.
- Live trading: Signal → risk checks → order manager → exchange; dry‑run/paper mode + small live mode.

### 3) Architecture (high level)
- Ingestor (REST/WS) → Parquet store → Research notebooks & Backtester → Strategy Signals → Live Trader (risk + orders) → Exchange
- Observability across all components (structured logs, metrics, alerts).

### 4) Milestones and acceptance criteria
1) Foundations (Week 1)
   - Deliver: Repo scaffolding, config, logging, Parquet writer/reader, schemas.
   - Accept: A small synthetic dataset round‑trips (write/read) and passes schema validation.
2) Data adapter (Weeks 2–3)
   - Deliver: Historical OHLCV ingestor; WS streamer with reconnect and rate‑limit handling.
   - Accept: N days of data for top symbols stored in Parquet; data QA report (gaps/dups) < 0.1% issues.
3) Indicators & stats (Weeks 4–5)
   - Deliver: Indicator library; rolling stats; correlation/covariance; parity notebooks on crypto.
   - Accept: Unit tests green; notebook charts/metrics match expectations.
4) Backtester v1 (Week 6)
   - Deliver: Crossover/MACD/BB with fees + basic slippage; PnL metrics and trade logs.
   - Accept: Reproducible runs with config snapshots; sample reports generated.
5) Live trading (Weeks 7–8)
   - Deliver: Live loop with risk checks, idempotent order manager, reconciliation; paper first, then small live.
   - Accept: 1–2 weeks stable paper trading; then a controlled live session with strict risk limits.

### 5) Work breakdown (modules)
- Data/adapters
  - Historical REST fetch, pagination, rate limits, retries; WS trades/order book with heartbeat and jittered backoff.
  - Parquet writer/reader; partitioning by dataset/symbol/date; schema validation.
- Indicators
  - SMA/EMA, RSI, ATR, MACD, Bollinger, VWAP; rolling windows by time (e.g., 14D/4H) for 24/7 markets.
- Statistics/portfolios (phase 2+)
  - Returns/covariance/correlation; optional PyPortfolioOpt or HRP for crypto baskets; simple volatility targeting.
- Strategies
  - Crossover, MACD, Bollinger; ATR‑based stops; liquidity filters; optional pairs (BTC/ETH) with ADF.
- Backtester
  - Vectorized engine; fees/slippage; trade logs; equity, drawdown, Sharpe, hit rate, payoff, turnover.
- Execution
  - Risk checks (max position, daily loss cut, exposure caps); idempotent client IDs; reconcile exchange state; safe rounding to tick/lot.
- Observability & ops
  - Structured logging; minimal metrics; alerts on disconnects, order rejects, drawdown breaches; config via env + YAML.

### 6) Non‑functional requirements (NFRs)
- Reliability: auto‑reconnect WS; retries with backoff; never drop data silently.
- Determinism: fixed seeds for backtests; snapshot configs with results.
- Security: API keys via env/secret store; no secrets in logs; principle of least privilege.
- Performance: use vectorized operations; optimize only after correctness.

### 7) Testing plan
- Unit: indicators, fees/slippage, order translation, schema checks.
- Integration: adapter against sandbox or recorded responses; backtests on synthetic known‑answer datasets.
- Property: invariants (PnL accounting, monotonic timestamps, no negative balances).
- E2E: paper‑mode live loop with deterministic inputs.

### 8) Risks and mitigations
- Exchange rate limits/outages → adaptive throttling, retries, and paper fallback.
- Data quality (gaps/dups) → QA checks; repair jobs; raw stream logs for replay.
- Overfitting → walk‑forward evaluation; OOS validation; conservative position sizing.
- Operational errors → dry‑run by default; strict guardrails; kill‑switch for daily loss.

### 9) Day‑1 to Day‑10 task list (actionable)
1) Scaffold repo folders and configs; add logging and Parquet utilities.
2) Implement schemas for OHLCV and trades; write/read with small synthetic data.
3) Build historical OHLCV ingestor for 1–2 symbols; store Parquet; add simple QA report.
4) Add WebSocket trade stream with reconnect; persist rolling Parquet files; validate monotonic timestamps.
5) Implement indicators (SMA/EMA/RSI/ATR/MACD/BB) and unit tests.
6) Create a parity notebook to replicate the simple indicator charts on crypto.
7) Build backtester v1 for crossover/MACD/BB with fees/slippage; output trade logs and metrics.
8) Implement risk checks and a minimal order manager (idempotent IDs, reconcile orders).
9) Wire paper trading loop; run for several days; monitor metrics and logs.
10) If stable, switch to very small live size with strict limits.

### 10) Deliverables (v1)
- Parquet datasets (historical and streamed), indicator library with tests, parity notebook, backtester reports, paper trading logs, and a minimal live trader with guardrails.


