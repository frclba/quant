## LLM Detailed Build Spec — Python Crypto Quant System (Mercado Bitcoin)

This is the exhaustive, implementation-ready specification for an LLM to build a Python-only crypto quant platform: ingest → analyze → backtest → live trade. Favor correctness, testability, and clarity. Implement in small, atomic edits with tests.

### 1) Objectives and scope
- Objective: Recreate and extend the stock notebook workflows for crypto (24/7 markets) and deliver a production-capable pipeline (data → research → backtest → live execution) using only Python.
- Scope v1: Spot markets on Mercado Bitcoin (MB). Historical OHLCV/trades, live trade stream, indicators, portfolio/risk stats, simple strategies (crossover/MACD/Bollinger), pairs cointegration (BTC/ETH etc.), vectorized backtests, paper trading, guarded live trading.
- Non-goals v1: Derivatives, complex order types beyond basic limit/market emulation, distributed systems, HFT latencies.

### 2) Architecture (end-to-end)
- Data sources: MB REST (historical), MB WebSocket (live trades/order book if available).
- Ingestion: Robust client with retries, backoff, reconnect, rate-limit awareness → raw events → normalized DataFrames → Parquet partitions.
- Research/analytics: Indicators, portfolio stats, cointegration and ML gating via notebooks and library modules.
- Backtest: Vectorized engine operating on Parquet OHLCV/trades; fee/slippage; metrics; trade logs.
- Execution: Signal generator → risk checks → order manager → exchange client; idempotency and reconciliation.
- Observability: Structured logs, lightweight metrics, anomaly alerts.

### 3) Directory structure (authoritative)
```
crypto_quant/
  data/
    adapters/mercado_bitcoin.py
    schemas.py
    writer.py
    reader.py
    quality.py
  indicators/ta_indicators.py
  stats/{portfolio.py, cointegration.py}
  portfolio/{optimizers.py, allocators.py}
  strategies/{crossover.py, macd.py, bollinger.py, pairs.py}
  backtest/{engine.py, rebalancer.py, slippage.py, fees.py, metrics.py}
  execution/{client_mb.py, order_manager.py, risk.py, live_loop.py}
  features/{pipelines.py, registry.py}
  utils/{time.py, config.py, logging.py, ids.py}
  scripts/{ingest_mb.py, backtest.py, trade_live.py, quality_report.py}
  data/parquet/  # gitignored
configs/
  default.yaml
tests/
  ...
notebooks/
  portfolio_crypto_frontier.ipynb
  pairs_crypto_cointegration.ipynb
  execution_sizing.ipynb
```

### 4) Configuration schema (YAML via pydantic-settings)
- Global: env (dev/test/prod), timezone (UTC), data_root (absolute path), log_level, paper_trading (bool).
- Symbols: list of symbols; quote currency (e.g., BRL/USDT); exclusions.
- Ingestion: rest_base_url, ws_url, rate_limits (requests/sec), retry policy (max_tries, backoff_jitter_ms).
- Partitions: dataset → path template, partition keys (dataset/symbol/date).
- Backtest: fees (taker/maker), slippage model params, cash, position caps, warmup windows.
- Live: risk thresholds (max_position, max_daily_loss, max_gross_exposure), order price bands, poll intervals.

### 5) Data sets and schemas (Parquet)
- ohlcv
  - Columns: ts:int64 (ns), open:float64, high:float64, low:float64, close:float64, volume:float64, symbol:str, exchange:str
  - Constraints: ts monotonic within file; (high ≥ max(open,close) ≥ min(open,close) ≥ low); volume ≥ 0
- trades
  - Columns: ts:int64, price:float64, size:float64, side:{buy,sell}, trade_id:str, symbol:str, exchange:str
  - Constraints: ts monotonic per symbol; size>0; price>0; trade_id unique within file
- orderbook_l2 (optional v1)
  - Columns: ts:int64, bids:list[{price:float64,size:float64}], asks:list[...], symbol, exchange
  - Constraints: best_bid < best_ask; price strictly ordered per side
- orders/fills (private)
  - Columns: ts:int64, order_id, client_id, side, type, price, size, status, fee, symbol, exchange
  - Constraints: client_id unique; status in lifecycle (new, open, partial, filled, canceled, rejected)

Partitioning scheme:
- data/parquet/{dataset}/{symbol}/{YYYY}/{MM}/{DD}/file-{HHmmss}.parquet

### 6) Ingestion (Mercado Bitcoin)
Responsibilities:
- REST historical OHLCV/trades fetch with pagination; store Parquet; fill gaps if any.
- WebSocket trades (and L2 if available): continuous stream with reconnect, jittered exponential backoff, heartbeat, and detection of stale streams; rolling file writer (e.g., rotate every N minutes or size).

API design (internal):
- REST
  - fetch_ohlcv(symbol:str, start:datetime, end:datetime, interval:str)->pd.DataFrame
  - fetch_trades(symbol:str, start:datetime, end:datetime)->pd.DataFrame
- WebSocket
  - stream_trades(symbols:list[str], on_event:Callable[[dict], None], stop_event:threading.Event)->None

Error handling & resilience:
- Rate limiting: token bucket per endpoint; if 429/backoff header present, respect; otherwise apply exponential backoff with jitter.
- Reconnect policy: backoff base 0.5s, cap 30s, jitter ±20%; resume from last seen ts where feasible; log session IDs.
- Data QA hooks: every write triggers sanity checks (monotonic ts, duplicates, schema types); write a daily `quality_report.json` via `scripts/quality_report.py`.

### 7) Indicators and technical analysis
- Implement SMA, EMA, RSI, ATR, MACD, Bollinger, VWAP using pandas; expose simple functions:
  - sma(df, column:str, window:str|int)->pd.Series
  - ema(df, column, span:int)->pd.Series
  - rsi(df, column, window:int)->pd.Series
  - atr(df, high:str, low:str, close:str, window:int)->pd.Series
  - macd(df, column, fast:int=12, slow:int=26, signal:int=9)->pd.DataFrame [macd, signal, hist]
  - bollinger(df, column, window:int, k:float=2.0)->pd.DataFrame [mid, upper, lower]
- All functions must be pure, vectorized, and avoid inline imports (global imports at top of file only).

### 8) Portfolio statistics and optimizers
- Stats (v1): log returns, cumulative returns, rolling mean/std/vol, covariance/correlation matrices; shrinkage covariance (Ledoit–Wolf) for stability.
- Optimizers (v2): Efficient frontier via PyPortfolioOpt; constraints (min/max weight, turnover cap); HRP allocator as baseline.
- Rebalancing simulator: periodic (weekly/monthly) plus drift threshold; include fees/slippage.

### 9) Cointegration and pairs
- Engle–Granger: OLS hedge ratio, residual, ADF on residual; p-value thresholds configurable.
- Johansen (v2): via statsmodels vecm for multi-asset relationships.
- Pairs engine: compute z-score of residual/spread; entry/exit thresholds; half-life estimate for holding horizon; stop-outs on excessive z.

### 10) Strategies (signal + sizing interfaces)
- Crossover: generate_signals(df)->pd.Series in {1,0,-1} using short/long SMA; position sizing via volatility targeting (σ target) or fixed fraction.
- MACD: signal cross; filters for trend/vol.
- Bollinger: touch/break + re-entry mean reversion; ATR stops.
- Pairs: spread z-score trades; neutral sizing by hedge ratio; max duration cap by half-life.
- Strategy interface contract:
  - generate_signals(df:pd.DataFrame, config:dict)->pd.Series
  - size_positions(df:pd.DataFrame, signals:pd.Series, config:dict)->pd.Series (position in units or notional)

### 11) Backtester v1 (vectorized)
- Inputs: OHLCV DataFrame, signal & sizing callables, fees, slippage, initial cash.
- Mechanics: apply trades at next bar open/close (configurable); deduct fees/slippage; update positions/cash; log trades.
- Outputs: BacktestResult with fields
  - trades: DataFrame [ts, symbol, side, qty, price, fee, slippage]
  - equity_curve: DataFrame [ts, equity]
  - metrics: dict {total_return, annualized_return, sharpe, max_dd, hit_rate, payoff, turnover}
- Slippage models: fixed bps, spread-based, participation-rate (v2). Fees: taker/maker %, minimum fee.
- Determinism: fixed seeds for any randomness; configs saved alongside results (JSON).

### 12) Execution layer (paper → live)
- client_mb.py (authenticated REST)
  - sign_request(path, params, body)->headers; place_order(symbol, side, type, price, size); cancel_order(id); get_open_orders(); get_fills(); get_balances().
- order_manager.py
  - Create deterministic client IDs (utils/ids.py) `f"{date}{symbol}{side}{nonce}"`.
  - Reconcile local vs exchange states: resolve orphan orders; re-query on mismatches; treat timeouts as unknown and retry carefully.
  - Replace vs cancel/new logic for price updates; protect against crossing top-of-book by price bands.
- risk.py
  - Checks: max position per asset, max gross exposure, max daily loss, max leverage (if applicable), per-order notional limits; circuit breaker.
- live_loop.py
  - Pull latest data/signals → risk → order_manager → client; supports paper mode (log-only) and live mode; small sleep cadence.

### 13) Observability & ops
- Logging schema fields: ts, level, component, symbol, action, status, reason, latency_ms, request_id, client_id.
- Metrics (lightweight): ingest lag, WS reconnects, backfill duration, backtest run time, orders placed/canceled/filled/rejected, fill ratio, slippage bps.
- Alerts: WS stale > threshold, rate-limit bursts, repeated request failures, daily drawdown breach, reconciliation mismatches.

### 14) Testing strategy (detailed)
- Unit tests: indicators, cointegration (on synthetic pairs), PnL bookkeeping, fee/slippage application, schema validators, ID generation.
- Integration tests: REST pagination with recorded responses; WS reconnect with fake server; end-to-end backtest on synthetic data with known results; execution flow with mocked client.
- Property tests: PnL accounting conservation; monotonic timestamps; portfolio weights sum to 1 within tolerance; no negative balances.
- Record/replay: persist WS streams for deterministic tests; include replayer utility.

### 15) Security
- API keys via environment variables; ensure no secrets in logs; scoped keys where possible.
- Clock sync: NTP or API time endpoint; refuse to sign if skew exceeds threshold.

### 16) Performance guidelines
- Use pandas/NumPy vectorization; prefer Polars (Py) for heavy joins/groupbys if needed later.
- File rotation: target Parquet files ~50–200 MB for efficient IO.

### 17) Milestones with granular tasks & DoD
1) Foundations
  - Implement utils/logging.py, config loader, data/schemas.py, writer/reader; add round-trip tests.
  - DoD: pytest green; sample Parquet files written/read with schema checks.
2) Ingestion
  - Implement REST OHLCV backfill; add QA checks; CLI `scripts/ingest_mb.py`.
  - Implement WS trade stream with reconnect/backoff; rolling writer; quality report.
  - DoD: N days for top symbols; <0.1% duplicates; reconnect tested.
3) Indicators & stats
  - Implement TA functions; rolling stats; correlation heatmaps.
  - DoD: unit-tested parity on toy series; parity notebook created.
4) Backtester v1
  - Implement engine + fees/slippage + metrics; run strategies; output reports.
  - DoD: deterministic results with config snapshots; trades CSV and metrics JSON produced.
5) Pairs & cointegration
  - Implement Engle–Granger utilities; pairs strategy with z-score entries; conservative stops.
  - DoD: notebook validates pairs on BTC/ETH; backtest result meets baseline thresholds.
6) Live trading (paper → small live)
  - Implement client_mb, order_manager, risk, live_loop; paper mode first.
  - DoD: 1–2 weeks stable paper run; then micro live run with strict limits.

### 18) Mapping stock notebooks → crypto (actionable)
- Aula 1: replicate returns/vol/Sharpe on crypto OHLCV; add vol targeting notebook section.
- Aula 2: implement efficient frontier + HRP on top N crypto; rebalance with cost; compare vs equal-weight.
- Aula 3–4: port indicators & simple algos; add ATR stops and liquidity filters; walk-forward.
- Aula 5: cointegration on majors; half-life; spread z-score logic; caution for structural breaks.

### 19) Definitions of done (global)
- Every module has unit tests; new data written with schema validation; logs include component and request IDs; configs reproducible; dry-run defaults on live.

### 20) Immediate next actions for the LLM
- Create module skeletons per directory structure with empty functions and docstrings.
- Implement utils/logging.py, data/schemas.py (pydantic models), data/writer.py/reader.py with tests.
- Add configs/default.yaml with minimal fields (no secrets) and config loader.


