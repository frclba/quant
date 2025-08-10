## LLM Build Brief — Python Crypto Quant (Mercado Bitcoin)

Single source of truth for an LLM to implement this project via iterative edits. Keep outputs small, atomic, and testable. Prefer creating/updating files over dumping long code in chat.

### Mission
- Build a Python-only crypto quant system: ingest data → compute indicators → backtest → live trade on Mercado Bitcoin with guardrails.

### Ground rules
- Language: Python 3.11+ only.
- Data format: Parquet via pyarrow. Timestamps in UTC.
- Secrets: never hardcode keys; load from env; redact in logs.
- Style: clear names, small functions, unit tests first for core utilities.
- Do not add README files (per user rule).

### References (read first)
- `docs/implementation_blueprint.md` — high-level plan and milestones.
- `docs/crypto_python_plan.md` — system plan and repo layout.
- `docs/crypto_python_plan_improvements.md` — enhancements and crypto mapping.

### Directory layout (must create as needed)
```
crypto_quant/
  data/
    adapters/mercado_bitcoin.py
    schemas.py
    writer.py
    reader.py
  indicators/ta_indicators.py
  stats/cointegration.py
  strategies/{crossover.py, macd.py, bollinger.py, pairs.py}
  backtest/{engine.py, slippage.py, fees.py, metrics.py}
  execution/{client_mb.py, order_manager.py, risk.py, live_loop.py}
  utils/{time.py, config.py, logging.py}
  scripts/{ingest_mb.py, backtest.py, trade_live.py}
  data/parquet/ (gitignored)
```

### Core schemas (Parquet)
- ohlcv: ts (int64 ns), open, high, low, close, volume (float64), symbol (str), exchange (str)
- trades: ts, price, size, side in {buy,sell}, trade_id (str), symbol, exchange
- orderbook_l2 (optional): ts, bids: list[price,size], asks: list[price,size]
- orders/fills (private): ts, order_id, client_id, side, type, price, size, status, fee, symbol

### Milestone tasks (LLM-executable)

1) Foundations
- Create `utils/logging.py`: function `get_logger(name:str)->logging.Logger` with JSON-ish structured logs.
- Create `data/schemas.py`: pydantic models for OHLCV and Trade rows; validate fields and dtypes.
- Create `data/writer.py`: functions `write_parquet(df:pd.DataFrame, path:str)` and partition helpers.
- Create `data/reader.py`: function `read_parquet(path:str, columns:Optional[list[str]]=None)`.
- DoD: round-trip test in `tests/test_io.py` writing and reading small synthetic frames.

2) Mercado Bitcoin adapter (public)
- `data/adapters/mercado_bitcoin.py`:
  - REST: `fetch_ohlcv(symbol:str, start:datetime, end:datetime, interval:str)->pd.DataFrame` (rate-limit safe, pagination if needed).
  - WS: `stream_trades(symbols:list[str], on_event:Callable)` using `websockets` with reconnect/backoff.
- `scripts/ingest_mb.py`: CLI to backfill OHLCV for given symbols and write Parquet partitions.
- DoD: integration test with mocked HTTP/WS; saved Parquet under `data/parquet/ohlcv/<symbol>/...`.

3) Indicators
- `indicators/ta_indicators.py`: SMA, EMA, RSI, ATR, MACD, Bollinger using pandas ops (no inline imports inside functions).
- DoD: unit tests in `tests/test_indicators.py` vs known small arrays.

4) Backtester v1 (vectorized)
- `backtest/engine.py`: API `run_backtest(data:pd.DataFrame, strategy:Callable, fees:Fees, slip:Slippage, config:dict)->BacktestResult`.
- `backtest/fees.py`, `backtest/slippage.py`, `backtest/metrics.py`: simple percentage fees, spread/participation slippage, metrics (PnL, equity, DD, Sharpe, hit rate).
- Strategies in `strategies/` expose `generate_signals(df)->pd.Series` and `size_positions(df, signals)->pd.Series`.
- DoD: `scripts/backtest.py` runs crossover/MACD/BB on sample data and writes a small report (JSON + CSV trades).

5) Live trading (paper first)
- `execution/client_mb.py`: authenticated REST (HMAC), `place_order`, `cancel_order`, `get_open_orders`, `get_fills`.
- `execution/order_manager.py`: idempotent client IDs, reconcile local vs exchange, replace vs cancel/new.
- `execution/risk.py`: checks for max position, max daily loss, notional limits, and circuit breaker.
- `execution/live_loop.py`: loop wiring signals → risk → orders; supports paper and live modes.
- DoD: `scripts/trade_live.py --paper` runs and logs actions without placing real orders.

### Non-functional expectations
- Reconnect WS on errors (exponential backoff with jitter); never lose ordering within a file (monotonic ts check).
- All backtests deterministic with fixed seeds and saved config snapshots.
- Logging includes timestamps, symbol, action, status, and error messages.

### Testing matrix (minimum)
- IO: round-trip Parquet, schema validation, partition correctness.
- Indicators: numeric equality on toy inputs; window edge cases.
- Backtester: PnL math reconciliation to trade logs; fee/slippage application tested separately.
- Adapter: HTTP pagination and WS reconnect mocked; rate-limit paths.
- Execution: order translation and idempotency IDs; reconciliation logic with mocked API.

### Delivery cadence
- Each PR focuses on one milestone sub-task with tests and a short summary.
- Keep configs in `configs/default.yaml` and load via `utils/config.py`.

### Out of scope (v1)
- Derivatives (funding/index prices), advanced portfolio optimizers (HRP/Markowitz), and deep ML; add later.


