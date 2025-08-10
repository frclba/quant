### What the notebooks actually do
- Stochastic modeling and returns: Brownian/Geometric Brownian motion, log returns, rolling mean/std, cumulative PnL.
- Portfolio theory: covariance/correlation matrices, random-weight portfolios, efficient-frontier exploration, risk/return diagrams.
- Technical indicators and simple strategies: SMA/EMA, MACD, RSI, ATR, Bollinger, crossover signals, trade slicing and PnL aggregation.
- Pairs trading: Engle–Granger cointegration workflow using OLS + ADF test; ranking by correlation.
- ML for signals: basic Decision Tree/Random Forest gating of crossover signals; K-Means clustering for asset selection.
- IO and infra: historical data via investpy/yfinance/HTML scraping; visualization; a TCP socket server to talk to MetaTrader 5 and return a linear regression fit for plotting.

Key Python deps: numpy, pandas, matplotlib/seaborn, scipy/signal, statsmodels (ADF, coint, OLS), scikit-learn (LinReg, KMeans, DT/RF), tqdm, investpy/yfinance/pandas-datareader.

### Can this be built in Rust or Go?
Yes, both can implement all functional pieces, but the effort and library maturity differs sharply.

- Rust strengths for this workload
  - DataFrame and columnar compute: Polars (native Rust, very fast, strong time-series ops).
  - Numerics/ML: ndarray/ndarray-linalg/nalgebra; SmartCore and Linfa (KMeans, DT/RF, LinReg); statrs for distributions; argmin for optimization.
  - Tech indicators: multiple crates (e.g., ta, talib bindings) cover SMA/EMA/MACD/RSI/ATR/BB.
  - Concurrency/perf: rayon/tokio for SIMD-ish parallelism and async IO; zero-cost abstractions and memory safety.
  - Interop: pyo3/maturin to call Python only where needed (e.g., ADF), or bind to C/BLAS/LAPACK.
  - Notebooks: evcxr_jupyter is usable; Polars integrates with Arrow/Parquet cleanly.

- Go strengths for this workload
  - Services, IO, and ops: goroutines for adapters, APIs, streaming; simple deploys; great tooling.
  - Numerics: gonum (mat, stat, optimize), go-talib for indicators, golearn/gorgonia for ML (basic).
  - DataFrames: gota (adequate but limited vs Polars).
  - Plotting and notebooks exist but are basic (gonum/plot, gophernotes).

- Python strengths for this workload
  - Fastest prototyping and richest quant/stats/ML ecosystem: pandas/Polars (Py), NumPy/SciPy, statsmodels (ADF, coint, OLS), scikit-learn (LinReg/KMeans/DT/RF), pandas-ta/TA-Lib.
  - Excellent notebooks/visualization; large community patterns for research-to-prod handoff.
  - Solid async IO (asyncio, httpx, websockets, uvloop) for exchange adapters; C/NumPy/CuPy backends for performance-critical blocks.

- Biggest gap areas
  - Cointegration/ADF: first-class, batteries-included equivalents to statsmodels are lacking in both; Rust has better building blocks to implement ADF correctly; Go has fewer high-level stats/TS libs. In either language you’ll likely:
    - Implement ADF + Engle–Granger (feasible), or
    - Call Python’s statsmodels via FFI (Rust pyo3 is mature; in Go this is clunkier).

### Which language is the best option?
- If you want one language to cover both analytics and a production engine, choose Rust. Its numerics + DataFrame + performance ecosystem is meaningfully stronger than Go for this kind of quant/stat workload, and it gives you execution-quality performance and safety.
- If your priority is distributed services, exchange adapters, ingestion APIs, and ops simplicity (and you’re okay to call out to Python/Rust for analytics), Go is a great choice for the outer system, but it will be slower to replicate the analytics/stat notebooks natively.
- If speed of research iteration and breadth of statistics/ML matter most (especially cointegration/ADF, feature engineering, modeling), Python is the most feasible and productive choice; many teams ship mid-frequency systems fully in Python with careful engineering.

Short answer: Rust is the better single-language choice for a quant system like these notebooks. Go is excellent for the surrounding service mesh and data plumbing.

### Feasibility: Python vs Rust vs Go (at-a-glance)
- Prototyping/iteration speed: Python > Rust ≥ Go
- Stats/econometrics (ADF/coint/OLS): Python (statsmodels) >> Rust (custom/bridge) ≥ Go (custom)
- ML breadth (KMeans/DT/RF and beyond): Python >> Rust ≥ Go
- DataFrames/time-series ergonomics: Python (pandas/Polars) ≥ Rust (Polars) >> Go (gota)
- Low-latency execution/perf determinism: Rust >> Go > Python
- IO/concurrency for adapters: Rust (tokio) ≈ Go (goroutines) ≥ Python (asyncio/uvloop)
- Packaging/deploy ops: Go ≈ Rust > Python
- Community examples for quant: Python >> Rust ≥ Go

### Mapping notebook features to Rust vs Go vs Python
- Price simulation, returns, rolling stats
  - Rust: ndarray/Polars rolling; statrs.
  - Go: gonum/stat, rolling via custom/gonum.
  - Python: NumPy/SciPy + pandas rolling; vectorized ops.
- Tech indicators (SMA/EMA/MACD/RSI/ATR/BB)
  - Rust: ta/ta-lib bindings.
  - Go: go-talib, community ta packages.
  - Python: pandas-ta, TA-Lib, vectorbt.
- Portfolio/Markowitz
  - Rust: Polars for returns/cov; argmin or custom for constrained search; ndarray-linalg.
  - Go: gonum/mat + gonum/optimize; random portfolios trivial.
  - Python: pandas/NumPy for returns/cov; cvxpy for constrained optimization.
- Cointegration (OLS + ADF)
  - Rust: OLS via ndarray-linalg/SmartCore; implement ADF or call Python via pyo3.
  - Go: OLS via gonum/stat/regression; ADF requires custom implementation or external call.
  - Python: statsmodels (ADF, coint) out-of-the-box; OLS/rolling regressions ready.
- K-Means, DecisionTree/RandomForest
  - Rust: Linfa/SmartCore.
  - Go: golearn (older), gorgonia (DL), some community RF/DT libs.
  - Python: scikit-learn; mature and feature-rich.
 - Data ingestion and exchange connectivity (Yahoo/B3 scraping, Mercado Bitcoin-style REST/WebSocket)
   - Rust: reqwest/scraper or yahoo_finance_api for historical; tokio + tokio-tungstenite for WebSocket market data; HMAC-SHA256 request signing (e.g., `hmac` + `sha2`) for private endpoints; optional time sync (NTP) and retry/backoff with `tower`.
   - Go: colly/goquery for scraping; net/http + gorilla/websocket for streams; HMAC signing via `crypto/hmac` + `sha256` for private endpoints.
  - Python: httpx/aiohttp for REST, websockets for streams, HMAC signing via `hmac`/`hashlib`; rich SDK patterns and examples.

### Recommended architecture
- Research/analytics
  - Rust with Polars + SmartCore/Linfa; use pyo3 to borrow statsmodels ADF initially.
  - Optionally keep Python notebooks for exploration while porting core computations to Rust crates for reuse.
- Data layer
  - Parquet on disk (Arrow), or ClickHouse/TimescaleDB; both Rust and Go have good clients.
 - Strategy/backtesting/execution
   - Rust engine: Polars for timeseries, rayon for parallel backtests, ta for indicators, SmartCore for ML gates, argmin for optimization.
   - Low-latency IO: tokio + exchange connectors; build a Mercado Bitcoin adapter (REST for account/order, WebSocket for ticker/trades/orderbook) with authenticated signing and idempotent order management.
- Services and orchestration
  - If you prefer, implement adapters, REST/WebSocket gateways, schedulers in Go; call Rust analytics via gRPC/FFI.

### Migration plan (lean)
 1) Port indicators, returns, and Markowitz sampling to Rust (Polars + ta + argmin).
 2) Implement OLS + ADF in Rust, or bridge to Python’s statsmodels via pyo3 as an interim.
 3) Implement a Mercado Bitcoin exchange adapter in Rust:
    - Public data: REST for historical candles; WebSocket streams (ticker/trades/orderbook) via tokio-tungstenite.
    - Private endpoints: HMAC-SHA256 signed REST requests for balances/orders; clock skew handling; retry/backoff.
    - Normalize to internal types (instrument, side, order type, fills) and persist raw + normalized to Parquet.
 4) Replace notebook data pulls with the adapter output and a small data service (Rust or Go) that writes Arrow/Parquet and exposes query endpoints.
 5) Build a Rust backtester that consumes Parquet market data and reproduces crossover/BB/MA strategies and PnL figures; validate against notebook outputs.
 6) Add K-Means and RF gates via Linfa/SmartCore; validate parity with sklearn.
 7) Wire live trading: strategy -> risk checks -> adapter (placing/canceling orders); add sanity guards (min notional, max exposure, rate limits) and structured logging/metrics.

Status: Parsed the notebooks and identified all techniques and dependencies. Mapped each to Rust/Go capabilities and highlighted ecosystem gaps. Recommendation provided with an incremental migration plan.

- Best single-language choice: Rust (numerics + performance + DataFrame maturity).
- Go remains great for service plumbing; use it alongside Rust if your team values operational simplicity.