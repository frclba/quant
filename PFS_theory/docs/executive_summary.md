## Executive summary: Building a practical quant system

### What we’re building
- A fast, reliable trading and research system inspired by the existing notebooks.
- Uses Mercado Bitcoin’s API for market data and live trading instead of MetaTrader.

### Recommended languages and roles
- Rust: Core engine for speed, safety, and reliability (data processing, backtesting, live execution).
- Python: Research and modeling (quick experiments, statistics, machine learning, notebooks).
- Go (optional): Lightweight services and integration pieces if we want very simple deployment for APIs and schedulers.

### Why this split works
- Rust keeps the live engine fast and stable when money is at risk.
- Python lets us iterate quickly on ideas and models.
- Both share the same data files, so research and production stay in sync.

### Data source
- Mercado Bitcoin: historical and live market data, plus order placement and account info.

### What we will deliver
- Market data adapter: downloads historical data and streams live prices; stores everything in analysis-friendly files.
- Analytics library: indicators (moving averages, RSI, MACD, Bollinger), portfolio math, and basic statistics.
- Backtester: runs strategies on past data to estimate performance and risk.
- Live trading service: turns strategy signals into orders with guardrails (position limits, max loss, etc.).
- Research workspace: notebooks to explore data and build models (e.g., cointegration, clustering, trees/forests).

### How we’ll get there (phased)
1) Foundations: indicators, returns, and portfolio basics; set up shared data format.
2) Data: build the Mercado Bitcoin adapter (historical + live) and store data consistently.
3) Backtesting: reproduce the simple strategies from the notebooks and validate results.
4) Research: python notebooks for cointegration and ML features; promote stable pieces into the engine.
5) Live trading: connect the engine to the exchange with safety checks and monitoring.
6) Improve: iterate on models, add risk controls, and expand strategy coverage.

### Key risks and how we handle them
- Statistics depth (e.g., cointegration tests): use Python where it’s strongest; migrate later if needed.
- Data quality and outages: store raw and cleaned data; add retries and alerts.
- Exchange limits and changes: rate‑limit requests, version adapters cleanly, and keep a sandbox mode.

### Outcome
- A system that is both practical and robust: quick to research in Python, safe and fast to run in Rust, with a clear path to production.


