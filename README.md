# bursa-qlib

**AI-Oriented Quantitative Research Platform for Bursa Malaysia**

A research and demonstration platform that applies Microsoft Qlib's AI-oriented quantitative investment framework to the Malaysian equity market.

## Purpose

1. **Personal learning vehicle** for end-to-end ML-driven portfolio research on Bursa Malaysia
2. **Demonstration** of institutional-grade analytics for Malaysian pension fund context

## Data Sources

| Source | Data | Frequency |
|--------|------|-----------|
| Yahoo Finance | OHLCV price data (.KL tickers) | Daily |
| BNM OpenAPI | OPR, FX rates, interbank rates | Daily/Per MPC |
| OpenDOSM | GDP, CPI, IPI, trade, labour | Monthly/Quarterly |

## Architecture

| Layer | Description | Status |
|-------|-------------|--------|
| 1. Data Platform | Data ingestion, Qlib conversion, validation | Code complete |
| 2. Alpha Research | Standard + MY-specific factors, model training | Planned |
| 4. Macro Regime Overlay | HMM-based regime detection | Planned |
| 5. Institutional Constraints | Shariah, sector caps, liquidity filters | Planned |
| 6. Anomaly Detection | Z-score, velocity, KNN surveillance | Planned |

## Malaysia-Specific Alpha Factors

- **Palm oil beta** — rolling beta to FCPO
- **FX sensitivity** — correlation to USD/MYR
- **Shariah compliance effect** — SC list entry/exit
- **GLC relative strength** — GLC vs private sector spread
- **Festive seasonality** — CNY, Hari Raya, year-end windows
- **OPR regime** — hiking/holding/cutting classification

## Quick Start

```bash
# Clone
git clone https://github.com/sediamembantu/bursa-qlib.git
cd bursa-qlib

# Install dependencies with uv
uv sync

# Run data pipeline
python scripts/01_fetch_data.py
python scripts/02_convert_qlib.py

# Train model
python scripts/03_train_model.py

# Backtest
python scripts/04_backtest.py
```

## Project Structure

```
bursa-qlib/
├── data/           # Layer 1: fetch, convert, validate
├── alpha/          # Layer 2: factors, models, backtest
├── regime/         # Layer 4: HMM regime detection
├── constraints/    # Layer 5: institutional constraints
├── anomaly/        # Layer 6: surveillance detection
├── scripts/        # End-to-end workflow (01-05)
├── dashboard/      # Streamlit demo
├── reference/      # Static data (Shariah list, GLC flags)
└── tests/          # Unit/integration tests
```

## Status

**M1 Complete** — Data pipeline operational.

See [Milestones](docs/milestones.md) for full timeline.

## License

MIT

## Author

Peter — 2026
