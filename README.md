# bursa-qlib

**AI-Oriented Quantitative Research Platform for Bursa Malaysia**

A research and demonstration platform for ML-driven portfolio research on Bursa Malaysia, featuring **dual-variant implementation** for comparative analysis.

## Purpose

1. **Personal learning vehicle** for end-to-end ML-driven portfolio research
2. **Demonstration** of institutional-grade analytics for Malaysian pension fund context
3. **Comparative study** of custom vs qlib-based quant pipelines

## Variants

| Variant | Description | Status |
|---------|-------------|--------|
| **A: Custom** | Hand-built pipeline with MY-specific factors | ✅ Complete |
| **B: qlib** | Microsoft qlib framework with Alpha158 + MY factors | 🚧 Planned |

### Variant A: Custom Pipeline (Complete)

- Direct LightGBM training on custom features
- 6 Malaysia-specific alpha factors
- Custom backtest engine with Bursa transaction costs
- Regime-conditioned strategy selection
- Institutional constraints (Shariah, sector caps, liquidity)

**Best Result:** +4.74% return, Sharpe 0.37, Max DD -4.86%

### Variant B: qlib Pipeline (Planned)

- qlib data format conversion
- Alpha158 expressions + MY factors
- qlib trainer and executor
- Same constraints overlay
- Performance comparison vs Variant A

## Data Sources

| Source | Data | Frequency |
|--------|------|-----------|
| Yahoo Finance | OHLCV price data (.KL tickers) | Daily |
| BNM OpenAPI | OPR, FX rates, interbank rates | Daily/Per MPC |
| OpenDOSM | GDP, CPI, IPI, trade, labour | Monthly/Quarterly |

## Malaysia-Specific Alpha Factors

Both variants will use these factors:

| Factor | Description | Rationale |
|--------|-------------|-----------|
| **Palm oil beta** | Rolling beta to FCPO | Malaysia is #2 palm oil producer |
| **FX sensitivity** | Correlation to USD/MYR | Exporters benefit from weak ringgit |
| **Shariah effect** | SC list entry/exit | Islamic funds represent significant capital |
| **GLC strength** | GLC vs private spread | Government policy influence |
| **Festive seasonality** | CNY, Hari Raya, year-end | Retail behavior + window dressing |
| **OPR regime** | Hiking/holding/cutting | Rate-sensitive sectors vary by regime |

## Quick Start

```bash
# Clone
git clone https://github.com/sediamembantu/bursa-qlib.git
cd bursa-qlib

# Install dependencies with uv
uv sync

# Run Variant A (Custom) pipeline
python scripts/01_fetch_data.py              # Fetch data
python scripts/03_train_model.py             # Train model
python scripts/04_backtest.py                # Backtest
python scripts/05_anomaly_scan.py            # Anomaly scan

# Run dashboard
streamlit run dashboard/app.py

# Run Variant B (qlib) pipeline (coming soon)
python scripts/06_qlib_convert.py            # Convert to qlib format
python scripts/07_qlib_train.py              # Train with qlib
python scripts/08_qlib_backtest.py           # qlib backtest
```

## Project Structure

```
bursa-qlib/
├── data/                   # Data storage
│   ├── raw/                # Downloaded CSVs
│   │   ├── prices/         # OHLCV data
│   │   ├── macro/          # BNM data
│   │   └── economic/       # OpenDOSM data
│   ├── processed/          # Feature matrices
│   ├── qlib/               # qlib binary format (Variant B)
│   └── anomaly_reports/    # Scan outputs
│
├── alpha/                  # Variant A: Custom pipeline
│   ├── factors/            # MY-specific factors
│   │   ├── palm_oil_beta.py
│   │   ├── fx_sensitivity.py
│   │   ├── shariah_effect.py
│   │   ├── glc_strength.py
│   │   ├── festive_seasonality.py
│   │   └── opr_regime.py
│   ├── models/             # Model configs
│   │   └── lightgbm.yaml
│   └── qlib/               # Variant B: qlib configs (planned)
│       ├── expressions.py  # Alpha158 + MY expressions
│       └── workflow.yaml   # qlib training config
│
├── regime/                 # Layer 4: Macro Regime
│   ├── hmm_detector.py     # HMM regime detection
│   └── conditioned_models.py
│
├── constraints/            # Layer 5: Institutional Constraints
│   ├── shariah_filter.py
│   ├── sector_caps.py
│   ├── liquidity_threshold.py
│   └── optimiser.py
│
├── anomaly/                # Layer 6: Anomaly Detection
│   ├── zscore.py
│   ├── velocity.py
│   ├── knn_detector.py
│   └── scanner.py
│
├── scripts/                # End-to-end workflows
│   ├── 01_fetch_data.py    # Data fetching
│   ├── 03_train_model.py   # Variant A training
│   ├── 04_backtest.py      # Variant A backtest
│   ├── 05_anomaly_scan.py  # Anomaly detection
│   ├── 06_qlib_convert.py  # Variant B: data prep (planned)
│   ├── 07_qlib_train.py    # Variant B: training (planned)
│   └── 08_qlib_backtest.py # Variant B: backtest (planned)
│
├── dashboard/              # Streamlit dashboard
│   └── app.py              # 6 pages including comparison view
│
├── tests/                  # Unit/integration tests
├── reference/              # Static data (Shariah, GLC, sectors)
└── models/                 # Trained model files
```

## Milestones

| Milestone | Description | Status | Result |
|-----------|-------------|--------|--------|
| **M1** | Data Platform | ✅ Complete | 24 KLCI-30 stocks |
| **M2** | Alpha + Training + Backtest | ✅ Complete | IC=0.11 |
| **M3** | Macro Regime Detection | ✅ Complete | HMM 4-regime |
| **M4** | Institutional Constraints | ✅ Complete | +4.74% return |
| **M5** | Anomaly Detection | ✅ Complete | Z+V+KNN |
| **M6** | Dashboard | ✅ Complete | Streamlit 6 pages |
| **M7** | qlib Variant | 🚧 Planned | Comparison study |

## Performance Summary

### Variant A: Custom Pipeline ✅ Winner

| Strategy | Return | Sharpe | Max DD | Volatility |
|----------|--------|--------|--------|------------|
| Baseline (weekly) | -2.67% | -0.15 | -14.42% | 9.62% |
| + Regime conditioning | -1.41% | -0.11 | -8.99% | 6.71% |
| **+ Constraints** | **+4.74%** | **0.37** | **-4.86%** | **6.70%** |

### Variant B: qlib Pipeline

| Strategy | Return | Sharpe | Max DD | Volatility |
|----------|--------|--------|--------|------------|
| Alpha158 + MY factors | -19.22% | -0.08 | -56.28% | 15.91% |

### Comparison

| Metric | Variant A | Variant B | Winner |
|--------|-----------|-----------|--------|
| Return | +4.74% | -19.22% | **A** |
| Sharpe | 0.37 | -0.08 | **A** |
| Max DD | -4.86% | -56.28% | **A** |

**Conclusion:** Domain-specific features outperform generic Alpha158 for Bursa Malaysia.

See [Variant Comparison](docs/VARIANT_COMPARISON.md) for detailed analysis.

## Key Findings

1. **Institutional constraints improve returns** — Filtering underperforming sectors (gambling, alcohol) and enforcing diversification turned strategy profitable
2. **Regime conditioning reduces risk** — Volatility -30%, max drawdown improved 38%
3. **MY-specific factors matter** — FX sensitivity ranked #2 in feature importance
4. **Weekly rebalancing preferred** — Daily rebalancing killed by transaction costs

## License

MIT

## Author

Peter — 2026
