# Proposed Structure for bursa-qlib

```
qlib_experiment/
├── README.md
├── pyproject.toml              # uv package config
├── uv.lock                     # pinned dependencies
├── config.py                   # centralised config (APIs, paths, universe)
├── tickers.py                  # KLCI-30 and extended universe definitions
│
├── data/                       # Layer 1: Data Platform
│   ├── raw/                    # Downloaded CSVs (price, macro, economic)
│   │   ├── prices/
│   │   ├── macro/
│   │   └── economic/
│   ├── qlib/                   # Qlib binary format output
│   ├── fetch/                  # Data fetchers
│   │   ├── yahoo_finance.py
│   │   ├── bnm_openapi.py
│   │   └── opendosm.py
│   ├── convert/                # CSV → Qlib binary
│   │   └── qlib_converter.py
│   └── validate/               # Data quality checks
│       └── validation.py
│
├── alpha/                      # Layer 2: Alpha Research
│   ├── factors/                # Malaysia-specific factors
│   │   ├── palm_oil_beta.py
│   │   ├── fx_sensitivity.py
│   │   ├── shariah_effect.py
│   │   ├── glc_strength.py
│   │   ├── festive_seasonality.py
│   │   └── opr_regime.py
│   ├── models/                 # Model configs and training
│   │   ├── lightgbm.yaml
│   │   ├── transformer.yaml
│   │   └── alstm.yaml
│   └── backtest/               # Bursa-specific backtest config
│       └── bursa_config.py
│
├── regime/                     # Layer 4: Macro Regime Overlay
│   ├── features.py             # Build feature matrix from BNM/DOSM
│   ├── hmm_detector.py         # HMM regime detection
│   └── conditioned_models.py   # Regime-conditioned model selection
│
├── constraints/                # Layer 5: Institutional Constraints
│   ├── shariah_filter.py
│   ├── sector_caps.py
│   ├── liquidity_threshold.py
│   └── optimiser.py            # Portfolio optimisation with constraints
│
├── anomaly/                    # Layer 6: Anomaly Detection
│   ├── zscore.py
│   ├── velocity.py
│   └── knn_detector.py
│
├── scripts/                    # End-to-end workflow scripts
│   ├── 01_fetch_data.py
│   ├── 02_convert_qlib.py
│   ├── 03_train_model.py
│   ├── 04_backtest.py
│   └── 05_anomaly_scan.py
│
├── notebooks/                  # Exploration and demos
│   └── exploration.ipynb
│
├── dashboard/                  # Streamlit dashboard
│   └── app.py
│
├── tests/                      # Unit and integration tests
│   ├── test_fetchers.py
│   ├── test_factors.py
│   └── test_validation.py
│
├── reference/                  # Static reference data
│   ├── shariah_list.csv
│   ├── klc_constituents.csv
│   ├── sector_mapping.csv
│   └── glc_ownership.csv
│
└── logs/                       # Pipeline logs
```

## Notes

- **Layer 3** is skipped (deferred per requirements)
- **config.py** centralises all API endpoints, paths, and universe definitions
- **scripts/** numbered for reproducible workflow
- **reference/** holds manually curated data (Shariah list, GLC flags)
- **uv** for dependency management (as specified)
