# bursa-qlib

Quant research workflows for Bursa Malaysia, with a custom factor pipeline and a
local qlib-style comparison workflow.

## Current Scope

- Variant A: custom LightGBM workflow with Malaysia-specific factors.
- Variant B: qlib-style local workflow built from source in this repo.
- Actual Microsoft `pyqlib` integration remains future research and is not
  required to run the checked-in code.

## Environment

This repository is intended to be run directly from source with `uv`.

```bash
git clone https://github.com/sediamembantu/bursa-qlib.git
cd bursa-qlib
uv sync --group dev
```

Notes:

- No market data is committed. The fetch scripts populate `data/raw/`.
- The project is configured for source execution, so use `uv run ...` from the
  repository root.

## Quick Start

Fetch data:

```bash
uv run python scripts/01_fetch_data.py
```

Run the custom workflow:

```bash
uv run python scripts/03_train_model.py
uv run python scripts/04_backtest.py
uv run python scripts/05_anomaly_scan.py
```

Run the qlib-style workflow:

```bash
uv run python scripts/06_qlib_calendar.py
uv run python scripts/06_qlib_convert.py
uv run python scripts/07_qlib_train.py
uv run python scripts/08_qlib_backtest.py
```

Run the dashboard:

```bash
uv run python scripts/run_dashboard.py
```

Run tests:

```bash
uv run pytest
```

## Data Sources

- Yahoo Finance for Bursa OHLCV data (`.KL` tickers).
- BNM OpenAPI for OPR and exchange-rate data.
- OpenDOSM for Malaysian macro and economic datasets.

## Repository Layout

```text
alpha/          Factor logic and qlib-style helpers
anomaly/        Anomaly detection workflows
constraints/    Portfolio constraint logic
dashboard/      Streamlit dashboard
data/           Fetchers, validation helpers, runtime data folders
multi_asset/    EPF-style multi-asset utilities
regime/         Regime detection and conditioned models
scripts/        End-to-end entry points
tests/          Smoke and regression tests
```

## Operational Notes

- `scripts/07_qlib_train.py` and `scripts/08_qlib_backtest.py` use the local
  `alpha/qlib/` helpers. They do not import `pyqlib`.
- `data/qlib/` contains parquet exports, a Bursa trading calendar, and a simple
  instrument list for the qlib-style workflow.
- Generated artifacts such as raw data, trained models, and backtest outputs are
  intentionally gitignored.

## Validation

The repo is expected to pass:

```bash
uv run pytest
uv run python -m compileall alpha anomaly constraints dashboard data multi_asset regime scripts tickers.py config.py
```

## License

MIT
