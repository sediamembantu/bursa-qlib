# M7: qlib Integration Specification

## Overview

Integrate Microsoft qlib as **Variant B** to compare against the existing custom pipeline (Variant A).

## Goals

1. **Compare frameworks** — Custom vs qlib performance
2. **Leverage qlib Alpha158** — 158 standard expressions + MY factors
3. **Validate findings** — Does qlib improve on custom results?
4. **Learning** — Understand qlib internals for future projects

## Technical Requirements

### Data Conversion

Convert existing CSV data to qlib binary format:

```
data/raw/prices/1155.csv → data/qlib/features/1155.bin
```

**Format:**
- Bin files with (date, open, high, low, close, volume)
- Calendar: Bursa trading days (skip public holidays)
- Instruments: 24 KLCI-30 tickers with data

### qlib Configuration

**File:** `alpha/qlib/qlib_config.yaml`

```yaml
qlib_version: 0.9
market: my
calendar: my_calendar
instruments: klc30

data_handler:
  class: Alpha158
  module_path: qlib.contrib.data.handler

  # Custom MY factors to append
  custom_features:
    - palm_oil_beta
    - fx_sensitivity
    - shariah_compliant
    - glc_flag
    - festive_window
    - opr_regime
```

### Alpha158 + MY Expressions

**File:** `alpha/qlib/expressions.py`

```python
# Standard Alpha158 (subset)
ALPHA158 = [
    "Ref($close, -1) / $close",  # 1-day return
    "Mean($close, 5) / $close",  # 5-day MA
    # ... 158 expressions
]

# Malaysia-specific (append)
MY_ALPHA = [
    "Corr($close, $palm_oil, 20)",     # Palm oil correlation
    "Corr($close, $usdmyr, 20)",       # FX sensitivity
    "$shariah_compliant",               # Shariah flag
    "($close - Mean($glc_index, 20))", # GLC spread
    # ... etc
]
```

### Training Script

**File:** `scripts/07_qlib_train.py`

```python
# Pseudo-code
from qlib.workflow import R
from qlib.contrib.model.gbdt import LGBModel

with R.start(experiment_name="bursa_qlib"):
    # Load qlib data
    dataset = DatasetH(handler=Alpha158Handler(...))
    
    # Train model
    model = LGBModel(**config)
    model.fit(dataset)
    
    # Save predictions
    R.save_objects(model=model)
```

### Backtest Script

**File:** `scripts/08_qlib_backtest.py`

```python
# Use qlib executor with same constraints
from qlib.backtest import backtest_executor

# Apply Shariah filter, sector caps, liquidity
# Compare results to Variant A
```

## File Structure (New)

```
bursa-qlib/
├── alpha/
│   └── qlib/                    # NEW
│       ├── __init__.py
│       ├── expressions.py       # Alpha158 + MY expressions
│       ├── handler.py           # Custom data handler
│       └── workflow.yaml        # qlib experiment config
│
├── scripts/
│   ├── 06_qlib_convert.py       # NEW: CSV → qlib binary
│   ├── 07_qlib_train.py         # NEW: Train with qlib
│   └── 08_qlib_backtest.py      # NEW: Backtest with qlib
│
├── data/
│   └── qlib/                    # NEW
│       ├── calendars/           # Trading calendar
│       ├── instruments/         # Instrument list
│       └── features/            # Binary feature files
│
└── dashboard/
    └── pages/
        └── 7_Comparison.py      # NEW: Variant A vs B
```

## Implementation Tasks

| # | Task | File | Est. Time |
|---|------|------|-----------|
| 1 | Create qlib calendar (MY trading days) | `data/qlib/calendars/` | 15 min |
| 2 | Convert CSV to qlib binary | `scripts/06_qlib_convert.py` | 30 min |
| 3 | Create Alpha158 + MY expressions | `alpha/qlib/expressions.py` | 30 min |
| 4 | Create data handler | `alpha/qlib/handler.py` | 20 min |
| 5 | Training script | `scripts/07_qlib_train.py` | 30 min |
| 6 | Backtest script | `scripts/08_qlib_backtest.py` | 30 min |
| 7 | Dashboard comparison page | `dashboard/pages/7_Comparison.py` | 20 min |
| 8 | Documentation update | `README.md` | 10 min |

**Total: ~3 hours**

## Success Criteria

| Metric | Variant A (Custom) | Variant B Target |
|--------|-------------------|------------------|
| Return | +4.74% | > +4.74% |
| Sharpe | 0.37 | > 0.37 |
| Max DD | -4.86% | < -4.86% |
| IC | 0.11 | > 0.11 |

## Risks

| Risk | Mitigation |
|------|------------|
| qlib learning curve | Start with minimal config |
| Data format issues | Test with 1 ticker first |
| Alpha158 may not suit MY | Add MY-specific expressions |
| Performance may not improve | Document learnings anyway |

## Notes

- Keep Variant A unchanged for comparison
- Use same test period for fair comparison
- Document any qlib-specific issues
- Consider this a learning exercise, not production
