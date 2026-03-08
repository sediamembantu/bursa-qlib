# Variant Comparison: bursa-qlib

## Fair Comparison on 2024-2025 Period

To properly compare Variant A (Custom) vs Variant B (qlib), both should be tested on the **same time period**.

### Variant B (qlib-style) - 2024-2025

| Metric | Value |
|--------|-------|
| Period | 2024-01-02 to 2025-12-31 |
| Trading Days | 489 |
| Total Return | **-2.88%** |
| Annual Return | -1.48% |
| Sharpe Ratio | -0.22 |
| Max Drawdown | -4.95% |
| Volatility | 6.68% |

### Variant A (Custom) - 2024-2025

**Status:** ⏳ Pending execution

To run:
```bash
python scripts/04_backtest.py \
    --model models/lightgbm_20260308_0816.txt \
    --start 2024-01-01 \
    --end 2025-12-31 \
    --capital 1000000
```

### Previous Results (Different Periods)

#### Variant A (Custom) - 2023-2024
| Metric | Value |
|--------|-------|
| Total Return | **+4.74%** |
| Sharpe | 0.37 |
| Max DD | -4.86% |

#### Variant B (qlib) - 2024-2026  
| Metric | Value |
|--------|-------|
| Total Return | -3.21% |
| Sharpe | -0.22 |
| Max DD | -5.58% |

## Key Observations

1. **Different time periods** make direct comparison difficult
2. **Variant B (qlib)** shows negative returns on 2024-2025 period
3. **Variant A (Custom)** showed positive returns on 2023-2024 period
4. Market conditions likely differed between periods

## Recommendation

Run both variants on **identical 2024-2025 period** for definitive comparison.

## Technical Differences

| Aspect | Variant A (Custom) | Variant B (qlib) |
|--------|-------------------|------------------|
| Framework | Custom pipeline | qlib-inspired |
| Features | 6 MY-specific factors | 16 factors (MY-only) |
| Training | Direct LightGBM | qlib-style workflow |
| Backtest | Custom engine | Custom with constraints |
| Constraints | Shariah, sector, liquidity | Same constraints |

## Conclusion

**Current Status:** 
- Variant A showed promise on 2023-2024 (+4.74%)
- Variant B underperformed on 2024-2025 (-2.88%)
- **Fair comparison requires same time period testing**

**Next Steps:**
1. Complete Variant A backtest on 2024-2025
2. Compare metrics side-by-side
3. Document findings
