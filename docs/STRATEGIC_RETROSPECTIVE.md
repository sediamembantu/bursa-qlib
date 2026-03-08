# Strategic Retrospective: Why EPF Wins

## Executive Summary

**Hypothesis:** Dynamic allocation + monthly rebalancing + trend following would improve returns by +10-15%

**Reality:** Only improved by +0.20%

**Root cause:** Can't fix a losing equity strategy with allocation alone

**Key insight:** EPF's advantage is global diversification, not active management

---

## The Setup

### Starting Point (After M8)

| Strategy | Return | Sharpe | Max DD |
|----------|--------|--------|--------|
| EPF | +13.09% | ∞ | 0% |
| Conservative Static | +2.01% | 0.24 | -8.57% |
| Equity-Only | -13.61% | -0.43 | -34.72% |

**Gap to EPF:** 11%

### Our Hypothesis (Phase 1)

We believed dynamic allocation would close the gap:

1. **Regime-based allocation:** +3-5%
   - Use HMM detector (already built in M3)
   - Be aggressive in risk-on, defensive in crisis
   
2. **Monthly rebalancing:** +3-4%
   - Reduce from 4,664 trades to ~1,200
   - Save ~75% on transaction costs
   
3. **Trend following:** +4-6%
   - 200-day moving average filter
   - Go defensive when below MA

**Expected combined improvement:** +10-15%
**Expected final return:** +12-17% (competitive with EPF!)

---

## What We Implemented

### scripts/11_dynamic_backtest_fixed.py

```python
ALLOCATION_RULES = {
    # (regime, trend) -> (equity, bonds, mm)
    ("risk_on", "bull"): (0.70, 0.20, 0.10),      # Aggressive
    ("risk_on", "bear"): (0.40, 0.40, 0.20),      # Cautious
    ("risk_off", "bull"): (0.50, 0.35, 0.15),     # Balanced
    ("risk_off", "bear"): (0.30, 0.50, 0.20),     # Conservative
    ("crisis", "bull"): (0.20, 0.55, 0.25),       # Defensive
    ("crisis", "bear"): (0.10, 0.60, 0.30),       # Crisis mode
}
```

**Implementation details:**
- HMM detector for regime (risk_on, risk_off, crisis, recovery)
- KLCI index for trend (200-day MA)
- Monthly rebalancing (20 trading days)
- Dynamic weights based on regime + trend

---

## The Results

### Reality Check

| Strategy | Return | Sharpe | Max DD |
|----------|--------|--------|--------|
| Dynamic Multi-Asset | **+2.21%** | 0.18 | -8.39% |
| Conservative Static | +2.01% | 0.24 | -8.57% |
| **Improvement** | **+0.20%** | -0.06 | +0.18% |

**Expected:** +10-15%
**Actual:** +0.20%
**Gap:** 99.3% shortfall

### Market Conditions (2024-2025)

| Signal | Distribution | Impact |
|--------|--------------|--------|
| Trend | 80% bear | Defensive allocation |
| Regime | 49% risk-on, 35% risk-off, 16% crisis | Mixed |
| KLCI | Declining | Bad for Malaysian equities |

### Weight Distribution

```
Average equity weight: 37.4% (vs 30% static)
Range: 10% to 70%
```

The strategy correctly stayed defensive but couldn't generate alpha.

---

## Why It Failed

### The Math Problem

```
Underlying equity return: -13.61%
Bonds: +3% (approx)
Money market: +2.5% (approx)

Even with perfect timing:
70% × -13.61% + 20% × 3% + 10% × 2.5% = -8.7%
30% × -13.61% + 50% × 3% + 20% × 2.5% = -1.6%

You can't escape negative alpha with allocation alone.
```

### The Feedback Loop

1. Equity strategy loses money → NAV declines
2. Market in bear trend → Defensive allocation
3. Defensive = lower equity exposure
4. Lower equity = lower returns (but also lower losses)
5. Still negative overall

### The Fundamental Issue

**We were trying to solve the wrong problem.**

The problem isn't "how to allocate better"
The problem is "our equity strategy loses money"

**If your stock picks lose -13.61%, no allocation strategy fixes that.**

---

## What ACTUALLY Works

### The EPF Advantage

| Factor | EPF | Our Strategy |
|--------|-----|--------------|
| Geography | Global | 100% Malaysia |
| Securities | 1000+ | 24 stocks |
| Asset classes | Equities, bonds, PE, RE, infra | Equities, bonds, MM |
| Costs | Minimal (scale) | High (4,664 trades) |
| Management | Professional team | Simple ML model |
| Time horizon | Decades | Monthly rebalance |

### Key Insight: Global Diversification

**S&P 500 (2024-2025):** ~+25%
**KLCI (2024-2025):** ~-10%

If we had 30% in S&P 500:
```
30% × 25% = +7.5% contribution
```

**That alone would close 68% of the gap to EPF.**

### Realistic Improvement Path

| Improvement | Expected Impact | Difficulty |
|------------|-----------------|------------|
| **Global diversification** | +7-10% | Medium (need ETF data) |
| Better equity alpha | +2-3% | Hard (research needed) |
| Options income | +1-2% | Medium |
| REITs/alternatives | +1-2% | Easy |
| **Total** | **+11-17%** | |

With global diversification alone, we'd reach +9-12% (nearly breakeven with EPF).

---

## Lessons Learned

### 1. Allocation ≠ Alpha

**Misconception:** "Dynamic allocation will improve returns"

**Reality:** Allocation manages risk, doesn't create alpha

- ✅ Risk reduction: -26% max drawdown
- ❌ Return improvement: +0.20% only

**Lesson:** Alpha comes from security selection, not allocation

### 2. Can't Time a Losing Strategy

**Misconception:** "We'll reduce exposure in bad times"

**Reality:** If the strategy loses money overall, even perfect timing can't fix it

```
Perfect timing on a -13.61% strategy = still negative
```

**Lesson:** Fix the underlying strategy first

### 3. Home Bias Kills

**Misconception:** "Malaysian market is enough"

**Reality:** Malaysia is 0.5% of global market cap

- Concentrated risk
- Limited diversification
- Underperformance vs global markets

**Lesson:** Global diversification is essential

### 4. Transaction Costs Matter

**Misconception:** "Weekly rebalancing keeps portfolio optimized"

**Reality:** 4,664 trades × ~0.15% = ~7% in costs

**Lesson:** Lower frequency = higher returns (for same strategy)

### 5. EPF Knows What They're Doing

**Misconception:** "I can beat EPF with ML"

**Reality:** EPF has:
- Global access
- Scale advantages
- Professional management
- Decades of experience
- Private markets access

**Lesson:** Retail investors should benchmark against EPF, not try to beat it

---

## The Honest Truth

### For Retail Investors

**EPF is the optimal choice for most people.**

Why:
- Guaranteed 6.15% with zero risk
- Professional management
- Global diversification
- No effort required
- Tax advantages

### When Active Management Makes Sense

Only if:
1. ✅ You can access global markets
2. ✅ You have edge in stock selection
3. ✅ You can tolerate higher risk
4. ✅ You have time to manage it
5. ✅ You accept you might underperform

### What bursa-qlib Achieved

Despite not beating EPF, we:
- ✅ Built complete ML pipeline (M1-M7)
- ✅ Implemented multi-asset portfolio (M8)
- ✅ Tested dynamic allocation (Phase 1)
- ✅ Proved risk reduction works (-26% max DD)
- ✅ Learned why EPF is hard to beat
- ✅ Documented lessons for future

**This is valuable research, even if the conclusion is "EPF wins".**

---

## What Would I Do Differently?

### If Starting Over

1. **Start with global diversification**
   - 40% Malaysia, 30% US, 10% Asia, 20% bonds/MM
   - Don't limit to single country

2. **Focus on cost reduction**
   - Quarterly rebalancing, not weekly
   - Target: <500 trades over 2 years

3. **Simpler model**
   - Factor investing > ML
   - Momentum, quality, value

4. **Better benchmark**
   - Compare to EPF from day 1
   - Test on same time period

5. **Accept limitations**
   - Can't access private markets
   - Can't match EPF's scale
   - Focus on what's achievable

---

## Next Steps

### Option A: Implement Global Diversification

**Effort:** Medium
**Expected improvement:** +7-10%

**Requirements:**
- S&P 500 ETF data (VOO/IVV)
- Asia ex-MY ETF data (AAXJ/EWM)
- Currency hedging consideration

**Result:** Would nearly close gap to EPF

### Option B: Accept and Move On

**Effort:** Zero
**Improvement:** N/A

**Rationale:**
- EPF is superior for passive investors
- Active management not worth the effort
- Time better spent on other projects

### Option C: Research Better Alpha

**Effort:** High
**Expected improvement:** +2-5%

**Requirements:**
- Factor research
- Fundamental analysis
- Alternative data

**Uncertain:** May or may not work

---

## Final Thoughts

### The Hardest Lesson

**Sometimes the answer is: "You can't beat the benchmark."**

And that's okay. Understanding *why* you can't beat it is valuable:

- EPF has structural advantages (global, scale, private markets)
- Our strategy had fundamental flaws (home bias, costs, bad period)
- Dynamic allocation manages risk, not alpha

### What Success Looks Like

**Not:** "I beat EPF by 1%"
**But:** "I understand the market, built skills, and made informed decisions"

### The Real Value

This project taught us:
- How to build ML pipelines
- How to backtest properly
- How to compare strategies fairly
- When to accept limitations
- Why benchmarks exist

**That's worth more than beating EPF by 1%.**

---

## Conclusion

**Phase 1 failed to deliver expected returns because:**
1. Underlying equity strategy lost money
2. Allocation can't fix negative alpha
3. Market conditions were unfavorable
4. We were solving the wrong problem

**What would actually work:**
1. Global diversification (+7-10%)
2. Better equity alpha (+2-3%)
3. Cost reduction (+1-2%)

**Honest assessment:**
- EPF wins for most investors
- Active management only makes sense with global access
- bursa-qlib succeeded as a learning project

**Project status:**
- M1-M8: Complete
- Phase 1: Complete (underperformed expectations)
- Phase 2: Available but not started (global diversification)

**Final words:**

> "The goal wasn't to beat EPF. The goal was to understand if we could. 
> The answer is 'not without global diversification.' 
> And that's a valuable answer."

---

**Document version:** 1.0
**Date:** 2026-03-08
**Author:** Galahad
**Project:** bursa-qlib
**Repo:** https://github.com/sediamembantu/bursa-qlib
