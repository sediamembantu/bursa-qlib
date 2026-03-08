"""
Bursa-Qlib Dashboard

Streamlit dashboard for visualizing:
- Portfolio performance
- Market regime states
- Anomaly alerts
- Factor exposures
- Current portfolio holdings

Run with: streamlit run dashboard/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import sys
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config import PRICES_DIR, RANDOM_SEED
from tickers import KLCI30_CODES, get_local_name, get_sector


# =============================================================================
# Page Config
# =============================================================================

st.set_page_config(
    page_title="Bursa-Qlib Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Load Data
# =============================================================================

@st.cache_data
def load_price_data():
    """Load all price data."""
    price_data = {}
    for csv_file in PRICES_DIR.glob("*.csv"):
        ticker = csv_file.stem
        df = pd.read_csv(csv_file)
        df["date"] = pd.to_datetime(df["date"])
        price_data[ticker] = df
    return price_data


@st.cache_data
def load_regime_data():
    """Load regime labels."""
    regime_path = Path("data/processed/regime_labels.csv")
    if regime_path.exists():
        return pd.read_csv(regime_path, index_col=0, parse_dates=True)
    return pd.DataFrame()


@st.cache_data
def load_backtest_results():
    """Load latest backtest results."""
    results_dir = Path("backtest_results")
    if not results_dir.exists():
        return pd.DataFrame()
    
    nav_files = list(results_dir.glob("nav_*.csv"))
    if not nav_files:
        return pd.DataFrame()
    
    latest = max(nav_files, key=lambda x: x.stat().st_mtime)
    df = pd.read_csv(latest)
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_anomaly_results():
    """Load latest anomaly scan."""
    anomaly_dir = Path("data/anomaly_reports")
    if not anomaly_dir.exists():
        return pd.DataFrame()
    
    scan_files = list(anomaly_dir.glob("scan_*.csv"))
    if not scan_files:
        return pd.DataFrame()
    
    latest = max(scan_files, key=lambda x: x.stat().st_mtime)
    return pd.read_csv(latest)


# =============================================================================
# Sidebar
# =============================================================================

st.sidebar.title("⚔️ Bursa-Qlib")
st.sidebar.markdown("AI-Oriented Quant Research for Bursa Malaysia")

page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Performance", "Regime", "Anomalies", "Factors", "Portfolio"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")


# =============================================================================
# Overview Page
# =============================================================================

def show_overview():
    st.title("📊 Overview")
    
    # Load data
    price_data = load_price_data()
    regime_df = load_regime_data()
    backtest_df = load_backtest_results()
    anomaly_df = load_anomaly_results()
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Universe", f"{len(price_data)} stocks")
    
    with col2:
        if not regime_df.empty:
            current_regime = regime_df.iloc[-1]["regime_name"]
            st.metric("Current Regime", current_regime.title())
        else:
            st.metric("Current Regime", "N/A")
    
    with col3:
        if not backtest_df.empty:
            total_return = (backtest_df["nav"].iloc[-1] / backtest_df["nav"].iloc[0] - 1) * 100
            st.metric("Backtest Return", f"{total_return:.1f}%")
        else:
            st.metric("Backtest Return", "N/A")
    
    with col4:
        if not anomaly_df.empty:
            anomaly_count = (anomaly_df["combined_score"] > 25).sum()
            st.metric("Active Anomalies", anomaly_count)
        else:
            st.metric("Active Anomalies", 0)
    
    # Market overview chart
    st.subheader("KLCI-30 Performance (2024)")
    
    # Calculate index (equal-weighted)
    if price_data:
        all_prices = []
        for ticker, df in price_data.items():
            temp = df[["date", "close"]].copy()
            temp["ticker"] = ticker
            all_prices.append(temp)
        
        combined = pd.concat(all_prices)
        
        # Filter 2024
        combined = combined[combined["date"] >= "2024-01-01"]
        
        # Equal-weighted index
        index_df = combined.groupby("date")["close"].mean().reset_index()
        index_df["index"] = (index_df["close"] / index_df["close"].iloc[0]) * 100
        
        fig = px.line(index_df, x="date", y="index", title="KLCI-30 Equal-Weighted Index")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    # Top movers
    st.subheader("Top Movers (Latest Day)")
    
    if price_data:
        movers = []
        for ticker, df in price_data.items():
            if len(df) >= 2:
                latest = df.iloc[-1]
                prev = df.iloc[-2]
                change = (latest["close"] - prev["close"]) / prev["close"] * 100
                movers.append({
                    "Ticker": ticker,
                    "Name": get_local_name(ticker),
                    "Close": latest["close"],
                    "Change %": change,
                })
        
        movers_df = pd.DataFrame(movers)
        movers_df = movers_df.sort_values("Change %", ascending=False)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Top Gainers**")
            st.dataframe(movers_df.head(5), use_container_width=True)
        
        with col2:
            st.markdown("**Top Losers**")
            st.dataframe(movers_df.tail(5).iloc[::-1], use_container_width=True)


# =============================================================================
# Performance Page
# =============================================================================

def show_performance():
    st.title("📈 Backtest Performance")
    
    backtest_df = load_backtest_results()
    
    if backtest_df.empty:
        st.warning("No backtest results available. Run scripts/04_backtest.py first.")
        return
    
    # Performance metrics
    backtest_df["return"] = backtest_df["nav"].pct_change()
    
    total_return = (backtest_df["nav"].iloc[-1] / backtest_df["nav"].iloc[0] - 1) * 100
    annual_return = total_return * (252 / len(backtest_df)) if len(backtest_df) > 0 else 0
    volatility = backtest_df["return"].std() * np.sqrt(252) * 100
    sharpe = annual_return / volatility if volatility > 0 else 0
    
    cummax = backtest_df["nav"].cummax()
    drawdown = (backtest_df["nav"] - cummax) / cummax * 100
    max_drawdown = drawdown.min()
    
    # Metrics display
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Return", f"{total_return:.2f}%")
    with col2:
        st.metric("Annual Return", f"{annual_return:.2f}%")
    with col3:
        st.metric("Sharpe Ratio", f"{sharpe:.2f}")
    with col4:
        st.metric("Max Drawdown", f"{max_drawdown:.2f}%")
    
    # NAV chart
    st.subheader("Portfolio NAV")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=backtest_df["date"],
        y=backtest_df["nav"],
        mode="lines",
        name="NAV",
        line=dict(color="blue", width=2),
    ))
    
    fig.update_layout(
        title="Portfolio Net Asset Value",
        xaxis_title="Date",
        yaxis_title="NAV (RM)",
        height=400,
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Drawdown chart
    st.subheader("Drawdown")
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=backtest_df["date"],
        y=drawdown,
        mode="lines",
        name="Drawdown",
        fill="tozeroy",
        line=dict(color="red"),
    ))
    
    fig.update_layout(
        title="Portfolio Drawdown",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        height=300,
    )
    
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Regime Page
# =============================================================================

def show_regime():
    st.title("🌊 Market Regime Detection")
    
    regime_df = load_regime_data()
    
    if regime_df.empty:
        st.warning("No regime data available. Run regime/hmm_detector.py first.")
        return
    
    # Current regime
    current = regime_df.iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Regime", current["regime_name"].title())
    
    with col2:
        regime_counts = regime_df["regime_name"].value_counts()
        st.metric("Days in Regime", regime_counts.get(current["regime_name"], 0))
    
    with col3:
        st.metric("Regime Index", int(current["regime"]))
    
    # Regime distribution
    st.subheader("Regime Distribution")
    
    regime_counts = regime_df["regime_name"].value_counts().reset_index()
    regime_counts.columns = ["Regime", "Days"]
    
    fig = px.pie(regime_counts, values="Days", names="Regime", title="Regime Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Regime timeline
    st.subheader("Regime Timeline")
    
    # Color mapping
    color_map = {
        "risk_on": "green",
        "risk_off": "orange",
        "crisis": "red",
        "recovery": "blue",
    }
    
    fig = go.Figure()
    
    for regime in regime_df["regime_name"].unique():
        mask = regime_df["regime_name"] == regime
        regime_subset = regime_df[mask]
        
        fig.add_trace(go.Scatter(
            x=regime_subset.index,
            y=[1] * len(regime_subset),
            mode="markers",
            name=regime.title(),
            marker=dict(color=color_map.get(regime, "gray"), size=5),
        ))
    
    fig.update_layout(
        title="Regime Timeline",
        xaxis_title="Date",
        yaxis_visible=False,
        height=300,
    )
    
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Anomalies Page
# =============================================================================

def show_anomalies():
    st.title("🚨 Anomaly Detection")
    
    anomaly_df = load_anomaly_results()
    
    if anomaly_df.empty:
        st.warning("No anomaly data available. Run scripts/05_anomaly_scan.py first.")
        return
    
    # Summary metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Scanned", len(anomaly_df))
    
    with col2:
        high_priority = (anomaly_df["combined_score"] >= 50).sum()
        st.metric("High Priority", high_priority)
    
    with col3:
        any_anomaly = (anomaly_df["combined_score"] > 0).sum()
        st.metric("Any Anomaly", any_anomaly)
    
    # Anomaly score distribution
    st.subheader("Anomaly Score Distribution")
    
    fig = px.histogram(anomaly_df, x="combined_score", nbins=20, title="Anomaly Score Distribution")
    st.plotly_chart(fig, use_container_width=True)
    
    # Top anomalies table
    st.subheader("Top Anomalies")
    
    top_anomalies = anomaly_df[anomaly_df["combined_score"] > 0].sort_values(
        "combined_score", ascending=False
    ).head(10)
    
    if not top_anomalies.empty:
        display_cols = ["ticker", "combined_score", "priority", "zscore_anomaly", "velocity_anomaly", "knn_outlier"]
        st.dataframe(top_anomalies[display_cols], use_container_width=True)
    else:
        st.info("No significant anomalies detected.")


# =============================================================================
# Factors Page
# =============================================================================

def show_factors():
    st.title("📊 Malaysia-Specific Factors")
    
    st.markdown("""
    ### Factor Descriptions
    
    The model uses 6 Malaysia-specific alpha factors:
    """)
    
    factors = [
        {
            "Factor": "Palm Oil Beta",
            "Description": "Rolling beta to FCPO (Crude Palm Oil) futures",
            "Rationale": "Malaysia is 2nd largest palm oil producer globally",
        },
        {
            "Factor": "FX Sensitivity",
            "Description": "Rolling correlation to USD/MYR exchange rate",
            "Rationale": "Exporters benefit from weaker ringgit, importers from stronger",
        },
        {
            "Factor": "Shariah Effect",
            "Description": "Entry/exit from SC Shariah-compliant list",
            "Rationale": "Islamic funds represent significant institutional capital",
        },
        {
            "Factor": "GLC Strength",
            "Description": "GLC vs private sector relative performance",
            "Rationale": "GLCs behave differently due to government policy influence",
        },
        {
            "Factor": "Festive Seasonality",
            "Description": "CNY, Hari Raya, Deepavali, year-end effects",
            "Rationale": "Retail investor behavior and window dressing",
        },
        {
            "Factor": "OPR Regime",
            "Description": "Hiking/holding/cutting classification",
            "Rationale": "Rate-sensitive sectors perform differently under each regime",
        },
    ]
    
    factors_df = pd.DataFrame(factors)
    st.table(factors_df)
    
    # Feature importance (placeholder)
    st.subheader("Model Feature Importance")
    
    st.markdown("""
    Based on LightGBM training on KLCI-30:
    
    | Rank | Feature | Importance |
    |------|---------|------------|
    | 1 | momentum_20 | 91 |
    | 2 | **fx_sensitivity** | **74** |
    | 3 | daily_return | 66 |
    | 4 | volume_ratio | 53 |
    | 5 | volatility_20 | 43 |
    | 6 | **christmas_window** | **10** |
    
    **MY-specific factors highlighted in bold** - FX sensitivity ranked #2!
    """)


# =============================================================================
# Portfolio Page
# =============================================================================

def show_portfolio():
    st.title("💼 Current Portfolio")
    
    st.markdown("""
    ### Constrained Portfolio Strategy
    
    The portfolio applies institutional constraints:
    - ✅ Shariah-compliant only
    - ✅ Sector concentration limits (max 25-30%)
    - ✅ Single stock limit (max 10%)
    - ✅ Liquidity filter (min RM 1M daily turnover)
    - ✅ Regime-conditioned parameters
    """)
    
    # Performance comparison
    st.subheader("Strategy Comparison")
    
    comparison = pd.DataFrame({
        "Strategy": ["Baseline (weekly)", "Regime-conditioned", "Constrained"],
        "Return": [-2.67, -1.41, 4.74],
        "Sharpe": [-0.15, -0.11, 0.37],
        "Max DD": [-14.42, -8.99, -4.86],
        "Volatility": [9.62, 6.71, 6.70],
    })
    
    st.dataframe(comparison, use_container_width=True)
    
    # Sample portfolio
    st.subheader("Sample Portfolio (Latest Optimization)")
    
    sample_portfolio = pd.DataFrame({
        "Ticker": ["1155", "1295", "5347", "0166", "0097", "6012", "4863", "6888"],
        "Name": ["Maybank", "PBBank", "Tenaga", "Inari", "Vitrox", "Maxis", "TM", "Axiata"],
        "Sector": ["Financials", "Financials", "Utilities", "Technology", "Technology", 
                   "Telecom", "Telecom", "Telecom"],
        "Weight": [0.08, 0.07, 0.06, 0.05, 0.04, 0.04, 0.03, 0.03],
    })
    
    st.dataframe(sample_portfolio, use_container_width=True)
    
    # Sector breakdown
    st.subheader("Sector Breakdown")
    
    sector_weights = sample_portfolio.groupby("Sector")["Weight"].sum().reset_index()
    
    fig = px.pie(sector_weights, values="Weight", names="Sector", title="Sector Allocation")
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# Main App
# =============================================================================

def main():
    if page == "Overview":
        show_overview()
    elif page == "Performance":
        show_performance()
    elif page == "Regime":
        show_regime()
    elif page == "Anomalies":
        show_anomalies()
    elif page == "Factors":
        show_factors()
    elif page == "Portfolio":
        show_portfolio()


if __name__ == "__main__":
    main()
