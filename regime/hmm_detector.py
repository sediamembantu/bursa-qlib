"""
Macro Regime Detection using Hidden Markov Models.

Detects market regimes from BNM/DOSM macroeconomic data:
- Risk-on: Low volatility, growth, bullish
- Risk-off: High uncertainty, defensive positioning
- Crisis: Extreme stress, sharp corrections
- Recovery: Post-crisis normalization

Uses HMM to identify latent states from observable macro features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MACRO_DIR,
    ECONOMIC_DIR,
    RANDOM_SEED,
)
from data.fetch.bnm_openapi import fetch_all_exchange_rates, fetch_opr_history
from data.fetch.opendosm import fetch_gdp, fetch_ipi


# =============================================================================
# Feature Engineering
# =============================================================================

def build_macro_feature_matrix(
    start_date: str = "2010-01-01",
    end_date: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Build feature matrix from macro data sources.
    
    Features:
    - OPR level and changes
    - USD/MYR exchange rate
    - GDP growth rate
    - IPI (Industrial Production Index)
    - Market volatility proxy
    
    Args:
        start_date: Start date
        end_date: End date
        use_cache: Use cached data if available
    
    Returns:
        DataFrame with date index and macro features
    """
    features = []
    
    # 1. OPR data (use historical from opr_regime.py)
    print("Loading OPR data...")
    from alpha.factors.opr_regime import load_opr_history
    opr_df = load_opr_history()
    if not opr_df.empty:
        # Expand to daily by forward-filling
        opr_daily = opr_df.set_index("date")[["rate"]].rename(columns={"rate": "opr"})
        # Create daily date range
        date_range = pd.date_range(opr_daily.index.min(), opr_daily.index.max(), freq="D")
        opr_daily = opr_daily.reindex(date_range, method="ffill")
        features.append(opr_daily)
        print(f"  Loaded {len(opr_daily)} days of OPR data")
    
    # 2. Exchange rate data (use Yahoo Finance for historical)
    print("Loading FX data...")
    try:
        import yfinance as yf
        fx_df = yf.download("MYRUSD=X", start="2015-01-01", progress=False)
        if not fx_df.empty:
            if isinstance(fx_df.columns, pd.MultiIndex):
                fx_df.columns = fx_df.columns.get_level_values(0)
            fx_df = fx_df.reset_index()
            fx_df = fx_df.rename(columns={"Date": "date", "Close": "usdmyr"})
            fx_df = fx_df[["date", "usdmyr"]].set_index("date")
            features.append(fx_df)
            print(f"  Loaded {len(fx_df)} days of FX data")
    except Exception as e:
        print(f"  Warning: Could not load FX data: {e}")
    
    # 3. IPI data
    print("Loading IPI data...")
    ipi_path = ECONOMIC_DIR / "ipi_ipi.parquet"
    if ipi_path.exists():
        import pyarrow.parquet as pq
        ipi_df = pq.read_table(ipi_path).to_pandas()
        if "date" in ipi_df.columns and "index" in ipi_df.columns:
            ipi_df["date"] = pd.to_datetime(ipi_df["date"])
            # Remove duplicates, keep last
            ipi_df = ipi_df.drop_duplicates(subset="date", keep="last")
            ipi_monthly = ipi_df[["date", "index"]].rename(columns={"index": "ipi"})
            ipi_monthly = ipi_monthly.set_index("date")
            features.append(ipi_monthly)
    
    # 4. GDP data
    print("Loading GDP data...")
    gdp_path = ECONOMIC_DIR / "gdp_gdp.parquet"
    if gdp_path.exists():
        import pyarrow.parquet as pq
        gdp_df = pq.read_table(gdp_path).to_pandas()
        # Find growth rate column
        if "date" in gdp_df.columns:
            gdp_df["date"] = pd.to_datetime(gdp_df["date"])
            # Remove duplicates
            gdp_df = gdp_df.drop_duplicates(subset="date", keep="last")
            # Look for growth or real GDP column
            growth_col = None
            for col in gdp_df.columns:
                if "growth" in col.lower() or "real" in col.lower():
                    growth_col = col
                    break
            if growth_col:
                gdp_monthly = gdp_df[["date", growth_col]].rename(columns={growth_col: "gdp_growth"})
                gdp_monthly = gdp_monthly.set_index("date")
                features.append(gdp_monthly)
    
    if not features:
        print("Warning: No macro data available")
        return pd.DataFrame()
    
    # Combine all features
    print("Combining features...")
    combined = pd.concat(features, axis=1, join="outer")
    
    # Fill missing values
    combined = combined.ffill().bfill()
    
    # Filter date range
    combined = combined.loc[start_date:]
    if end_date:
        combined = combined.loc[:end_date]
    
    # Add derived features
    if "opr" in combined.columns:
        combined["opr_change"] = combined["opr"].diff()
        combined["opr_ma_30"] = combined["opr"].rolling(30).mean()
    
    if "usdmyr" in combined.columns:
        combined["usdmyr_return"] = combined["usdmyr"].pct_change()
        combined["usdmyr_vol_30"] = combined["usdmyr_return"].rolling(30).std()
    
    if "ipi" in combined.columns:
        combined["ipi_yoy"] = combined["ipi"].pct_change(252)  # Year-over-year
    
    # Drop NaN rows
    combined = combined.dropna()
    
    print(f"Built feature matrix: {len(combined)} rows, {len(combined.columns)} features")
    
    return combined


# =============================================================================
# HMM Regime Detection
# =============================================================================

def train_hmm_regime(
    features: pd.DataFrame,
    n_regimes: int = 4,
    feature_cols: Optional[list] = None,
) -> Tuple[object, np.ndarray]:
    """
    Train HMM to detect market regimes.
    
    Args:
        features: DataFrame with macro features
        n_regimes: Number of regimes (states)
        feature_cols: Columns to use for HMM
    
    Returns:
        Tuple of (trained model, regime labels)
    """
    from hmmlearn import hmm
    
    if feature_cols is None:
        # Default features for regime detection
        feature_cols = [
            "opr_change",
            "usdmyr_return",
            "usdmyr_vol_30",
        ]
        # Only use available columns
        feature_cols = [c for c in feature_cols if c in features.columns]
    
    if not feature_cols:
        raise ValueError("No valid features for HMM")
    
    print(f"Training HMM with {n_regimes} regimes using: {feature_cols}")
    
    # Prepare data
    X = features[feature_cols].values
    
    # Normalize
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Train HMM
    model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=100,
        random_state=RANDOM_SEED,
    )
    
    model.fit(X_scaled)
    
    # Predict regimes
    regimes = model.predict(X_scaled)
    
    # Calculate regime statistics
    print("\nRegime Statistics:")
    for i in range(n_regimes):
        mask = regimes == i
        count = mask.sum()
        pct = count / len(regimes) * 100
        print(f"  Regime {i}: {count} days ({pct:.1f}%)")
    
    return model, regimes, scaler


def label_regimes(
    features: pd.DataFrame,
    regimes: np.ndarray,
    regime_names: Optional[dict] = None,
) -> pd.DataFrame:
    """
    Label regimes with descriptive names based on characteristics.
    
    Args:
        features: Feature DataFrame
        regimes: Regime labels from HMM
        regime_names: Mapping of regime index to name
    
    Returns:
        DataFrame with regime column added
    """
    df = features.copy()
    df["regime"] = regimes
    
    # Analyze each regime to assign names
    if regime_names is None:
        regime_names = {}
        
        # Calculate average characteristics per regime
        for i in range(regimes.max() + 1):
            mask = df["regime"] == i
            
            # Determine regime type based on features
            avg_vol = df.loc[mask, "usdmyr_vol_30"].mean() if "usdmyr_vol_30" in df.columns else 0
            avg_fx_ret = df.loc[mask, "usdmyr_return"].mean() if "usdmyr_return" in df.columns else 0
            avg_opr_chg = df.loc[mask, "opr_change"].mean() if "opr_change" in df.columns else 0
            
            # Simple classification based on available features
            if "usdmyr_vol_30" in df.columns and avg_vol > df["usdmyr_vol_30"].quantile(0.75):
                name = "crisis"
            elif "usdmyr_vol_30" in df.columns and avg_vol > df["usdmyr_vol_30"].quantile(0.5):
                name = "risk_off"
            elif "opr_change" in df.columns and avg_opr_chg < 0:
                name = "recovery"
            else:
                name = "risk_on"
            
            regime_names[i] = name
            print(f"  Regime {i} -> {name} (vol={avg_vol:.6f}, fx_ret={avg_fx_ret:.6f}, opr_chg={avg_opr_chg:.4f})")
    
    # Map regime numbers to names
    df["regime_name"] = df["regime"].map(regime_names)
    
    return df, regime_names


def get_regime_for_date(
    date: pd.Timestamp,
    regime_df: pd.DataFrame,
) -> Tuple[int, str]:
    """
    Get regime for a specific date.
    
    Args:
        date: Target date
        regime_df: DataFrame with regime column
    
    Returns:
        Tuple of (regime index, regime name)
    """
    # Find most recent regime
    mask = regime_df.index <= date
    if not mask.any():
        return 0, "unknown"
    
    row = regime_df.loc[mask].iloc[-1]
    regime = int(row["regime"])
    name = row.get("regime_name", f"regime_{regime}")
    
    return regime, name


# =============================================================================
# Main Pipeline
# =============================================================================

def run_regime_detection(
    n_regimes: int = 4,
    save_output: bool = True,
) -> pd.DataFrame:
    """
    Run full regime detection pipeline.
    
    Args:
        n_regimes: Number of regimes
        save_output: Save results to file
    
    Returns:
        DataFrame with regime labels
    """
    print("=" * 60)
    print("MACRO REGIME DETECTION")
    print("=" * 60)
    
    # Build features
    features = build_macro_feature_matrix()
    
    if features.empty:
        print("Error: No features available")
        return pd.DataFrame()
    
    # Train HMM
    model, regimes, scaler = train_hmm_regime(features, n_regimes=n_regimes)
    
    # Label regimes
    regime_df, regime_names = label_regimes(features, regimes)
    
    # Save
    if save_output:
        output_dir = Path(__file__).parent.parent / "data" / "processed"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / "regime_labels.csv"
        regime_df[["regime", "regime_name"]].to_csv(output_path)
        print(f"\nSaved regime labels to: {output_path}")
    
    print("\nDone!")
    
    return regime_df


if __name__ == "__main__":
    regime_df = run_regime_detection()
    
    if not regime_df.empty:
        print("\n" + "=" * 60)
        print("REGIME SUMMARY")
        print("=" * 60)
        print(regime_df.groupby("regime_name").size())
