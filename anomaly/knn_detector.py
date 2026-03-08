"""
KNN Anomaly Detection

Detects outliers using K-Nearest Neighbors approach.

Method:
1. Compute feature vector for each stock
2. Find K nearest neighbors
3. Outlier score = average distance to K neighbors

Features used:
- Price momentum
- Volume characteristics
- Volatility
- Correlation to market
"""

import pandas as pd
import numpy as np
from typing import Optional, List, Dict, Tuple
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RANDOM_SEED


# =============================================================================
# Feature Engineering for KNN
# =============================================================================

def extract_knn_features(
    price_df: pd.DataFrame,
    window: int = 20,
) -> pd.Series:
    """
    Extract features for KNN anomaly detection.
    
    Args:
        price_df: DataFrame with OHLCV data
        window: Feature calculation window
    
    Returns:
        Series of features
    """
    df = price_df.tail(window * 2).copy()  # Need extra history
    
    if len(df) < window:
        return pd.Series()
    
    # Returns
    df["return"] = df["close"].pct_change()
    
    features = {}
    
    # 1. Return statistics
    features["return_mean"] = df["return"].tail(window).mean()
    features["return_std"] = df["return"].tail(window).std()
    features["return_skew"] = df["return"].tail(window).skew()
    
    # 2. Volatility
    features["volatility"] = df["return"].tail(window).std() * np.sqrt(252)
    features["volatility_ratio"] = (
        df["return"].tail(window // 2).std() / 
        df["return"].tail(window).std()
    ) if df["return"].tail(window).std() > 0 else 1.0
    
    # 3. Volume characteristics
    features["volume_mean"] = df["volume"].tail(window).mean()
    features["volume_std"] = df["volume"].tail(window).std()
    features["volume_trend"] = (
        df["volume"].tail(window // 2).mean() / 
        df["volume"].tail(window).mean()
    ) if df["volume"].tail(window).mean() > 0 else 1.0
    
    # 4. Price levels
    features["price_momentum"] = df["close"].iloc[-1] / df["close"].iloc[-window] - 1
    features["price_position"] = (
        (df["close"].iloc[-1] - df["low"].tail(window).min()) /
        (df["high"].tail(window).max() - df["low"].tail(window).min())
    ) if df["high"].tail(window).max() > df["low"].tail(window).min() else 0.5
    
    # 5. Range characteristics
    features["avg_range"] = ((df["high"] - df["low"]) / df["close"]).tail(window).mean()
    features["range_trend"] = (
        ((df["high"] - df["low"]) / df["close"]).tail(window // 2).mean() /
        ((df["high"] - df["low"]) / df["close"]).tail(window).mean()
    )
    
    return pd.Series(features)


def build_feature_matrix(
    price_data: Dict[str, pd.DataFrame],
    window: int = 20,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build feature matrix for all stocks.
    
    Args:
        price_data: Dictionary of ticker -> price DataFrame
        window: Feature calculation window
    
    Returns:
        Tuple of (feature DataFrame, list of valid tickers)
    """
    features_list = []
    valid_tickers = []
    
    for ticker, df in price_data.items():
        features = extract_knn_features(df, window)
        
        if len(features) > 0 and not features.isna().any():
            features_list.append(features)
            valid_tickers.append(ticker)
    
    if not features_list:
        return pd.DataFrame(), []
    
    feature_df = pd.DataFrame(features_list, index=valid_tickers)
    
    return feature_df, valid_tickers


# =============================================================================
# KNN Outlier Detection
# =============================================================================

def detect_knn_outliers(
    feature_df: pd.DataFrame,
    k: int = 5,
    contamination: float = 0.1,
) -> pd.DataFrame:
    """
    Detect outliers using KNN distance.
    
    Args:
        feature_df: Feature matrix (stocks x features)
        k: Number of neighbors
        contamination: Expected proportion of outliers
    
    Returns:
        DataFrame with outlier scores and labels
    """
    if feature_df.empty:
        return pd.DataFrame()
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(feature_df)
    
    # Fit KNN
    knn = NearestNeighbors(n_neighbors=k + 1, metric='euclidean')
    knn.fit(X)
    
    # Get distances to k nearest neighbors
    distances, indices = knn.kneighbors(X)
    
    # Outlier score = average distance to k neighbors (excluding self)
    outlier_scores = distances[:, 1:].mean(axis=1)
    
    # Determine threshold
    threshold = np.percentile(outlier_scores, (1 - contamination) * 100)
    
    # Create result DataFrame
    result = pd.DataFrame({
        "ticker": feature_df.index,
        "outlier_score": outlier_scores,
        "is_outlier": (outlier_scores > threshold).astype(int),
    })
    
    result = result.set_index("ticker")
    
    # Add feature columns for reference
    for col in feature_df.columns:
        result[col] = feature_df[col]
    
    return result.sort_values("outlier_score", ascending=False)


# =============================================================================
# Cross-Sectional Anomaly Detection
# =============================================================================

def detect_cross_sectional_anomalies(
    price_data: Dict[str, pd.DataFrame],
    window: int = 20,
    k: int = 5,
    contamination: float = 0.1,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Detect cross-sectional anomalies across universe.
    
    Identifies stocks behaving differently from peers.
    
    Args:
        price_data: Dictionary of ticker -> price DataFrame
        window: Feature calculation window
        k: KNN neighbors
        contamination: Expected outlier proportion
    
    Returns:
        Tuple of (results DataFrame, list of outlier tickers)
    """
    # Build feature matrix
    feature_df, valid_tickers = build_feature_matrix(price_data, window)
    
    if feature_df.empty:
        return pd.DataFrame(), []
    
    # Detect outliers
    outliers_df = detect_knn_outliers(feature_df, k, contamination)
    
    # Get outlier tickers
    outlier_tickers = outliers_df[outliers_df["is_outlier"] == 1].index.tolist()
    
    return outliers_df, outlier_tickers


# =============================================================================
# Anomaly Interpretation
# =============================================================================

def interpret_anomaly(
    ticker: str,
    features: pd.Series,
    feature_df: pd.DataFrame,
) -> str:
    """
    Interpret why a stock is flagged as anomalous.
    
    Compares stock's features to universe median.
    
    Args:
        ticker: Stock code
        features: Stock's feature values
        feature_df: Full feature matrix
    
    Returns:
        Interpretation string
    """
    interpretations = []
    
    # Compare to median
    for feature in features.index:
        stock_val = features[feature]
        median_val = feature_df[feature].median()
        
        # Check if significantly different
        if median_val != 0:
            diff_pct = (stock_val - median_val) / abs(median_val)
            
            if abs(diff_pct) > 1.0:  # More than 100% different
                direction = "high" if diff_pct > 0 else "low"
                interpretations.append(f"{feature} {direction} ({diff_pct:.0%})")
    
    if not interpretations:
        return "No clear pattern"
    
    return "; ".join(interpretations[:3])  # Top 3


if __name__ == "__main__":
    print("KNN Anomaly Detection Test")
    print("=" * 50)
    
    # Load test data
    from pathlib import Path
    
    price_dir = Path("data/raw/prices")
    price_data = {}
    
    for csv_file in price_dir.glob("*.csv"):
        ticker = csv_file.stem
        df = pd.read_csv(csv_file)
        df["date"] = pd.to_datetime(df["date"])
        price_data[ticker] = df
    
    print(f"\nLoaded {len(price_data)} stocks")
    
    # Detect cross-sectional anomalies
    results, outliers = detect_cross_sectional_anomalies(
        price_data,
        window=20,
        k=5,
        contamination=0.15,  # 15% expected outliers
    )
    
    print(f"\nDetected {len(outliers)} outliers ({len(outliers)/len(price_data)*100:.1f}%)")
    
    if not results.empty:
        print("\nTop Outliers:")
        top_outliers = results[results["is_outlier"] == 1].head(5)
        
        for ticker, row in top_outliers.iterrows():
            print(f"  {ticker}: score={row['outlier_score']:.3f}")
            print(f"    return_mean={row['return_mean']:.4f}, volatility={row['volatility']:.2%}")
    
    # Build feature matrix for interpretation
    feature_df, _ = build_feature_matrix(price_data, window=20)
    
    if not feature_df.empty and outliers:
        print("\nInterpretation for top outlier:")
        ticker = outliers[0]
        features = feature_df.loc[ticker]
        interpretation = interpret_anomaly(ticker, features, feature_df)
        print(f"  {ticker}: {interpretation}")
