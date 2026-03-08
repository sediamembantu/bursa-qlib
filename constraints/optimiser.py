"""
Portfolio Optimizer with Institutional Constraints

Combines all constraint modules to produce a final portfolio:
1. Shariah filter
2. Sector concentration limits
3. Liquidity constraints
4. Position size limits

Optimization objective: Maximize expected return subject to constraints.
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import (
    MAX_STOCK_WEIGHT,
    MAX_SECTOR_WEIGHT,
    MIN_DAILY_TURNOVER,
    RANDOM_SEED,
)
from constraints.shariah_filter import (
    is_shariah_compliant,
    apply_shariah_filter,
    enforce_shariah_portfolio,
)
from constraints.sector_caps import (
    enforce_sector_limits,
    enforce_stock_limits,
    calculate_sector_weights,
    check_diversification,
)
from constraints.liquidity_threshold import (
    screen_for_liquidity,
    adjust_for_liquidity,
)


# =============================================================================
# Portfolio Optimizer
# =============================================================================

class ConstrainedPortfolioOptimizer:
    """
    Portfolio optimizer with institutional constraints.
    
    Steps:
    1. Filter to Shariah-compliant (optional)
    2. Filter to liquid stocks
    3. Rank by prediction
    4. Apply sector limits
    5. Apply position limits
    6. Ensure diversification
    """
    
    def __init__(
        self,
        shariah_compliant_only: bool = True,
        max_stock_weight: float = MAX_STOCK_WEIGHT,
        max_sector_weight: float = MAX_SECTOR_WEIGHT,
        min_turnover: float = MIN_DAILY_TURNOVER,
        min_stocks: int = 10,
        top_n_candidates: int = 30,
    ):
        """
        Initialize optimizer.
        
        Args:
            shariah_compliant_only: Enforce Shariah compliance
            max_stock_weight: Maximum single stock weight
            max_sector_weight: Maximum sector weight
            min_turnover: Minimum daily turnover (RM millions)
            min_stocks: Minimum number of stocks
            top_n_candidates: Number of candidates to consider
        """
        self.shariah_compliant_only = shariah_compliant_only
        self.max_stock_weight = max_stock_weight
        self.max_sector_weight = max_sector_weight
        self.min_turnover = min_turnover
        self.min_stocks = min_stocks
        self.top_n = top_n_candidates
        
        # Cache
        self.shariah_list = None
        self.sector_mapping = None
    
    def optimize(
        self,
        predictions: Dict[str, float],
        price_data: Dict[str, pd.DataFrame],
        portfolio_value: float = 1_000_000,
        verbose: bool = False,
    ) -> Dict[str, float]:
        if verbose:
            print(f"Starting optimization with {len(predictions)} candidates")
        
        # Step 1: Shariah filter
        if self.shariah_compliant_only:
            candidates = apply_shariah_filter(list(predictions.keys()), self.shariah_list)
            if verbose:
                print(f"After Shariah filter: {len(candidates)}")
        else:
            candidates = list(predictions.keys())
        
        # Step 2: Liquidity filter
        liquid, illiquid_reasons = screen_for_liquidity(
            candidates,
            price_data,
            self.min_turnover,
        )
        if verbose:
            print(f"After liquidity filter: {len(liquid)}")
        
        # Step 3: Rank by prediction
        ranked = sorted(
            [(t, predictions[t]) for t in liquid],
            key=lambda x: x[1],
            reverse=True,
        )[:self.top_n]
        
        if not ranked:
            return {}
        
        # Step 4: Initial equal weights
        weights = {t: 1.0 / len(ranked) for t, _ in ranked}
        
        # Step 5: Adjust for liquidity
        weights = adjust_for_liquidity(weights, price_data, portfolio_value)
        
        # Step 6: Apply sector limits
        weights = enforce_sector_limits(weights, self.sector_mapping)
        
        # Step 7: Apply position limits
        weights = enforce_stock_limits(weights, self.max_stock_weight)
        
        # Step 8: Check diversification
        is_diversified, num_positions = check_diversification(weights, self.min_stocks)
        
        if not is_diversified and verbose:
            print(f"Warning: Only {num_positions} positions (min: {self.min_stocks})")
        
        # Step 9: Final normalization
        total = sum(weights.values())
        if total > 0:
            weights = {t: w / total for t, w in weights.items()}
        
        if verbose:
            print(f"Final portfolio: {len(weights)} stocks")
            sector_weights = calculate_sector_weights(weights, self.sector_mapping)
            print("Sector breakdown:")
            for sector, w in sorted(sector_weights.items(), key=lambda x: -x[1]):
                print(f"  {sector}: {w:.1%}")
        
        return weights
    
    def get_portfolio_stats(
        self,
        weights: Dict[str, float],
    ) -> Dict:
        """
        Get portfolio statistics.
        
        Args:
            weights: Portfolio weights
        
        Returns:
            Dictionary of statistics
        """
        is_diversified, num_positions = check_diversification(weights, self.min_stocks)
        sector_weights = calculate_sector_weights(weights, self.sector_mapping)
        
        return {
            "num_positions": num_positions,
            "is_diversified": is_diversified,
            "max_weight": max(weights.values()) if weights else 0,
            "num_sectors": len(sector_weights),
            "sector_concentration": max(sector_weights.values()) if sector_weights else 0,
            "shariah_compliant": self.shariah_compliant_only,
        }


# =============================================================================
# Convenience Functions
# =============================================================================

def optimize_portfolio(
    predictions: Dict[str, float],
    price_data: Dict[str, pd.DataFrame],
    shariah_compliant: bool = True,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Optimize portfolio with default institutional constraints.
    
    Args:
        predictions: Ticker -> prediction score
        price_data: Ticker -> price DataFrame
        shariah_compliant: Enforce Shariah compliance
        verbose: Print progress
    
    Returns:
        Dictionary of ticker -> weight
    """
    optimizer = ConstrainedPortfolioOptimizer(
        shariah_compliant_only=shariah_compliant,
    )
    
    return optimizer.optimize(predictions, price_data, verbose=verbose)


if __name__ == "__main__":
    print("Constrained Portfolio Optimizer Test")
    print("=" * 50)
    
    # Load test data
    from pathlib import Path
    import pandas as pd
    import lightgbm as lgb
    
    # Load pre-computed features
    df = pd.read_parquet("data/processed/backtest_features.parquet")
    
    # Add derived features
    df['daily_return'] = df.groupby('ticker')['close'].pct_change()
    df['volatility_20'] = df.groupby('ticker')['daily_return'].transform(lambda x: x.rolling(20).std())
    df['momentum_20'] = df.groupby('ticker')['close'].transform(lambda x: x / x.shift(20) - 1)
    df['volume_ratio'] = df.groupby('ticker')['volume'].transform(lambda x: x / x.rolling(20).mean())
    
    # Load model
    model = lgb.Booster(model_file="models/lightgbm_20260308_0816.txt")
    
    feature_cols = [
        'open', 'high', 'low', 'close', 'volume',
        'daily_return', 'volatility_20', 'momentum_20', 'volume_ratio',
        'palm_oil_beta', 'fx_sensitivity',
        'shariah_compliant', 'shariah_event',
        'glc_flag', 'glc_spread',
        'cny_window', 'hari_raya_window', 'deepavali_window',
        'christmas_window', 'year_end', 'any_festive',
        'opr_rate', 'opr_regime', 'opr_hiking', 'opr_cutting', 'opr_holding',
    ]
    
    df['prediction'] = model.predict(df[feature_cols].fillna(0))
    
    # Get latest predictions
    latest_date = df['date'].max()
    latest = df[df['date'] == latest_date]
    predictions = dict(zip(latest['ticker'], latest['prediction']))
    
    # Load price data
    price_dir = Path("data/raw/prices")
    price_data = {}
    for csv_file in price_dir.glob("*.csv"):
        ticker = csv_file.stem
        pdf = pd.read_csv(csv_file)
        pdf['date'] = pd.to_datetime(pdf['date'])
        price_data[ticker] = pdf
    
    # Optimize
    print(f"\nOptimizing portfolio as of {latest_date}")
    print(f"Candidates: {len(predictions)}")
    
    optimizer = ConstrainedPortfolioOptimizer(
        shariah_compliant_only=True,
    )
    
    weights = optimizer.optimize(predictions, price_data, verbose=True)
    
    print("\n" + "=" * 50)
    print("FINAL PORTFOLIO")
    print("=" * 50)
    
    for ticker, weight in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {ticker}: {weight:.2%}")
    
    stats = optimizer.get_portfolio_stats(weights)
    print(f"\nPortfolio Stats:")
    print(f"  Positions: {stats['num_positions']}")
    print(f"  Diversified: {stats['is_diversified']}")
    print(f"  Max weight: {stats['max_weight']:.2%}")
    print(f"  Sectors: {stats['num_sectors']}")
    print(f"  Max sector: {stats['sector_concentration']:.1%}")
