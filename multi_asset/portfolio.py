"""
Multi-Asset Portfolio Module

Extends bursa-qlib to include bonds, money market, and other asset classes
for realistic portfolio construction and comparison vs EPF.

Asset Classes:
- Equities (bursa-qlib strategy)
- Malaysian Government Securities (MGS)
- Money Market (T-Bills, FDs)
- REITs (optional)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from alpha.factors.opr_regime import load_opr_history


class MultiAssetPortfolio:
    """
    Multi-asset portfolio with Malaysian instruments.
    """
    
    def __init__(
        self,
        equity_weight: float = 0.5,
        bond_weight: float = 0.35,
        money_market_weight: float = 0.15,
        rebalance_freq: int = 20,  # Monthly
    ):
        """
        Initialize portfolio.
        
        Args:
            equity_weight: Weight for equities (default 50%)
            bond_weight: Weight for MGS bonds (default 35%)
            money_market_weight: Weight for money market (default 15%)
            rebalance_freq: Rebalancing frequency in days
        """
        assert abs(equity_weight + bond_weight + money_market_weight - 1.0) < 0.01, \
            "Weights must sum to 1.0"
        
        self.equity_weight = equity_weight
        self.bond_weight = bond_weight
        self.money_market_weight = money_market_weight
        self.rebalance_freq = rebalance_freq
        
    def fetch_bond_returns(self, start_date: str = "2024-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
        """
        Fetch MGS (Malaysian Government Securities) returns.
        
        Uses 10-year MGS yield as proxy.
        Annual yield converted to daily returns.
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with bond returns
        """
        # Load OPR history
        opr = load_opr_history()
        
        if opr.empty:
            print("⚠️  No OPR data, using default 3%")
            dates = pd.date_range(start_date, end_date, freq='D')
            bond_returns = pd.DataFrame({
                'date': dates,
                'yield': 0.03,  # 3% annual
            })
        else:
            # Use OPR + spread for MGS yield
            # MGS typically OPR + 1-2%
            opr['date'] = pd.to_datetime(opr['date'])
            opr = opr.sort_values('date')
            
            # Create daily series
            date_range = pd.date_range(start_date, end_date, freq='D')
            bond_returns = pd.DataFrame({'date': date_range})
            
            # Merge with OPR (forward fill)
            bond_returns = bond_returns.merge(opr[['date', 'rate']], on='date', how='left')
            bond_returns['rate'] = bond_returns['rate'].fillna(method='ffill')
            
            # Add spread for MGS
            bond_returns['yield'] = (bond_returns['rate'] + 0.015) / 100  # Convert to decimal
        
        # Convert annual yield to daily return
        bond_returns['daily_return'] = bond_returns['yield'] / 252
        bond_returns['cumulative_return'] = (1 + bond_returns['daily_return']).cumprod()
        
        return bond_returns
    
    def fetch_money_market_returns(self, start_date: str = "2024-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
        """
        Fetch money market returns (T-Bills, FDs).
        
        Uses OPR rate as proxy for money market returns.
        
        Args:
            start_date: Start date
            end_date: End date
        
        Returns:
            DataFrame with money market returns
        """
        opr = load_opr_history()
        
        if opr.empty:
            print("⚠️  No OPR data, using default 2.5%")
            dates = pd.date_range(start_date, end_date, freq='D')
            mm_returns = pd.DataFrame({
                'date': dates,
                'yield': 0.025,  # 2.5% annual
            })
        else:
            opr['date'] = pd.to_datetime(opr['date'])
            opr = opr.sort_values('date')
            
            # Create daily series
            date_range = pd.date_range(start_date, end_date, freq='D')
            mm_returns = pd.DataFrame({'date': date_range})
            
            # Merge with OPR (forward fill)
            mm_returns = mm_returns.merge(opr[['date', 'rate']], on='date', how='left')
            mm_returns['rate'] = mm_returns['rate'].fillna(method='ffill')
            
            # Money market typically OPR - 0.5% (lower than policy rate)
            mm_returns['yield'] = (mm_returns['rate'] - 0.005) / 100  # Convert to decimal
        
        # Convert annual yield to daily return
        mm_returns['daily_return'] = mm_returns['yield'] / 252
        mm_returns['cumulative_return'] = (1 + mm_returns['daily_return']).cumprod()
        
        return mm_returns
    
    def combine_assets(
        self,
        equity_nav: pd.DataFrame,
        bond_returns: pd.DataFrame,
        mm_returns: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combine asset classes into multi-asset portfolio.
        
        Args:
            equity_nav: DataFrame with equity NAV (columns: date, nav)
            bond_returns: DataFrame with bond returns
            mm_returns: DataFrame with money market returns
        
        Returns:
            DataFrame with combined portfolio NAV
        """
        # Normalize dates
        equity_nav['date'] = pd.to_datetime(equity_nav['date'])
        bond_returns['date'] = pd.to_datetime(bond_returns['date'])
        mm_returns['date'] = pd.to_datetime(mm_returns['date'])
        
        # Start with equity dates
        combined = equity_nav[['date', 'nav']].copy()
        combined = combined.rename(columns={'nav': 'equity_nav'})
        
        # Normalize equity NAV to start at 1.0
        combined['equity_nav'] = combined['equity_nav'] / combined['equity_nav'].iloc[0]
        
        # Merge bond and money market
        combined = combined.merge(
            bond_returns[['date', 'cumulative_return']],
            on='date',
            how='left'
        )
        combined = combined.rename(columns={'cumulative_return': 'bond_nav_raw'})
        
        combined = combined.merge(
            mm_returns[['date', 'cumulative_return']],
            on='date',
            how='left'
        )
        combined = combined.rename(columns={'cumulative_return': 'mm_nav_raw'})
        
        # Fill forward
        combined['bond_nav_raw'] = combined['bond_nav_raw'].ffill()
        combined['mm_nav_raw'] = combined['mm_nav_raw'].ffill()
        
        # Normalize bond and MM to start at 1.0 on equity start date
        combined['bond_nav'] = combined['bond_nav_raw'] / combined['bond_nav_raw'].iloc[0]
        combined['mm_nav'] = combined['mm_nav_raw'] / combined['mm_nav_raw'].iloc[0]
        
        # Calculate weighted portfolio NAV
        combined['portfolio_nav'] = (
            self.equity_weight * combined['equity_nav'] +
            self.bond_weight * combined['bond_nav'] +
            self.money_market_weight * combined['mm_nav']
        )
        
        return combined


def get_epf_benchmark(start_date: str = "2024-01-01", end_date: str = "2025-12-31") -> pd.DataFrame:
    """
    Create EPF benchmark with 6.15% annual return (2025 rate).
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        DataFrame with EPF NAV
    """
    dates = pd.date_range(start_date, end_date, freq='D')
    daily_return = 0.0615 / 365  # EPF compounds daily
    
    epf = pd.DataFrame({'date': dates})
    epf['daily_return'] = daily_return
    epf['nav'] = (1 + epf['daily_return']).cumprod()
    
    return epf


if __name__ == "__main__":
    print("Testing multi-asset portfolio...")
    
    # Create portfolio (50/35/15 allocation)
    portfolio = MultiAssetPortfolio(
        equity_weight=0.5,
        bond_weight=0.35,
        money_market_weight=0.15,
    )
    
    # Fetch bond returns
    print("\nFetching bond returns...")
    bond_returns = portfolio.fetch_bond_returns("2024-01-01", "2025-12-31")
    print(f"Bond returns: {len(bond_returns)} days")
    print(bond_returns.head())
    
    # Fetch money market returns
    print("\nFetching money market returns...")
    mm_returns = portfolio.fetch_money_market_returns("2024-01-01", "2025-12-31")
    print(f"Money market returns: {len(mm_returns)} days")
    print(mm_returns.head())
    
    # Get EPF benchmark
    print("\nEPF benchmark...")
    epf = get_epf_benchmark("2024-01-01", "2025-12-31")
    print(f"EPF: {len(epf)} days")
    print(epf.head())
