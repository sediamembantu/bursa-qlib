"""
OpenDOSM data fetcher for Malaysian economic statistics.

Datasets:
- GDP (Gross Domestic Product)
- CPI (Consumer Price Index)
- IPI (Industrial Production Index)
- Trade (Exports, Imports)
- Labour Force
"""

from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq
import requests

from config import (
    ECONOMIC_DIR,
    OPENDOSM_BASE,
    OPENDOSM_DATASETS,
)


def fetch_parquet_dataset(
    dataset_name: str,
    output_dir: Path = ECONOMIC_DIR,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch a parquet dataset from OpenDOSM.
    
    Args:
        dataset_name: Key from OPENDOSM_DATASETS
        output_dir: Directory to save downloaded file
        cache: If True, use cached file if available
    
    Returns:
        DataFrame with the dataset
    """
    if dataset_name not in OPENDOSM_DATASETS:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    relative_path = OPENDOSM_DATASETS[dataset_name]
    url = f"{OPENDOSM_BASE}/{relative_path}"
    
    # Create output path
    filename = relative_path.replace("/", "_")
    output_path = output_dir / filename
    
    # Check cache
    if cache and output_path.exists():
        print(f"Loading cached {dataset_name} from {output_path}")
        table = pq.read_table(output_path)
        return table.to_pandas()
    
    print(f"Fetching {dataset_name} from OpenDOSM...")
    print(f"  URL: {url}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        # Save to file
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_path, "wb") as f:
            f.write(response.content)
        
        print(f"  Saved to {output_path}")
        
        # Read parquet
        table = pq.read_table(output_path)
        df = table.to_pandas()
        
        print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
        return df
        
    except Exception as e:
        print(f"  Error fetching {dataset_name}: {e}")
        return pd.DataFrame()


def fetch_gdp(cache: bool = True) -> pd.DataFrame:
    """Fetch GDP data."""
    return fetch_parquet_dataset("gdp", cache=cache)


def fetch_cpi(cache: bool = True) -> pd.DataFrame:
    """Fetch CPI data."""
    return fetch_parquet_dataset("cpi", cache=cache)


def fetch_ipi(cache: bool = True) -> pd.DataFrame:
    """Fetch IPI data."""
    return fetch_parquet_dataset("ipi", cache=cache)


def fetch_trade(cache: bool = True) -> pd.DataFrame:
    """Fetch trade data."""
    return fetch_parquet_dataset("trade", cache=cache)


def fetch_labour(cache: bool = True) -> pd.DataFrame:
    """Fetch labour force data."""
    return fetch_parquet_dataset("labour", cache=cache)


def fetch_all_economic_data(cache: bool = True) -> dict[str, pd.DataFrame]:
    """
    Fetch all economic datasets.
    
    Returns:
        Dictionary mapping dataset name to DataFrame
    """
    datasets = {}
    
    for name in OPENDOSM_DATASETS.keys():
        df = fetch_parquet_dataset(name, cache=cache)
        if not df.empty:
            datasets[name] = df
    
    return datasets


def main():
    """Fetch all OpenDOSM economic data."""
    print("=" * 60)
    print("Fetching OpenDOSM Economic Data")
    print("=" * 60)
    
    datasets = fetch_all_economic_data()
    
    print(f"\nFetched {len(datasets)} datasets:")
    for name, df in datasets.items():
        print(f"  - {name}: {len(df)} rows")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
