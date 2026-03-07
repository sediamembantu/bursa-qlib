"""
Centralised configuration for bursa-qlib.
All paths, API endpoints, and settings defined here.
"""

from pathlib import Path

# =============================================================================
# Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
QLIB_DIR = DATA_DIR / "qlib"
REFERENCE_DIR = PROJECT_ROOT / "reference"
LOGS_DIR = PROJECT_ROOT / "logs"

# Raw data subdirectories
PRICES_DIR = RAW_DIR / "prices"
MACRO_DIR = RAW_DIR / "macro"
ECONOMIC_DIR = RAW_DIR / "economic"

# =============================================================================
# Universe Configuration
# =============================================================================

# Primary universe: KLCI-30 constituents
KLCI_UNIVERSE = "klci30"

# Extended universe: up to 100 Main Market stocks
EXTENDED_UNIVERSE = "main_market"

# =============================================================================
# Date Ranges
# =============================================================================

# Historical data start date
START_DATE = "2010-01-01"

# Default training/validation split
TRAIN_END_DATE = "2022-12-31"
VALID_START_DATE = "2023-01-01"

# =============================================================================
# API Endpoints
# =============================================================================

# BNM OpenAPI base URL
BNM_API_BASE = "https://api.bnm.gov.my/public"

# BNM API endpoints
BNM_ENDPOINTS = {
    "opr": "/opr",
    "exchange_rate": "/exchange-rate",
    "interbank": "/interbank-swap",
    "klibor": "/klibor-rate",
}

# OpenDOSM base URL (parquet files)
OPENDOSM_BASE = "https://storage.dosm.gov.my"

# OpenDOSM datasets
OPENDOSM_DATASETS = {
    "gdp": "gdp/gdp_monthly.parquet",
    "cpi": "cpi/cpi.parquet",
    "ipi": "ipi/ipi.parquet",
    "trade": "trade/trade_msic.parquet",
    "labour": "labour/labourforce.parquet",
}

# =============================================================================
# Yahoo Finance Configuration
# =============================================================================

# Malaysia ticker suffix
YF_SUFFIX = ".KL"

# Rate limiting (requests per batch)
YF_BATCH_SIZE = 10
YF_SLEEP_SECONDS = 1

# =============================================================================
# Bursa Market Parameters
# =============================================================================

# Minimum lot size
LOT_SIZE = 100

# Daily price limit (%)
DAILY_PRICE_LIMIT = 0.30

# Default transaction costs
COMMISSION_RATE = 0.001  # 0.1%
STAMP_DUTY_RATE = 0.001  # 0.1% (capped)
CLEARING_FEE_RATE = 0.0003  # 0.03%

# =============================================================================
# Qlib Configuration
# =============================================================================

# Qlib data provider
QLIB_PROVIDER = "LocalProvider"

# Feature frequency
QLIB_FREQ = "day"

# =============================================================================
# Model Configuration
# =============================================================================

# Random seed for reproducibility
RANDOM_SEED = 42

# Default model parameters
MODEL_CONFIGS = {
    "lightgbm": {
        "num_leaves": 31,
        "learning_rate": 0.05,
        "n_estimators": 200,
        "max_depth": 6,
    },
    "transformer": {
        "d_model": 64,
        "nhead": 4,
        "num_layers": 2,
        "dropout": 0.1,
    },
    "alstm": {
        "hidden_size": 64,
        "num_layers": 2,
        "dropout": 0.1,
    },
}

# =============================================================================
# Institutional Constraints
# =============================================================================

# Maximum sector concentration
MAX_SECTOR_WEIGHT = 0.25

# Maximum single stock weight
MAX_STOCK_WEIGHT = 0.10

# Minimum daily turnover (RM millions)
MIN_DAILY_TURNOVER = 1.0

# Minimum dividend yield for income strategy
MIN_DIVIDEND_YIELD = 0.02

# =============================================================================
# Logging Configuration
# =============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
