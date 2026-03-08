"""
Ticker definitions for Bursa Malaysia universe.
KLCI-30 constituents and extended Main Market stocks.

Yahoo Finance uses stock codes (e.g., 1155.KL) not ticker names.
"""

from config import YF_SUFFIX

# =============================================================================
# KLCI-30 Constituents (as of 2026)
# Format: {YAHOO_CODE: LOCAL_NAME}
# =============================================================================

KLCI30_CODES = {
    # Financials
    "1155": "MAYBANK",
    "1295": "PBBANK",
    "1023": "CIMB",
    "1066": "RHBBANK",
    "5398": "AMBANK",
    "5819": "HLBANK",
    "1082": "HLFG",
    
    # Telecommunications
    "6012": "MAXIS",
    "6947": "DIGI",
    "4863": "TM",
    "6888": "AXIATA",
    
    # Plantations
    "1961": "IOICORP",
    "5285": "SIME",
    "2445": "KLK",
    
    # Consumer
    "4707": "NESTLE",
    "1817": "DLADY",
    "3689": "F&N",
    
    # Utilities & Energy
    "5347": "TENAGA",
    "7277": "DIALOG",
    "6033": "PETGAS",
    "5183": "PCHEM",
    
    # Industrial & Gaming
    "3182": "GENTING",
    "4715": "GENM",
    
    # Conglomerates
    "4588": "UMW",
    
    # Technology
    "0166": "INARI",
    "0097": "VITROX",
    
    # Healthcare
    "6683": "IHH",
    
    # Construction
    "5398": "GAMUDA",
}

# List of Yahoo codes for KLCI-30
KLCI30 = list(KLCI30_CODES.keys())


# =============================================================================
# Extended Universe (Liquid Main Market Stocks)
# =============================================================================

EXTENDED_CODES = {
    # Additional financials
    "5099": "AFFIN",
    "6888": "LPI",
    
    # Plantations
    "2125": "FIMACORP",
    
    # Property
    "5204": "SPSETIA",
    "3087": "MAHSING",
    "1773": "UEMS",
    "8206": "ECOWLD",
    
    # Construction
    "3336": "IJM",
    "5398": "SUNCON",
    
    # Industrial - Gloves
    "5168": "HARTA",
    "7113": "TOPGLOV",
    "7153": "KOSSAN",
    
    # Technology
    "7106": "PENTA",
    "0084": "NOTION",
    "5006": "UNISEM",
    "3872": "MPI",
    "0183": "GREATEC",
    
    # Consumer
    "7081": "QL",
    "3222": "PADINI",
    "5284": "MRDIY",
    
    # Energy
    "5681": "PETDAG",
    "3043": "PETRONM",
    
    # Telecommunications
    "5113": "TIMECOM",
    
    # Tobacco & Beverage
    "4162": "BAT",
    "2629": "HEINEKEN",
    "2836": "CARLSBG",
}


# =============================================================================
# GLC Flag (Government-Linked Companies)
# =============================================================================

GLC_COMPANIES = {
    # Khazanah-backed
    "1155": "MAYBANK",  # Maybank
    "1023": "CIMB",
    "6888": "AXIATA",
    "4863": "TM",
    "5347": "TENAGA",
    "6683": "IHH",
    "1773": "UEMS",
    
    # PNB-backed
    "1295": "PBBANK",
    "5285": "SIME",
    "4588": "UMW",
    "5204": "SPSETIA",
    
    # Petronas
    "6033": "PETGAS",
    "5183": "PCHEM",
    "5681": "PETDAG",
}


# =============================================================================
# Sector Classification
# =============================================================================

SECTOR_MAPPING = {
    # Financials
    "1155": "Financials", "1295": "Financials", "1023": "Financials",
    "1066": "Financials", "5398": "Financials", "5819": "Financials",
    "1082": "Financials", "5099": "Financials", "6888": "Financials",
    
    # Telecommunications
    "6012": "Telecommunications", "6947": "Telecommunications",
    "4863": "Telecommunications", "6888": "Telecommunications",
    "5113": "Telecommunications",
    
    # Plantations
    "1961": "Plantations", "5285": "Plantations", "2445": "Plantations",
    "2125": "Plantations",
    
    # Consumer
    "4707": "Consumer Staples", "1817": "Consumer Staples",
    "3689": "Consumer Staples", "3222": "Consumer Discretionary",
    "5284": "Consumer Discretionary", "7081": "Consumer Staples",
    
    # Utilities & Energy
    "5347": "Utilities", "7277": "Energy",
    "6033": "Energy", "5183": "Energy", "5681": "Energy",
    "3043": "Energy",
    
    # Industrial
    "3182": "Consumer Discretionary", "4715": "Consumer Discretionary",
    "5168": "Healthcare", "7113": "Healthcare", "7153": "Healthcare",
    
    # Technology
    "0166": "Technology", "0097": "Technology", "7106": "Technology",
    "0084": "Technology", "5006": "Technology", "3872": "Technology",
    "0183": "Technology",
    
    # Healthcare
    "6683": "Healthcare",
    
    # Construction
    "5398": "Industrials", "3336": "Industrials",
    
    # Conglomerates
    "4588": "Industrials",
    
    # Property
    "5204": "Real Estate", "3087": "Real Estate",
    "1773": "Real Estate", "8206": "Real Estate",
    
    # Tobacco & Beverage
    "4162": "Consumer Staples", "2629": "Consumer Staples",
    "2836": "Consumer Staples",
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_yahoo_ticker(code: str) -> str:
    """Convert Bursa stock code to Yahoo Finance format."""
    return f"{code}{YF_SUFFIX}"


def get_local_name(code: str) -> str:
    """Get local name for a stock code."""
    return KLCI30_CODES.get(code, EXTENDED_CODES.get(code, "Unknown"))


def get_all_tickers(universe: str = "klci30") -> list[str]:
    """Get all stock codes for a given universe."""
    if universe == "klci30":
        return KLCI30
    elif universe == "extended":
        return list(set(KLCI30 + list(EXTENDED_CODES.keys())))
    elif universe == "all":
        return list(set(KLCI30 + list(EXTENDED_CODES.keys())))
    else:
        raise ValueError(f"Unknown universe: {universe}")


def is_glc(code: str) -> bool:
    """Check if a company is a Government-Linked Company."""
    return code in GLC_COMPANIES


def get_sector(code: str) -> str:
    """Get sector classification for a stock code."""
    return SECTOR_MAPPING.get(code, "Unknown")


# =============================================================================
# Shariah Non-Compliant Tickers
# =============================================================================

# Known Shariah-non-compliant stocks
NON_COMPLIANT_TICKERS = {
    "4162",  # BAT (British American Tobacco)
    "2629",  # HEINEKEN
    "2836",  # CARLSBG
    "4715",  # GENM (Genting Malaysia - gambling)
    "3182",  # GENTING (gambling)
}


def is_shariah_non_compliant(code: str) -> bool:
    """Check if ticker is known to be non-Shariah compliant."""
    return code in NON_COMPLIANT_TICKERS
