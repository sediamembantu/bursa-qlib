"""
Ticker definitions for Bursa Malaysia universe.
KLCI-30 constituents and extended Main Market stocks.
"""

from config import YF_SUFFIX

# =============================================================================
# KLCI-30 Constituents (as of 2026)
# =============================================================================

KLCI30 = [
    # Financials
    "MAYANG",       # Maybank
    "PBBANK",       # Public Bank
    "CIMB",         # CIMB Group
    "RHBBANK",      # RHB Bank
    "AMBANK",       # AmBank Group
    
    # Telecommunications
    "MAXIS",        # Maxis
    "DIGI",         # Digi.com
    "TM",           # Telekom Malaysia
    "AXIATA",       # Axiata Group
    
    # Plantations
    "IOICORP",      # IOI Corporation
    "SIME",         # Sime Darby Plantation
    "KLK",          # Kuala Lumpur Kepong
    
    # Banking & Finance (continued)
    "HLBANK",       # Hong Leong Bank
    "HLFG",         # Hong Leong Financial Group
    
    # Consumer
    "NESTLE",       # Nestle Malaysia
    "DLADY",        # Dutch Lady Milk
    "F&N",          # Fraser & Neave
    
    # Utilities
    "TENAGA",       # Tenaga Nasional
    "DIALOG",       # Dialog Group
    
    # Industrial
    "PETRONAS",     # Petronas Gas
    "PCHEM",        # Petronas Chemicals
    "GENTING",      # Genting Berhad
    "GENM",         # Genting Malaysia
    
    # Property
    "UMW",          # UMW Holdings
    
    # Technology
    "INARI",        # Inari Amertron
    "VITROX",       # Vitrox Corp
    
    # Healthcare
    "IHH",          # IHH Healthcare
    
    # Construction
    "GAMUDA",       # Gamuda
    "MRCB",         # Malaysia Building Society
]


# =============================================================================
# Extended Universe (Liquid Main Market Stocks)
# =============================================================================

EXTENDED_UNIVERSE = [
    # Additional financials
    "AFFIN", "BAF", "BPPLAB", "LPI",
    
    # Plantations
    "SDPLANT", "FIMACORP", "SARAWAK",
    
    # Property
    "SPSETIA", "MAHSING", "UEMS", "ECOWLD", "IOIPROP", "TAMBUN",
    
    # Construction
    "IJM", "SUNCON", "GADANG", "WEIDA",
    
    # Industrial
    "HARTA", "TOPGLOV", "KOSSAN", "SUPERMX",  # Glove sector
    "PADINI", "FIPPER", "MRDIY",
    
    # Technology
    "PENTA", "NOTION", "UNISEM", "MPI", "GREATEC",
    
    # Consumer
    "QL", "LEONGHUP", "LAYHONG", "APOLLO", "HUPSENG",
    
    # Energy
    "PETDAG", "PETRONM", "SEALINK",
    
    # Telecommunications
    "TIMECOM", "CNASIA",
    
    # Transport & Logistics
    "POS", "TAS", "WESTPORT",
    
    # Healthcare
    "PHARMA", "DKSH", "APEX",
    
    # Media
    "MEDIA", "STAR",
    
    # Others
    "BAT", "JTINTER", "HEINEKEN", "CARLSBG", "FRASER", "MRCB",
]


# =============================================================================
# GLC Flag (Government-Linked Companies)
# =============================================================================

GLC_COMPANIES = {
    # Khazanah-backed
    "MAYANG", "CIMB", "AXIATA", "TM", "TENAGA", "IHH", "UEMS", "MAHB",
    
    # PNB-backed
    "PBBANK", "SIME", "MAYANG", "UMW", "SPSETIA",
    
    # EPF-backed
    "MAYANG", "CIMB", "PBBANK", "TENAGA",
    
    # Other GLCs
    "PETRONAS", "PCHEM", "PETDAG", "MRCB",
}


# =============================================================================
# Sector Classification
# =============================================================================

SECTOR_MAPPING = {
    # Financials
    "MAYANG": "Financials", "PBBANK": "Financials", "CIMB": "Financials",
    "RHBBANK": "Financials", "AMBANK": "Financials", "HLBANK": "Financials",
    "HLFG": "Financials", "AFFIN": "Financials", "BAF": "Financials",
    "LPI": "Financials",
    
    # Telecommunications
    "MAXIS": "Telecommunications", "DIGI": "Telecommunications",
    "TM": "Telecommunications", "AXIATA": "Telecommunications",
    "TIMECOM": "Telecommunications",
    
    # Plantations
    "IOICORP": "Plantations", "SIME": "Plantations", "KLK": "Plantations",
    "SDPLANT": "Plantations", "FIMACORP": "Plantations",
    
    # Consumer
    "NESTLE": "Consumer Staples", "DLADY": "Consumer Staples",
    "F&N": "Consumer Staples", "PADINI": "Consumer Discretionary",
    "MRDIY": "Consumer Discretionary", "QL": "Consumer Staples",
    
    # Utilities
    "TENAGA": "Utilities", "DIALOG": "Energy",
    
    # Energy
    "PETRONAS": "Energy", "PCHEM": "Energy", "PETDAG": "Energy",
    "PETRONM": "Energy",
    
    # Industrial
    "GENTING": "Consumer Discretionary", "GENM": "Consumer Discretionary",
    "HARTA": "Healthcare", "TOPGLOV": "Healthcare", "KOSSAN": "Healthcare",
    
    # Technology
    "INARI": "Technology", "VITROX": "Technology", "PENTA": "Technology",
    "NOTION": "Technology", "UNISEM": "Technology", "MPI": "Technology",
    "GREATEC": "Technology",
    
    # Healthcare
    "IHH": "Healthcare", "PHARMA": "Healthcare",
    
    # Construction
    "GAMUDA": "Industrials", "MRCB": "Industrials", "IJM": "Industrials",
    "SUNCON": "Industrials",
    
    # Property
    "UMW": "Industrials", "SPSETIA": "Real Estate", "MAHSING": "Real Estate",
    "UEMS": "Real Estate", "ECOWLD": "Real Estate",
    
    # Tobacco & Beverage
    "BAT": "Consumer Staples", "HEINEKEN": "Consumer Staples",
    "CARLSBG": "Consumer Staples",
}


# =============================================================================
# Helper Functions
# =============================================================================

def get_yahoo_ticker(ticker: str) -> str:
    """Convert Bursa ticker to Yahoo Finance format."""
    return f"{ticker}{YF_SUFFIX}"


def get_all_tickers(universe: str = "klci30") -> list[str]:
    """Get all tickers for a given universe."""
    if universe == "klci30":
        return KLCI30
    elif universe == "extended":
        return KLCI30 + EXTENDED_UNIVERSE
    elif universe == "all":
        return list(set(KLCI30 + EXTENDED_UNIVERSE))
    else:
        raise ValueError(f"Unknown universe: {universe}")


def is_glc(ticker: str) -> bool:
    """Check if a company is a Government-Linked Company."""
    return ticker in GLC_COMPANIES


def get_sector(ticker: str) -> str:
    """Get sector classification for a ticker."""
    return SECTOR_MAPPING.get(ticker, "Unknown")
