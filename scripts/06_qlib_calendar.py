"""
Bursa Malaysia Trading Calendar for qlib

Generates trading calendar excluding:
- Weekends (Sat, Sun)
- Malaysian public holidays
"""

import pandas as pd
from datetime import date
from pathlib import Path

# Malaysian public holidays (2020-2026)
# Major holidays that affect trading
MALAYSIAN_HOLIDAYS = [
    # 2024
    "2024-01-01",  # New Year
    "2024-02-01",  # Federal Territory Day
    "2024-02-08",  # Chinese New Year
    "2024-02-09",
    "2024-04-10",  # Hari Raya Puasa
    "2024-05-01",  # Labour Day
    "2024-05-22",  # Wesak Day
    "2024-06-17",  # Hari Raya Haji
    "2024-07-07",  # Awal Muharram
    "2024-08-31",  # National Day
    "2024-09-16",  # Malaysia Day
    "2024-10-31",  # Deepavali
    "2024-11-01",  # Deepavali (some states)
    "2024-12-25",  # Christmas
    
    # 2025 (projected)
    "2025-01-01",  # New Year
    "2025-01-29",  # Chinese New Year
    "2025-01-30",
    "2025-03-31",  # Hari Raya Puasa (projected)
    "2025-05-01",  # Labour Day
    "2025-05-12",  # Wesak Day (projected)
    "2025-06-07",  # Hari Raya Haji (projected)
    "2025-06-27",  # Awal Muharram
    "2025-08-31",  # National Day
    "2025-09-16",  # Malaysia Day
    "2025-10-20",  # Deepavali (projected)
    "2025-12-25",  # Christmas
    
    # 2026 (projected)
    "2026-01-01",  # New Year
    "2026-02-17",  # Chinese New Year (projected)
    "2026-02-18",
    "2026-03-20",  # Hari Raya Puasa (projected)
    "2026-05-01",  # Labour Day
    "2026-05-02",  # Wesak Day (projected)
    "2026-05-28",  # Hari Raya Haji (projected)
    "2026-06-16",  # Awal Muharram
    "2026-08-31",  # National Day
    "2026-09-16",  # Malaysia Day
    "2026-11-08",  # Deepavali (projected)
    "2026-12-25",  # Christmas
]


def generate_calendar(
    start_date: str = "2020-01-01",
    end_date: str = "2026-12-31",
) -> pd.DatetimeIndex:
    """
    Generate Bursa Malaysia trading calendar.
    
    Args:
        start_date: Start date
        end_date: End date
    
    Returns:
        DatetimeIndex of trading days
    """
    # Generate all weekdays
    all_dates = pd.date_range(start=start_date, end=end_date, freq="B")
    
    # Convert holidays to datetime
    holidays = pd.to_datetime(MALAYSIAN_HOLIDAYS)
    
    # Filter out holidays
    trading_days = all_dates[~all_dates.isin(holidays)]
    
    return trading_days


def save_calendar(output_dir: Path = None):
    """Save calendar to qlib format."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent.parent / "data" / "qlib" / "calendars"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    calendar = generate_calendar()
    
    # qlib calendar format: one date per line (YYYY-MM-DD)
    output_file = output_dir / "day.txt"
    
    with open(output_file, "w") as f:
        for date in calendar:
            f.write(date.strftime("%Y-%m-%d") + "\n")
    
    print(f"Calendar saved: {output_file}")
    print(f"Trading days: {len(calendar)}")
    
    return output_file


if __name__ == "__main__":
    save_calendar()
