"""Pinned, independently sourced SEC annual statement spot checks.

Source figures are millions USD, converted to raw USD for Yahoo comparison.
Checks apply only to the named fiscal YEAR; future periods use new filings.
Apple 2025 10-K: https://d18rn0p25nwr6d.cloudfront.net/CIK-0000320193/c636d8a7-8025-47d2-9b13-bcf5465343b3.html
NVIDIA 2026 10-K: https://www.sec.gov/Archives/edgar/data/1045810/000104581026000021/nvda-20260125.htm
"""
from pathlib import Path
import pandas as pd
import pytest

DATA = Path(__file__).resolve().parents[1] / "universe.csv"

@pytest.mark.parametrize("ticker,year,field,millions", [
    ("AAPL", 2025, "revenue", 416161),
    ("AAPL", 2025, "gross_profit_reported", 195201),
    ("AAPL", 2025, "total_assets", 359241),
    ("NVDA", 2026, "revenue", 215938),
    ("NVDA", 2026, "gross_profit_reported", 153463),
    ("NVDA", 2026, "total_assets", 206803),
])
def test_yahoo_fields_against_sec_10k(ticker,year,field,millions):
    df = pd.read_csv(DATA)
    record = df.loc[df.ticker.eq(ticker)]
    if record.empty:
        pytest.skip("Company not yet collected")
    r=record.iloc[0]
    if str(r.fiscal_year_end)[:4] != str(year):
        pytest.skip("Latest financial year has advanced; re-anchor to newer SEC filing")
    observed=pd.to_numeric(r[field],errors="coerce")
    assert pd.notna(observed), f"{ticker} missing {field}"
    assert observed == pytest.approx(millions * 1_000_000, abs=2_000_000)
