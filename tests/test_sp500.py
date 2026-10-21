"""Deterministic S&P 500 membership and financial-quality checks."""
from io import StringIO
import pandas as pd
from sp500_pipeline import normalize_symbol, anomaly_flags, complete_fraction

def test_normalize_dot_class_tickers():
    assert normalize_symbol("BRK.B")=="BRK-B"
    assert normalize_symbol("BF.B")=="BF-B"

def test_flag_missing_cap_currency_and_stale_statements():
    sample={"ticker":"ZZZ","total_assets":1_000,"revenue":400,
            "gross_profit_reported":300,"cash_and_equivalents":1500,
            "market_cap":None,"market_currency":"USD","financial_currency":"CAD",
            "fiscal_year_end":"2015-12-31","net_income":15,
            "operating_cash_flow":None,"common_equity":300}
    findings=anomaly_flags(sample)
    codes={x["issue"] for x in findings}
    assert "MISSING_MARKET_CAP" in codes
    assert "CASH_OUTSIDE_ASSETS" in codes
    assert "CURRENCY_MISMATCH" in codes
    assert "STALE_ANNUAL_STATEMENTS" in codes

def test_data_completeness_does_not_impute_zeros():
    data=pd.DataFrame([
      dict(gross_profit_reported=20,net_income=10,operating_cash_flow=15,total_assets=100,common_equity=60),
      dict(gross_profit_reported=None,net_income=20,operating_cash_flow=15,total_assets=110,common_equity=70)])
    assert complete_fraction(data)==0.5

def test_dashboard_contains_charts_and_anomaly_report():
    from pathlib import Path
    source=(Path(__file__).resolve().parents[1]/"app.py").read_text()
    assert "Value versus profitability" in source
    assert "Data quality and detailed methodology" in source
    assert "sp500_constituents.csv" in source
    assert "Export full audit" in source
