"""Research thesis calculations remain transparent and guard missing values."""
import numpy as np
import pandas as pd
from research_workbench import describe, scenario, report

def row(**updates):
    x=dict(ticker="FOO",name="Example Inc",sector="Information Technology",
      industry="Software",industrial_applicable=True,market_cap=1_000_000_000,
      revenue=500_000_000,prior_revenue=400_000_000,net_income=50_000_000,
      operating_cash_flow=60_000_000,financial_currency="USD",price_asof="2026-10-21",
      fiscal_year_end="2025-12-31",GPA=.35,GPA_sector_pct=85,BM=.2,
      BM_sector_pct=75,STA=-.01,SNOA=np.nan,SNOA_sector_pct=np.nan,
      Beneish_M=np.nan,Altman_Z=np.nan,BUY_candidate=True,SELL_candidate=False,
      SELL_flag_count=0,conflicting_signals=False)
    x.update(updates)
    return pd.Series(x)

def peers():
    return pd.DataFrame([row(ticker=f"C{i}",market_cap=1_000_000_000+i*100_000_000)
                         for i in range(10)])

def test_conditional_long_has_evidence():
    x=describe(row(),peers())
    assert x["verdict"]=="Long research candidate"
    assert x["ps"]==2
    assert len(x["strengths"])>0
    assert "missing" not in x["verdict"].lower()

def test_accounting_flag_alone_is_not_short_recommendation():
    x=describe(row(BUY_candidate=False,SELL_candidate=True,SELL_flag_count=1),peers())
    assert x["verdict"]=="Accounting risk watchlist"

def test_two_accounting_flags_mark_review_not_automatic_short():
    x=describe(row(BUY_candidate=False,SELL_candidate=True,SELL_flag_count=2),peers())
    assert x["verdict"]=="Short-risk research candidate"

def test_conflicting_signals_are_not_given_an_automatic_long():
    x=describe(row(SELL_candidate=True,conflicting_signals=True),peers())
    assert x["verdict"]=="Mixed signals"

def test_scenarios_are_exact_accounting_arithmetic():
    d=scenario(row(),[("Bear",1),("Base",2),("Bull",3)])
    assert list(d["Implied market cap"])==[500e6,1e9,1.5e9]
    assert list(d["Implied change"])==[-.5,0,.5]

def test_missing_revenue_blocks_implied_value():
    d=scenario(row(revenue=np.nan),[("Bear",1)])
    assert pd.isna(d.iloc[0]["Implied market cap"])

def test_export_preserves_unsupported_research_as_incomplete():
    r=row()
    x=describe(r,peers())
    text=report(r,x)
    assert "Not yet provided" in text
    assert "not a trade instruction" in text
    assert "financials/" in text
