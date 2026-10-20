"""ACCT 3343 equity screening models.

Verified course methods (phase 1): GPA, B/M and STA.
SNOA, Beneish PMAN, and Altman Z must be checked against course notes
before they are implemented. All demo records are synthetic.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

GICS_SECTORS = [
    "Information Technology", "Health Care", "Financials",
    "Consumer Discretionary", "Consumer Staples", "Industrials",
    "Energy", "Materials", "Utilities", "Real Estate",
    "Communication Services",
]
INCOMPATIBLE = {"Financials", "Real Estate"}
BASE_FIELDS = [
    "ticker", "name", "sector", "industry", "exchange",
    "market_cap", "market_currency", "financial_currency",
    "price_asof", "fiscal_year_end", "data_source", "refreshed_at",
    "revenue", "cost_of_revenue", "gross_profit_reported",
    "net_income", "operating_cash_flow", "total_assets", "common_equity",
]
FINANCIAL_FIELDS = [
    "market_cap", "revenue", "cost_of_revenue", "gross_profit_reported",
    "net_income", "operating_cash_flow", "total_assets", "common_equity",
]

def safe_divide(numerator, denominator):
    """Missing/nonfinite values stay missing, denominator must be positive."""
    n = pd.to_numeric(numerator, errors="coerce")
    d = pd.to_numeric(denominator, errors="coerce").where(lambda x: x > 0)
    result = n / d
    return result.where(np.isfinite(result))

def prepare(data, min_sector_n=10, min_industry_n=10):
    """Calculate auditable raw ratios and full-universe peer benchmarks."""
    d = data.copy()
    for field in BASE_FIELDS:
        if field not in d:
            d[field] = np.nan
    for field in FINANCIAL_FIELDS:
        d[field] = pd.to_numeric(d[field], errors="coerce")
    for field in ["ticker", "name", "sector", "industry",
                  "financial_currency", "market_currency"]:
        d[field] = d[field].fillna("").astype(str).str.strip()
    d["ticker"] = d["ticker"].str.upper()
    d = d.loc[d["ticker"].ne("")].drop_duplicates("ticker").reset_index(drop=True)
    d["industrial_applicable"] = ~d["sector"].isin(INCOMPATIBLE)
    d["currency_matches"] = (
        d["market_currency"].ne("") & d["financial_currency"].ne("")
        & d["market_currency"].str.upper().eq(d["financial_currency"].str.upper())
    )
    provided = d["revenue"].notna() & d["cost_of_revenue"].notna()
    d["gross_profit"] = (d["revenue"] - d["cost_of_revenue"]).where(
        provided, d["gross_profit_reported"]
    )
    d["gross_profit_basis"] = np.where(
        provided, "Revenue minus cost of revenue",
        np.where(d["gross_profit_reported"].notna(), "Reported gross profit", "Missing")
    )
    d["GPA"] = safe_divide(d["gross_profit"], d["total_assets"])
    d["BM"] = safe_divide(d["common_equity"], d["market_cap"])
    d.loc[(d["common_equity"] <= 0) | ~d["currency_matches"], "BM"] = np.nan
    d["STA"] = safe_divide(
        d["net_income"] - d["operating_cash_flow"], d["total_assets"]
    )
    d["ROA"] = safe_divide(d["net_income"], d["total_assets"])
    d["CFOA"] = safe_divide(d["operating_cash_flow"], d["total_assets"])
    for metric in ["GPA", "BM", "STA", "ROA", "CFOA"]:
        valid = d["industrial_applicable"] & d[metric].notna()
        d[metric + "_sector_pct"] = np.nan
        d[metric + "_sector_median"] = np.nan
        d[metric + "_sector_mean"] = np.nan
        d[metric + "_sector_count"] = np.nan
        d[metric + "_industry_pct"] = np.nan
        d[metric + "_industry_count"] = np.nan
        if not valid.any():
            continue
        peers = d.loc[valid, ["sector", "industry", metric]]
        sector = peers.groupby("sector")[metric]
        industry = peers.groupby(["sector", "industry"])[metric]
        sc = sector.transform("count")
        ic = industry.transform("count")
        d.loc[valid, metric + "_sector_count"] = sc
        d.loc[valid, metric + "_sector_median"] = sector.transform("median")
        d.loc[valid, metric + "_sector_mean"] = sector.transform("mean")
        d.loc[valid, metric + "_industry_count"] = ic
        d.loc[valid, metric + "_sector_pct"] = (
            sector.rank(pct=True).mul(100).where(sc >= min_sector_n)
        )
        d.loc[valid, metric + "_industry_pct"] = (
            industry.rank(pct=True).mul(100).where(ic >= min_industry_n)
        )
    return d

@dataclass(frozen=True)
class ScreenSettings:
    min_gpa_pct: float = 60
    min_bm_pct: float = 60
    min_buy_score: float = 65
    gpa_weight: float = 0.60
    min_sta_pct: float = 80
    min_sta_raw: float = 0.0

def classify(d, settings):
    """Independent BUY (profitability/value) and SELL (earnings quality)."""
    result = d.copy()
    w = settings.gpa_weight
    result["BUY_score"] = (
        result["GPA_sector_pct"] * w + result["BM_sector_pct"] * (1 - w)
    )
    result["BUY_candidate"] = (
        result["industrial_applicable"]
        & result["GPA_sector_pct"].ge(settings.min_gpa_pct)
        & result["BM_sector_pct"].ge(settings.min_bm_pct)
        & result["BUY_score"].ge(settings.min_buy_score)
    )
    result["STA_risk_flag"] = (
        result["industrial_applicable"]
        & result["STA"].gt(settings.min_sta_raw)
        & result["STA_sector_pct"].ge(settings.min_sta_pct)
    )
    result["SELL_flag_count"] = result["STA_risk_flag"].astype(int)
    result["SELL_candidate"] = result["STA_risk_flag"]
    result["conflicting_signals"] = (
        result["BUY_candidate"] & result["SELL_candidate"]
    )
    return result

def filter_universe(df, sectors=None, industry=None, min_cap=0, exclude=None):
    d = df
    if sectors is not None:
        d = d.loc[d["sector"].isin(sectors)]
    if industry and industry != "All industries":
        d = d.loc[d["industry"].eq(industry)]
    d = d.loc[d["market_cap"].ge(min_cap)]
    if exclude:
        d = d.loc[~d["ticker"].isin({t.strip().upper() for t in exclude})]
    return d.copy()

def integrity_report(df):
    return {
        "companies": int(len(df)),
        "invalid_sector": int((~df["sector"].isin(GICS_SECTORS)).sum()),
        "currency_mismatch_or_missing": int((~df["currency_matches"]).sum()),
        "GPA_available": int(df["GPA"].notna().sum()),
        "BM_available": int(df["BM"].notna().sum()),
        "STA_available": int(df["STA"].notna().sum()),
    }
