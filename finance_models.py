"""Auditable Phase-1 course ratios and independent BUY/SELL screening rules.

GPA, B/M, STA, ROA, CFOA plus course-defined SNOA and published
Beneish M-score and Altman variants. Missing inputs remain missing.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from advanced_models import compute_advanced

GICS_SECTORS = [
    "Information Technology", "Health Care", "Financials",
    "Consumer Discretionary", "Consumer Staples", "Industrials",
    "Energy", "Materials", "Utilities", "Real Estate",
    "Communication Services",
]

# These industries require an alternative/verified course treatment for the
# *industrial-company* accounting screen. They remain visible in the universe.
INDUSTRIAL_INCOMPATIBLE = {"Financials", "Real Estate"}

METADATA_COLUMNS = [
    "ticker", "name", "sector", "industry", "exchange", "market_cap",
    "market_currency", "financial_currency", "price_asof", "fiscal_year_end",
    "prior_fiscal_year_end", "refreshed_at", "data_source",
]
RAW_COLUMNS = [
    "revenue", "cost_of_revenue", "gross_profit_reported", "net_income",
    "operating_cash_flow", "total_assets", "common_equity",
    "current_assets", "current_liabilities", "total_liabilities",
    "retained_earnings", "ebit", "cash_and_equivalents",
    "accounts_receivable", "net_ppe", "depreciation", "sga",
    "total_debt", "prior_revenue", "prior_cost_of_revenue",
    "prior_net_income", "prior_operating_cash_flow", "prior_total_assets",
    "prior_current_assets", "prior_current_liabilities",
    "prior_total_liabilities", "prior_common_equity", "prior_receivables",
    "prior_net_ppe", "prior_depreciation", "prior_sga", "prior_total_debt",
    "short_term_debt", "long_term_debt", "minority_interest", "preferred_equity",
    "prior_short_term_debt", "prior_long_term_debt",
    "prior_minority_interest", "prior_preferred_equity",
]
UNIVERSE_COLUMNS = METADATA_COLUMNS + RAW_COLUMNS


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Validates finite numerator and strictly positive denominator."""
    n = pd.to_numeric(numerator, errors="coerce")
    d = pd.to_numeric(denominator, errors="coerce")
    v = n.div(d.where(d > 0))
    return v.where(np.isfinite(v))


def prepare(df: pd.DataFrame, min_sector_n: int = 10,
            min_industry_n: int = 10) -> pd.DataFrame:
    """Calculate only validated Phase-1 ratios; retain missing values as NaN."""
    data = df.copy()
    for col in UNIVERSE_COLUMNS:
        if col not in data.columns:
            data[col] = np.nan
    for col in RAW_COLUMNS + ["market_cap"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    for col in ["ticker", "name", "sector", "industry", "market_currency", "financial_currency"]:
        data[col] = data[col].fillna("").astype(str).str.strip()
    data["ticker"] = data["ticker"].str.upper()
    data = data.loc[data["ticker"].ne("")].drop_duplicates(subset="ticker").reset_index(drop=True)

    # Sector restrictions do NOT remove these stocks from the source universe.
    data["industrial_applicable"] = ~data["sector"].isin(INDUSTRIAL_INCOMPATIBLE)
    data["has_matching_currency"] = (
        data["market_currency"].ne("")
        & data["financial_currency"].ne("")
        & data["market_currency"].str.upper().eq(data["financial_currency"].str.upper())
    )
    # Prefer raw Revenue - Cost of Revenue; if both are absent, an independently
    # reported Gross Profit may be used, visibly marked in 'gross_profit_basis'.
    computed_gp = data["revenue"] - data["cost_of_revenue"]
    valid_components = data["revenue"].notna() & data["cost_of_revenue"].notna()
    data["gross_profit"] = computed_gp.where(valid_components, data["gross_profit_reported"])
    data["gross_profit_basis"] = np.select(
        [valid_components, data["gross_profit_reported"].notna()],
        ["Revenue - COGS", "Reported gross profit"], default="Unavailable"
    )
    ta = data["total_assets"]
    data["GPA"] = safe_ratio(data["gross_profit"], ta)
    data["BM"] = safe_ratio(data["common_equity"], data["market_cap"])
    # Negative book equity is not a meaningful 'cheap value' score.
    data.loc[(data["common_equity"] <= 0) | ~data["has_matching_currency"], "BM"] = np.nan
    data["STA"] = safe_ratio(data["net_income"] - data["operating_cash_flow"], ta)
    data["ROA"] = safe_ratio(data["net_income"], ta)
    data["CFOA"] = safe_ratio(data["operating_cash_flow"], ta)
    advanced = data.apply(compute_advanced,axis=1)
    for col in advanced.columns:
        data[col] = advanced[col]

    # The benchmark is built from the WHOLE eligible universe, not the dashboard
    # post-filter subset, avoiding look-ahead to user-selected filtering rules.
    for metric in ["GPA", "BM", "STA", "ROA", "CFOA", "SNOA", "Beneish_M"]:
        eligible = data["industrial_applicable"] & data[metric].notna()
        tmp = data.loc[eligible, ["sector", "industry", metric]]
        s_grp = tmp.groupby("sector")[metric]
        i_grp = tmp.groupby(["sector", "industry"])[metric]
        data[metric + "_sector_count"] = np.nan
        data[metric + "_sector_median"] = np.nan
        data[metric + "_sector_mean"] = np.nan
        data[metric + "_sector_pct"] = np.nan
        data[metric + "_industry_count"] = np.nan
        data[metric + "_industry_pct"] = np.nan
        if not tmp.empty:
            s_count = s_grp.transform("count")
            i_count = i_grp.transform("count")
            data.loc[eligible, metric + "_sector_count"] = s_count
            data.loc[eligible, metric + "_sector_median"] = s_grp.transform("median")
            data.loc[eligible, metric + "_sector_mean"] = s_grp.transform("mean")
            data.loc[eligible, metric + "_industry_count"] = i_count
            # Assign via original indices, guarding peer sample size.
            s_pct = s_grp.rank(pct=True) * 100
            i_pct = i_grp.rank(pct=True) * 100
            data.loc[s_pct.index, metric + "_sector_pct"] = s_pct.where(s_count >= min_sector_n)
            data.loc[i_pct.index, metric + "_industry_pct"] = i_pct.where(i_count >= min_industry_n)
    return data


@dataclass(frozen=True)
class ScreenSettings:
    buy_gpa_pct: float = 60.0
    buy_bm_pct: float = 60.0
    buy_score_min: float = 65.0
    buy_gpa_weight: float = 0.60
    sell_sta_pct: float = 80.0
    sell_sta_floor: float = 0.0
    sell_min_flags: int = 1
    sell_snoa_pct: float = 80.0
    sell_beneish_cutoff: float = -1.78


def classify(prepared: pd.DataFrame, settings: ScreenSettings) -> pd.DataFrame:
    """Separate signals; no inverse-BUY rule is ever used for SELL."""
    if not 0 <= settings.buy_gpa_weight <= 1:
        raise ValueError("BUY GPA weight must be 0..1")
    if settings.sell_min_flags < 1:
        raise ValueError("SELL minimum flags must be >= 1")
    d = prepared.copy()
    w = settings.buy_gpa_weight
    d["BUY_score"] = w * d["GPA_sector_pct"] + (1.0 - w) * d["BM_sector_pct"]
    d["BUY_candidate"] = (
        d["industrial_applicable"]
        & d["GPA_sector_pct"].ge(settings.buy_gpa_pct)
        & d["BM_sector_pct"].ge(settings.buy_bm_pct)
        & d["BUY_score"].ge(settings.buy_score_min)
    )
    d["STA_risk_flag"] = (
        d["industrial_applicable"]
        & d["STA"].gt(settings.sell_sta_floor)
        & d["STA_sector_pct"].ge(settings.sell_sta_pct)
    )
    d["SNOA_risk_flag"] = (
        d["industrial_applicable"]
        & d["SNOA_sector_pct"].ge(settings.sell_snoa_pct)
    )
    d["Beneish_risk_flag"] = (
        d["industrial_applicable"]
        & d["Beneish_M"].gt(settings.sell_beneish_cutoff)
    )
    d["Altman_risk_flag"] = (
        d["industrial_applicable"] & d["Altman_zone"].eq("Distress")
    )
    d["SELL_flag_count"] = (
        d["STA_risk_flag"].astype(int) + d["SNOA_risk_flag"].astype(int)
        + d["Beneish_risk_flag"].astype(int) + d["Altman_risk_flag"].astype(int)
    )
    d["SELL_candidate"] = d["SELL_flag_count"].ge(settings.sell_min_flags)
    d["conflicting_signals"] = d["BUY_candidate"] & d["SELL_candidate"]
    return d


def scope_filter(prepared: pd.DataFrame, sectors: list[str] | None = None,
                 industry: str = "All industries", min_cap: float = 0,
                 excluded_tickers: set[str] | None = None) -> pd.DataFrame:
    """Apply UI filters AFTER universe-relative percentiles are established."""
    d = prepared
    if sectors is not None:
        d = d.loc[d["sector"].isin(sectors)]
    if industry != "All industries":
        d = d.loc[d["industry"].eq(industry)]
    if min_cap > 0:
        d = d.loc[d["market_cap"].ge(min_cap)]
    if excluded_tickers:
        excluded = {str(t).strip().upper() for t in excluded_tickers}
        d = d.loc[~d["ticker"].isin(excluded)]
    return d.copy()


def validate_universe(df: pd.DataFrame) -> dict:
    """Human-readable data integrity report for dashboard and tests."""
    if df.empty:
        return {"rows": 0, "duplicate_tickers": 0, "invalid_sector_rows": 0,
                "unmatched_currency_rows": 0, "usable_gpa": 0, "usable_bm": 0,
                "usable_sta": 0, "usable_snoa": 0,
                "usable_beneish": 0,"usable_altman": 0}
    return {
        "rows": int(len(df)),
        "duplicate_tickers": int(df["ticker"].duplicated().sum()),
        "invalid_sector_rows": int((~df["sector"].isin(GICS_SECTORS)).sum()),
        "unmatched_currency_rows": int((~df["has_matching_currency"]).sum()),
        "usable_gpa": int(df["GPA"].notna().sum()),
        "usable_bm": int(df["BM"].notna().sum()),
        "usable_sta": int(df["STA"].notna().sum()),
        "usable_snoa": int(df["SNOA"].notna().sum()),
        "usable_beneish": int(df["Beneish_M"].notna().sum()),
        "usable_altman": int(df["Altman_Z"].notna().sum()),
    }
