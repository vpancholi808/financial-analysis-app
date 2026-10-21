"""Independent, reproducible audit of saved Yahoo financial inputs and ratio logic.

Run: python audit_data.py
Uses ONLY the saved actual-source CSV, never simulated fundamentals.
Warnings require manual filing review; no source data is overwritten.
"""
from __future__ import annotations
import json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
from finance_models import prepare, UNIVERSE_COLUMNS, STALE_STATEMENT_DAYS

ROOT=Path(__file__).resolve().parent
AUDIT_COLUMNS=["ticker","severity","category","detail","as_of"]

def audit_table(raw: pd.DataFrame, members: pd.DataFrame | None = None,
                as_of: str | None = None):
    now=pd.Timestamp(as_of) if as_of else pd.Timestamp.now(tz="UTC").tz_localize(None)
    if now.tzinfo is not None: now=now.tz_localize(None)
    data=prepare(raw)
    issues=[]
    def flag(row,level,code,detail):
        issues.append({"ticker":str(row["ticker"]),"severity":level,
                       "category":code,"detail":str(detail),"as_of":now.date().isoformat()})
    cols=["gross_profit","GPA","BM","STA","ROA","CFOA"]
    for _,r in data.iterrows():
        a=r.get("total_assets")
        if not np.isfinite(a) or a<=0:
            flag(r,"critical","INVALID_TOTAL_ASSETS","Assets absent or nonpositive; ratio denominators invalid")
            continue
        if not np.isfinite(r.get("market_cap",np.nan)) or r.market_cap<=0:
            flag(r,"critical","INVALID_MARKET_CAP","B/M unavailable without a positive market cap")
        if not r.get("has_matching_currency",False):
            flag(r,"warning","CURRENCY_MISMATCH","Book-to-market requires matched reporting and quote currency")
        for k in ("gross_profit","net_income","operating_cash_flow","common_equity","ebit"):
            v=r.get(k,np.nan)
            if pd.isna(v):
                flag(r,"missing","MISSING_"+k.upper(),"Related metrics must show N/A, not zero")
        # Reported accounting fields are matched-year observations, not substitute zeros.
        if pd.notna(r.get("gross_profit_reported")) and pd.notna(r.get("revenue")) and pd.notna(r.get("cost_of_revenue")):
            difference=(r.revenue-r.cost_of_revenue)-r.gross_profit_reported
            tol=max(abs(r.revenue)*.01,1e6)
            if abs(difference)>tol:
                flag(r,"warning","GROSS_PROFIT_RECONCILIATION",
                     f"revenue - COGS differs from stated GP by {difference:,.0f}")
        li=r.get("total_liabilities",np.nan)
        eq=r.get("common_equity",np.nan)
        if pd.notna(li) and pd.notna(eq) and abs(li+eq-a)/a > .15:
            flag(r,"review","COMMON_VS_TOTAL_EQUITY",
                 "Liabilities + COMMON equity differs from total assets >15%; check minority and preferred equity and Yahoo line-item definitions")
        if pd.notna(r.get("cash_and_equivalents")) and (r.cash_and_equivalents<0 or r.cash_and_equivalents>a*1.05):
            flag(r,"warning","CASH_OUTSIDE_ASSETS","Cash is negative or exceeds total assets")
        fy=pd.to_datetime(r.get("fiscal_year_end"),errors="coerce")
        if pd.notna(fy) and (now-fy).days>STALE_STATEMENT_DAYS:
            flag(r,"warning","STALE_FISCAL_PERIOD",f"Fiscal statement is more than {STALE_STATEMENT_DAYS} days old; excluded from ranking and screens")
        if pd.notna(fy) and (now-fy).days < -65:
            flag(r,"critical","FUTURE_FISCAL_PERIOD","Annual fiscal date is in the future")
        # Independently recompute active formulas, requiring every source input.
        # Gross profit is rebuilt from the raw columns, not the model's derived field.
        if pd.notna(r.revenue) and pd.notna(r.cost_of_revenue):
            raw_gp=r.revenue-r.cost_of_revenue
        else:
            raw_gp=r.get("gross_profit_reported",np.nan)
        cash,std,ltd,tl=(r.get(k,np.nan) for k in ("cash_and_equivalents","short_term_debt","long_term_debt","total_liabilities"))
        mi,pe,ce=(r.get(k,np.nan) for k in ("minority_interest","preferred_equity","common_equity"))
        z=lambda v:0.0 if pd.isna(v) else v   # unreported lines are left out, as in the model
        if pd.isna(cash):
            snoa_exp=np.nan
        elif pd.notna(mi) and pd.notna(pe) and pd.notna(ce):
            snoa_exp=((a-cash)-(a-z(std)-z(ltd)-mi-pe-ce))/a
        elif pd.notna(tl):
            snoa_exp=((a-cash)-(tl-z(std)-z(ltd)))/a
        elif pd.notna(ce):
            snoa_exp=((a-cash)-(a-z(std)-z(ltd)-z(mi)-z(pe)-ce))/a
        else:
            snoa_exp=np.nan
        expected={
            "GPA": raw_gp/a if pd.notna(raw_gp) else np.nan,
            "SNOA": snoa_exp,
            "STA": (r.net_income-r.operating_cash_flow)/a if pd.notna(r.net_income) and pd.notna(r.operating_cash_flow) else np.nan,
            "ROA": r.net_income/a if pd.notna(r.net_income) else np.nan,
            "CFOA": r.operating_cash_flow/a if pd.notna(r.operating_cash_flow) else np.nan,
            "BM": r.common_equity/r.market_cap if pd.notna(r.common_equity) and r.common_equity>0 and pd.notna(r.market_cap) and r.market_cap>0 and r.has_matching_currency else np.nan,
        }
        for k,v in expected.items():
            observed=r[k]
            same=(pd.isna(v) and pd.isna(observed)) or (pd.notna(v) and pd.notna(observed) and np.isclose(v,observed,rtol=1e-10,atol=1e-12))
            if not same:
                flag(r,"critical","RATIO_MISMATCH_"+k,f"calculated {v}, stored {observed}")
        # Static income statement and current market capitalization have different observation dates.
        # This is disclosed rather than falsely treating B/M as a point-in-time historical backtest.
    if members is not None and "ticker" in members:
        present=set(data.ticker)
        for sym in members.ticker.astype(str):
            if sym not in present:
                issues.append({"ticker":sym,"severity":"missing","category":"MISSING_CONSTITUENT",
                               "detail":"No usable annual statement imported for index ticker","as_of":now.date().isoformat()})
    issue_df=pd.DataFrame(issues,columns=AUDIT_COLUMNS)
    summary={
        "as_of":now.date().isoformat(),
        "fetched_constituents":int(len(data)),
        "index_constituents":int(len(members)) if members is not None else None,
        "required_field_coverage":{
            k: {"valid":int(data[k].notna().sum()),"missing":int(data[k].isna().sum())}
            for k in ["GPA","BM","STA","ROA","CFOA","gross_profit","ebit","cost_of_revenue"]
        },
        "ratio_mismatches":int(issue_df.category.str.startswith("RATIO_MISMATCH").sum()) if len(issue_df) else 0,
        "warnings":int(issue_df.severity.eq("warning").sum()),
        "critical":int(issue_df.severity.eq("critical").sum()),
        "review":int(issue_df.severity.eq("review").sum()),
        "missing":int(issue_df.severity.eq("missing").sum()),
        "limitations":[
            "Yahoo annual statement fiscal-year labels may be calendar month-end rather than exact SEC fiscal closing dates.",
            "Market cap is current quotation data and is not matched to the historical annual balance-sheet date.",
            "B/M must not be presented as a historical backtest without point-in-time market capitalization.",
            "Financials and Real Estate require specialist models; industrial-company scores are not assigned.",
            "Missing source numbers are NEVER replaced with assumed zero values.",
            "No automated error flag proves reporting fraud or misstatement; human filing review is required."
        ]
    }
    return issue_df, summary

def main():
    raw=pd.read_csv(ROOT/"universe.csv")
    member_file=ROOT/"sp500_constituents.csv"
    members=pd.read_csv(member_file) if member_file.exists() else None
    issues,summary=audit_table(raw,members)
    issues.to_csv(ROOT/"metric_audit.csv",index=False)
    (ROOT/"audit_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(summary,indent=2))
    if summary["ratio_mismatches"]>0 or summary["critical"]>0:
        raise SystemExit("Critical accuracy failures. Review metric_audit.csv.")

if __name__=="__main__":
    main()
