"""Evidence-first long/short research framework for the S&P 500 screen.

Every numeric output is calculated from the saved Yahoo financial statements.
Candidate classifications are NOT final recommendations to trade. Scenario
multiples are user assumptions, not analyst forecasts.
"""
from __future__ import annotations
import math
import numpy as np
import pandas as pd

def n(row, key):
    try:
        x=float(row.get(key,np.nan))
        return x if math.isfinite(x) else np.nan
    except (ValueError,TypeError):return np.nan

def ratio(a,b):
    return a/b if np.isfinite(a) and np.isfinite(b) and b>0 else np.nan

def money(x,currency="USD"):
    if not np.isfinite(x):return "N/A"
    ax=abs(x)
    fmt=(f"{ax/1e12:,.2f}T" if ax>=1e12 else
         f"{ax/1e9:,.2f}B" if ax>=1e9 else
         f"{ax/1e6:,.2f}M" if ax>=1e6 else
         f"{ax/1e3:,.1f}K" if ax>=1e3 else f"{ax:,.0f}")
    prefix="$" if currency=="USD" else currency+" "
    return ("-" if x<0 else "")+prefix+fmt

def pct(x):
    return "N/A" if not np.isfinite(x) else f"{100*x:,.1f}%"

def change(current,previous):
    return ratio(current,previous)-1 if np.isfinite(ratio(current,previous)) else np.nan

def sector_multiple(row, peers, minimum=5):
    """Median market-cap / latest-annual-revenue among comparable-sector names.

    This is a cross-sectional, mixed-date statistic, not trailing P/S.
    """
    if not isinstance(peers,pd.DataFrame) or peers.empty:return np.nan,0
    sector=str(row.get("sector",""))
    t=peers.loc[peers.sector.eq(sector)]
    if "industrial_applicable" in t:t=t.loc[t.industrial_applicable]
    vals=pd.to_numeric(t["market_cap"],errors="coerce")/pd.to_numeric(t["revenue"],errors="coerce")
    vals=vals.where(pd.to_numeric(t["revenue"],errors="coerce")>0).replace([np.inf,-np.inf],np.nan).dropna()
    # Non-equal market/reporting currencies cannot be compared this way.
    if "has_matching_currency" in t:
        vals=vals.loc[t.loc[vals.index,"has_matching_currency"].astype(bool)]
    return (float(vals.median()),len(vals)) if len(vals)>=minimum else (np.nan,len(vals))

def describe(row,peers):
    """Prepared research facts and conflicts; conclusions are conditional."""
    revenue=n(row,"revenue");cap=n(row,"market_cap");ni=n(row,"net_income")
    cf=n(row,"operating_cash_flow")
    ps=ratio(cap,revenue)
    ey=ratio(ni,cap)
    cfy=ratio(cf,cap)
    gp=n(row,"GPA");bm=n(row,"BM");sta=n(row,"STA")
    rev_growth=change(revenue,n(row,"prior_revenue"))
    median,peer_n=sector_multiple(row,peers)
    if not bool(row.get("industrial_applicable",False)):
        verdict="Specialist review"
        reason="Industrial accounting screens are not appropriate for this sector."
    elif bool(row.get("conflicting_signals",False)):
        verdict="Mixed signals"
        reason="The company satisfies both a quality/value screen and an accounting-risk screen."
    elif bool(row.get("BUY_candidate",False)) and np.isfinite(ni) and ni>0 and np.isfinite(cf) and cf>0:
        verdict="Long research candidate"
        reason="Quality and book-value screening conditions are met, with positive earnings and operating cash flow."
    elif bool(row.get("SELL_candidate",False)) and int(row.get("SELL_flag_count",0))>=2:
        verdict="Short-risk research candidate"
        reason="Multiple independent accounting warning signals warrant a bearish thesis investigation."
    elif bool(row.get("SELL_candidate",False)):
        verdict="Accounting risk watchlist"
        reason="An accounting warning alone is insufficient to establish a short-sale thesis."
    else:
        verdict="No qualified thesis"
        reason="The saved fundamentals do not satisfy the current research selection rules."

    strengths=[];risks=[];gaps=[]
    if np.isfinite(gp):
        strengths.append(f"Gross profitability is {pct(gp)} of total assets.")
        if np.isfinite(n(row,"GPA_sector_pct")):
            strengths.append(f"Gross-profitability rank is the {n(row,'GPA_sector_pct'):.0f}th sector percentile.")
    else:gaps.append("Gross profit unavailable: no GPA conclusion.")
    if np.isfinite(bm):
        strengths.append(f"Book-to-market is {bm:.2f}x (current market cap / latest annual book equity comparison).")
        if np.isfinite(n(row,"BM_sector_pct")):
            strengths.append(f"Book-to-market sits at the {n(row,'BM_sector_pct'):.0f}th sector percentile.")
    else:gaps.append("Comparable positive common equity or market cap unavailable.")
    if np.isfinite(rev_growth):
        (strengths if rev_growth>=0 else risks).append(f"Fiscal-year revenue growth was {pct(rev_growth)}.")
    else:gaps.append("Matched prior fiscal-year revenue unavailable.")
    if np.isfinite(cf):
        (strengths if cf>0 else risks).append(f"Annual operating cash flow is {money(cf,str(row.get('financial_currency') or 'USD'))} ({pct(cfy)} of current market cap).")
    else:gaps.append("Annual operating cash flow missing.")
    if np.isfinite(ni):
        (strengths if ni>0 else risks).append(f"Annual net income is {money(ni,str(row.get('financial_currency') or 'USD'))}; annual-income yield vs. current market cap is {pct(ey)}.")
    else:gaps.append("Annual net income missing.")
    if np.isfinite(ps):
        label=f"Market cap / latest annual sales is {ps:.2f}x"
        if np.isfinite(median):
            label+=f" versus {median:.2f}x sector median across {peer_n} eligible names"
            (risks if ps>median else strengths).append(label+".")
        else:
            strengths.append(label+"; no comparable sector median available.")
    else:gaps.append("Positive annual revenue and positive current market cap required for sales valuation.")
    if np.isfinite(sta) and sta>0:
        risks.append(f"Scaled accruals are {pct(sta)}: net income exceeds operating cash flow relative to assets.")
    if np.isfinite(n(row,"SNOA_sector_pct")) and n(row,"SNOA_sector_pct")>=80:
        risks.append(f"SNOA is in the {n(row,'SNOA_sector_pct'):.0f}th percentile of sector peers.")
    if bool(row.get("Beneish_risk_flag",False)):
        risks.append(f"Beneish M-score {n(row,'Beneish_M'):.2f} breaches the active manipulation-risk screen; not proof of fraud.")
    if bool(row.get("Altman_risk_flag",False)):
        risks.append(f"Altman model shows a distress-zone score of {n(row,'Altman_Z'):.2f}; confirm business-type applicability.")
    if bool(row.get("conflicting_signals",False)):
        risks.append("Opposing BUY and SELL screen signals require qualitative resolution.")
    if not np.isfinite(n(row,"Beneish_M")):
        gaps.append("Beneish unavailable; required two-year statement components missing.")
    if not np.isfinite(n(row,"SNOA")):
        gaps.append("SNOA unavailable; one or more operating-asset/liability components missing.")
    if not np.isfinite(n(row,"Altman_Z")):
        gaps.append("Altman model unavailable or not appropriate for this industry.")
    gaps.append("Catalysts, market expectations, forecasts, stock-borrow cost and SEC filing review are not contained in these annual statements.")
    return {"verdict":verdict,"reason":reason,"strengths":strengths,"risks":risks,
            "gaps":gaps,"ps":ps,"sector_ps":median,"peers":peer_n,
            "earnings_yield":ey,"cashflow_yield":cfy,"revenue_growth":rev_growth}

def scenario(row,multiples):
    """Revenue x assumed P/S => hypothetical market capitalization, NOT share price."""
    revenue=n(row,"revenue");current=n(row,"market_cap")
    out=[]
    for label,multiple in multiples:
        value=revenue*multiple if np.isfinite(revenue) and revenue>0 else np.nan
        delta=(value/current-1) if np.isfinite(value) and np.isfinite(current) and current>0 else np.nan
        out.append({"Scenario":label,"Sales multiple":multiple,"Implied market cap":value,
                    "Implied change":delta})
    return pd.DataFrame(out)

def report(row,findings,assumptions=None,analyst=None):
    """Full, source-labeled working paper. Human research sections never fabricated."""
    assumptions=assumptions or {}
    analyst=analyst or {}
    ticker=str(row.get("ticker","")).upper()
    quote=f"https://finance.yahoo.com/quote/{ticker}/"
    entries=[
      f"# {ticker} | Investment research working paper","",
      f"**Company:** {row.get('name','')}",
      f"**Sector:** {row.get('sector','')} | **Industry:** {row.get('industry','')}",
      f"**Research classification:** {findings['verdict']} (conditional, not a trade instruction)",
      f"**Basis:** {findings['reason']}",
      f"**Statements fiscal year:** {row.get('fiscal_year_end','N/A')}",
      f"**Market-cap snapshot:** {row.get('price_asof','N/A')}",
      f"**Source:** Yahoo Finance via yfinance, as retrieved {row.get('refreshed_at','N/A')}",
      f"**Financial statements:** {quote}financials/",
      f"**Balance sheet:** {quote}balance-sheet/",
      f"**Cash flow:** {quote}cash-flow/","",
      "## Financial and valuation evidence"
    ]
    entries += ["- "+v for v in findings["strengths"]] or ["- No positive claim supported by current data."]
    entries += ["","## Bear case and accounting concerns"]
    entries += ["- "+v for v in findings["risks"]] or ["- No qualifying warning in available indicators; absence of evidence is not evidence of safety."]
    entries += ["","## Missing evidence and objections"]
    entries += ["- "+v for v in findings["gaps"]]
    if assumptions:
        entries += ["","## Hypothetical price-to-sales scenarios (not forecasts)"]
        for k,val in assumptions.items():
            entries.append(f"- {k}: {val}")
    entries += ["","## Analyst-reviewed research (human input)"]
    for k in ("position","thesis","catalyst","counterargument","invalidation","filing_reference"):
        entries.append(f"- **{k.replace('_',' ').title()}:** {analyst.get(k,'') or 'Not yet provided'}")
    entries += ["","## Conclusion",
        "The screening classification is conditional. A supported long or short recommendation requires analyst-verified filings, valuation, catalysts, and a falsifiable alternative case. These are not supplied by an accounting screen alone.",
        "","*Academic analysis; not an investment recommendation.*"]
    return "\n".join(entries)
