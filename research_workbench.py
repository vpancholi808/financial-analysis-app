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
        strengths.append(f"Book-to-market is {bm:.2f} (latest annual common book equity / current market cap).")
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
    # Risk bullets follow the active screen flags, so they always agree with the
    # user's slider settings instead of hard-coded thresholds.
    if bool(row.get("STA_risk_flag",False)):
        risks.append(f"Scaled accruals are {pct(sta)}: net income exceeds operating cash flow relative to assets "
                     f"({n(row,'STA_sector_pct'):.0f}th sector percentile).")
    if bool(row.get("SNOA_risk_flag",False)):
        risks.append(f"SNOA is in the {n(row,'SNOA_sector_pct'):.0f}th percentile of sector peers.")
    if bool(row.get("Beneish_outlier",False)):
        gaps.append(f"Beneish M-score is not used: {row.get('Beneish_status','component outlier')} (usually a near-zero prior-year figure).")
    if bool(row.get("Beneish_risk_flag",False)):
        risks.append(f"Beneish M-score {n(row,'Beneish_M'):.2f} breaches the active manipulation-risk screen; not proof of fraud.")
    if bool(row.get("Altman_risk_flag",False)):
        risks.append(f"Altman model shows a distress-zone score of {n(row,'Altman_Z'):.2f}; confirm business-type applicability.")
    if str(row.get("Altman_zone","")).startswith("Review"):
        gaps.append("Altman Z double-prime not used: negative book equity (often from buybacks) is outside the model's design.")
    if bool(row.get("conflicting_signals",False)):
        risks.append("Opposing BUY and SELL screen signals require qualitative resolution.")
    if not np.isfinite(n(row,"Beneish_M")):
        gaps.append("Beneish unavailable; required two-year statement components missing.")
    if not np.isfinite(n(row,"SNOA")):
        gaps.append("SNOA unavailable; cash, short- or long-term debt, or total liabilities missing.")
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
    """Full, source-labeled working paper; generated and analyst-edited sections are labelled."""
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
    position=analyst.get("position","") or "Not yet provided"
    conviction=analyst.get("conviction","")
    entries += ["","## Recommendation",
        f"**{position}**" + (f" · {conviction} conviction" if conviction else ""),"",
        f"*Sections edited by the analyst: {analyst.get('edited','none (generated case)')}. Unedited sections are generated "
        "from the screen evidence above.*","",
        "## Investment case"]
    for k,label in (("thesis","Thesis"),("catalyst","Catalyst and timing"),("counterargument","Strongest counterargument"),
                    ("invalidation","What would invalidate the thesis"),("filing_reference","Filings")):
        entries += ["",f"### {label}",analyst.get(k,"") or "Not yet provided"]
    entries += ["","## Conclusion",
        "The recommendation follows fixed rules applied to annual financial statements and a current market-cap snapshot: "
        "long when the company ranks highly on gross profitability and book-to-market with cash-backed earnings and no "
        "accounting-risk checks, short / avoid when enough independent accounting-risk checks trigger. It contains no price "
        "target, forecast or catalyst information beyond the next annual report, so it should be verified against the 10-K.",
        "","*Academic analysis; not investment advice.*"]
    return "\n".join(entries)

SEC_10K_SEARCH="https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K&dateb=&owner=include&count=40"

def _ord(p):
    if not np.isfinite(p):return "n/a"
    v=int(round(p));suf="th" if 10<=v%100<=20 else {1:"st",2:"nd",3:"rd"}.get(v%10,"th")
    return f"{v}{suf}"

def _next_annual_report(row):
    """Next fiscal year-end after the saved one, and ~60 days later when its
    10-K is due (large accelerated filers), rolled forward past the quote date."""
    fye=pd.to_datetime(row.get("fiscal_year_end"),errors="coerce")
    if pd.isna(fye):return None,None
    asof=pd.to_datetime(row.get("price_asof"),errors="coerce")
    if pd.isna(asof):asof=pd.Timestamp.now().normalize()
    nxt=fye+pd.DateOffset(years=1)
    while nxt+pd.DateOffset(days=60)<asof:
        nxt=nxt+pd.DateOffset(years=1)
    return nxt,nxt+pd.DateOffset(days=60)

def investment_case(row,findings,settings=None):
    """Rule-based recommendation built only from the screen evidence.

    Long: passes the long screen with positive earnings and cash flow and no
    accounting-risk check triggered. Short / avoid: risk-flagged by the active
    screen (enough independent checks) and not a long candidate. Everything
    else is No position, with the reason stated. Conviction rises with the
    strength of the evidence and when valuation agrees (P/S vs sector median).
    Catalyst, invalidation tests and filing link are derived from the data;
    no price target or return forecast is produced.
    """
    sta_pct=getattr(settings,"sell_sta_pct",80.);snoa_pct=getattr(settings,"sell_snoa_pct",80.)
    cutoff=getattr(settings,"sell_beneish_cutoff",-1.78);need=int(getattr(settings,"sell_min_flags",2))
    gpa_min=getattr(settings,"buy_gpa_pct",60.);bm_min=getattr(settings,"buy_bm_pct",60.)
    t=str(row.get("ticker","")).upper();name=str(row.get("name",t)).strip().rstrip(".")
    ni=n(row,"net_income");cf=n(row,"operating_cash_flow");flags=int(n(row,"SELL_flag_count") if np.isfinite(n(row,"SELL_flag_count")) else 0)
    ps,sps=findings.get("ps",np.nan),findings.get("sector_ps",np.nan)
    rel=(ps/sps-1) if np.isfinite(ps) and np.isfinite(sps) and sps>0 else np.nan
    growth=findings.get("revenue_growth",np.nan)
    gpa,gpa_p=n(row,"GPA"),n(row,"GPA_sector_pct");bm,bm_p=n(row,"BM"),n(row,"BM_sector_pct")
    score=n(row,"BUY_score");sta=n(row,"STA");m=n(row,"Beneish_M");z=n(row,"Altman_Z")
    gpa_med=n(row,"GPA_sector_median");sector=str(row.get("sector",""))
    fired=[lbl for lbl,key in (("accruals","STA_risk_flag"),("net operating assets","SNOA_risk_flag"),
                                ("Beneish M-score","Beneish_risk_flag"),("Altman distress","Altman_risk_flag")) if bool(row.get(key,False))]
    nxt,due=_next_annual_report(row)
    report_txt=(f"the next annual report (fiscal year ending about {nxt:%b %Y}, expected by about {due:%b %Y})"
                if nxt is not None else "the next annual report")
    val_txt=("" if not np.isfinite(rel) else
             f" The shares trade at {ps:.1f}× sales, {abs(rel)*100:.0f}% {'below' if rel<0 else 'above'} the {sector} median of {sps:.1f}×.")
    filing=SEC_10K_SEARCH.format(ticker=t)
    base=dict(filing_reference=filing)

    if not bool(row.get("industrial_applicable",True)):
        return dict(base,position="No position",conviction="",headline="Specialist sector: no rule-based call",
            summary=f"{name} is a bank, insurer or REIT; the accounting screens used here do not apply, so no recommendation is generated.",
            thesis="Not applicable: industrial accounting screens (GPA, B/M, accruals, SNOA, Beneish, Altman) are not valid for this sector.",
            catalyst="Not applicable.",counterargument="Not applicable.",
            invalidation="Not applicable.")
    if bool(row.get("stale_statements",False)):
        return dict(base,position="No position",conviction="",headline="Stale data: no call until statements refresh",
            summary=f"{name}'s latest saved statements are too old to compare with today's market value.",
            thesis="No recommendation: the saved annual statements are more than about 15 months older than the market-cap quote.",
            catalyst=f"Refresh when {report_txt} is loaded.",counterargument="Not applicable.",invalidation="Not applicable.")

    is_long=bool(row.get("BUY_candidate",False)) and np.isfinite(ni) and ni>0 and np.isfinite(cf) and cf>0
    is_short=bool(row.get("SELL_candidate",False)) and flags>=max(need,1)

    if is_long and is_short:
        return dict(base,position="No position",conviction="",headline="Mixed signals: no call",
            summary=(f"{name} passes the long screen (long score {score:.0f}) but also triggers {flags} accounting-risk "
                     f"check{'s' if flags!=1 else ''} ({', '.join(fired)}). The evidence points both ways."),
            thesis=(f"Quality and value ranks are strong (gross profitability {_ord(gpa_p)} and book-to-market {_ord(bm_p)} "
                    f"sector percentile), but {', '.join(fired)} signal possible earnings-quality problems. No position until the "
                    "accounting concern is resolved in the filings."),
            catalyst=f"Resolution in {report_txt}: whether the flagged accounting measures normalise.",
            counterargument="Either side can be argued; the screen alone cannot rank which signal dominates.",
            invalidation="A long case becomes valid if the risk checks clear; a short case if quality or value ranks fall below the long thresholds.")

    if is_long:
        clean=flags==0
        pts=sum([score>=85,np.isfinite(rel) and rel<=0,clean,np.isfinite(sta) and sta<=0,np.isfinite(growth) and growth>=0])
        conviction="High" if pts>=4 else "Moderate" if pts>=3 else "Low"
        thesis=(f"Long {name}. It is among the most profitable and cheapest companies in {sector}: gross profit is "
                f"{gpa*100:.1f}% of assets ({_ord(gpa_p)} sector percentile"
                + (f", sector median {gpa_med*100:.1f}%" if np.isfinite(gpa_med) else "") + ") and book-to-market is "
                f"{bm:.2f} ({_ord(bm_p)} percentile), giving a long score of {score:.0f}/100. Profitable, cheap-on-book firms "
                "have historically outperformed (the gross-profitability and value premia). Earnings are backed by cash: "
                f"net income {money(ni)}, operating cash flow {money(cf)}"
                + (f", accruals {sta*100:.1f}% of assets" if np.isfinite(sta) else "") + "."
                + (f" Revenue grew {growth*100:.1f}% last fiscal year." if np.isfinite(growth) else "") + val_txt
                + (" No accounting-risk check is triggered." if clean else f" One risk check to watch: {', '.join(fired)}."))
        catalyst=(f"{report_txt[0].upper()+report_txt[1:]}: continued high gross profitability and cash-backed earnings should keep the "
                  "company in its sector's top ranks while the market re-rates the discount to book value.")
        counter=[]
        if np.isfinite(rel) and rel>0:counter.append(f"valuation is not cheap on sales ({ps:.1f}× vs {sps:.1f}× sector median)")
        if np.isfinite(growth) and growth<0:counter.append(f"revenue fell {abs(growth)*100:.1f}% last year")
        if fired:counter.append(f"{', '.join(fired)} triggered")
        if np.isfinite(bm):counter.append("a high book-to-market can reflect a value trap (assets the market expects to earn poorly)")
        counterargument=("Strongest objections: "+"; ".join(counter)+".") if counter else "No material counter-evidence in the saved statements."
        invalidation=(f"Exit or reverse if the next annual report shows gross profitability falling below the {_ord(gpa_min)} sector percentile, "
                      f"book-to-market rank below the {_ord(bm_min)}, accruals turning positive into the top {100-sta_pct:.0f}% of the sector, "
                      f"or {need} or more accounting-risk checks triggering.")
        return dict(base,position="Long",conviction=conviction,headline=f"Long · {conviction} conviction",
            summary=(f"Long: top-ranked on gross profitability and book-to-market in {sector} (long score {score:.0f}/100), "
                     "with positive, cash-backed earnings" + (", and no accounting red flags." if clean else f"; one risk check to watch ({', '.join(fired)}).")),
            thesis=thesis,catalyst=catalyst,counterargument=counterargument,invalidation=invalidation)

    if is_short:
        pts=sum([flags>=3,np.isfinite(rel) and rel>0,np.isfinite(growth) and growth<0,np.isfinite(ni) and ni<=0,
                 bool(row.get("Beneish_risk_flag",False))])
        conviction="High" if pts>=3 else "Moderate" if pts>=1 else "Low"
        detail=[]
        if bool(row.get("STA_risk_flag",False)):
            detail.append(f"accruals are {sta*100:.1f}% of assets ({_ord(n(row,'STA_sector_pct'))} sector percentile): reported profit is running ahead of cash")
        if bool(row.get("SNOA_risk_flag",False)):
            detail.append(f"net operating assets are {n(row,'SNOA')*100:.0f}% of assets ({_ord(n(row,'SNOA_sector_pct'))} percentile): a bloated balance sheet that tends to precede weaker earnings")
        if bool(row.get("Beneish_risk_flag",False)):
            detail.append(f"the Beneish M-score is {m:.2f}, above the {cutoff:.2f} cutoff")
        if bool(row.get("Altman_risk_flag",False)):
            detail.append(f"the Altman score of {z:.2f} is in the distress zone")
        thesis=(f"Short / avoid {name}. {flags} independent accounting-risk checks trigger: " + "; ".join(detail) + "."
                + (f" Revenue changed {growth*100:+.1f}% last fiscal year." if np.isfinite(growth) else "") + val_txt
                + " Firms with these accounting signals have historically underperformed as earnings quality catches up with reported profit.")
        catalyst=(f"{report_txt[0].upper()+report_txt[1:]}: weaker cash conversion, write-downs or restated items would confirm the accounting signals.")
        counter=[]
        if np.isfinite(gpa_p) and gpa_p>=60:counter.append(f"gross profitability is still strong ({_ord(gpa_p)} sector percentile)")
        if np.isfinite(cf) and cf>0:counter.append(f"operating cash flow remains positive ({money(cf)})")
        if np.isfinite(rel) and rel<0:counter.append(f"the shares already trade below the sector sales multiple ({ps:.1f}× vs {sps:.1f}×)")
        if np.isfinite(growth) and growth>0:counter.append(f"revenue is growing ({growth*100:.1f}%)")
        counter.append("accounting signals flag risk, not certainty; short positions also carry borrow cost and squeeze risk")
        counterargument="Strongest objections: "+"; ".join(counter)+"."
        invalidation=(f"Cover if the next annual report shows fewer than {need} risk checks triggering — for example accruals back "
                      f"at or below zero, net operating assets below the {_ord(snoa_pct)} sector percentile, an M-score below "
                      f"{cutoff:.2f}, or an Altman score out of the distress zone.")
        return dict(base,position="Short / avoid",conviction=conviction,headline=f"Short / avoid · {conviction} conviction",
            summary=(f"Short / avoid: {flags} accounting-risk checks trigger ({', '.join(fired)})"
                     + (f"; trades {abs(rel)*100:.0f}% {'above' if rel>0 else 'below'} the sector sales multiple." if np.isfinite(rel) else ".")),
            thesis=thesis,catalyst=catalyst,counterargument=counterargument,invalidation=invalidation)

    near=[]
    if np.isfinite(gpa_p):near.append(f"gross profitability {_ord(gpa_p)} percentile (long needs ≥ {_ord(gpa_min)})")
    if np.isfinite(bm_p):near.append(f"book-to-market {_ord(bm_p)} percentile (needs ≥ {_ord(bm_min)})")
    near.append(f"{flags} of 4 risk checks triggered (short needs ≥ {need})")
    return dict(base,position="No position",conviction="",headline="No position: criteria not met",
        summary=f"No position: {name} meets neither the long nor the short criteria — " + "; ".join(near) + ".",
        thesis=(f"No position in {name}. The accounting evidence is not strong enough either way: " + "; ".join(near) + "."
                + val_txt),
        catalyst=f"Re-evaluate after {report_txt}.",
        counterargument="A qualitative view (products, management, guidance) could still justify a position; the accounting screen does not.",
        invalidation="A long case opens if both ranks pass the long thresholds; a short case if enough risk checks trigger.")

def sector_ps_medians(peers, minimum=5):
    """Sector median market cap / revenue, computed once (same rule as sector_multiple)."""
    if not isinstance(peers,pd.DataFrame) or peers.empty:return {}
    t=peers
    if "industrial_applicable" in t:t=t.loc[t.industrial_applicable.astype(bool)]
    if "has_matching_currency" in t:t=t.loc[t.has_matching_currency.astype(bool)]
    rev=pd.to_numeric(t["revenue"],errors="coerce");cap=pd.to_numeric(t["market_cap"],errors="coerce")
    ps=(cap/rev).where(rev>0).replace([np.inf,-np.inf],np.nan)
    g=ps.groupby(t["sector"])
    return {s:float(m) for s,m in g.median().items() if g.count()[s]>=minimum}

REC_LABEL={"Long":"▲ Long","Short / avoid":"▼ Short / avoid","No position":"○ No position"}

def recommendation_table(classified, peers, settings=None):
    """investment_case() for every row: position, conviction, label and one-line
    summary, index-aligned with `classified`."""
    medians=sector_ps_medians(peers)
    out=[]
    for idx,row in classified.iterrows():
        rev=n(row,"revenue");cap=n(row,"market_cap")
        findings={"ps":ratio(cap,rev),"sector_ps":medians.get(str(row.get("sector","")),np.nan),
                  "revenue_growth":change(rev,n(row,"prior_revenue"))}
        case=investment_case(row,findings,settings)
        pos=case["position"]
        not_rated=pos=="No position" and (not bool(row.get("industrial_applicable",True)) or bool(row.get("stale_statements",False)))
        label=("– Not rated" if not_rated else REC_LABEL.get(pos,pos)+(f" · {case['conviction']}" if case["conviction"] else ""))
        out.append({"rec_position":"Not rated" if not_rated else pos,"rec_conviction":case["conviction"],
                    "rec_label":label,"rec_summary":case["summary"]})
    return pd.DataFrame(out,index=classified.index)
