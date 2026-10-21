"""S&P 500 Research Terminal: simple overview, screening and evidence-based research."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from finance_models import GICS_SECTORS, ScreenSettings, prepare, classify, scope_filter
from research_workbench import describe, scenario, report, money, pct, n
from audit_data import audit_table

ROOT=Path(__file__).resolve().parent
st.set_page_config(page_title="S&P 500 | Financial Research",page_icon="📈",layout="wide",
                   initial_sidebar_state="expanded")
st.markdown("""<style>
.block-container{max-width:1420px;padding-top:1rem;padding-bottom:2rem}
h1{font-size:2.15rem!important;letter-spacing:-.04em}
h2{font-size:1.5rem!important}
[data-testid="stSidebar"]{border-right:1px solid #344453}
[data-testid="stMetric"]{background:#172536;border:1px solid #32445a;border-radius:10px;padding:13px 16px}
[data-testid="stMetricLabel"]{font-size:.78rem}
[data-testid="stDataFrame"]{border-radius:9px}
div[data-testid="stAlert"]{border-radius:9px}
.small-note{color:#9bacbd;font-size:.82rem}
</style>""",unsafe_allow_html=True)

@st.cache_data(ttl=1800)
def load():
    data=pd.read_csv(ROOT/"universe.csv")
    members=pd.read_csv(ROOT/"sp500_constituents.csv")
    data=data.loc[data.ticker.astype(str).isin(set(members.ticker))].copy()
    config=json.loads((ROOT/"config.json").read_text())
    banned=set(config.get("excluded_tickers",[]))
    return data.loc[~data.ticker.isin(banned)].copy(),members

@st.cache_data(ttl=1800)
def calculate(df):
    return prepare(df,min_sector_n=10,min_industry_n=10)

def cash_label(r,k):
    return money(n(r,k),str(r.get("financial_currency") or "USD"))
def style_chart(fig,height=None):
    fig.update_layout(template="plotly_dark",paper_bgcolor="#0D1724",plot_bgcolor="#172536",
                      margin=dict(l=8,r=12,b=16,t=52),font_size=12)
    if height:fig.update_layout(height=height)
    return fig
def table(df,cols,label=None):
    if label:st.subheader(label)
    if df.empty:
        st.info("No qualifying companies. Adjust the filters.")
        return
    shown=df[cols].copy()
    for key in ("market_cap","revenue","operating_cash_flow"):
        if key in shown:shown[key]=shown[key].apply(lambda x:money(float(x)) if pd.notna(x) else "N/A")
    st.dataframe(shown,hide_index=True,use_container_width=True,height=min(490,45+35*len(shown)))

try:
    raw,members=load()
except (OSError,ValueError,KeyError) as exc:
    st.error("Financial statements are not available in this deployment. Verify the collection workflow.")
    st.stop()
if raw.empty:
    st.info("No usable S&P 500 financial statements have been saved yet.")
    st.stop()
base=calculate(raw)
if base.empty:
    st.error("The saved financial dataset is empty after validation.")
    st.stop()

st.markdown("### ◈ FINANCIAL ANALYSIS")
st.caption("S&P 500 equity research · Financial statements: Yahoo Finance via yfinance")
with st.sidebar:
    st.title("Research Terminal")
    page=st.radio("Navigate",["S&P 500 Screener","Company Overview","Research Desk"])
    st.divider()
    sectors=st.multiselect("Sector",sorted(base.sector.dropna().unique()),default=sorted(base.sector.dropna().unique()))
    industries=sorted(base.loc[base.sector.isin(sectors),"industry"].dropna().unique())
    industry=st.selectbox("Industry",["All industries"]+industries)
    cap_min=st.number_input("Minimum market cap ($B)",min_value=0.,value=0.,step=5.)
    with st.expander("Screening criteria"):
        gpa=st.slider("Profitability percentile",0,100,60,5)
        bm=st.slider("Book-to-market percentile",0,100,60,5)
        score=st.slider("Minimum long screen score",0,100,65,5)
        w=st.slider("Profitability weight",0.,1.,.6,.05)
        sta=st.slider("Accruals risk percentile",0,100,80,5)
        snoa=st.slider("SNOA risk percentile",0,100,80,5)
        beneish=st.number_input("Beneish risk cutoff",value=-1.78,step=.01)
        riskcount=st.slider("Minimum bearish indicators",1,4,2)
    st.caption(f"Latest dataset: {len(base)} of {len(members)} S&P 500 listings")
    with st.expander("Sources and methodology"):
        st.write("Annual company statements are matched by fiscal year. Market cap is a more recent quote snapshot; ratios are cross-sectional, not point-in-time backtests.")
        st.markdown("[Financial statements at Yahoo Finance](https://finance.yahoo.com/)")
        st.markdown("[GitHub model and data audit](https://github.com/vpancholi808/financial-analysis-app)")
settings=ScreenSettings(buy_gpa_pct=gpa,buy_bm_pct=bm,buy_score_min=score,buy_gpa_weight=w,
                        sell_sta_pct=sta,sell_snoa_pct=snoa,sell_beneish_cutoff=beneish,sell_min_flags=riskcount)
selected=scope_filter(base,sectors,industry,cap_min*1e9)
view=classify(selected,settings)
longs=view.loc[view.BUY_candidate].sort_values("BUY_score",ascending=False)
shorts=view.loc[view.SELL_candidate].sort_values(["SELL_flag_count","STA"],ascending=[False,False])

def choose_company(df,key):
    if df.empty:st.info("No companies match your filters.");st.stop()
    labels={str(r.ticker):f"{r.ticker}  ·  {r['name']}" for _,r in df.iterrows()}
    return st.selectbox("Search a company",sorted(labels),format_func=lambda s:labels[s],key=key)

if page=="S&P 500 Screener":
    st.title("Discover S&P 500 opportunities")
    st.caption("Filter and compare companies using profitability, book value and accounting quality.")
    a,b,c,d=st.columns(4)
    a.metric("Companies",len(view))
    b.metric("Long-screen candidates",len(longs))
    c.metric("Accounting-risk candidates",len(shorts))
    d.metric("Data coverage",f"{len(base)/len(members):.1%}")
    t1,t2,t3=st.tabs(["Opportunity map","Ranked companies","Sector comparison"])
    with t1:
        plotted=view.dropna(subset=["GPA_sector_pct","BM_sector_pct"]).copy()
        if len(plotted):
            plotted["Screen"]=np.select([plotted.conflicting_signals,plotted.BUY_candidate,plotted.SELL_candidate],
                  ["Mixed","Long","Risk"],default="Other")
            fig=px.scatter(plotted,x="BM_sector_pct",y="GPA_sector_pct",color="Screen",
                    hover_name="ticker",hover_data=["name","sector","GPA","BM"],
                    title="Value versus profitability · sector percentiles",
                    color_discrete_map={"Long":"#18b790","Risk":"#e07870","Mixed":"#f0be65","Other":"#70859a"})
            st.plotly_chart(style_chart(fig,470),use_container_width=True)
            st.caption("Higher = greater book-value exposure or profitability relative to sector peers; not a guaranteed return.")
        else:st.info("Not enough complete peer observations for this chart.")
    with t2:
        side=st.radio("Show",["Long candidates","Accounting-risk candidates"],horizontal=True)
        if side=="Long candidates":
            table(longs,["ticker","name","sector","market_cap","GPA","BM","BUY_score"])
        else:table(shorts,["ticker","name","sector","STA","SNOA","Beneish_M","Altman_Z","SELL_flag_count"])
    with t3:
        agg=view.loc[view.industrial_applicable].groupby("sector",as_index=False).agg(
            Count=("ticker","size"),Median_GPA=("GPA","median"),Median_STA=("STA","median"))
        if len(agg):
            agg["Median_GPA"]=agg.Median_GPA*100
            fig=px.bar(agg.sort_values("Median_GPA"),x="Median_GPA",y="sector",orientation="h",
                      title="Median gross profitability (% of assets)",hover_data=["Count"],
                      color_discrete_sequence=["#22a88f"])
            st.plotly_chart(style_chart(fig,420),use_container_width=True)
        st.caption("Banks and REITs remain visible in the index, but excluded from industrial factor comparisons.")
    with st.expander("Data quality and detailed methodology"):
        issues,summary=audit_table(raw,members)
        st.write(f"Source coverage: {len(raw)}/{len(members)} securities. Source field gaps and review flags are shown below.")
        st.json(summary["required_field_coverage"])
        if not issues.empty:
            st.dataframe(issues,hide_index=True,use_container_width=True)
            st.download_button("Export full audit",issues.to_csv(index=False),"data_quality.csv","text/csv")

elif page=="Company Overview":
    ticker=choose_company(view,"overview_select")
    r=view.loc[view.ticker.eq(ticker)].iloc[0]
    st.title(f"{r['name']} ({ticker})")
    st.caption(f"{r.sector} · {r.industry} · FY {r.fiscal_year_end}")
    a,b,c,d=st.columns(4)
    a.metric("Market cap",money(n(r,"market_cap")))
    b.metric("Annual revenue",cash_label(r,"revenue"))
    c.metric("Gross profitability",pct(n(r,"GPA")))
    d.metric("Operating cash flow",cash_label(r,"operating_cash_flow"))
    t1,t2,t3=st.tabs(["Overview","Financial statements","Accounting ratios"])
    with t1:
        figures=[]
        for title,current,previous in [("Revenue","revenue","prior_revenue"),
                 ("Net income","net_income","prior_net_income"),
                 ("Operating cash flow","operating_cash_flow","prior_operating_cash_flow")]:
            for label,key in [("Previous FY",previous),("Latest FY",current)]:
                if np.isfinite(n(r,key)):figures.append(dict(Metric=title,Period=label,USD_billions=n(r,key)/1e9))
        if figures:
            fig=px.bar(pd.DataFrame(figures),x="Metric",y="USD_billions",color="Period",barmode="group",
                       title="Annual financial performance · billions in reporting currency",
                       color_discrete_sequence=["#68809a","#23b497"])
            st.plotly_chart(style_chart(fig,435),use_container_width=True)
        st.caption("These are annual financial-statement comparisons, not quarterly figures or stock-price returns.")
    with t2:
        metrics=[("Revenue","revenue"),("Cost of revenue","cost_of_revenue"),("Gross profit","gross_profit"),
                 ("Net income","net_income"),("Operating cash flow","operating_cash_flow"),
                 ("Total assets","total_assets"),("Common equity","common_equity")]
        st.dataframe(pd.DataFrame({"Financial item":[x for x,_ in metrics],
                    "Latest fiscal year":[cash_label(r,k) for _,k in metrics]}),
                    hide_index=True,use_container_width=True)
        st.caption("Yahoo Finance figures, matched annual period. Totals may have issuer-specific definitions.")
    with t3:
        one,two,three=st.columns(3)
        for i,(label,key,as_percent) in enumerate([
                ("Return on assets","ROA",True),("Operating CF / Assets","CFOA",True),
                ("Book to market","BM",False),("Accruals","STA",True),
                ("Scaled net operating assets","SNOA",True),("Beneish M-score","Beneish_M",False),
                ("Altman Z","Altman_Z",False)]):
            (one,two,three)[i%3].metric(label,pct(n(r,key)) if as_percent else
                                      f"{n(r,key):.2f}" if np.isfinite(n(r,key)) else "N/A")
        st.caption(f"Altman model: {r.Altman_model} · {r.Altman_zone}; accounting signals require filing review.")
        comp=[]
        for key in ["GPA","BM","STA","ROA","CFOA","SNOA"]:
            score=n(r,key+"_sector_pct")
            if np.isfinite(score):comp.append({"Metric":key,"Peer percentile":score})
        if comp:
            fig=px.bar(pd.DataFrame(comp),x="Metric",y="Peer percentile",
                       title="Where this company sits versus its sector peers",
                       color_discrete_sequence=["#20b99a"],text_auto=".0f")
            fig.update_yaxes(range=[0,100])
            st.plotly_chart(style_chart(fig,360),use_container_width=True)
    with st.expander("Source and methodology"):
        st.caption(f"Yahoo Finance via yfinance · FY {r.fiscal_year_end} · quote snapshot {r.price_asof} · retrieved {r.refreshed_at}")
        st.markdown(f"[Review source financial statements](https://finance.yahoo.com/quote/{ticker}/financials/)")
        st.caption("Market capitalization and latest FY figures use different dates. Do not treat the derived ratios as historical backtests.")

else:
    st.title("Investment Research")
    st.caption("Evidence-based long and bearish cases · screening candidates, not automatic trade recommendations")
    a,b=st.columns(2)
    with a:table(longs.head(8),["ticker","name","BUY_score"],"Long research shortlist")
    with b:table(shorts.head(8),["ticker","name","SELL_flag_count"],"Accounting-risk shortlist")
    ticker=choose_company(view,"research_select")
    r=view.loc[view.ticker.eq(ticker)].iloc[0]
    findings=describe(r,base)
    st.subheader(f"{r['name']} · {ticker}")
    st.markdown(f"**Research status: {findings['verdict']}**")
    st.caption(findings["reason"])
    a,b,c,d=st.columns(4)
    a.metric("Market cap / annual sales",f"{findings['ps']:.2f}x" if np.isfinite(findings["ps"]) else "N/A")
    b.metric("Sector median",f"{findings['sector_ps']:.2f}x" if np.isfinite(findings["sector_ps"]) else "N/A")
    c.metric("Earnings yield",pct(findings["earnings_yield"]))
    d.metric("Operating CF yield",pct(findings["cashflow_yield"]))
    left,right=st.columns(2)
    with left:
        st.markdown("#### What supports a long case")
        if findings["strengths"]:
            for text in findings["strengths"]:st.write("• "+text)
        else:st.caption("No quantitative strengths established.")
    with right:
        st.markdown("#### What supports a bearish case")
        if findings["risks"]:
            for text in findings["risks"]:st.write("• "+text)
        else:st.caption("No quantified accounting risks under current criteria.")
    st.subheader("Valuation sensitivity")
    st.caption("Illustrative sales multiples are editable assumptions—not forecasts or a price target.")
    low,mid,high=st.columns(3)
    with low:bear=st.number_input("Bear case P/S",min_value=0.1,value=2.0,step=0.5)
    with mid:base_multiple=st.number_input("Base case P/S",min_value=0.1,value=4.0,step=0.5)
    with high:bull=st.number_input("Bull case P/S",min_value=0.1,value=6.0,step=0.5)
    scen=scenario(r,[("Bear",bear),("Base",base_multiple),("Bull",bull)])
    chart=scen.loc[scen["Implied change"].notna()].copy()
    if len(chart):
        chart["Implied change (%)"]=chart["Implied change"]*100
        fig=px.bar(chart,x="Scenario",y="Implied change (%)",color="Scenario",
                   text_auto=".1f",title="Hypothetical market-cap change relative to today's market cap",
                   color_discrete_sequence=["#d27773","#8f9faf","#26b69b"])
        st.plotly_chart(style_chart(fig,330),use_container_width=True)
    st.caption("Revenue is annual historical data; market cap is a recent snapshot. Neither the multiples nor implied changes account for new earnings, share issuance or dilution.")
    with st.expander("What could invalidate this conclusion?",expanded=True):
        for gap in findings["gaps"][:8]:st.write("• "+gap)
    st.subheader("Finish the investment case")
    st.caption("A defensible final recommendation needs the analyst's specific thesis, catalyst, opposing argument and an actual filing reference.")
    position=st.selectbox("Your position",["Under review","Long","Short / avoid","No position"])
    thesis=st.text_area("Investment thesis: why is the market mispricing this company?",height=95)
    catalyst=st.text_input("Catalyst and expected timing")
    counter=st.text_area("Strongest counterargument",height=75)
    invalid=st.text_input("What evidence would invalidate the thesis?")
    filing=st.text_input("SEC filing or issuer annual-report URL")
    sections=dict(position=position,thesis=thesis,catalyst=catalyst,
                  counterargument=counter,invalidation=invalid,filing_reference=filing)
    require=[thesis.strip(),catalyst.strip(),counter.strip(),invalid.strip(),filing.strip()]
    if all(require):
        st.success("Working paper is ready for analyst review; verify source references before using it as a recommendation.")
    else:st.info("Research draft incomplete: supply thesis, catalyst, counterargument, invalidation and source filing.")
    assumptions={"Bear P/S":bear,"Base P/S":base_multiple,"Bull P/S":bull}
    st.download_button("Download research working paper",report(r,findings,assumptions,sections),
                       file_name=f"{ticker}_investment_research.md",mime="text/markdown")
    st.markdown(f"[Open Yahoo Finance financial statements for {ticker}](https://finance.yahoo.com/quote/{ticker}/financials/)")

st.divider()
st.caption("ACCT 3343 · Yahoo Finance financial-statement snapshot · Academic research; financial models and scenarios are not investment advice")
