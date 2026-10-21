"""S&P 500 Research Terminal: plain-language screening, company profiles and evidence-based research.

Presentation layer only. All ratios, percentiles and BUY/SELL screens come from
finance_models / advanced_models / research_workbench unchanged; ui_format.py
holds the display helpers (number formats, grades, labels, tooltips).
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from finance_models import GICS_SECTORS, ScreenSettings, prepare, classify, scope_filter
from research_workbench import describe, scenario, report, money, pct, n, change
from audit_data import audit_table
from ui_format import (METRICS, MISSING, TONE_HEX, GOOD, BAD, CAUTION, NEUTRAL, NA,
                       fmt_money, fmt_pct, fmt_multiple, fmt_number, fmt_metric, ordinal,
                       standing, grade_from_percentile, altman_status, screen_status,
                       verdict_tone, badge, tone_css, risk_checks)

ROOT=Path(__file__).resolve().parent
PAGES=["S&P 500 Screener","Company Profile","Research Desk","Glossary & Data"]
PAGE_CAPTIONS=["Find long and accounting-risk candidates","One company versus its sector",
               "Build an evidence-based case","Definitions, sources, data quality"]
ACCENT="#3E7FE6"; PRIOR="#9AA5B1"

st.set_page_config(page_title="S&P 500 | Financial Research",page_icon="📈",layout="wide",
                   initial_sidebar_state="expanded")
# Layout-only CSS: colours come from the theme so light and dark both work.
st.markdown("""<style>
.block-container{max-width:1380px;padding-top:2.2rem;padding-bottom:2.5rem}
[data-testid="stMetricValue"]{font-size:1.75rem}
[data-testid="stMetricLabel"] p{font-size:.85rem;opacity:.85}
[data-testid="stCaptionContainer"]{line-height:1.45}
.eyebrow{font-size:.78rem;letter-spacing:.08em;text-transform:uppercase;opacity:.65;margin-bottom:-.4rem}
.lede{font-size:1.02rem;line-height:1.55;margin:.2rem 0 .6rem 0}
@media (max-width:640px){.block-container{padding-left:1rem;padding-right:1rem;padding-top:1rem}
 [data-testid="stMetricValue"]{font-size:1.3rem}
 /* KPI rows: 2 x 2 grid on phones instead of four full-width stacked cards */
 [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]){flex-direction:row!important;flex-wrap:wrap!important;gap:.6rem!important}
 [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stMetric"]) > [data-testid="stColumn"]{flex:1 1 calc(50% - .6rem)!important;min-width:calc(50% - .6rem)!important;width:calc(50% - .6rem)!important}}
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

@st.cache_data(ttl=1800)
def cached_audit(raw_df,members_df):
    """Same audit_table output, cached so slider moves do not re-run the full audit."""
    return audit_table(raw_df,members_df)

SNOA_NOTE=("Net operating assets (SNOA) needs total assets, cash and total liabilities. When Yahoo omits the "
           "minority-interest or preferred-equity line, their combined value is taken from the balance sheet "
           "(assets − liabilities − common equity), which gives the identical course formula. A debt line Yahoo does "
           "not report is left out of the formula and the value is labelled \"partial inputs\".")

def altman_short(model):
    m=str(model)
    return "Z (manufacturer)" if m.startswith("Original") else "Z″ (non-manufacturer)" if m.startswith("Z double") else MISSING

# ------------------------------------------------------------------ helpers
def style_chart(fig,height=None):
    """Theme-neutral Plotly layout; Streamlit's theme supplies light/dark colours."""
    fig.update_layout(margin=dict(l=8,r=12,b=12,t=84),font_size=12,
                      title=dict(y=0.98,yanchor="top",x=0,xanchor="left",pad=dict(l=4)),
                      legend=dict(orientation="h",yanchor="bottom",y=1.0,xanchor="left",x=0,title_text=""),
                      hoverlabel=dict(font_size=12))
    if height:fig.update_layout(height=height)
    return fig

def show_chart(fig,height=None):
    st.plotly_chart(style_chart(fig,height),width="stretch",theme="streamlit",
                    config={"displayModeBar":False})

def table_height(rows,cap=560):
    return int(min(cap,40+35*max(rows,1)))

def styled(df,tone_cols):
    """Colour badge-text cells (▲/✓ green, ▼/⚠ red, ● amber); everything else uses theme text."""
    cols=[c for c in tone_cols if c in df.columns]
    return df.style.map(tone_css,subset=cols) if cols else df

def help_for(key):
    return METRICS[key].tooltip

def md(text):
    """Escape '$' so currency amounts are not rendered as LaTeX by st.markdown."""
    return str(text).replace("$","\\$")

def lede(text):
    st.markdown(f"<p class='lede'>{text}</p>",unsafe_allow_html=True)

def eyebrow(text):
    st.markdown(f"<p class='eyebrow'>{text}</p>",unsafe_allow_html=True)

def go_to(ticker,page=None):
    st.session_state["focus"]=ticker
    for k in ("overview_select","research_select"):st.session_state.pop(k,None)
    if page:st.session_state["nav"]=page

def selection_actions(event,data,key):
    """When a table row is selected, offer jumps to the profile / research pages."""
    try:rows=list(event.selection.rows)
    except AttributeError:
        try:rows=list(event["selection"]["rows"])
        except (KeyError,TypeError):rows=[]
    if not rows:
        st.caption("Tip: select a row to open that company's profile or research page.")
        return
    t=str(data.iloc[rows[0]]["Ticker"])
    a,b,_=st.columns([1,1,2])
    a.button(f"Open {t} profile →",key=key+"_p",on_click=go_to,args=(t,"Company Profile"),width="stretch")
    b.button(f"Research {t} →",key=key+"_r",on_click=go_to,args=(t,"Research Desk"),width="stretch")

def tone_counts_line(items):
    flagged=sum(i["tone"]==BAD for i in items);na=sum(i["tone"]==NA for i in items)
    available=len(items)-na
    review=sum(i["tone"]==CAUTION for i in items)
    s=f"**{flagged} of {available}** available checks flagged"
    s+=f" · {review} for review (not counted)" if review else ""
    return s+(f" · {na} not available (missing inputs)" if na else "")

# ------------------------------------------------------------------ data
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

asof=str(base.price_asof.dropna().max()) if base.price_asof.notna().any() else "n/a"
fy_min=str(base.fiscal_year_end.dropna().min()) if base.fiscal_year_end.notna().any() else "n/a"
fy_max=str(base.fiscal_year_end.dropna().max()) if base.fiscal_year_end.notna().any() else "n/a"
coverage=len(base)/len(members) if len(members) else np.nan

# ------------------------------------------------------------------ sidebar
all_sectors=sorted(base.sector.dropna().unique())
with st.sidebar:
    st.markdown("### S&P 500 Research")
    st.caption("ACCT 3343 · accounting-based equity screen")
    page=st.radio("Go to",PAGES,captions=PAGE_CAPTIONS,key="nav")
    st.divider()
    st.markdown("**Filters**")
    sectors=st.multiselect("Sectors",all_sectors,default=[],placeholder="All sectors",
                           help="Leave empty to include every sector.")
    active_sectors=sectors or all_sectors
    industries=sorted(base.loc[base.sector.isin(active_sectors),"industry"].dropna().unique())
    industry=st.selectbox("Industry",["All industries"]+industries)
    cap_min=st.number_input("Minimum market cap ($B)",min_value=0.,value=0.,step=5.,
                            help="Companies below this market capitalization are hidden.")
    with st.expander("Screen settings"):
        st.caption("**Long screen** — strong gross profitability *and* cheap book-to-market versus sector peers.")
        gpa=st.slider("Min. profitability rank (GPA percentile)",0,100,60,5,
                      help="Sector percentile of gross profit / assets a long candidate must reach.")
        bm=st.slider("Min. value rank (B/M percentile)",0,100,60,5,
                     help="Sector percentile of book equity / market cap a long candidate must reach.")
        score=st.slider("Min. long score",0,100,65,5,
                        help="Long score = weight × GPA percentile + (1 − weight) × B/M percentile.")
        w=st.slider("Profitability weight in long score",0.,1.,.6,.05)
        st.caption("**Accounting-risk screen** — independent warning checks; not the inverse of the long screen.")
        sta=st.slider("Accruals flag at percentile ≥",0,100,80,5,
                      help="Flags positive accruals (profit above operating cash) at or above this sector percentile.")
        snoa=st.slider("Net operating assets flag at percentile ≥",0,100,80,5)
        beneish=st.number_input("Beneish M-score cutoff",value=-1.78,step=.01,
                                help="M-scores above this value are flagged. −1.78 is the published default.")
        riskcount=st.slider("Checks needed to be risk-flagged",1,4,2,
                            help="How many of the 4 accounting-risk checks must trigger. Only checks with complete inputs can fire. "
                                 "Beneish index outliers and Altman results for negative-equity firms are shown for review and never count.")
    st.caption(f"Dataset: {len(base)} of {len(members)} S&P 500 listings · market caps as of {asof}")
    with st.expander("Sources and methodology"):
        st.write("Annual company statements are matched by fiscal year. Market cap is a more recent quote snapshot; ratios are cross-sectional, not point-in-time backtests.")
        st.markdown("[Financial statements at Yahoo Finance](https://finance.yahoo.com/)")
        st.markdown("[GitHub model and data audit](https://github.com/vpancholi808/financial-analysis-app)")

settings=ScreenSettings(buy_gpa_pct=gpa,buy_bm_pct=bm,buy_score_min=score,buy_gpa_weight=w,
                        sell_sta_pct=sta,sell_snoa_pct=snoa,sell_beneish_cutoff=beneish,sell_min_flags=riskcount)
selected=scope_filter(base,active_sectors,industry,cap_min*1e9)
view=classify(selected,settings)
longs=view.loc[view.BUY_candidate].sort_values("BUY_score",ascending=False)
shorts=view.loc[view.SELL_candidate].sort_values(["SELL_flag_count","STA"],ascending=[False,False])

def choose_company(df,key,label="Company"):
    if df.empty:st.info("No companies match your filters.");st.stop()
    labels={str(r.ticker):f"{r.ticker}  ·  {r['name']}" for _,r in df.iterrows()}
    options=sorted(labels)
    if key in st.session_state and st.session_state[key] not in labels:
        st.session_state.pop(key)
    focus=st.session_state.get("focus")
    index=options.index(focus) if focus in labels else 0
    def remember():st.session_state["focus"]=st.session_state[key]
    return st.selectbox(label,options,index=index,format_func=lambda s:labels[s],key=key,
                        on_change=remember,help="Type a ticker or company name to search.")

# ------------------------------------------------------------------ column configs
CC=st.column_config
def cfg_common():
    return {"Ticker":CC.TextColumn("Ticker",pinned=True,width="small"),
            "Company":CC.TextColumn("Company",pinned=True,width="medium"),
            "Sector":CC.TextColumn("Sector",width="medium"),
            "Market cap":CC.NumberColumn("Market cap",format="$%,.1fB",help="Latest market capitalization in US$ billions (quote snapshot).")}
def pct_col(key,label=None):
    return CC.NumberColumn(label or METRICS[key].label,format="%.1f%%",help=help_for(key))
def rank_col(label,help_text):
    return CC.ProgressColumn(label,min_value=0,max_value=100,format="%.0f",help=help_text,width="small")

def screen_frame(df):
    status=[screen_status(r)[0] for _,r in df.iterrows()]
    return pd.DataFrame({
        "Ticker":df.ticker.values,"Company":df["name"].values,"Sector":df.sector.values,
        "Market cap":(df.market_cap/1e9).values,"Status":status,
        "GPA":(df.GPA*100).values,"GPA rank":df.GPA_sector_pct.values,
        "B/M":df.BM.values,"B/M rank":df.BM_sector_pct.values,
        "Long score":df.BUY_score.values,
        "Accruals":(df.STA*100).values,"Risk flags":df.SELL_flag_count.values,
        "Signals fired":[", ".join(k for k,f in (("Accruals",r.STA_risk_flag),("SNOA",r.SNOA_risk_flag),
                                                 ("Beneish",r.Beneish_risk_flag),("Altman",r.Altman_risk_flag)) if bool(f)) or MISSING
                         for _,r in df.iterrows()],
        "Altman Z":df.Altman_Z.values,"Altman model":[altman_short(m) for m in df.Altman_model],
        "Distress zone":[altman_status(z)[0] for z in df.Altman_zone],
        "FY end":df.fiscal_year_end.astype(str).replace({"nan":MISSING}).values,
    })

SCREEN_CFG={**cfg_common(),
    "Status":CC.TextColumn("Screen result",help="▲ passes the long screen · ▼ risk-flagged · ◆ both · ● neither · – bank/insurer/REIT (not scored)."),
    "GPA":pct_col("GPA","Gross profitability"),
    "GPA rank":rank_col("Profitability rank","Gross profitability (GPA) percentile within its sector; 100 = most profitable."),
    "B/M":CC.NumberColumn("Book-to-market",format="%.2f",help=help_for("BM")),
    "B/M rank":rank_col("Value rank","Book-to-market percentile within its sector; 100 = cheapest relative to book value."),
    "Long score":rank_col("Long score","Weighted blend of the profitability and value ranks (0–100)."),
    "Accruals":pct_col("STA","Accruals"),
    "Risk flags":CC.NumberColumn("Risk flags",format="%d of 4",help="Accounting-risk checks triggered: accruals, net operating assets, Beneish M-score, Altman distress."),
    "Signals fired":CC.TextColumn("Signals fired",help="Which accounting-risk checks triggered for this company."),
    "Altman Z":CC.NumberColumn("Altman score",format="%.2f",help=help_for("Altman_Z")),
    "Altman model":CC.TextColumn("Altman model",help="Z (manufacturer): distress < 1.81, grey 1.81–2.99. Z″ (non-manufacturer): distress < 1.1, grey 1.1–2.6."),
    "Distress zone":CC.TextColumn("Distress zone (Altman)",help=help_for("Altman_Z")),
    "FY end":CC.TextColumn("FY end",help="Fiscal year-end of the annual statements used. Market cap is a later quote snapshot.",width="small"),
}

# ================================================================== pages
if page=="S&P 500 Screener":
    eyebrow("S&P 500 · accounting screen")
    st.title("Screener")
    n_conf=int(view.conflicting_signals.sum())
    n_spec=int((~view.industrial_applicable).sum())
    summary=(f"Of the **{len(view)}** companies in view, **{len(longs)}** pass the long screen "
             f"(strong gross profitability *and* cheap book-to-market versus sector peers) and "
             f"**{len(shorts)}** trip at least {riskcount} of 4 accounting-risk checks")
    summary+=f"; {n_conf} show both signals." if n_conf else "."
    if n_spec:summary+=f" {n_spec} banks, insurers and REITs are listed but not scored."
    n_stale=int((view.industrial_applicable & view.stale_statements).sum())
    if n_stale:summary+=f" {n_stale} with statements older than about 15 months {'is' if n_stale==1 else 'are'} not scored until the data refreshes."
    st.markdown(summary)
    a,b,c,d=st.columns(4)
    a.metric("Companies in view",f"{len(view):,}",border=True,
             help=f"{len(base)} of {len(members)} current S&P 500 listings have usable statements ({fmt_pct(coverage)} coverage). Filters in the sidebar narrow this.")
    b.metric("▲ Long candidates",len(longs),border=True,
             help=f"GPA rank ≥ {gpa}, B/M rank ≥ {bm} and long score ≥ {score}. A screen result, not a recommendation.")
    c.metric("▼ Risk-flagged",len(shorts),border=True,
             help=f"At least {riskcount} of 4 accounting-risk checks triggered. Warnings require filing review; not proof of misstatement.")
    d.metric("Data as of",asof,border=True,
             help=f"Market caps are a quote snapshot on this date. Annual statements have fiscal years ending {fy_min} to {fy_max}.")
    st.caption(f"Coverage {fmt_pct(coverage)} of current S&P 500 listings · statements: Yahoo Finance via yfinance · "
               "a screen result is a starting point for research, not investment advice.")

    t1,t2,t3,t4,t5=st.tabs(["▲ Long candidates","▼ Risk flags","Opportunity map","Sector medians","All companies"])
    with t1:
        st.markdown("Ranked by **long score**. Bars show where each company sits **within its own sector** (100 = best).")
        if longs.empty:st.info("No qualifying companies. Adjust the filters or screen settings.")
        else:
            data=screen_frame(longs)[["Ticker","Company","Sector","Market cap","Long score","GPA rank","GPA","B/M rank","B/M","Risk flags","FY end"]]
            ev=st.dataframe(data,column_config=SCREEN_CFG,hide_index=True,width="stretch",
                            height=table_height(len(data)),placeholder=MISSING,
                            on_select="rerun",selection_mode="single-row",key="tbl_long")
            selection_actions(ev,data,"tbl_long")
    with t2:
        st.markdown("Each check is a separate warning sign. **⚠ Flag** = triggered, **✓ Clear** = passed, **–** = input not reported.")
        if shorts.empty:st.info("No qualifying companies. Adjust the filters or screen settings.")
        else:
            rows=[]
            for _,r in shorts.iterrows():
                ch=risk_checks(r,settings)
                rows.append({"Ticker":r.ticker,"Company":r["name"],"Sector":r.sector,"Risk flags":r.SELL_flag_count,
                             "Accruals check":ch[0]["label"],"Net op. assets check":ch[1]["label"],
                             "Manipulation check":ch[2]["label"],"Distress check":ch[3]["label"],
                             "Accruals":n(r,"STA")*100,"SNOA value":n(r,"SNOA")*100,"M-score":n(r,"Beneish_M"),
                             "Altman Z":n(r,"Altman_Z"),"Altman model":altman_short(r.Altman_model),
                             "Distress zone":altman_status(r.Altman_zone)[0],"FY end":str(r.fiscal_year_end)})
            data=pd.DataFrame(rows)
            st.caption(SNOA_NOTE)
            cfg={**SCREEN_CFG,
                 "Accruals check":CC.TextColumn("Accruals",help=help_for("STA")),
                 "Net op. assets check":CC.TextColumn("Net op. assets",help=help_for("SNOA")),
                 "Manipulation check":CC.TextColumn("Beneish M",help=help_for("Beneish_M")),
                 "Distress check":CC.TextColumn("Altman distress",help=help_for("Altman_Z")),
                 "Accruals":pct_col("STA","Accruals value"),
                 "SNOA value":pct_col("SNOA","Net op. assets value"),
                 "M-score":CC.NumberColumn("M-score",format="%.2f",help=help_for("Beneish_M"))}
            ev=st.dataframe(styled(data,["Accruals check","Net op. assets check","Manipulation check","Distress check","Distress zone"]),
                            column_config=cfg,hide_index=True,width="stretch",
                            height=table_height(len(data)),placeholder=MISSING,
                            on_select="rerun",selection_mode="single-row",key="tbl_risk")
            selection_actions(ev,data,"tbl_risk")
    with t3:
        plotted=view.dropna(subset=["GPA_sector_pct","BM_sector_pct"]).copy()
        if len(plotted):
            plotted["Screen"]=np.select([plotted.conflicting_signals,plotted.BUY_candidate,plotted.SELL_candidate],
                  ["Mixed signals","Long candidate","Risk-flagged"],default="No signal")
            plotted["GPA (%)"]=plotted.GPA*100
            order=["Long candidate","Risk-flagged","Mixed signals","No signal"]
            fig=px.scatter(plotted,x="BM_sector_pct",y="GPA_sector_pct",color="Screen",symbol="Screen",
                    category_orders={"Screen":order},
                    color_discrete_map={"Long candidate":TONE_HEX[GOOD],"Risk-flagged":TONE_HEX[BAD],
                                        "Mixed signals":TONE_HEX[CAUTION],"No signal":TONE_HEX[NEUTRAL]},
                    symbol_map={"Long candidate":"triangle-up","Risk-flagged":"triangle-down",
                                "Mixed signals":"diamond","No signal":"circle"},
                    hover_name="name",custom_data=["ticker","sector","GPA (%)","BM"],
                    title="Value versus profitability, ranked within each sector",
                    labels={"BM_sector_pct":"Value rank in sector (book-to-market percentile) →",
                            "GPA_sector_pct":"Profitability rank in sector (GPA percentile) →"})
            fig.update_traces(marker=dict(size=9,line=dict(width=0)),
                hovertemplate="<b>%{hovertext}</b> (%{customdata[0]})<br>%{customdata[1]}<br>"
                              "Profitability rank %{y:.0f} · GPA %{customdata[2]:.1f}%<br>"
                              "Value rank %{x:.0f} · B/M %{customdata[3]:.2f}<extra></extra>")
            for tr in fig.data:
                if tr.name=="No signal":tr.marker.opacity=.35
            fig.add_vline(x=bm,line_dash="dot",line_width=1,line_color=PRIOR)
            fig.add_hline(y=gpa,line_dash="dot",line_width=1,line_color=PRIOR)
            fig.add_annotation(x=100,y=100,xanchor="right",yanchor="bottom",showarrow=False,
                               text="Profitable and cheap vs peers",font=dict(size=11,color=TONE_HEX[GOOD]))
            fig.update_xaxes(range=[-2,103]);fig.update_yaxes(range=[-2,106])
            show_chart(fig,500)
            st.caption(f"Dotted lines mark the current thresholds (value rank ≥ {bm}, profitability rank ≥ {gpa}); long candidates "
                       f"must also reach a long score of {score}. Higher = greater book value or profitability relative to sector peers; not a guaranteed return.")
        else:st.info("Not enough complete peer observations for this chart.")
    with t4:
        ind=view.loc[view.industrial_applicable]
        if len(ind):
            agg=ind.groupby("sector",as_index=False).agg(
                Companies=("ticker","size"),Median_GPA=("GPA","median"),Median_BM=("BM","median"),
                Median_ROA=("ROA","median"),Median_STA=("STA","median"),
                Long=("BUY_candidate","sum"),Risk=("SELL_candidate","sum"))
            agg["Median_GPA"]=agg.Median_GPA*100
            fig=px.bar(agg.sort_values("Median_GPA"),x="Median_GPA",y="sector",orientation="h",
                      title="Median gross profitability by sector (% of assets)",
                      custom_data=["Companies"],text="Median_GPA",
                      labels={"Median_GPA":"Median gross profit / assets (%)","sector":""},
                      color_discrete_sequence=[ACCENT])
            fig.update_traces(texttemplate="%{text:.1f}%",textposition="outside",cliponaxis=False,
                              hovertemplate="<b>%{y}</b><br>Median GPA %{x:.1f}%<br>%{customdata[0]} companies<extra></extra>")
            show_chart(fig,420)
            tbl=agg.sort_values("Median_GPA",ascending=False).rename(columns={"sector":"Sector"})
            tbl["Median_ROA"]=tbl.Median_ROA*100;tbl["Median_STA"]=tbl.Median_STA*100
            st.dataframe(tbl,hide_index=True,width="stretch",placeholder=MISSING,column_config={
                "Sector":CC.TextColumn("Sector",pinned=True),
                "Companies":CC.NumberColumn("Companies",format="%d"),
                "Median_GPA":pct_col("GPA","Median gross profitability"),
                "Median_BM":CC.NumberColumn("Median book-to-market",format="%.2f",help=help_for("BM")),
                "Median_ROA":pct_col("ROA","Median return on assets"),
                "Median_STA":pct_col("STA","Median accruals"),
                "Long":CC.NumberColumn("▲ Long candidates",format="%d"),
                "Risk":CC.NumberColumn("▼ Risk-flagged",format="%d")})
        st.caption("Banks, insurers and REITs remain visible in the index, but are excluded from industrial factor comparisons.")
    with t5:
        data=screen_frame(view.sort_values("market_cap",ascending=False))
        st.markdown(f"All **{len(data)}** companies in view, largest first. Click a column header to sort.")
        ev=st.dataframe(styled(data,["Status","Distress zone"]),column_config=SCREEN_CFG,hide_index=True,
                        width="stretch",height=560,placeholder=MISSING,
                        on_select="rerun",selection_mode="single-row",key="tbl_all")
        selection_actions(ev,data,"tbl_all")
        export=view[["ticker","name","sector","industry","market_cap","GPA","GPA_sector_pct","BM","BM_sector_pct",
                     "BUY_score","BUY_candidate","STA","SNOA","Beneish_M","Altman_Z","Altman_zone",
                     "SELL_flag_count","SELL_candidate","conflicting_signals"]]
        st.download_button("Download this screen (CSV)",export.to_csv(index=False),"sp500_screen.csv","text/csv")
    st.caption("Definitions, formulas and the data-quality audit are on the **Glossary & Data** page.")

elif page=="Company Profile":
    pick,_=st.columns([3,2])
    with pick:ticker=choose_company(view,"overview_select")
    r=view.loc[view.ticker.eq(ticker)].iloc[0]
    cur=str(r.get("financial_currency") or "USD")
    eyebrow(f"{r.sector} · {r.industry}")
    st.title(f"{r['name']} ({ticker})")
    status,tone=screen_status(r)
    z_label,z_tone=altman_status(r.Altman_zone)
    if bool(r.get("stale_statements",False)) and bool(r.industrial_applicable):
        z_label,z_tone="– not scored (stale data)",NA
    st.markdown(f"{badge(status,tone)} &nbsp; {badge('Altman: '+z_label,z_tone)} &nbsp; "
                f"{badge('FY ending '+str(r.fiscal_year_end),NA)}")
    rev_g=change(n(r,"revenue"),n(r,"prior_revenue"))
    ni_g=change(n(r,"net_income"),n(r,"prior_net_income"))
    cf_g=change(n(r,"operating_cash_flow"),n(r,"prior_operating_cash_flow"))
    yoy=lambda g:(fmt_pct(g,signed=True)+" vs prior FY") if np.isfinite(g) else None
    a,b,c,d=st.columns(4)
    a.metric("Market cap",fmt_money(n(r,"market_cap")),border=True,help=f"Quote snapshot {r.price_asof}.")
    b.metric("Revenue",fmt_money(n(r,"revenue"),cur),yoy(rev_g),border=True,help="Latest fiscal-year revenue; change vs the matched prior fiscal year.")
    c.metric("Net income",fmt_money(n(r,"net_income"),cur),yoy(ni_g),border=True,help="Change shown only when the prior-year figure is positive.")
    d.metric("Operating cash flow",fmt_money(n(r,"operating_cash_flow"),cur),yoy(cf_g),border=True)

    st.subheader("Scorecard versus sector peers")
    if not bool(r.industrial_applicable):
        st.info("Banks, insurers and REITs need specialist models: sector ranks, grades and accounting-risk checks are not assigned. Company ratios are still shown.")
    elif bool(r.get("stale_statements",False)):
        st.warning(f"The latest annual statements (FY ending {r.fiscal_year_end}) are {int(r.statement_age_days)} days older than the market-cap quote, "
                   "so this company is not ranked or screened until its data refreshes. Company ratios are still shown.")
    if str(r.get("share_class_of","")):
        st.caption(f"Share class of {r.share_class_of}: the issuer is counted once in sector benchmarks.")
    rows=[]
    for key in ["GPA","BM","ROA","CFOA","STA","SNOA","Beneish_M"]:
        m=METRICS[key];p=n(r,key+"_sector_pct")
        g,_=grade_from_percentile(p,m.higher_is_better)
        if key=="Beneish_M" and bool(r.get("Beneish_outlier",False)):
            g="● Review"
        rows.append({"Metric":m.label,"Company":fmt_metric(key,n(r,key)),
                     "Sector median":fmt_metric(key,n(r,key+"_sector_median")),
                     "Standing":standing(p,m.higher_is_better),"Grade":g,
                     "Better when":"↑ Higher" if m.higher_is_better else "↓ Lower",
                     "Raw percentile":ordinal(p)})
    # Altman is judged by absolute zones (not sector rank); show score, model and zone together.
    z_txt=fmt_number(n(r,"Altman_Z"))
    if np.isfinite(n(r,"Altman_Z")):z_txt+=" (Z)" if str(r.Altman_model).startswith("Original") else " (Z″)"
    rows.append({"Metric":METRICS["Altman_Z"].label,"Company":z_txt,"Sector median":MISSING,
                 "Standing":np.nan,"Grade":altman_status(r.Altman_zone)[0].split(":")[0],"Better when":"↑ Higher",
                 "Raw percentile":"zone-based"})
    card=pd.DataFrame(rows)
    st.dataframe(styled(card,["Grade"]),hide_index=True,width="stretch",placeholder=MISSING,
        column_config={"Metric":CC.TextColumn("Metric",width=230),
            "Company":CC.TextColumn(ticker,help="This company's latest annual value."),
            "Sector median":CC.TextColumn("Sector median",help=f"Median across scored {r.sector} companies in the full S&P 500 dataset."),
            "Standing":CC.ProgressColumn("Standing in sector",min_value=0,max_value=100,format="%.0f",width="medium",help="Direction-adjusted sector percentile: 100 = most favourable. For accruals, net operating assets and the M-score, lower raw values rank higher."),
            "Grade":CC.TextColumn("Grade",help="▲ Strong = top third of sector · ● Neutral = middle third · ▼ Weak = bottom third (direction-adjusted). Altman uses its published zones: Z distress < 1.81, grey 1.81–2.99; Z″ distress < 1.1, grey 1.1–2.6."),
            "Better when":CC.TextColumn("Better when",width="small"),
            "Raw percentile":CC.TextColumn("Raw percentile",help="Unadjusted sector percentile of the metric value.",width="small")})
    st.caption(f"Grades for accruals, net operating assets and the M-score are inverted so ▲ Strong always means favourable "
               f"(lower raw value than peers). Beneish screen cutoff: {beneish:.2f}. — = not available.")
    if pd.isna(r.SNOA):st.caption(SNOA_NOTE)
    elif str(r.get("SNOA_basis","")).startswith("Partial"):
        st.caption(f"Net operating assets (SNOA) uses partial inputs: {str(r.SNOA_basis).split(': ',1)[1]}.")
    with st.expander("What do these metrics mean?"):
        for key in ["GPA","BM","ROA","CFOA","STA","SNOA","Beneish_M","Altman_Z"]:
            m=METRICS[key];st.markdown(f"**{m.label}** — {m.help} *{m.direction}.*")

    st.subheader("Accounting-risk checks")
    checks=risk_checks(r,settings)
    st.markdown(tone_counts_line(checks)+(f" · risk-flagged when ≥ {riskcount}" ))
    cols=st.columns(4)
    for col,chk in zip(cols,checks):
        with col.container(border=True):
            st.markdown(f"{badge(chk['label'],chk['tone'])}")
            st.markdown(f"**{chk['check']}**")
            st.caption(md(chk["detail"]))

    t1,t2=st.tabs(["Statement trends","Financial statements"])
    with t1:
        def trend(items,title):
            figs=[]
            for label,cur_k,prev_k in items:
                cv,pv=n(r,cur_k),n(r,prev_k)
                g=change(cv,pv)
                x=label+(f"<br><sup>{fmt_pct(g,signed=True)} YoY</sup>" if np.isfinite(g) else "")
                if np.isfinite(pv):figs.append(dict(Item=x,Period=f"Prior FY ({r.prior_fiscal_year_end})" if pd.notna(r.prior_fiscal_year_end) else "Prior FY",B=pv/1e9))
                if np.isfinite(cv):figs.append(dict(Item=x,Period=f"Latest FY ({r.fiscal_year_end})",B=cv/1e9))
            if not figs:
                st.info("No matched annual figures available.");return
            df=pd.DataFrame(figs)
            fig=px.bar(df,x="Item",y="B",color="Period",barmode="group",title=title,text="B",
                       labels={"B":f"{cur} billions","Item":""},
                       color_discrete_sequence=[PRIOR,ACCENT])
            fig.update_traces(texttemplate="%{text:,.1f}",textposition="outside",cliponaxis=False,
                              hovertemplate="%{x}<br>%{fullData.name}: %{y:,.2f}B<extra></extra>")
            show_chart(fig,380)
        left,right=st.columns(2)
        with left:trend([("Revenue","revenue","prior_revenue"),("Net income","net_income","prior_net_income"),
                         ("Operating cash flow","operating_cash_flow","prior_operating_cash_flow")],
                        "Income and cash flow")
        with right:trend([("Total assets","total_assets","prior_total_assets"),
                          ("Total liabilities","total_liabilities","prior_total_liabilities"),
                          ("Common equity","common_equity","prior_common_equity")],"Balance sheet")
        st.caption("Annual financial-statement comparisons, not quarterly figures or stock-price returns. Missing years are omitted, never drawn as zero.")
    with t2:
        items=[("Income statement","Revenue","revenue","prior_revenue"),
               ("Income statement","Cost of revenue","cost_of_revenue","prior_cost_of_revenue"),
               ("Income statement","Gross profit","gross_profit",None),
               ("Income statement","EBIT","ebit",None),
               ("Income statement","SG&A expense","sga","prior_sga"),
               ("Income statement","Depreciation","depreciation","prior_depreciation"),
               ("Income statement","Net income","net_income","prior_net_income"),
               ("Cash flow","Operating cash flow","operating_cash_flow","prior_operating_cash_flow"),
               ("Balance sheet","Total assets","total_assets","prior_total_assets"),
               ("Balance sheet","Cash and equivalents","cash_and_equivalents",None),
               ("Balance sheet","Accounts receivable","accounts_receivable","prior_receivables"),
               ("Balance sheet","Current assets","current_assets","prior_current_assets"),
               ("Balance sheet","Net PP&E","net_ppe","prior_net_ppe"),
               ("Balance sheet","Current liabilities","current_liabilities","prior_current_liabilities"),
               ("Balance sheet","Total liabilities","total_liabilities","prior_total_liabilities"),
               ("Balance sheet","Total debt","total_debt","prior_total_debt"),
               ("Balance sheet","Retained earnings","retained_earnings",None),
               ("Balance sheet","Common equity","common_equity","prior_common_equity")]
        latest_lbl=f"Latest FY ({r.fiscal_year_end})";prior_lbl=f"Prior FY ({r.prior_fiscal_year_end})" if pd.notna(r.prior_fiscal_year_end) else "Prior FY"
        st.dataframe(pd.DataFrame([{"Statement":s,"Item":label,latest_lbl:fmt_money(n(r,k),cur),
                                    prior_lbl:fmt_money(n(r,p),cur) if p else MISSING,
                                    "Change":fmt_pct(change(n(r,k),n(r,p)),signed=True) if p else MISSING}
                                   for s,label,k,p in items]),
                     hide_index=True,width="stretch",height=table_height(len(items),700),
                     column_config={"Item":CC.TextColumn("Item",pinned=True)})
        st.caption(f"Yahoo Finance figures, matched annual period. Totals may have issuer-specific definitions. "
                   f"Gross profit basis: {r.gross_profit_basis}. — = not reported, not comparable, or prior year not stored.")
    with st.expander("Source and methodology"):
        st.caption(f"Yahoo Finance via yfinance · FY {r.fiscal_year_end} · quote snapshot {r.price_asof} · retrieved {r.refreshed_at}")
        st.markdown(f"[Review source financial statements](https://finance.yahoo.com/quote/{ticker}/financials/)")
        st.caption("Market capitalization and latest FY figures use different dates. Do not treat the derived ratios as historical backtests.")

elif page=="Research Desk":
    eyebrow("Long and bearish cases")
    st.title("Research Desk")
    st.caption("Evidence-based long and bearish cases · screening candidates, not automatic trade recommendations")
    def shortlist(df,cols,cfg,title,key,empty):
        st.markdown(f"**{title}**")
        if df.empty:st.info(empty);return
        data=df.head(8)
        tickers=list(data.Ticker)
        def pick():
            try:rows=st.session_state[key].selection.rows
            except (KeyError,AttributeError):rows=[]
            if rows:go_to(tickers[rows[0]])
        st.dataframe(data[cols],column_config=cfg,hide_index=True,width="stretch",
                     height=table_height(len(data)),placeholder=MISSING,
                     on_select=pick,selection_mode="single-row",key=key)
    a,b=st.columns(2)
    with a:shortlist(screen_frame(longs),["Ticker","Company","Long score"],SCREEN_CFG,"▲ Long research shortlist","rs_long",
                     "No qualifying companies. Adjust the filters.")
    with b:shortlist(screen_frame(shorts),["Ticker","Company","Risk flags"],SCREEN_CFG,"▼ Accounting-risk shortlist","rs_short",
                     "No qualifying companies. Adjust the filters.")
    st.caption("Select a row to analyse that company below, or search for any company.")
    st.divider()
    ticker=choose_company(view,"research_select","Company to research")
    r=view.loc[view.ticker.eq(ticker)].iloc[0]
    findings=describe(r,base)
    st.subheader(f"{r['name']} · {ticker}")
    st.markdown(f"Research status: {badge(findings['verdict'],verdict_tone(findings['verdict']))}")
    st.caption(md(findings["reason"]))
    ps,sps=findings["ps"],findings["sector_ps"]
    vs=None
    if np.isfinite(ps) and np.isfinite(sps) and sps>0:
        vs=f"{fmt_pct(ps/sps-1,0,signed=True)} vs sector"
    a,b,c,d=st.columns(4)
    a.metric("Price / sales",fmt_multiple(ps),vs,delta_color="inverse",border=True,
             help="Current market cap ÷ latest annual revenue. Lower = cheaper per dollar of sales.")
    b.metric("Sector median P/S",fmt_multiple(sps),border=True,
             help=f"Median across {findings['peers']} comparable {r.sector} companies.")
    c.metric("Earnings yield",fmt_pct(findings["earnings_yield"]),border=True,
             help="Annual net income ÷ current market cap. Higher = more earnings per dollar invested. Mixed dates: historical FY earnings vs today's market cap.")
    d.metric("Cash-flow yield",fmt_pct(findings["cashflow_yield"]),border=True,
             help="Annual operating cash flow ÷ current market cap. Mixed dates: historical FY cash flow vs today's market cap.")
    left,right=st.columns(2)
    with left.container(border=True):
        st.markdown("#### ▲ What supports a long case")
        if findings["strengths"]:st.markdown(md("\n".join("- "+t for t in findings["strengths"])))
        else:st.caption("No quantitative strengths established.")
    with right.container(border=True):
        st.markdown("#### ▼ What supports a bearish case")
        if findings["risks"]:st.markdown(md("\n".join("- "+t for t in findings["risks"])))
        else:st.caption("No quantified accounting risks under current criteria.")
    with st.expander("What could invalidate this conclusion?",expanded=True):
        st.markdown(md("\n".join("- "+g for g in findings["gaps"][:8])))

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
        colors=[TONE_HEX[GOOD] if v>=0 else TONE_HEX[BAD] for v in chart["Implied change (%)"]]
        fig=go.Figure(go.Bar(x=chart.Scenario,y=chart["Implied change (%)"],marker_color=colors,
                             text=[fmt_pct(v,1,signed=True) for v in chart["Implied change"]],textposition="outside",
                             cliponaxis=False,customdata=chart["Sales multiple"],
                             hovertemplate="%{x} case · %{customdata:.1f}× sales<br>%{y:+.1f}% vs today's market cap<extra></extra>"))
        fig.add_hline(y=0,line_width=1,line_color=PRIOR)
        fig.update_layout(title="Implied market-cap change versus today",yaxis_title="Change vs current market cap (%)",showlegend=False)
        c1,c2=st.columns([3,2])
        with c1:show_chart(fig,330)
        with c2:
            st.markdown(" ")
            st.dataframe(pd.DataFrame({"Scenario":scen.Scenario,"P/S":[fmt_multiple(v) for v in scen["Sales multiple"]],
                                       "Implied market cap":[fmt_money(v) for v in scen["Implied market cap"]],
                                       "vs today":[fmt_pct(v,1,signed=True) for v in scen["Implied change"]]}),
                         hide_index=True,width="stretch")
            st.caption(md(f"Today's market cap: {fmt_money(n(r,'market_cap'))} · latest annual revenue: {fmt_money(n(r,'revenue'))} · "
                          f"current P/S {fmt_multiple(ps)} (sector median {fmt_multiple(sps)}). A scenario at the current P/S implies 0% change."))
    st.caption("Revenue is annual historical data; market cap is a recent snapshot. Neither the multiples nor implied changes account for new earnings, share issuance or dilution.")

    st.subheader("Finish the investment case")
    st.caption("A defensible final recommendation needs the analyst's specific thesis, catalyst, opposing argument and an actual filing reference.")
    # Drafts are stored per company, so text written for one ticker can never
    # appear in (or be exported with) another company's working paper.
    drafts=st.session_state.setdefault("research_drafts",{}).setdefault(ticker,{})
    positions=["Under review","Long","Short / avoid","No position"]
    def draft_field(widget,field,label,**kw):
        v=widget(label,value=drafts.get(field,""),key=f"rd_{field}_{ticker}",**kw)
        drafts[field]=v
        return v
    with st.container(border=True):
        pos_default=positions.index(drafts["position"]) if drafts.get("position") in positions else 0
        position=st.selectbox("Your position",positions,index=pos_default,key=f"rd_position_{ticker}")
        drafts["position"]=position
        thesis=draft_field(st.text_area,"thesis","Investment thesis: why is the market mispricing this company?",height=95)
        catalyst=draft_field(st.text_input,"catalyst","Catalyst and expected timing")
        counter=draft_field(st.text_area,"counter","Strongest counterargument",height=75)
        invalid=draft_field(st.text_input,"invalid","What evidence would invalidate the thesis?")
        filing=draft_field(st.text_input,"filing","SEC filing or issuer annual-report URL")
    sections=dict(position=position,thesis=thesis,catalyst=catalyst,
                  counterargument=counter,invalidation=invalid,filing_reference=filing)
    require=[thesis.strip(),catalyst.strip(),counter.strip(),invalid.strip(),filing.strip()]
    done=sum(bool(x) for x in require)
    st.progress(done/len(require),text=f"{done} of {len(require)} required sections complete")
    if all(require):
        st.success("Working paper is ready for analyst review; verify source references before using it as a recommendation.")
    else:st.info("Research draft incomplete: supply thesis, catalyst, counterargument, invalidation and source filing.")
    assumptions={"Bear P/S":bear,"Base P/S":base_multiple,"Bull P/S":bull}
    st.download_button("Download research working paper",report(r,findings,assumptions,sections),
                       file_name=f"{ticker}_investment_research.md",mime="text/markdown",type="primary")
    st.markdown(f"[Open Yahoo Finance financial statements for {ticker}](https://finance.yahoo.com/quote/{ticker}/financials/)")

else:
    eyebrow("Reference")
    st.title("Glossary & Data")
    lede("How each number is defined, how grades and colours are assigned, where the data comes from, and what is missing.")
    t1,t2,t3=st.tabs(["Metrics","Grades & screens","Data quality"])
    with t1:
        metric_list=list(METRICS.values())
        for i in range(0,len(metric_list),2):
            cols=st.columns(2)
            for col,m in zip(cols,metric_list[i:i+2]):
                with col.container(border=True):
                    arrow="↑ Higher is better" if m.higher_is_better else "↓ Lower is better"
                    st.markdown(f"**{m.name}** · `{m.acronym}` &nbsp; {badge(arrow,NA)}")
                    st.markdown(m.help)
                    st.caption(f"Formula: {m.formula}")
        st.caption("Altman uses the original Z for public manufacturers (distress < 1.81, grey 1.81–2.99) and Z″ for other operating companies "
                   "(distress < 1.1, grey 1.1–2.6). Beneish is a screening score, not a calibrated probability of fraud. "
                   "Financials and Real Estate are excluded from the industrial models.")
    with t2:
        st.markdown("**Colours always mean the same thing**")
        st.markdown(f"{badge('▲ Strong / ✓ Clear / Safe','good')} favourable &nbsp; "
                    f"{badge('● Neutral / Grey zone / Mixed','caution')} caution &nbsp; "
                    f"{badge('▼ Weak / ⚠ Flag / Distress','bad')} unfavourable &nbsp; "
                    f"{badge('– n/a','na')} not available or not applicable")
        st.markdown(f"""
- **Sector grades** use each company's percentile *within its own sector* (minimum 10 peers), adjusted so that 100 is always best:
  top third = ▲ Strong, middle third = ● Neutral, bottom third = ▼ Weak.
- **Long candidate** (current settings): profitability rank ≥ {gpa}, value rank ≥ {bm} and long score ≥ {score},
  where long score = {w:.2f} × profitability rank + {1-w:.2f} × value rank.
- **Risk-flagged**: at least {riskcount} of 4 independent checks trigger — accruals positive and ≥ {sta}th sector percentile;
  net operating assets ≥ {snoa}th percentile; Beneish M-score above {beneish:.2f}; Altman distress zone.
- **Review, not a flag** (● Review): a Beneish score driven by an extreme index (one of the eight indices outside its plausible
  band, usually a near-zero prior-year figure), or an Altman Z″ score for a company with negative book equity. Altman is not
  applied to regulated utilities, and the manufacturer model (Z) is chosen from an explicit list of GICS sub-industries.
- **Not scored**: banks, insurers and REITs (specialist models needed) and companies whose latest statements are more than
  about 15 months older than the market-cap quote. Dual share classes (e.g. GOOG/GOOGL) count once in sector benchmarks.
- **Mixed signals**: a company passing both screens; resolve qualitatively.
- A screen result is a starting point for research. It is **not** a recommendation to buy or sell.""")
    with t3:
        st.markdown("#### Data quality and detailed methodology")
        issues,summary=cached_audit(raw,members)
        st.write(f"Source coverage: {len(raw)}/{len(members)} securities. Source field gaps and review flags are shown below.")
        st.markdown("**Model coverage** — companies with every input a model needs (missing inputs stay missing):")
        mc=pd.DataFrame([{"Model":METRICS[k].label,"Available":int(base[k].notna().sum()),"Of":len(base)} for k in
                         ["GPA","BM","STA","SNOA","Beneish_M","Altman_Z"]])
        mc["Coverage"]=100*mc.Available/mc.Of
        st.dataframe(mc,hide_index=True,width="stretch",column_config={
            "Coverage":CC.ProgressColumn("Coverage",min_value=0,max_value=100,format="%.0f%%")})
        st.caption(SNOA_NOTE)
        st.markdown("**Source field coverage**")
        cov=pd.DataFrame([{"Field":k,"Available":v["valid"],"Missing":v["missing"],
                           "Coverage":100*v["valid"]/max(v["valid"]+v["missing"],1)}
                          for k,v in summary["required_field_coverage"].items()])
        st.dataframe(cov,hide_index=True,width="stretch",column_config={
            "Coverage":CC.ProgressColumn("Coverage",min_value=0,max_value=100,format="%.0f%%")})
        a,b,c,d=st.columns(4)
        a.metric("Critical",summary["critical"],border=True);b.metric("Warnings",summary["warnings"],border=True)
        c.metric("Review",summary["review"],border=True);d.metric("Missing fields",summary["missing"],border=True)
        if not issues.empty:
            st.dataframe(issues,hide_index=True,width="stretch",height=360)
            st.download_button("Export full audit",issues.to_csv(index=False),"data_quality.csv","text/csv")
        with st.expander("Known limitations"):
            st.markdown("\n".join("- "+x for x in summary["limitations"]))
        st.markdown("[Financial statements at Yahoo Finance](https://finance.yahoo.com/) · "
                    "[GitHub model and data audit](https://github.com/vpancholi808/financial-analysis-app)")

st.divider()
st.caption("ACCT 3343 · Yahoo Finance financial-statement snapshot · Academic research; financial models and scenarios are not investment advice")
