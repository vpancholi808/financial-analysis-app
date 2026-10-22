"""S&P 500 Research Terminal: plain-language screening, company profiles and evidence-based research.

Presentation layer only. All ratios, percentiles and BUY/SELL screens come from
finance_models / advanced_models / research_workbench unchanged; ui_format.py
holds the display helpers (number formats, grades, labels, tooltips).
"""
from pathlib import Path
import json
import hashlib
import importlib
import os
import sys
import time

# Streamlit keeps imported modules in memory between runs. When new code is
# deployed (e.g. a merge that adds a function), app.py is re-read but a helper
# module may not be, which raises ImportError on the live site. Reload any
# local module whose file changed since it was loaded, dependencies first.
for _name in ("advanced_models","finance_models","ui_format","research_workbench","presentation","audit_data"):
    _mod=sys.modules.get(_name)
    if _mod is None:continue
    try:
        if os.path.getmtime(_mod.__file__)>getattr(_mod,"_loaded_at",0):
            importlib.reload(_mod);_mod._loaded_at=time.time()
    except (OSError,ImportError,TypeError):pass
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from finance_models import GICS_SECTORS, ScreenSettings, prepare, classify, scope_filter
from research_workbench import describe, scenario, report, money, pct, n, change, investment_case, recommendation_table
from audit_data import audit_table
import presentation as pr
for _name in ("advanced_models","finance_models","ui_format","research_workbench","presentation","audit_data"):
    if _name in sys.modules and not hasattr(sys.modules[_name],"_loaded_at"):
        sys.modules[_name]._loaded_at=time.time()
from ui_format import (METRICS, MISSING, TONE_HEX, GOOD, BAD, CAUTION, NEUTRAL, NA,
                       fmt_money, fmt_pct, fmt_multiple, fmt_number, fmt_metric, ordinal,
                       standing, grade_from_percentile, altman_status, screen_status,
                       verdict_tone, badge, tone_css, risk_checks)

ROOT=Path(__file__).resolve().parent
PAGES=["S&P 500 Screener","Company Profile","Research Desk","Glossary & Data"]
PAGE_CAPTIONS=["Buy and avoid ideas, every company, sectors, map","One company, answer first",
               "Build an evidence-based case","Definitions, sources, data quality"]
ACCENT="#3E7FE6"; PRIOR="#9AA5B1"

st.set_page_config(page_title="S&P 500 | Financial Research",page_icon="📈",layout="wide",
                   initial_sidebar_state="auto")
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
    st.markdown("**Show me**")
    sectors=st.multiselect("Sectors",all_sectors,default=[],placeholder="All sectors",
                           help="Leave empty to include every sector.")
    active_sectors=sectors or all_sectors
    industries=sorted(base.loc[base.sector.isin(active_sectors),"industry"].dropna().unique())
    industry=st.selectbox("Industry",["All industries"]+industries)
    cap_min=st.number_input("Smallest company to include (market value, $B)",min_value=0.,value=0.,step=5.,
                            help="Companies with a smaller market value are hidden.")
    st.markdown("**How strict is the screen?**")
    gpa_top=st.slider("A buy must be in its sector's top … for profitability",5,100,40,5,format="%d%%",
                      help="Gross profit per $1 of assets, ranked within the sector. 40% = the company must beat 60% of its sector peers.")
    bm_top=st.slider("… and in the top … for cheapness",5,100,40,5,format="%d%%",
                     help="Book value against market value, ranked within the sector. 40% = cheaper than 60% of its sector peers.")
    riskcount=st.slider("Avoid when at least this many warning signs show",1,4,2,
                        help="Out of 4 independent accounting checks: profit ahead of cash, heavy balance sheet, manipulation score, distress score. "
                             "Checks that can't be computed, or that are under review, never count.")
    with st.expander("Advanced settings"):
        score=st.slider("Minimum combined score for a buy",0,100,65,5,
                        help="Combined score = weight × profitability rank + (1 − weight) × value rank, on a 0–100 scale.")
        w=st.slider("Weight of profitability in the combined score",0.,1.,.6,.05)
        sta_worst=st.slider("Profit-ahead-of-cash warning: sector's worst …",5,100,20,5,format="%d%%",
                            help="Flags positive accruals (profit above operating cash flow) in this share of the sector with the highest accruals.")
        snoa_worst=st.slider("Heavy-balance-sheet warning: sector's heaviest …",5,100,20,5,format="%d%%",
                             help="Flags net operating assets in this share of the sector with the heaviest balance sheets.")
        beneish=st.number_input("Manipulation-score warning line (Beneish M)",value=-1.78,step=.01,
                                help="Scores above this line are flagged. −1.78 is the published default; it is a warning, not proof of fraud.")
    gpa=100-gpa_top;bm=100-bm_top;sta=100-sta_worst;snoa=100-snoa_worst
    st.caption(f"{len(base)} of {len(members)} S&P 500 companies · market values from {asof}")
    with st.expander("Sources and methodology"):
        st.write("Annual company statements are matched by fiscal year. Market cap is a more recent quote snapshot; ratios are cross-sectional, not point-in-time backtests.")
        st.markdown("[Financial statements at Yahoo Finance](https://finance.yahoo.com/)")
        st.markdown("[GitHub model and data audit](https://github.com/vpancholi808/financial-analysis-app)")

settings=ScreenSettings(buy_gpa_pct=gpa,buy_bm_pct=bm,buy_score_min=score,buy_gpa_weight=w,
                        sell_sta_pct=sta,sell_snoa_pct=snoa,sell_beneish_cutoff=beneish,sell_min_flags=riskcount)
selected=scope_filter(base,active_sectors,industry,cap_min*1e9)
view=classify(selected,settings)
view=view.join(recommendation_table(view,base,settings))
try:theme_type=st.context.theme.type
except Exception:theme_type=None
P=pr.palette(theme_type)
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
        "Ticker":df.ticker.values,"Company":df["name"].values,"Recommendation":df.rec_label.values,
        "Sector":df.sector.values,"Market cap":(df.market_cap/1e9).values,"Status":status,
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
    "Recommendation":CC.TextColumn("Recommendation",width="medium",
        help="Rule-based call with conviction. ▲ Long: passes the long screen with positive, cash-backed earnings and no risk check. "
             "▼ Short / avoid: enough independent accounting-risk checks trigger. ○ No position: neither, or both. "
             "– Not rated: bank, insurer, REIT or stale data. Full reasoning on the Research Desk."),
    "Status":CC.TextColumn("Screen result",help="▲ passes the long screen · ▼ risk-flagged · ◆ both · ● neither · – bank/insurer/REIT (not scored)."),
    "GPA":pct_col("GPA","Gross profitability"),
    "GPA rank":rank_col("Profitability rank","Gross profitability (GPA) percentile within its sector; 100 = most profitable."),
    "B/M":CC.NumberColumn("Book-to-market",format="%.2f",help=help_for("BM")),
    "B/M rank":rank_col("Value rank","Book-to-market percentile within its sector; 100 = cheapest relative to book value."),
    "Long score":rank_col("Combined score","Weighted blend of the profitability and value ranks (0–100)."),
    "Accruals":pct_col("STA","Accruals"),
    "Risk flags":CC.NumberColumn("Warning signs",format="%d of 4",help="Warning signs showing: profit ahead of cash, heavy balance sheet, manipulation score, distress score."),
    "Signals fired":CC.TextColumn("Signals fired",help="Which accounting-risk checks triggered for this company."),
    "Altman Z":CC.NumberColumn("Altman score",format="%.2f",help=help_for("Altman_Z")),
    "Altman model":CC.TextColumn("Altman model",help="Z (manufacturer): distress < 1.81, grey 1.81–2.99. Z″ (non-manufacturer): distress < 1.1, grey 1.1–2.6."),
    "Distress zone":CC.TextColumn("Distress zone (Altman)",help=help_for("Altman_Z")),
    "FY end":CC.TextColumn("FY end",help="Fiscal year-end of the annual statements used. Market cap is a later quote snapshot.",width="small"),
}

# ================================================================== pages
if page=="S&P 500 Screener":
    counts={k:int(view.rec_position.eq(k).sum()) for k in ["Long","Short / avoid","No position","Not rated"]}
    eyebrow(f"S&P 500 · data as of {asof}")
    st.title("Screener")
    summary=(f"Of **{len(view)}** companies, **{counts['Long']} look like buys**: very profitable and cheap for their sector, "
             f"with earnings backed by cash. **{counts['Short / avoid']} are worth avoiding**: {riskcount} or more accounting warning signs.")
    st.markdown(summary)
    st.html(pr.split_bar_html(counts,P))
    conv_rank={"High":0,"Moderate":1,"Low":2,"":3}

    t1,t2,t3,t4=st.tabs(["Ideas","All companies","By sector","Map"])
    with t1:
        def idea_cards(pos,title,tone_word,empty):
            d=view.loc[view.rec_position.eq(pos)]
            d=d.assign(_c=d.rec_conviction.map(conv_rank)).sort_values(
                ["_c","BUY_score" if pos=="Long" else "SELL_flag_count"],ascending=[True,False])
            st.markdown(f"#### {title}")
            if d.empty:
                st.info(empty);return
            for _,r in d.head(5).iterrows():
                with st.container(border=True):
                    ev=pr.evidence_label(pos,r.rec_conviction)
                    colour={"Strong evidence":tone_word,"Moderate evidence":"orange"}.get(ev,"gray")
                    st.markdown(f"**{r.ticker}** &nbsp;{r['name']} &nbsp; :{colour}-badge[{ev}]")
                    st.markdown(md(pr.why_line(r,settings)))
                    a,b=st.columns([3,2],vertical_alignment="center")
                    a.caption(f"{r.sector} · market value {pr.money(r.market_cap)}")
                    b.button(f"Open {r.ticker} →",key=f"idea_{pos}_{r.ticker}",type="tertiary",
                             on_click=go_to,args=(r.ticker,"Company Profile"),width="stretch")
            if len(d)>5:st.caption(f"Showing the strongest 5 of {len(d)}. All of them are in **All companies**.")
        left,right=st.columns(2,gap="large")
        with left:idea_cards("Long","▲ Best buy ideas","green","No company passes the buy screen with these settings. Try loosening them in the sidebar.")
        with right:idea_cards("Short / avoid","▼ Companies to avoid","red","No company shows enough warning signs with these settings.")
    with t2:
        opts=["all","Long","Short / avoid","No position","Not rated"]
        names={"all":f"All {len(view)}","Long":f"▲ Buy {counts['Long']}","Short / avoid":f"▼ Avoid {counts['Short / avoid']}",
               "No position":f"○ No clear signal {counts['No position']}","Not rated":f"– Not rated {counts['Not rated']}"}
        f1,f2,f3=st.columns([5,2,1.3],vertical_alignment="center")
        with f1:pick=st.segmented_control("Show",opts,format_func=lambda k:names[k],default="all",key="rec_filter",label_visibility="collapsed")
        with f2:query=st.text_input("Search",placeholder="Search ticker or company",key="rec_search",label_visibility="collapsed")
        with f3:more=st.toggle("More columns",key="rec_more")
        pick=pick or "all"
        shown=view if pick=="all" else view.loc[view.rec_position.eq(pick)]
        if query:
            qq=query.strip().lower()
            shown=shown.loc[shown.ticker.str.lower().str.contains(qq,regex=False)|shown["name"].str.lower().str.contains(qq,regex=False)]
        rec_order={"Long":0,"Short / avoid":1,"No position":2,"Not rated":3}
        shown=shown.assign(_r=shown.rec_position.map(rec_order),_c=shown.rec_conviction.map(conv_rank)) \
                   .sort_values(["_r","_c","market_cap"],ascending=[True,True,False])
        data=pd.DataFrame({
            "Ticker":shown.ticker.values,"Company":shown["name"].values,
            "Call":[pr.call_label(p_) for p_ in shown.rec_position],
            "Evidence":["Near miss" if pr.is_near_miss(r,settings) else pr.evidence_label(r.rec_position,r.rec_conviction).replace(" evidence","")
                        for _,r in shown.iterrows()],
            "Why":[pr.why_line(r,settings) for _,r in shown.iterrows()],
            "Data":[pr.data_status(r)[0] for _,r in shown.iterrows()],
            "Market cap":(shown.market_cap/1e9).values})
        cols=["Ticker","Company","Call","Evidence","Market cap","Data","Why"]
        if more or pick=="Short / avoid":
            data["Warning signs"]=[", ".join(pr.warnings_fired(r)) or MISSING for _,r in shown.iterrows()]
            cols.insert(4,"Warning signs")
        if more:
            data["Profitability rank"]=shown.GPA_sector_pct.values;data["Value rank"]=shown.BM_sector_pct.values
            data["Combined score"]=shown.BUY_score.values
            data["Distress zone"]=[altman_status(z)[0] for z in shown.Altman_zone]
            data["FY end"]=shown.fiscal_year_end.astype(str).replace({"nan":MISSING}).values
            cols+=["Profitability rank","Value rank","Combined score","Distress zone","FY end"]
        cfg={**cfg_common(),
             "Ticker":CC.TextColumn("Ticker",pinned=True,width=70),
             "Company":CC.TextColumn("Company",pinned=True,width=190),
             "Call":CC.TextColumn("Call",width=120,help="▲ Buy: passes the buy screen with cash-backed profits and no warning signs. "
                                  "▼ Avoid: enough independent warning signs. ○ No clear signal: neither. – Not rated: bank, insurer, REIT or out-of-date data."),
             "Evidence":CC.TextColumn("Evidence",width=95,help="How strongly the data supports the call. Near miss: within 10 points of both buy ranks, no warning signs."),
             "Why":CC.TextColumn("Why",width=560,help="Hover a cell to read the full sentence."),
             "Market cap":CC.NumberColumn("Market value",format="$%,.1fB",width=110,help="Latest market value in US$ billions."),
             "Data":CC.TextColumn("Data",width=130,help="Complete: all lines from Yahoo Finance. Partly from 10-K: blanks filled from the SEC filing for the same year. "
                                  "From FY…10-K: a newer annual report taken from the SEC. Out of date: not scored."),
             "Warning signs":CC.TextColumn("Warning signs",width="medium"),
             "Profitability rank":rank_col("Profitability rank","Gross profit per $1 of assets, percentile within the sector; 100 = most profitable."),
             "Value rank":rank_col("Value rank","Book value against market value, percentile within the sector; 100 = cheapest."),
             "Combined score":rank_col("Combined score","Weighted blend of the profitability and value ranks (0–100)."),
             "Distress zone":CC.TextColumn("Distress zone",help=help_for("Altman_Z")),
             "FY end":CC.TextColumn("FY end",width="small")}
        if data.empty:st.info("No companies match. Clear the search or pick another filter.")
        else:
            ev=st.dataframe(styled(data[cols],["Call","Distress zone"]),column_config=cfg,hide_index=True,width="stretch",
                            height=table_height(len(data)),placeholder=MISSING,
                            on_select="rerun",selection_mode="single-row",key="tbl_all")
            selection_actions(ev,data,"tbl_all")
        calls=view.loc[view.rec_position.isin(["Long","Short / avoid"])].assign(
            _r=lambda d:d.rec_position.map(rec_order),_c=lambda d:d.rec_conviction.map(conv_rank)).sort_values(["_r","_c"])
        recs=pd.DataFrame({"Ticker":calls.ticker,"Company":calls["name"],"Sector":calls.sector,
                           "Call":[pr.CALL_NAME[p_] for p_ in calls.rec_position],
                           "Evidence":[pr.evidence_label(p_,c_) for p_,c_ in zip(calls.rec_position,calls.rec_conviction)],
                           "Why":[pr.why_line(r,settings) for _,r in calls.iterrows()],
                           "Warning signs":[", ".join(pr.warnings_fired(r)) for _,r in calls.iterrows()],
                           "Profitability rank":calls.GPA_sector_pct.round(1),"Value rank":calls.BM_sector_pct.round(1),
                           "Combined score":calls.BUY_score.round(1),"Market value (USD)":calls.market_cap,
                           "Data":[pr.data_status(r)[0] for _,r in calls.iterrows()],
                           "FY end":calls.fiscal_year_end,"Market value date":calls.price_asof})
        export=view[["ticker","name","sector","industry","rec_position","rec_conviction","market_cap","GPA","GPA_sector_pct","BM","BM_sector_pct",
                     "BUY_score","BUY_candidate","STA","SNOA","Beneish_M","Altman_Z","Altman_zone",
                     "SELL_flag_count","SELL_candidate","conflicting_signals"]].assign(
                         call=lambda d:d.rec_position.map(pr.CALL_NAME),
                         evidence=lambda d:[pr.evidence_label(p_,c_) for p_,c_ in zip(d.rec_position,d.rec_conviction)]
                     ).drop(columns=["rec_position","rec_conviction"])
        d1,d2=st.columns(2)
        d1.download_button(f"Download buy / avoid list ({len(recs)} companies, CSV)",recs.to_csv(index=False),
                           "sp500_recommendations.csv","text/csv",type="primary",width="stretch",disabled=recs.empty)
        d2.download_button("Download everything (CSV)",export.to_csv(index=False),"sp500_screen.csv","text/csv",width="stretch")
    with t3:
        st.markdown("Every company is ranked only against its own sector, so each row stands alone. "
                    "The *typical* columns are that sector's median: the yardstick its companies are measured against.")
        tbl=pr.sector_table_html(view,P)
        if tbl:st.html(tbl)
        else:st.info("No scored companies match the filters.")
        st.caption("Banks, insurers and REITs are listed in the app but not scored: these accounting measures don't fit how they make money.")
    with t4:
        plotted=view.dropna(subset=["GPA_sector_pct","BM_sector_pct"]).copy()
        if len(plotted):
            plotted["Call"]=plotted.rec_position.map({"Long":"Buy","Short / avoid":"Avoid"}).fillna("No clear signal")
            plotted["GPA (%)"]=plotted.GPA*100
            order=["Buy","Avoid","No clear signal"]
            st.markdown("Each dot is a company, placed by where it ranks **within its own sector**. Buys come from the shaded top-right corner.")
            fig=px.scatter(plotted,x="BM_sector_pct",y="GPA_sector_pct",color="Call",symbol="Call",
                    category_orders={"Call":order},
                    color_discrete_map={"Buy":TONE_HEX[GOOD],"Avoid":TONE_HEX[BAD],"No clear signal":TONE_HEX[NEUTRAL]},
                    symbol_map={"Buy":"triangle-up","Avoid":"triangle-down","No clear signal":"circle"},
                    hover_name="name",custom_data=["ticker","sector","GPA (%)","BM"],
                    title="Where each company ranks in its own sector",
                    labels={"BM_sector_pct":"Cheaper than sector peers (value rank) →",
                            "GPA_sector_pct":"More profitable than sector peers (profitability rank) →"})
            fig.update_traces(marker=dict(size=9,line=dict(width=0)),
                hovertemplate="<b>%{hovertext}</b> (%{customdata[0]})<br>%{customdata[1]}<br>"
                              "Profitability rank %{y:.0f} · %{customdata[2]:.0f}¢ gross profit per $1 of assets<br>"
                              "Value rank %{x:.0f}<extra></extra>")
            for tr in fig.data:
                if tr.name=="No clear signal":tr.marker.opacity=.35
            fig.add_shape(type="rect",x0=bm,x1=103,y0=gpa,y1=106,line_width=0,fillcolor=TONE_HEX[GOOD],opacity=.10,layer="below")
            fig.add_annotation(x=102,y=105,xanchor="right",yanchor="top",showarrow=False,
                               text="Buy zone",font=dict(size=12,color=TONE_HEX[GOOD]))
            fig.update_xaxes(range=[-2,103]);fig.update_yaxes(range=[-2,106])
            show_chart(fig,520)
            st.caption(f"The shaded corner uses the current settings: top {gpa_top}% for profitability and top {bm_top}% for cheapness. "
                       f"A buy must also reach a combined score of {score}, have cash-backed profits and show no warning signs.")
        else:st.info("Not enough companies with both ranks for this chart.")
    st.caption("Definitions, formulas and the data-quality audit are on the **Glossary & Data** page.")

elif page=="Company Profile":
    pick,_=st.columns([3,2])
    with pick:ticker=choose_company(view,"overview_select")
    r=view.loc[view.ticker.eq(ticker)].iloc[0]
    cur=str(r.get("financial_currency") or "USD")
    eyebrow(f"{r.sector} · {r.industry}")
    st.title(f"{r['name']} ({ticker})")
    ds,_=pr.data_status(r)
    st.markdown(f":gray-badge[Data: {ds}] &nbsp; :gray-badge[Annual report: year ending {r.fiscal_year_end}] &nbsp; "
                f":gray-badge[Market value {md(pr.money(r.market_cap))} on {r.price_asof}]")
    findings=describe(r,base)
    st.html(pr.bottom_line_html(r,findings,settings,P))
    if str(r.get("share_class_of","")):
        st.caption(f"Share class of {r.share_class_of}: the company is counted once in sector comparisons.")

    rev_g=change(n(r,"revenue"),n(r,"prior_revenue"))
    ni_g=change(n(r,"net_income"),n(r,"prior_net_income"))
    cf_g=change(n(r,"operating_cash_flow"),n(r,"prior_operating_cash_flow"))
    yoy=lambda g:(fmt_pct(g,signed=True)+" vs prior year") if np.isfinite(g) else None
    a,b,c,d=st.columns(4)
    a.metric("Market value",fmt_money(n(r,"market_cap")),border=True,help=f"Quote snapshot {r.price_asof}.")
    b.metric("Revenue",fmt_money(n(r,"revenue"),cur),yoy(rev_g),border=True,help="Latest fiscal-year revenue; change vs the matched prior fiscal year.")
    c.metric("Net income",fmt_money(n(r,"net_income"),cur),yoy(ni_g),border=True,help="Change shown only when the prior-year figure is positive.")
    d.metric("Operating cash flow",fmt_money(n(r,"operating_cash_flow"),cur),yoy(cf_g),border=True)

    st.subheader("The evidence")
    if not bool(r.industrial_applicable):
        st.info("Banks, insurers and REITs need specialist models, so this company isn't scored. Its figures are still shown below.")
    elif bool(r.get("stale_statements",False)):
        st.warning(f"The latest annual figures (year ending {r.fiscal_year_end}) are {int(r.statement_age_days)} days older than the market value, "
                   "so this company isn't scored until its data refreshes. Its figures are still shown below.")
    else:
        st.caption(f"Each answer has two parts: a fixed test where one exists, and how {ticker} compares with the other "
                   f"{r.sector} companies. The dots are those peers, the line marks the typical peer, and the shaded side is the better side.")
        st.html(pr.evidence_html(pr.evidence_rows(r,base,settings),P))
    with st.expander("What do these measures mean?"):
        for key in ["GPA","BM","ROA","CFOA","STA","SNOA","Beneish_M","Altman_Z"]:
            m=METRICS[key];st.markdown(f"**{m.label}** — {m.help} *{m.direction}.*")
        st.caption(SNOA_NOTE)

    st.subheader("Financial statements")
    t1,t2=st.tabs(["Year-over-year charts","Statement lines"])
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
        sec_fields=str(r.get("sec_filled_fields","") or "")
        if sec_fields.startswith("all"):
            st.caption(f"Statements from the company's latest 10-K (SEC EDGAR XBRL) · FY {r.fiscal_year_end} · market cap from Yahoo Finance, quote snapshot {r.price_asof}")
        else:
            st.caption(f"Yahoo Finance via yfinance · FY {r.fiscal_year_end} · quote snapshot {r.price_asof} · retrieved {r.refreshed_at}")
            if sec_fields:
                st.caption("Lines Yahoo left blank, filled from the same fiscal year's 10-K (SEC EDGAR): "
                           + ", ".join(f.replace("_"," ") for f in sec_fields.split(";")))
        st.markdown(f"[Review source financial statements](https://finance.yahoo.com/quote/{ticker}/financials/) · "
                    f"[SEC filings](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=10-K)")
        st.caption("Market capitalization and latest FY figures use different dates. Do not treat the derived ratios as historical backtests.")

elif page=="Research Desk":
    eyebrow("Buy and avoid cases")
    st.title("Research Desk")
    st.caption("A rule-based call from the accounting evidence, with an editable investment case · verify against the filings before acting")
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
    with a:shortlist(screen_frame(longs),["Ticker","Company","Long score"],SCREEN_CFG,"▲ Buy candidates","rs_long",
                     "No qualifying companies. Adjust the filters.")
    with b:shortlist(screen_frame(shorts),["Ticker","Company","Risk flags"],SCREEN_CFG,"▼ Companies to avoid","rs_short",
                     "No qualifying companies. Adjust the filters.")
    st.caption("Select a row to analyse that company below, or search for any company.")
    st.divider()
    ticker=choose_company(view,"research_select","Company to research")
    r=view.loc[view.ticker.eq(ticker)].iloc[0]
    findings=describe(r,base)
    case=investment_case(r,findings,settings)
    st.subheader(f"{r['name']} · {ticker}")
    st.html(pr.bottom_line_html(r,findings,settings,P))
    st.caption("Rule-based call from the accounting screens and valuation below (current screen settings). "
               "No price target or return forecast. Verify against the 10-K before relying on it.")
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

    st.subheader("Investment case")
    st.caption("Pre-filled from the analysis above. Edit any section; your edits are kept for this company, and "
               "the working paper records whether each section is generated or analyst-edited.")
    # Drafts are stored per company, so text written for one ticker can never
    # appear in (or be exported with) another company's working paper. Only
    # analyst edits are stored; untouched fields follow the generated case.
    drafts=st.session_state.setdefault("research_drafts",{}).setdefault(ticker,{})
    generated={"position":case["position"],"thesis":case["thesis"],"catalyst":case["catalyst"],
               "counter":case["counterargument"],"invalid":case["invalidation"],"filing":case["filing_reference"]}
    def draft_field(widget,field,label,**kw):
        gen=generated[field]
        v=widget(label,value=drafts.get(field,gen),key=f"rd_{field}_{ticker}_{hashlib.md5(gen.encode()).hexdigest()[:8]}",**kw)
        if v!=gen:drafts[field]=v
        else:drafts.pop(field,None)
        return v
    positions=["Long","Short / avoid","No position","Under review"]
    with st.container(border=True):
        pos_value=drafts.get("position",generated["position"])
        position=st.selectbox("Call",positions,index=positions.index(pos_value) if pos_value in positions else 2,
                              format_func=lambda p_:pr.CALL_NAME.get(p_,p_),
                              key=f"rd_position_{ticker}_{generated['position']}")
        if position!=generated["position"]:drafts["position"]=position
        else:drafts.pop("position",None)
        thesis=draft_field(st.text_area,"thesis","Investment thesis",height=170)
        catalyst=draft_field(st.text_area,"catalyst","Catalyst and expected timing",height=80)
        counter=draft_field(st.text_area,"counter","Strongest counterargument",height=95)
        invalid=draft_field(st.text_area,"invalid","What evidence would invalidate the thesis?",height=80)
        filing=draft_field(st.text_input,"filing","SEC filing or issuer annual-report URL")
        edited=sorted(drafts)
        if edited:
            st.caption("Analyst-edited: "+", ".join(edited))
            if st.button("Reset to the generated case",key=f"rd_reset_{ticker}"):
                drafts.clear();st.rerun()
    sections=dict(position=pr.CALL_NAME.get(position,position),thesis=thesis,catalyst=catalyst,
                  counterargument=counter,invalidation=invalid,filing_reference=filing,
                  conviction=pr.EVIDENCE.get(case["conviction"],"").replace(" evidence","") if position==case["position"] else "",
                  edited=", ".join(sorted(drafts)) or "none (generated case)")
    require=[thesis.strip(),catalyst.strip(),counter.strip(),invalid.strip(),filing.strip()]
    if not all(require):
        st.info("Some sections are empty: thesis, catalyst, counterargument, invalidation and filing reference make a complete case.")
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
