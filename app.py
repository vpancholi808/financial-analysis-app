"""ACCT 3343 — US equity fundamental analysis, Streamlit Phase 1."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from finance_models import GICS_SECTORS, ScreenSettings, prepare, classify, filter_universe, integrity_report

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Financial Analysis App | ACCT 3343", page_icon="📊", layout="wide")

@st.cache_data
def load_real():
    file = ROOT / "universe.csv"
    return pd.read_csv(file) if file.exists() else pd.DataFrame()

@st.cache_data
def demo_data():
    """Synthetic fixtures only. Never confuse with market data."""
    rng = np.random.default_rng(3343)
    rows = []
    industries = {
      "Information Technology":"Software", "Health Care":"Medical Devices",
      "Financials":"Banks", "Consumer Discretionary":"Retail",
      "Consumer Staples":"Food", "Industrials":"Machinery",
      "Energy":"Oil & Gas", "Materials":"Chemicals", "Utilities":"Electricity",
      "Real Estate":"REITs", "Communication Services":"Media",
    }
    for sector in GICS_SECTORS:
        n = 24 if sector == "Information Technology" else 12
        for _ in range(n):
            assets = float(rng.uniform(2e9, 200e9))
            rev = assets * float(rng.uniform(0.25, 1.8))
            gp = rev * float(rng.uniform(0.18, 0.8))
            equity = assets * float(rng.uniform(0.10, 0.75))
            income = rev * float(rng.uniform(-0.08, 0.23))
            ocf = income + assets * float(rng.normal(0.01, 0.08))
            rows.append(dict(
                ticker=f"DEMO{len(rows)+1:04d}", name=f"Synthetic Company {len(rows)+1:04d}",
                sector=sector, industry=industries[sector], exchange="DEMO",
                market_cap=equity*float(rng.uniform(0.65, 8)),
                market_currency="USD", financial_currency="USD",
                price_asof="2026-10-20", fiscal_year_end="2025-12-31",
                data_source="SYNTHETIC — NOT REAL FINANCIAL DATA",
                refreshed_at="2026-10-20", revenue=rev, cost_of_revenue=rev-gp,
                net_income=income, operating_cash_flow=ocf,
                total_assets=assets, common_equity=equity
            ))
    return pd.DataFrame(rows)

def fmt(x):
    return "N/A" if pd.isna(x) else f"{x:,.3f}"

def cap(x):
    return "N/A" if pd.isna(x) else f"${x / 1e9:,.2f}B"

st.title("US Equity Fundamental Analysis")
st.caption("ACCT 3343 · Independent BUY and SELL screens · Sector-adjusted research")

real = load_real()
with st.sidebar:
    st.header("Financial Analysis App")
    demo = st.checkbox("Use synthetic demonstration data", value=real.empty,
                       help="Synthetic companies do NOT meet university submission requirements.")
    page = st.radio("Navigation", ["Stock Screener", "Company Fundamentals", "Investment Research"])
data = demo_data() if demo else real
if data.empty:
    st.warning("No real financial dataset is present. Supply valid records in universe.csv or enable the synthetic demo.")
    st.stop()
if demo:
    st.error("DEMONSTRATION ONLY: All DEMO tickers and financial values are invented. No real stocks are evaluated.")
else:
    st.info("Real-data mode. Verify source reporting dates, currency, company coverage, and the accuracy of each accounting input.")
try:
    config = json.loads((ROOT / "config.json").read_text())
    excluded = set(config.get("excluded_tickers", []))
except (OSError, ValueError):
    excluded = set()
if not excluded:
    st.warning("The 60-second stock-pitch company's ticker has not yet been excluded. Configure config.json before submitting the project.")

base = prepare(data)
with st.sidebar:
    st.divider()
    st.subheader("Universe filters")
    options = [s for s in GICS_SECTORS if s in base.sector.unique()]
    sectors = st.multiselect("Sector", options, default=options)
    industry_options = sorted(base.loc[base.sector.isin(sectors), "industry"].dropna().unique().tolist())
    industry = st.selectbox("Industry", ["All industries"] + industry_options)
    min_cap = st.number_input("Minimum market capitalization (USD billions)", min_value=0.0, value=0.0, step=1.0)
    st.divider()
    st.subheader("BUY: profitability + value")
    gpa_limit = st.slider("Gross profitability: minimum sector percentile", 0, 100, 60, 5)
    bm_limit = st.slider("Book-to-market: minimum sector percentile", 0, 100, 60, 5)
    buy_limit = st.slider("Combined BUY score cutoff", 0, 100, 65, 5)
    weight = st.slider("Gross profitability weight", 0.0, 1.0, 0.60, 0.05)
    st.divider()
    st.subheader("SELL: earnings quality")
    sta_limit = st.slider("Scaled accruals: minimum sector percentile", 0, 100, 80, 5)
    sta_floor = st.slider("Scaled accruals: minimum raw ratio", 0.0, 0.50, 0.0, 0.01)
    st.caption("Only one verified SELL red flag is enabled. SNOA, Beneish and Altman require course-specific formula confirmation.")

settings = ScreenSettings(gpa_limit, bm_limit, buy_limit, weight, sta_limit, sta_floor)
view = classify(filter_universe(base, sectors, industry, min_cap*1e9, excluded), settings)
buys = view.loc[view.BUY_candidate].sort_values("BUY_score", ascending=False)
sells = view.loc[view.SELL_candidate].sort_values("STA", ascending=False)

if page == "Stock Screener":
    a,b,c,d = st.columns(4)
    a.metric("Companies screened", len(view))
    b.metric("BUY candidates", len(buys))
    c.metric("SELL candidates", len(sells))
    d.metric("Overlapping signals", int(view.conflicting_signals.sum()))
    if not demo and len(base) < 500:
        st.error(f"Assignment coverage incomplete: {len(base)} rows, 500+ needed unless using approved single-industry alternative.")
    st.subheader("BUY candidates — GPA and book-to-market")
    st.dataframe(buys[["ticker","name","sector","market_cap","GPA","BM","GPA_sector_pct","BM_sector_pct","BUY_score"]],
                 use_container_width=True, hide_index=True)
    st.download_button("Export BUY candidates", buys.to_csv(index=False), "buy_candidates.csv", "text/csv")
    st.subheader("SELL candidates — positive accrual red flags")
    st.dataframe(sells[["ticker","name","sector","market_cap","STA","STA_sector_pct","SELL_flag_count","BUY_candidate"]],
                 use_container_width=True, hide_index=True)
    st.download_button("Export SELL candidates", sells.to_csv(index=False), "sell_candidates.csv", "text/csv")
    if not view.empty:
        counts = view.groupby("sector", as_index=False).size()
        st.plotly_chart(px.bar(counts, x="size", y="sector", orientation="h", title="Sector distribution"), use_container_width=True)
        scatter = view.dropna(subset=["GPA_sector_pct","BM_sector_pct"]).copy()
        if not scatter.empty:
            scatter["Screen"] = np.select(
                [scatter.conflicting_signals,scatter.BUY_candidate,scatter.SELL_candidate],
                ["Both","BUY","SELL"],default="No signal")
            st.plotly_chart(px.scatter(scatter, x="BM_sector_pct",y="GPA_sector_pct",
                color="Screen",hover_data=["ticker","sector","industry"],
                title="Sector-relative profitability and value"),use_container_width=True)
    with st.expander("Formulas, methodology and validation"):
        st.markdown("""
        - GPA = (revenue − cost of revenue) / total assets
        - B/M = common equity / market capitalization, provided equity is positive and currencies match
        - STA = (net income − operating cash flow) / total assets
        - BUY = sector percentiles for GPA and B/M, each passing cutoffs; weighted sum passing composite cutoff
        - SELL = high positive STA relative to sector; **not** a reversal of BUY
        - Sectors need at least 10 eligible observations for percentiles; Financials and Real Estate are not industrial-screen eligible.
        - Missing financial inputs remain missing, rather than being set to zero.
        """)
        st.json(integrity_report(base))
elif page == "Company Fundamentals":
    if view.empty:
        st.info("No companies satisfy the current filters.")
    else:
        ticker = st.selectbox("Ticker", sorted(view.ticker.tolist()))
        r = view.loc[view.ticker.eq(ticker)].iloc[0]
        st.header(f"{r['name']} ({ticker})")
        st.caption(f"{r['sector']} · {r['industry']} · Fiscal year {r['fiscal_year_end']} · Snapshot {r['price_asof']}")
        a,b,c,d = st.columns(4)
        a.metric("Market cap",cap(r.market_cap))
        b.metric("GPA",fmt(r.GPA))
        c.metric("B/M",fmt(r.BM))
        d.metric("STA",fmt(r.STA))
        st.subheader("Accounting evidence and peer comparisons")
        cols = ["GPA","BM","STA","ROA","CFOA"]
        evidence = pd.DataFrame([{
            "Metric":m, "Value":r[m], "Sector median":r[m+"_sector_median"],
            "Sector percentile":r[m+"_sector_pct"],
            "Industry percentile":r[m+"_industry_pct"],
            "Sector sample size":r[m+"_sector_count"],
        } for m in cols])
        st.dataframe(evidence,hide_index=True,use_container_width=True)
        st.subheader("Source inputs")
        fields = ["revenue","cost_of_revenue","gross_profit","total_assets","common_equity",
                  "market_cap","net_income","operating_cash_flow","data_source","refreshed_at"]
        st.dataframe(pd.DataFrame({"Input":fields,"Value":[r[c] for c in fields]}),
                     hide_index=True,use_container_width=True)
        if not r.industrial_applicable:
            st.warning("Bank/REIT accounting differs from industrial companies; the standard screen is disabled for this sector.")
        if not r.currency_matches:
            st.warning("B/M unavailable because financial-statement and market currencies are incompatible or missing.")
        st.warning("SNOA, Beneish PMAN and Altman Z await course formulas and verification.")
else:
    st.header("Investment Research")
    st.caption("Quantitative screens provide leads; final research conclusions require actual statements and human review.")
    if view.empty:
        st.info("No companies satisfy current filters.")
    else:
        summary = pd.concat([buys.head(10),sells.head(10)]).drop_duplicates("ticker")
        st.dataframe(summary[["ticker","name","sector","GPA","BM","STA","BUY_score","conflicting_signals"]],
                     hide_index=True,use_container_width=True)
        ticker=st.selectbox("Review ticker",sorted(view.ticker.tolist()))
        r=view.loc[view.ticker.eq(ticker)].iloc[0]
        st.markdown(f"**{ticker}: quantitative evidence only**")
        st.write(f"GPA {fmt(r.GPA)}; B/M {fmt(r.BM)}; STA {fmt(r.STA)}.")
        st.info("Write a specific thesis, accounting evidence, valuation, risks and conclusion after validating the source data. This is not an automatically approved investment case.")

st.divider()
st.caption("Phase 1 · Screening candidates are not recommendations to trade. Course models and actual company coverage remain incomplete.")
