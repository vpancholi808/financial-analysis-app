"""Financial Analysis App v2: real Yahoo financials + hybrid research interface."""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from finance_models import (GICS_SECTORS, ScreenSettings, prepare, classify,
                            scope_filter, validate_universe)
from research import evidence, metric, research_report, growth
from build_universe import extract_company

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title='Financial Analysis | Equity Research', page_icon='📈',
                   layout='wide', initial_sidebar_state='expanded')

st.markdown('''<style>
.block-container {padding-top:1.15rem; padding-bottom:2rem; max-width:1550px;}
h1 {letter-spacing:-.045em; font-size:2.25rem!important;}
h2 {letter-spacing:-.025em; font-size:1.45rem!important;}
h3 {font-size:1.15rem!important;}
[data-testid="stMetric"] {background:#172636; border:1px solid #2B4054; border-radius:11px; padding:14px 18px;}
[data-testid="stMetricLabel"] {color:#A2B2C4;}
[data-testid="stSidebar"] {border-right:1px solid #2B4054;background:#101E2C;} 
[data-testid="stDataFrame"] {border:1px solid #30465A;border-radius:10px;}
[data-testid="stAlert"] {border-radius:10px;}
a {color:#51D1BF!important;}
.app-kicker {font-size:.78rem;letter-spacing:.16em;color:#79CABA;font-weight:650;}
.app-sub {font-size:.92rem;color:#91A7BC;margin-top:-.5rem;}
.panel-intro {background:#172636; padding:14px 18px; border:1px solid #2B4054;
 border-radius:10px; color:#BCCBDD; font-size:.94rem;}
</style>''', unsafe_allow_html=True)

@st.cache_data(ttl=1800)
def saved_universe():
    file = ROOT / 'universe.csv'
    return pd.read_csv(file) if file.exists() else pd.DataFrame()

@st.cache_data
def load_config():
    try:
        return json.loads((ROOT / 'config.json').read_text())
    except (OSError, json.JSONDecodeError):
        return {}

def normal_tickers(text):
    import re
    tokens = re.split(r'[\s,;]+', text.strip().upper())
    return tuple(dict.fromkeys(s for s in tokens if re.fullmatch(r'[A-Z0-9.^-]{1,16}', s)))

def money(n):
    if pd.isna(n):return 'N/A'
    x=abs(float(n))
    value=(f'{x/1e12:,.2f}T' if x>=1e12 else f'{x/1e9:,.2f}B' if x>=1e9 else f'{x/1e6:,.2f}M' if x>=1e6 else f'{x/1e3:,.1f}K' if x>=1e3 else f'{x:,.0f}')
    return ('-' if n<0 else '')+'

def fmt_value(v, pct=False):
    return 'N/A' if pd.isna(v) else (f'{v*100:,.1f}%' if pct else f'{v:,.3f}')

def numeric_table(df, cols, order_col=None):
    if df.empty:
        st.info('No companies meet these screening rules. Adjust the filters or load a larger universe.')
        return
    view = df[cols].copy()
    spec = {}
    for key in ['GPA', 'BM', 'STA', 'ROA', 'CFOA']:
        if key in view: spec[key] = st.column_config.NumberColumn(key, format='%.3f')
    for key in ['GPA_sector_pct', 'BM_sector_pct','STA_sector_pct','BUY_score']:
        if key in view: spec[key] = st.column_config.NumberColumn(key, format='%.1f')
    if 'market_cap' in view:
        view['market_cap']=view['market_cap'].map(money)
        spec['market_cap']=st.column_config.TextColumn('Market Cap (USD)')
    st.dataframe(view, use_container_width=True, hide_index=True, column_config=spec)

def ticker_selector(df, key):
    names = {r.ticker: f'{r.ticker} · {r["name"]}' for _,r in df.iterrows()}
    return st.selectbox('Choose company', sorted(names), format_func=lambda s:names[s], key=key)

def company_source_line(r):
    st.caption(f'Source: {r.get("data_source", "Unknown")}  ·  Fiscal year: {r.get("fiscal_year_end", "N/A")}  ·  Quote snapshot: {r.get("price_asof", "N/A")}  ·  Refreshed: {r.get("refreshed_at", "N/A")}')

st.markdown('<div class="app-kicker">EQUITY RESEARCH WORKSTATION / ACCT 3343</div>', unsafe_allow_html=True)
st.title('Financial Analysis Terminal')
st.markdown('<div class="app-sub">Real financial statements · Independent BUY / SELL screening · Sector-relative evidence</div>', unsafe_allow_html=True)
config=load_config()
excluded=set(str(s).strip().upper() for s in config.get('excluded_tickers', []))

@st.cache_data(ttl=90)
def collection_state():
    try: return json.loads((ROOT/'collection_state.json').read_text())
    except (OSError, ValueError): return {}

with st.sidebar:
    st.markdown('### ◈ FINANCIAL ANALYSIS')
    page=st.radio('Workspace',['Stock Screener','Company Fundamentals','Investment Research'])
    st.divider()
    st.markdown('#### Full US Equity Universe')
    data=saved_universe()
    state=collection_state()
    st.metric('Verified companies',f'{len(data):,} / 1,000')
    st.progress(min(1.0,len(data)/1000))
    st.caption('Source: actual annual Yahoo Finance statements')
    st.caption('Excluded pitch company: **TSM (TSMC)**')
    if state:
        st.caption(f"Processed {int(state.get('cursor',0)):,} candidates; last refresh {state.get('updated_at','pending')}")
    st.caption('Data collection runs in GitHub Actions. No synthetic records are included.')

if data.empty:
    st.warning('Waiting for verified financial statements. Data collection runs in GitHub Actions; no synthetic results are shown.')
    st.markdown('[Check Yahoo Finance collection status](https://github.com/vpancholi808/financial-analysis-app/actions)')
    st.stop()

base=prepare(data, min_sector_n=10, min_industry_n=10)
if base.empty:
    st.error('The dataset has no eligible tickers.')
    st.stop()

with st.sidebar:
    st.markdown('#### Universe filters')
    sector_options=[s for s in GICS_SECTORS if s in base.sector.unique()]
    sectors=st.multiselect('GICS sectors',sector_options,default=sector_options)
    industries=sorted(base.loc[base.sector.isin(sectors),'industry'].dropna().unique().tolist())
    industry=st.selectbox('Industry',['All industries']+industries)
    min_cap=st.number_input('Minimum market cap · USD billions',min_value=0.,value=0.,step=1.)
    with st.expander('BUY factor thresholds',expanded=True):
        gpa_pct=st.slider('GPA · sector percentile ≥',0,100,60,5)
        bm_pct=st.slider('B/M · sector percentile ≥',0,100,60,5)
        buy_score=st.slider('BUY composite score ≥',0,100,65,5)
        gpa_weight=st.slider('Weight on GPA',0.,1.,.60,.05)
    with st.expander('SELL earnings-quality rules',expanded=False):
        sta_pct=st.slider('STA · sector percentile ≥',0,100,80,5)
        sta_floor=st.slider('STA · absolute ratio >',0.,.5,0.,.01)
        st.caption('Only STA is active until the course-defined SNOA, Beneish and Altman models are verified.')

settings=ScreenSettings(buy_gpa_pct=gpa_pct,buy_bm_pct=bm_pct,buy_score_min=buy_score,
                        buy_gpa_weight=gpa_weight,sell_sta_pct=sta_pct,sell_sta_floor=sta_floor)
filtered=scope_filter(base,sectors,industry,min_cap*1e9,excluded)
view=classify(filtered,settings)
buys=view.loc[view.BUY_candidate].sort_values('BUY_score',ascending=False)
sells=view.loc[view.SELL_candidate].sort_values('STA',ascending=False)

status='SAVED REAL YAHOO FINANCIALS'
st.caption(f'{status} · {len(base)} fetched company records · {len(filtered)} shown after filters · no synthetic securities')
if 'TSM' not in excluded:
    st.error('TSM must be excluded in config.json.')
if len(base)<500:
    st.warning(f'Collection in progress: {len(base)} real company records. The assignment requires 500+ unless the approved single-industry alternative is used. Small peer samples may suppress sector percentiles.')

if page=='Stock Screener':
    col1,col2,col3,col4=st.columns(4)
    col1.metric('Companies screened',len(view))
    col2.metric('BUY candidates',len(buys))
    col3.metric('SELL candidates',len(sells))
    col4.metric('Conflicting signals',int(view.conflicting_signals.sum()))
    st.subheader('Fundamental opportunity map')
    a,b=st.columns([2.1,1])
    with a:
        scatter=view.dropna(subset=['GPA_sector_pct','BM_sector_pct']).copy()
        if scatter.empty:
            st.info('Opportunity map requires at least 10 eligible companies per sector with both GPA and B/M. Raw company fundamentals are available on the next page.')
        else:
            scatter['Signal']=np.select([scatter.conflicting_signals,scatter.BUY_candidate,scatter.SELL_candidate],
                                        ['Conflicting','BUY candidate','SELL candidate'],default='Unclassified')
            fig=px.scatter(scatter,x='BM_sector_pct',y='GPA_sector_pct',color='Signal',
                           hover_name='ticker',hover_data=['name','sector','industry','GPA','BM'],
                           title='Gross profitability vs. book-to-market percentile',
                           color_discrete_map={'BUY candidate':'#20B59A','SELL candidate':'#DA7B76',
                                               'Conflicting':'#E0B66F','Unclassified':'#8897A9'})
            fig.update_layout(template='plotly_dark',paper_bgcolor='#0B1520',plot_bgcolor='#172636',
                              xaxis_title='B/M sector percentile',yaxis_title='GPA sector percentile',
                              margin=dict(l=10,r=15,b=25,t=55))
            st.plotly_chart(fig,use_container_width=True)
    with b:
        counts=view.groupby('sector').size().reset_index(name='Count')
        if not counts.empty:
            fig=px.bar(counts.sort_values('Count'),x='Count',y='sector',orientation='h',
                       title='Coverage by sector',color_discrete_sequence=['#20B59A'])
            fig.update_layout(template='plotly_dark',paper_bgcolor='#0B1520',plot_bgcolor='#172636',
                              margin=dict(l=10,r=5,t=55,b=15),height=410)
            st.plotly_chart(fig,use_container_width=True)
    tab_buy,tab_sell,tab_sector=st.tabs(['BUY · Quality & Value','SELL · Accounting Risk','Sector Explorer'])
    with tab_buy:
        st.caption('Two independent core inputs: gross profitability and book-to-market value, both relative to eligible sector peers.')
        numeric_table(buys,['ticker','name','sector','market_cap','GPA','BM','GPA_sector_pct','BM_sector_pct','BUY_score'])
        st.download_button('Export BUY results',buys.to_csv(index=False),'buy_candidates.csv','text/csv')
    with tab_sell:
        st.caption('Risk screen uses positive scaled total accruals. The remaining course SELL models are not silently substituted.')
        numeric_table(sells,['ticker','name','sector','STA','STA_sector_pct','SELL_flag_count','BUY_candidate'])
        st.download_button('Export SELL results',sells.to_csv(index=False),'sell_candidates.csv','text/csv')
    with tab_sector:
        eligible=base.loc[base.industrial_applicable]
        med=eligible.groupby('sector').agg(n=('ticker','size'),median_GPA=('GPA','median'),
                                            median_BM=('BM','median'),median_STA=('STA','median'))
        med=med.reset_index()
        for field in ['median_GPA','median_STA']:
            med[field]=med[field].apply(lambda n:'N/A' if pd.isna(n) else f'{n*100:.2f}%')
        med['median_BM']=med['median_BM'].apply(lambda n:'N/A' if pd.isna(n) else f'{n:.2f}x')
        st.dataframe(med,use_container_width=True,hide_index=True)
        st.caption('These are full-universe peer benchmarks; banks and REITs are not ranked with industrial companies.')
    with st.expander('Calculation methodology & data-quality audit'):
        st.markdown('''**GPA** = (revenue − cost of revenue) / assets · **B/M** = positive common equity / market cap (same currency) · **STA** = (net income − operating cash flow) / assets. Sector percentiles are calculated before user filtering. No missing inputs are filled with zero. When there are fewer than 10 valid peers, percentiles are shown as unavailable.''')
        st.json(validate_universe(base))

elif page=='Company Fundamentals':
    if view.empty:
        st.info('No companies match the current filters.')
        st.stop()
    ticker=ticker_selector(view,'fund_company')
    r=view.loc[view.ticker.eq(ticker)].iloc[0]
    st.subheader(f'{r["name"]}  /  {ticker}')
    st.caption(f'{r.sector} · {r.industry} · {r.exchange} · Currency: {r.financial_currency}')
    company_source_line(r)
    a,b,c,d=st.columns(4)
    a.metric('Market cap',money(r.market_cap))
    b.metric('GPA',fmt_value(r.GPA,True))
    c.metric('B/M',fmt_value(r.BM))
    d.metric('STA',fmt_value(r.STA,True))
    t1,t2,t3=st.tabs(['Financial dashboard','Peer benchmarks','Audit inputs'])
    with t1:
        x,y=st.columns([1.6,1])
        with x:
            s=[]
            for label,current,prior in [('Revenue','revenue','prior_revenue'),('Net income','net_income','prior_net_income'),('Operating cash flow','operating_cash_flow','prior_operating_cash_flow')]:
                for period,field in [('Prior FY',prior),('Latest FY',current)]:
                    if pd.notna(r.get(field)):
                        s.append({'Statement':label,'Period':period,'Value':r[field]})
            if s:
                fig=px.bar(pd.DataFrame(s),x='Statement',y='Value',color='Period',barmode='group',
                           color_discrete_sequence=['#8898AA','#20B59A'],title='Matched annual statement trends')
                fig.update_layout(template='plotly_dark',paper_bgcolor='#0B1520',plot_bgcolor='#172636')
                st.plotly_chart(fig,use_container_width=True)
            else:st.info('Prior-year statement figures are unavailable.')
        with y:
            st.markdown('##### Accounting indicators')
            for k,label in [('ROA','Return on assets'),('CFOA','Cash flow on assets')]:
                st.metric(label,fmt_value(r[k],True))
            st.metric('Revenue growth',fmt_value(growth(r.revenue,r.get('prior_revenue')),True))
            st.caption('Growth omitted when the prior base is missing or nonpositive.')
    with t2:
        peers=pd.DataFrame([{'Metric':m,'Raw':r[m],
                             'Sector median':r[m+'_sector_median'],
                             'Sector percentile':r[m+'_sector_pct'],
                             'Industry percentile':r[m+'_industry_pct'],
                             'Sector observations':r[m+'_sector_count']} for m in ['GPA','BM','STA','ROA','CFOA']])
        st.dataframe(peers,hide_index=True,use_container_width=True)
        if not r.industrial_applicable:st.warning('Specialist accounting is required for this sector; industrial rankings are deliberately unavailable.')
    with t3:
        fields=['revenue','cost_of_revenue','gross_profit','total_assets','common_equity','market_cap',
                'net_income','operating_cash_flow','prior_revenue','prior_net_income','prior_operating_cash_flow',
                'fiscal_year_end','prior_fiscal_year_end','data_source','refreshed_at']
        st.dataframe(pd.DataFrame({'Source field':fields,'Value':[money(r.get(k,np.nan)) if k in {'revenue','cost_of_revenue','gross_profit','total_assets','common_equity','market_cap','net_income','operating_cash_flow','prior_revenue','prior_net_income','prior_operating_cash_flow'} else str(r.get(k,'')) for k in fields]}),
                     hide_index=True,use_container_width=True)
        st.caption('Financial line item definitions can differ across issuers. Verify against SEC/issuer filings.')
    st.info('SNOA, Beneish PMAN and Altman Z are awaiting the exact ACCT 3343 methodology; no proxy scores are displayed.')
    st.markdown(f'[Review {ticker} financial statements at Yahoo Finance](https://finance.yahoo.com/quote/{ticker}/financials/)')

else:
    st.subheader('Investment Research Desk')
    st.markdown('''<div class="panel-intro">A screening signal is a research lead, not a trading recommendation. This desk separates verified numeric evidence, accounting red flags, missing information and a human-reviewed investment conclusion.</div>''',unsafe_allow_html=True)
    st.write('')
    l,rcol=st.columns(2)
    with l:
        st.markdown('##### BUY watchlist · Quality and Value')
        numeric_table(buys.head(10),['ticker','name','GPA','BM','BUY_score'])
    with rcol:
        st.markdown('##### SELL watchlist · Earnings Quality')
        numeric_table(sells.head(10),['ticker','name','STA','STA_sector_pct'])
    if view.empty:
        st.info('No companies match the current universe filters.')
        st.stop()
    ticker=ticker_selector(view,'research_company')
    r=view.loc[view.ticker.eq(ticker)].iloc[0]
    company_source_line(r)
    if r.conflicting_signals:st.warning('CONFLICT: This company meets both the value/quality screen and an earnings-quality risk flag.')
    ev=evidence(r)
    left,right=st.columns(2)
    with left:
        st.markdown('##### Fundamental evidence')
        for x in ev['facts']:st.markdown(f'- {x}')
        if not ev['facts']:st.info('Insufficient numeric evidence.')
    with right:
        st.markdown('##### Red flags and limitations')
        for x in ev['risks']:st.markdown(f'- {x}')
        if not ev['risks']:st.caption('No quantified red flags in available models; unmodeled risks remain.')
    with st.expander('Missing information and required due diligence',expanded=True):
        for x in ev['missing']:st.markdown(f'- {x}')
    note=st.text_area('Analyst-reviewed investment case / conclusion',key='note_'+ticker,
                      placeholder='Cite actual filings, discuss valuation, earnings quality, sector conditions, catalysts and a reasoned conclusion. This text is your own reviewed research.',height=135)
    st.download_button('Download company research brief (.md)',research_report(r,note),
                       file_name=f'{ticker}_research.md',mime='text/markdown')
    st.markdown(f'[Underlying Yahoo Finance statements for {ticker}](https://finance.yahoo.com/quote/{ticker}/financials/)')

st.divider()
st.caption('ACCT 3343 · Real records only · Market data and financial-reporting dates may differ · Screen outputs are not guaranteed investment outcomes.')
+value

def fmt_value(v, pct=False):
    return 'N/A' if pd.isna(v) else (f'{v*100:,.1f}%' if pct else f'{v:,.3f}')

def numeric_table(df, cols, order_col=None):
    if df.empty:
        st.info('No companies meet these screening rules. Adjust the filters or load a larger universe.')
        return
    view = df[cols].copy()
    spec = {}
    for key in ['GPA', 'BM', 'STA', 'ROA', 'CFOA']:
        if key in view: spec[key] = st.column_config.NumberColumn(key, format='%.3f')
    for key in ['GPA_sector_pct', 'BM_sector_pct','STA_sector_pct','BUY_score']:
        if key in view: spec[key] = st.column_config.NumberColumn(key, format='%.1f')
    if 'market_cap' in view: spec['market_cap'] = st.column_config.NumberColumn('Market Cap (USD)', format='$%.0f')
    st.dataframe(view, use_container_width=True, hide_index=True, column_config=spec)

def ticker_selector(df, key):
    names = {r.ticker: f'{r.ticker} · {r["name"]}' for _,r in df.iterrows()}
    return st.selectbox('Choose company', sorted(names), format_func=lambda s:names[s], key=key)

def company_source_line(r):
    st.caption(f'Source: {r.get("data_source", "Unknown")}  ·  Fiscal year: {r.get("fiscal_year_end", "N/A")}  ·  Quote snapshot: {r.get("price_asof", "N/A")}  ·  Refreshed: {r.get("refreshed_at", "N/A")}')

st.markdown('<div class="app-kicker">EQUITY RESEARCH WORKSTATION / ACCT 3343</div>', unsafe_allow_html=True)
st.title('Financial Analysis Terminal')
st.markdown('<div class="app-sub">Real financial statements · Independent BUY / SELL screening · Sector-relative evidence</div>', unsafe_allow_html=True)
config=load_config()
excluded=set(str(s).strip().upper() for s in config.get('excluded_tickers', []))

@st.cache_data(ttl=90)
def collection_state():
    try: return json.loads((ROOT/'collection_state.json').read_text())
    except (OSError, ValueError): return {}

with st.sidebar:
    st.markdown('### ◈ FINANCIAL ANALYSIS')
    page=st.radio('Workspace',['Stock Screener','Company Fundamentals','Investment Research'])
    st.divider()
    st.markdown('#### Full US Equity Universe')
    data=saved_universe()
    state=collection_state()
    st.metric('Verified companies',f'{len(data):,} / 1,000')
    st.progress(min(1.0,len(data)/1000))
    st.caption('Source: actual annual Yahoo Finance statements')
    st.caption('Excluded pitch company: **TSM (TSMC)**')
    if state:
        st.caption(f"Processed {int(state.get('cursor',0)):,} candidates; last refresh {state.get('updated_at','pending')}")
    st.caption('Data collection runs in GitHub Actions. No synthetic records are included.')

if data.empty:
    st.warning('Waiting for verified financial statements. Data collection runs in GitHub Actions; no synthetic results are shown.')
    st.markdown('[Check Yahoo Finance collection status](https://github.com/vpancholi808/financial-analysis-app/actions)')
    st.stop()

base=prepare(data, min_sector_n=10, min_industry_n=10)
if base.empty:
    st.error('The dataset has no eligible tickers.')
    st.stop()

with st.sidebar:
    st.markdown('#### Universe filters')
    sector_options=[s for s in GICS_SECTORS if s in base.sector.unique()]
    sectors=st.multiselect('GICS sectors',sector_options,default=sector_options)
    industries=sorted(base.loc[base.sector.isin(sectors),'industry'].dropna().unique().tolist())
    industry=st.selectbox('Industry',['All industries']+industries)
    min_cap=st.number_input('Minimum market cap · USD billions',min_value=0.,value=0.,step=1.)
    with st.expander('BUY factor thresholds',expanded=True):
        gpa_pct=st.slider('GPA · sector percentile ≥',0,100,60,5)
        bm_pct=st.slider('B/M · sector percentile ≥',0,100,60,5)
        buy_score=st.slider('BUY composite score ≥',0,100,65,5)
        gpa_weight=st.slider('Weight on GPA',0.,1.,.60,.05)
    with st.expander('SELL earnings-quality rules',expanded=False):
        sta_pct=st.slider('STA · sector percentile ≥',0,100,80,5)
        sta_floor=st.slider('STA · absolute ratio >',0.,.5,0.,.01)
        st.caption('Only STA is active until the course-defined SNOA, Beneish and Altman models are verified.')

settings=ScreenSettings(buy_gpa_pct=gpa_pct,buy_bm_pct=bm_pct,buy_score_min=buy_score,
                        buy_gpa_weight=gpa_weight,sell_sta_pct=sta_pct,sell_sta_floor=sta_floor)
filtered=scope_filter(base,sectors,industry,min_cap*1e9,excluded)
view=classify(filtered,settings)
buys=view.loc[view.BUY_candidate].sort_values('BUY_score',ascending=False)
sells=view.loc[view.SELL_candidate].sort_values('STA',ascending=False)

status='SAVED REAL YAHOO FINANCIALS'
st.caption(f'{status} · {len(base)} fetched company records · {len(filtered)} shown after filters · no synthetic securities')
if not excluded:
    st.warning('The 60-second stock-pitch company has not yet been excluded in config.json. Complete this before submission.')
if len(base)<500:
    st.warning(f'Collection in progress: {len(base)} real company records. The assignment requires 500+ unless the approved single-industry alternative is used. Small peer samples may suppress sector percentiles.')

if page=='Stock Screener':
    col1,col2,col3,col4=st.columns(4)
    col1.metric('Companies screened',len(view))
    col2.metric('BUY candidates',len(buys))
    col3.metric('SELL candidates',len(sells))
    col4.metric('Conflicting signals',int(view.conflicting_signals.sum()))
    st.subheader('Fundamental opportunity map')
    a,b=st.columns([2.1,1])
    with a:
        scatter=view.dropna(subset=['GPA_sector_pct','BM_sector_pct']).copy()
        if scatter.empty:
            st.info('Opportunity map requires at least 10 eligible companies per sector with both GPA and B/M. Raw company fundamentals are available on the next page.')
        else:
            scatter['Signal']=np.select([scatter.conflicting_signals,scatter.BUY_candidate,scatter.SELL_candidate],
                                        ['Conflicting','BUY candidate','SELL candidate'],default='Unclassified')
            fig=px.scatter(scatter,x='BM_sector_pct',y='GPA_sector_pct',color='Signal',
                           hover_name='ticker',hover_data=['name','sector','industry','GPA','BM'],
                           title='Gross profitability vs. book-to-market percentile',
                           color_discrete_map={'BUY candidate':'#20B59A','SELL candidate':'#DA7B76',
                                               'Conflicting':'#E0B66F','Unclassified':'#8897A9'})
            fig.update_layout(template='plotly_dark',paper_bgcolor='#0B1520',plot_bgcolor='#172636',
                              xaxis_title='B/M sector percentile',yaxis_title='GPA sector percentile',
                              margin=dict(l=10,r=15,b=25,t=55))
            st.plotly_chart(fig,use_container_width=True)
    with b:
        counts=view.groupby('sector').size().reset_index(name='Count')
        if not counts.empty:
            fig=px.bar(counts.sort_values('Count'),x='Count',y='sector',orientation='h',
                       title='Coverage by sector',color_discrete_sequence=['#20B59A'])
            fig.update_layout(template='plotly_dark',paper_bgcolor='#0B1520',plot_bgcolor='#172636',
                              margin=dict(l=10,r=5,t=55,b=15),height=410)
            st.plotly_chart(fig,use_container_width=True)
    tab_buy,tab_sell,tab_sector=st.tabs(['BUY · Quality & Value','SELL · Accounting Risk','Sector Explorer'])
    with tab_buy:
        st.caption('Two independent core inputs: gross profitability and book-to-market value, both relative to eligible sector peers.')
        numeric_table(buys,['ticker','name','sector','market_cap','GPA','BM','GPA_sector_pct','BM_sector_pct','BUY_score'])
        st.download_button('Export BUY results',buys.to_csv(index=False),'buy_candidates.csv','text/csv')
    with tab_sell:
        st.caption('Risk screen uses positive scaled total accruals. The remaining course SELL models are not silently substituted.')
        numeric_table(sells,['ticker','name','sector','STA','STA_sector_pct','SELL_flag_count','BUY_candidate'])
        st.download_button('Export SELL results',sells.to_csv(index=False),'sell_candidates.csv','text/csv')
    with tab_sector:
        eligible=base.loc[base.industrial_applicable]
        med=eligible.groupby('sector').agg(n=('ticker','size'),median_GPA=('GPA','median'),
                                            median_BM=('BM','median'),median_STA=('STA','median'))
        st.dataframe(med,use_container_width=True)
        st.caption('These are full-universe peer benchmarks; banks and REITs are not ranked with industrial companies.')
    with st.expander('Calculation methodology & data-quality audit'):
        st.markdown('''**GPA** = (revenue − cost of revenue) / assets · **B/M** = positive common equity / market cap (same currency) · **STA** = (net income − operating cash flow) / assets. Sector percentiles are calculated before user filtering. No missing inputs are filled with zero. When there are fewer than 10 valid peers, percentiles are shown as unavailable.''')
        st.json(validate_universe(base))

elif page=='Company Fundamentals':
    if view.empty:
        st.info('No companies match the current filters.')
        st.stop()
    ticker=ticker_selector(view,'fund_company')
    r=view.loc[view.ticker.eq(ticker)].iloc[0]
    st.subheader(f'{r["name"]}  /  {ticker}')
    st.caption(f'{r.sector} · {r.industry} · {r.exchange} · Currency: {r.financial_currency}')
    company_source_line(r)
    a,b,c,d=st.columns(4)
    a.metric('Market cap',money(r.market_cap))
    b.metric('GPA',fmt_value(r.GPA,True))
    c.metric('B/M',fmt_value(r.BM))
    d.metric('STA',fmt_value(r.STA,True))
    t1,t2,t3=st.tabs(['Financial dashboard','Peer benchmarks','Audit inputs'])
    with t1:
        x,y=st.columns([1.6,1])
        with x:
            s=[]
            for label,current,prior in [('Revenue','revenue','prior_revenue'),('Net income','net_income','prior_net_income'),('Operating cash flow','operating_cash_flow','prior_operating_cash_flow')]:
                for period,field in [('Prior FY',prior),('Latest FY',current)]:
                    if pd.notna(r.get(field)):
                        s.append({'Statement':label,'Period':period,'Value':r[field]})
            if s:
                fig=px.bar(pd.DataFrame(s),x='Statement',y='Value',color='Period',barmode='group',
                           color_discrete_sequence=['#8898AA','#20B59A'],title='Matched annual statement trends')
                fig.update_layout(template='plotly_dark',paper_bgcolor='#0B1520',plot_bgcolor='#172636')
                st.plotly_chart(fig,use_container_width=True)
            else:st.info('Prior-year statement figures are unavailable.')
        with y:
            st.markdown('##### Accounting indicators')
            for k,label in [('ROA','Return on assets'),('CFOA','Cash flow on assets')]:
                st.metric(label,fmt_value(r[k],True))
            st.metric('Revenue growth',fmt_value(growth(r.revenue,r.get('prior_revenue')),True))
            st.caption('Growth omitted when the prior base is missing or nonpositive.')
    with t2:
        peers=pd.DataFrame([{'Metric':m,'Raw':r[m],
                             'Sector median':r[m+'_sector_median'],
                             'Sector percentile':r[m+'_sector_pct'],
                             'Industry percentile':r[m+'_industry_pct'],
                             'Sector observations':r[m+'_sector_count']} for m in ['GPA','BM','STA','ROA','CFOA']])
        st.dataframe(peers,hide_index=True,use_container_width=True)
        if not r.industrial_applicable:st.warning('Specialist accounting is required for this sector; industrial rankings are deliberately unavailable.')
    with t3:
        fields=['revenue','cost_of_revenue','gross_profit','total_assets','common_equity','market_cap',
                'net_income','operating_cash_flow','prior_revenue','prior_net_income','prior_operating_cash_flow',
                'fiscal_year_end','prior_fiscal_year_end','data_source','refreshed_at']
        st.dataframe(pd.DataFrame({'Source field':fields,'Value':[r.get(k,np.nan) for k in fields]}),
                     hide_index=True,use_container_width=True)
        st.caption('Financial line item definitions can differ across issuers. Verify against SEC/issuer filings.')
    st.info('SNOA, Beneish PMAN and Altman Z are awaiting the exact ACCT 3343 methodology; no proxy scores are displayed.')
    st.markdown(f'[Review {ticker} financial statements at Yahoo Finance](https://finance.yahoo.com/quote/{ticker}/financials/)')

else:
    st.subheader('Investment Research Desk')
    st.markdown('''<div class="panel-intro">A screening signal is a research lead, not a trading recommendation. This desk separates verified numeric evidence, accounting red flags, missing information and a human-reviewed investment conclusion.</div>''',unsafe_allow_html=True)
    st.write('')
    l,rcol=st.columns(2)
    with l:
        st.markdown('##### BUY watchlist · Quality and Value')
        numeric_table(buys.head(10),['ticker','name','GPA','BM','BUY_score'])
    with rcol:
        st.markdown('##### SELL watchlist · Earnings Quality')
        numeric_table(sells.head(10),['ticker','name','STA','STA_sector_pct'])
    if view.empty:
        st.info('No companies match the current universe filters.')
        st.stop()
    ticker=ticker_selector(view,'research_company')
    r=view.loc[view.ticker.eq(ticker)].iloc[0]
    company_source_line(r)
    if r.conflicting_signals:st.warning('CONFLICT: This company meets both the value/quality screen and an earnings-quality risk flag.')
    ev=evidence(r)
    left,right=st.columns(2)
    with left:
        st.markdown('##### Fundamental evidence')
        for x in ev['facts']:st.markdown(f'- {x}')
        if not ev['facts']:st.info('Insufficient numeric evidence.')
    with right:
        st.markdown('##### Red flags and limitations')
        for x in ev['risks']:st.markdown(f'- {x}')
        if not ev['risks']:st.caption('No quantified red flags in available models; unmodeled risks remain.')
    with st.expander('Missing information and required due diligence',expanded=True):
        for x in ev['missing']:st.markdown(f'- {x}')
    note=st.text_area('Analyst-reviewed investment case / conclusion',key='note_'+ticker,
                      placeholder='Cite actual filings, discuss valuation, earnings quality, sector conditions, catalysts and a reasoned conclusion. This text is your own reviewed research.',height=135)
    st.download_button('Download company research brief (.md)',research_report(r,note),
                       file_name=f'{ticker}_research.md',mime='text/markdown')
    st.markdown(f'[Underlying Yahoo Finance statements for {ticker}](https://finance.yahoo.com/quote/{ticker}/financials/)')

st.divider()
st.caption('ACCT 3343 · Real records only · Market data and financial-reporting dates may differ · Screen outputs are not guaranteed investment outcomes.')
