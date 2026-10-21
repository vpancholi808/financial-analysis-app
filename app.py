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
from sp500_pipeline import anomaly_flags
from audit_data import audit_table

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

def money(n):
    if pd.isna(n): return 'N/A'
    x=abs(float(n))
    val=(f'{x/1e12:,.2f}T' if x>=1e12 else
         f'{x/1e9:,.2f}B' if x>=1e9 else
         f'{x/1e6:,.2f}M' if x>=1e6 else
         f'{x/1e3:,.1f}K' if x>=1e3 else f'{x:,.0f}')
    return ('-' if n<0 else '')+'$'+val

def fmt_value(v, pct=False):
    return 'N/A' if pd.isna(v) else (f'{v*100:,.1f}%' if pct else f'{v:,.3f}')

def numeric_table(df, cols, order_col=None):
    if df.empty:
        st.info('No companies meet these screening rules. Adjust the filters or load a larger universe.')
        return
    view = df[cols].copy()
    spec = {}
    for key in ['GPA', 'BM', 'STA', 'ROA', 'CFOA', 'SNOA', 'Altman_Z', 'Beneish_M']:
        if key in view: spec[key] = st.column_config.NumberColumn(key, format='%.3f')
    for key in ['GPA_sector_pct', 'BM_sector_pct','STA_sector_pct','SNOA_sector_pct','BUY_score']:
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
    members_path=ROOT/'sp500_constituents.csv'
    member_count=0
    if members_path.exists():
        membership=pd.read_csv(members_path)
        member_count=len(membership)
        data=data.loc[data.ticker.astype(str).isin(set(membership.ticker))].copy()
    else:
        data=data.iloc[0:0].copy()
    st.metric('S&P 500 statements imported',f'{len(data):,} / {member_count:,}' if member_count else 'Awaiting index list')
    st.progress(min(1.0,len(data)/member_count) if member_count else 0.0)
    st.caption('S&P 500 constituents · Yahoo Finance financials · TSM excluded')
    if state:
        st.caption(f"Imported {int(state.get('fetched_records',len(data))):,}; collection status: {state.get('status','unknown')}")
    st.caption('Full-universe collection is managed through GitHub Actions.')

if data.empty:
    st.warning('S&P 500 financial collection has not published its first matched statements. Previous non-index stocks are hidden; no synthetic data is displayed.')
    st.markdown('[Check collection workflow](https://github.com/vpancholi808/financial-analysis-app/actions)')
    st.stop()

base=prepare(data, min_sector_n=10, min_industry_n=10)
if base.empty:
    st.error('The dataset has no eligible tickers.')
    st.stop()

@st.cache_data(ttl=1800,show_spinner=False)
def completed_audit(source_data, source_members):
    return audit_table(source_data,source_members)
audit_issues,audit_summary=completed_audit(data,membership)

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
        snoa_pct=st.slider('SNOA · sector percentile ≥',0,100,80,5)
        beneish_cut=st.number_input('Beneish M-score screen cutoff',value=-1.78,step=.01)
        min_flags=st.slider('Minimum distinct SELL warning signals',1,4,1)
        st.caption('Signals: high STA, high SNOA, Beneish M-score, model-specific Altman distress. Missing inputs do not trigger a flag.')

settings=ScreenSettings(buy_gpa_pct=gpa_pct,buy_bm_pct=bm_pct,buy_score_min=buy_score,
                        buy_gpa_weight=gpa_weight,sell_sta_pct=sta_pct,sell_sta_floor=sta_floor,
                        sell_snoa_pct=snoa_pct,sell_beneish_cutoff=beneish_cut,sell_min_flags=min_flags)
filtered=scope_filter(base,sectors,industry,min_cap*1e9,excluded)
view=classify(filtered,settings)
buys=view.loc[view.BUY_candidate].sort_values('BUY_score',ascending=False)
sells=view.loc[view.SELL_candidate].sort_values(['SELL_flag_count','STA'],ascending=[False,False])

status='S&P 500 · SAVED REAL YAHOO FINANCIALS'
st.caption(f'{status} · {len(base)} fetched company records · {len(filtered)} shown after filters · no synthetic securities')
if 'TSM' not in excluded:
    st.error('TSM must be excluded in config.json.')
if len(base)<member_count:
    st.warning(f'S&P 500 coverage: {len(base):,} of {member_count:,} constituent listings ({len(base)/member_count:.1%}). Missing statements are not scored. Screen rankings are provisional until coverage is sufficient.')

if page=='Stock Screener':
    col1,col2,col3,col4=st.columns(4)
    col1.metric('Companies screened',len(view))
    col2.metric('BUY candidates',len(buys))
    col3.metric('SELL candidates',len(sells))
    col4.metric('Conflicting signals',int(view.conflicting_signals.sum()))
    audit=audit_issues
    a,b,c=st.columns(3)
    a.metric('Constituent coverage',f'{len(base)/member_count:.1%}' if member_count else 'N/A')
    b.metric('Missing S&P 500 statements',max(0,member_count-len(base)))
    c.metric('Accounting review flags',audit_summary['warnings']+audit_summary['review']+audit_summary['critical'])
    if not audit.empty and {'ticker','severity','category'}.issubset(audit):
        with st.expander('Data quality · flagged abnormalities and missing constituents'):
            st.caption('Flags invite verification against original SEC or issuer filings. Missing inputs are not silently imputed.')
            st.dataframe(audit[['ticker','severity','category','detail']].head(500),hide_index=True,use_container_width=True)
            st.download_button('Download full data-quality audit',audit.to_csv(index=False),'sp500_data_quality.csv','text/csv')
    st.subheader('Financial factor visualizations')
    if not view.empty:
        left,right=st.columns(2)
        with left:
            valid=view.loc[view.GPA.notna() & view.STA.notna()].copy()
            if not valid.empty:
                valid['Gross profitability (%)']=valid.GPA*100
                valid['Accruals / assets (%)']=valid.STA*100
                fig=px.scatter(valid,x='Gross profitability (%)',y='Accruals / assets (%)',color='sector',
                    hover_name='ticker',hover_data=['name'],title='Profitability vs. earnings-quality risk',
                    color_discrete_sequence=px.colors.qualitative.Safe)
                fig.update_layout(template='plotly_dark',paper_bgcolor='#0B1520',plot_bgcolor='#172636')
                st.plotly_chart(fig,use_container_width=True)
        with right:
            quality=view.groupby('sector',as_index=False).agg(Companies=('ticker','size'),GPA_median=('GPA','median'))
            quality['GPA_median']=quality.GPA_median*100
            fig=px.bar(quality.sort_values('GPA_median'),x='GPA_median',y='sector',
                orientation='h',title='Median gross profitability (%) by sector',
                color_discrete_sequence=['#20B59A'])
            fig.update_layout(template='plotly_dark',paper_bgcolor='#0B1520',plot_bgcolor='#172636')
            st.plotly_chart(fig,use_container_width=True)
        if {'GPA','STA','BM'}.issubset(view):
            factors=view[['GPA','BM','STA']].replace([np.inf,-np.inf],np.nan).dropna(how='all')
            if len(factors)>=5:
                fig=px.box(factors.melt(var_name='Factor',value_name='Ratio'),x='Factor',y='Ratio',
                    color='Factor',points='outliers',title='Accounting factor distributions · possible outliers')
                fig.update_layout(template='plotly_dark',paper_bgcolor='#0B1520',plot_bgcolor='#172636')
                st.plotly_chart(fig,use_container_width=True)
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
        st.caption('Independent risk indicators: high accruals, SNOA, Beneish M-score and model-appropriate Altman distress. Values unavailable due to missing fields stay N/A.')
        numeric_table(sells,['ticker','name','sector','STA','SNOA','Beneish_M','Altman_Z','Altman_model','SELL_flag_count','BUY_candidate'])
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
        if not audit.empty and 'severity' in audit:
            st.write('Detected data-quality issues by type')
            problems=audit.groupby(['severity','category']).size().reset_index(name='Count').sort_values('Count',ascending=False).head(15)
            if len(problems): st.plotly_chart(px.bar(problems,x='Count',y='category',color='severity',orientation='h'),use_container_width=True)
    with st.expander('Calculation methodology & data-quality audit'):
        st.markdown('''**GPA** = (revenue − cost of revenue) / assets · **B/M** = positive common equity / market cap (same currency) · **STA** = (net income − operating cash flow) / assets. Sector percentiles are calculated before user filtering. No missing inputs are filled with zero. When there are fewer than 10 valid peers, percentiles are shown as unavailable.''')
        st.json(validate_universe(base))
        st.json(audit_summary)

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
    a2,b2,c2=st.columns(3)
    a2.metric('SNOA',fmt_value(r.SNOA,True))
    b2.metric('Altman Z / Z double-prime',fmt_value(r.Altman_Z))
    c2.metric('Beneish M-score',fmt_value(r.Beneish_M))
    st.caption(f'Altman: {r.Altman_model} · {r.Altman_zone}; Beneish: {r.Beneish_status} (risk screen, not a probability of fraud)')
    flags=anomaly_flags(r.to_dict())
    if flags:
        st.warning('Data quality review: '+', '.join(x['issue'] for x in flags[:5]))
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
                             'Sector observations':r[m+'_sector_count']} for m in ['GPA','BM','STA','ROA','CFOA','SNOA','Beneish_M']])
        st.dataframe(peers,hide_index=True,use_container_width=True)
        observed=peers.loc[peers['Sector percentile'].notna()].copy()
        if len(observed):
            fig=px.bar(observed,x='Metric',y='Sector percentile',title='Company percentile position among S&P sector peers',text_auto='.0f',color='Metric')
            fig.update_yaxes(range=[0,100])
            fig.update_layout(template='plotly_dark',paper_bgcolor='#0B1520',plot_bgcolor='#172636',showlegend=False)
            st.plotly_chart(fig,use_container_width=True)
        if not r.industrial_applicable:st.warning('Specialist accounting is required for this sector; industrial rankings are deliberately unavailable.')
    with t3:
        fields=['revenue','cost_of_revenue','gross_profit','total_assets','common_equity','market_cap',
                'net_income','operating_cash_flow','prior_revenue','prior_net_income','prior_operating_cash_flow',
                'fiscal_year_end','prior_fiscal_year_end','data_source','refreshed_at']
        st.dataframe(pd.DataFrame({'Source field':fields,'Value':[money(r.get(k,np.nan)) if k in {'revenue','cost_of_revenue','gross_profit','total_assets','common_equity','market_cap','net_income','operating_cash_flow','prior_revenue','prior_net_income','prior_operating_cash_flow'} else str(r.get(k,'')) for k in fields]}),
                     hide_index=True,use_container_width=True)
        st.caption('Financial line item definitions can differ across issuers. Verify against SEC/issuer filings.')
    with st.expander('Advanced models · required data and exact formulas'):
        st.markdown('**SNOA** = [(Total assets − cash) − (Total assets − ST debt − LT debt − minority interest − preferred equity − common equity)] / Total assets. All named source components are required; missing is NOT treated as zero.')
        st.markdown('**Altman**: Original public-manufacturer Z or non-manufacturer Z double-prime, according to business classification. Banks and REITs are N/A. The manufacturer form uses CURRENT market capitalization against annual statement figures; treat its score as a screening signal, not a historical backtest.')
        st.markdown('**Beneish**: Eight-variable 1999 M-score, with an adjustable -1.78 screening threshold. Net income is used as a flagged proxy for continuing-operations income in TATA; no calibrated manipulation probability is claimed.')
        st.caption('Every model shows N/A unless the required Yahoo financial inputs are present. Sub-industry model assignment warrants analyst review.')
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
        numeric_table(sells.head(10),['ticker','name','STA','SNOA','Beneish_M','Altman_Z','SELL_flag_count'])
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
