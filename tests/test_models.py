from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from finance_models import safe_ratio, prepare, classify, scope_filter, ScreenSettings
from build_universe import shared_fiscal_dates, extract_company, series_at, build
from research import evidence, growth, research_report


def row(ticker='AAA', sector='Information Technology', **changes):
    d = dict(ticker=ticker, name='Test real-like fixture', sector=sector, industry='Software',
             exchange='NMS', market_cap=500.0, market_currency='USD', financial_currency='USD',
             revenue=800., cost_of_revenue=300., total_assets=1000.,
             net_income=80., operating_cash_flow=35., common_equity=200.,
             fiscal_year_end='2025-12-31',price_asof='2026-10-20',data_source='yfinance fixture')
    d.update(changes)
    return d


def cohort():
    return pd.DataFrame([row(ticker=f'R{i}',cost_of_revenue=200+17*i,
                             common_equity=70+10*i,net_income=120+8*i,
                             operating_cash_flow=140-2*i) for i in range(12)])


def test_manual_gpa_bm_sta():
    v=prepare(pd.DataFrame([row()]),min_sector_n=1).iloc[0]
    assert v.GPA == pytest.approx(.5)
    assert v.BM == pytest.approx(.4)
    assert v.STA == pytest.approx(.045)
    assert v.ROA == pytest.approx(.08)
    assert v.CFOA == pytest.approx(.035)


def test_missing_not_zero_and_bad_denominator():
    v=prepare(pd.DataFrame([row(net_income=np.nan,common_equity=np.nan,total_assets=-100)]),min_sector_n=1).iloc[0]
    assert pd.isna(v.GPA) and pd.isna(v.STA) and pd.isna(v.BM)
    assert safe_ratio(pd.Series([1,1]),pd.Series([0,-1])).isna().all()


def test_currency_guard():
    v=prepare(pd.DataFrame([row(financial_currency='CAD')]),min_sector_n=1).iloc[0]
    assert pd.isna(v.BM) and v.GPA == pytest.approx(.5)


def test_sector_not_comparable():
    d=prepare(pd.DataFrame([row('BANK',sector='Financials'),row('REIT',sector='Real Estate')]),min_sector_n=1)
    s=classify(d,ScreenSettings(buy_gpa_pct=0,buy_bm_pct=0,buy_score_min=0,sell_sta_pct=0,sell_sta_floor=-1))
    assert not s.BUY_candidate.any() and not s.SELL_candidate.any()


def test_benchmarks_stable_across_filters():
    prepared=prepare(cohort(),min_sector_n=10)
    original=prepared.loc[prepared.ticker=='R10','GPA_sector_pct'].iloc[0]
    subset=scope_filter(prepared,min_cap=0).loc[prepared.ticker=='R10']
    assert subset.iloc[0].GPA_sector_pct==original


def test_small_cohort_has_no_percentiles():
    d=prepare(cohort().head(4),min_sector_n=10)
    assert d.GPA_sector_pct.isna().all()
    assert not classify(d,ScreenSettings()).BUY_candidate.any()


def test_buy_and_sell_independent_and_conflicting():
    d=classify(prepare(cohort()),ScreenSettings(
        buy_gpa_pct=0,buy_bm_pct=0,buy_score_min=0,
        sell_sta_pct=0,sell_sta_floor=-1))
    assert d.BUY_candidate.all() and d.SELL_candidate.all()
    assert d.conflicting_signals.all()


def test_percentile_changes_buy_count():
    base=prepare(cohort())
    easy=classify(base,ScreenSettings(buy_gpa_pct=0,buy_bm_pct=0,buy_score_min=0))
    strict=classify(base,ScreenSettings(buy_gpa_pct=95,buy_bm_pct=95,buy_score_min=95))
    assert easy.BUY_candidate.sum()>strict.BUY_candidate.sum()


def test_exclusion_and_missing_cap_research_visibility():
    d=prepare(pd.DataFrame([row('ONE',market_cap=np.nan),row('TWO')]),min_sector_n=1)
    selected=scope_filter(d,excluded_tickers={'TWO'},min_cap=0)
    assert list(selected.ticker)==['ONE']


def test_statement_date_intersection():
    yr1=pd.Timestamp('2025-12-31');yr0=pd.Timestamp('2024-12-31')
    income=pd.DataFrame({yr1:[100],yr0:[80]},index=['Total Revenue'])
    bal=pd.DataFrame({yr0:[400]},index=['Total Assets'])
    cash=pd.DataFrame({yr1:[10],yr0:[20]},index=['Operating Cash Flow'])
    assert shared_fiscal_dates(income,bal,cash)==[yr0]
    assert series_at(income,yr0,['Total Revenue'])==80


class FakeYahooTicker:
    def __init__(self):
        y1=pd.Timestamp('2025-12-31');y0=pd.Timestamp('2024-12-31')
        self.info={'sector':'Technology','industry':'Semiconductors','longName':'Verified Fixture',
                   'quoteType':'EQUITY','currency':'USD','financialCurrency':'USD',
                   'marketCap':1_000_000_000, 'exchange':'NMS'}
        self.income_stmt=pd.DataFrame({y1:[1_200.,400.,130.],y0:[1_000.,380.,90.]},
                                      index=['Total Revenue','Cost Of Revenue','Net Income'])
        self.balance_sheet=pd.DataFrame({y1:[2_000.,600.],y0:[1_700.,500.]},
                                        index=['Total Assets','Common Stock Equity'])
        self.cashflow=pd.DataFrame({y1:[170.],y0:[120.]},index=['Operating Cash Flow'])


def test_yahoo_extraction_matches_fiscal_dates_and_ratios():
    item=extract_company(FakeYahooTicker(),{'symbol':'AAPL'})
    assert item['ticker']=='AAPL'
    assert item['fiscal_year_end']=='2025-12-31'
    assert item['prior_fiscal_year_end']=='2024-12-31'
    assert item['revenue']==1200.
    assert item['prior_revenue']==1000.
    r=prepare(pd.DataFrame([item]),min_sector_n=1).iloc[0]
    assert r.GPA==pytest.approx(.4)
    assert r.STA==pytest.approx((130-170)/2000)
    assert r.BM==pytest.approx(600/1_000_000_000)


def test_yahoo_rejects_etf_and_missing_periods():
    fake=FakeYahooTicker();fake.info['quoteType']='ETF'
    assert extract_company(fake,{'symbol':'BAD'}) is None
    fake=FakeYahooTicker();fake.cashflow=pd.DataFrame()
    assert extract_company(fake,{'symbol':'BAD'}) is None


def test_research_output_provenance_and_missing_limitations():
    r=classify(prepare(pd.DataFrame([row(prior_revenue=700.)]),min_sector_n=1),
               ScreenSettings()).iloc[0]
    report=research_report(r,'Pending analyst review')
    assert '2025-12-31' in report and 'yfinance fixture' in report
    assert 'SNOA' in report and 'Beneish' in report
    assert 'Pending analyst review' in report
    assert '800' in report
    assert growth(800,700)==pytest.approx(1/7)
    assert np.isnan(growth(800,-100))


def test_no_demo_generated_or_referenced():
    src=(Path(__file__).resolve().parents[1]/'app.py').read_text()
    assert 'demo_data(' not in src
    assert 'Synthetic Company' not in src
    assert 'Synthetic Company' not in (Path(__file__).resolve().parents[1]/'universe.csv').read_text()


def test_share_classes_count_once_in_benchmarks():
    rows=cohort()
    twin=rows.iloc[[3]].copy();twin["ticker"]="R3B"
    with_twin=prepare(pd.concat([rows,twin],ignore_index=True))
    alone=prepare(rows)
    t=with_twin.set_index("ticker")
    assert t.loc["R3B","share_class_of"]=="R3" and not t.loc["R3B","issuer_primary"]
    # The duplicate listing does not move anyone's sector percentile.
    assert np.allclose(t.loc[alone.ticker,"GPA_sector_pct"].values,alone.GPA_sector_pct.values)
    assert t.loc["R3B","GPA_sector_pct"]==pytest.approx(t.loc["R3","GPA_sector_pct"])


def test_stale_statements_visible_but_not_screened():
    rows=cohort()
    rows.loc[0,"fiscal_year_end"]="2024-06-30"   # ~840 days before the quote date
    d=classify(prepare(rows),ScreenSettings(buy_gpa_pct=0,buy_bm_pct=0,buy_score_min=0,
                                           sell_sta_pct=0,sell_sta_floor=-1))
    r=d.set_index("ticker").loc["R0"]
    assert bool(r.stale_statements) and not bool(r.scoreable)
    assert pd.notna(r.GPA) and pd.isna(r.GPA_sector_pct)
    assert not bool(r.BUY_candidate) and not bool(r.SELL_candidate)
    assert d.loc[d.ticker.ne("R0"),"BUY_candidate"].any()


def test_prepare_handles_empty_frame():
    d=prepare(cohort().iloc[:0])
    assert len(d)==0 and "SNOA" in d.columns and "Altman_zone" in d.columns
