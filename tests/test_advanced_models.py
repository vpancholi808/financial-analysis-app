"""Hand-calculated checks for course SNOA and published Altman/Beneish."""
import math
import numpy as np
import pandas as pd
import pytest
from advanced_models import snoa, altman, beneish
from finance_models import prepare, classify, ScreenSettings

def sample(**kw):
    d=dict(ticker="TST",sector="Information Technology",industry="Semiconductors",
       market_currency="USD",financial_currency="USD",
       total_assets=1000.,cash_and_equivalents=100.,
       short_term_debt=50.,long_term_debt=150.,minority_interest=20.,
       preferred_equity=10.,common_equity=300.,market_cap=1500.,
       current_assets=400.,current_liabilities=200.,retained_earnings=100.,
       ebit=150.,total_liabilities=700.,revenue=1200.,cost_of_revenue=600.,
       accounts_receivable=120.,net_ppe=300.,depreciation=60.,sga=120.,
       net_income=150.,operating_cash_flow=100.,
       prior_revenue=1000.,prior_cost_of_revenue=500.,prior_total_assets=900.,
       prior_receivables=100.,prior_current_assets=350.,prior_net_ppe=250.,
       prior_depreciation=50.,prior_sga=100.,prior_current_liabilities=180.,
       prior_long_term_debt=280.)
    d.update(kw)
    return d

def test_exact_snoa_user_formula():
    # Operating assets = 1000 - 100 = 900
    # Operating liabilities = 1000 - 50 - 150 - 20 - 10 - 300 = 470
    assert snoa(sample())==pytest.approx(.43)

def test_snoa_does_not_invent_zero_minority_interest():
    assert math.isnan(snoa(sample(minority_interest=np.nan)))

def test_public_manufacturer_altman_hand_check():
    z,model,zone=altman(sample())
    expected=1.2*(200/1000)+1.4*(100/1000)+3.3*(150/1000)+.6*(1500/700)+1200/1000
    assert z==pytest.approx(expected)
    assert zone=="Lower distress signal"
    assert "manufacturer" in model

def test_nonmanufacturer_variant_differs():
    z,model,zone=altman(sample(industry="Internet Services"))
    expected=6.56*.2+3.26*.1+6.72*.15+1.05*(300/700)
    assert z==pytest.approx(expected)
    assert "double-prime" in model

def test_altman_not_appropriate_for_banks():
    z,_,zone=altman(sample(sector="Financials"))
    assert math.isnan(z)
    assert zone=="Not applicable"

def test_beneish_eight_components_and_independent_score():
    score,terms,status=beneish(sample())
    assert len(terms)==8
    assert terms["DSRI"]==pytest.approx(1)
    assert terms["GMI"]==pytest.approx(1)
    assert terms["AQI"]==pytest.approx(.9)
    assert terms["SGI"]==pytest.approx(1.2)
    assert terms["DEPI"]==pytest.approx(1)
    assert terms["SGAI"]==pytest.approx(1)
    assert terms["TATA"]==pytest.approx(.05)
    expected=(-4.84+.92*terms["DSRI"]+.528*terms["GMI"]+
              .404*terms["AQI"]+.892*terms["SGI"]+.115*terms["DEPI"]-
              .172*terms["SGAI"]+4.679*terms["TATA"]-.327*terms["LVGI"])
    assert score==pytest.approx(expected)
    assert status=="Below screen cutoff"

def test_beneish_incomplete_prior_inputs_stays_missing():
    score,_,status=beneish(sample(prior_sga=np.nan))
    assert math.isnan(score)
    assert status=="Insufficient input"

def test_advanced_columns_and_separate_risk_flags():
    d=prepare(pd.DataFrame([sample()]),min_sector_n=1)
    assert d.iloc[0].SNOA==pytest.approx(.43)
    assert pd.notna(d.iloc[0].Altman_Z)
    assert pd.notna(d.iloc[0].Beneish_M)
    result=classify(d,ScreenSettings(sell_sta_pct=0,sell_sta_floor=0,
                                    sell_snoa_pct=0,sell_beneish_cutoff=-10))
    assert bool(result.iloc[0].Beneish_risk_flag)
    assert int(result.iloc[0].SELL_flag_count)>=2
