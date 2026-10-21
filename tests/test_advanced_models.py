"""Hand-calculated checks for course SNOA and published Altman/Beneish."""
import math
import numpy as np
import pandas as pd
import pytest
from advanced_models import (snoa, snoa_with_basis, altman, beneish, is_manufacturer,
                             SNOA_COURSE, SNOA_IDENTITY)
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
    # With MI missing, MI + PE comes from the balance sheet (TA - TL - CE = 0 here),
    # so operating liabilities = TL - STD - LTD = 500 and SNOA = (900 - 500) / 1000.
    value,basis=snoa_with_basis(sample(minority_interest=np.nan))
    assert value==pytest.approx(.40)
    assert basis==SNOA_IDENTITY
    # The balance-sheet value (MI + PE = 0) is used, not the reported PE line with MI = 0 (0.41).
    assert value!=pytest.approx(snoa(sample(minority_interest=0.)))

def test_snoa_leaves_out_unreported_debt_and_labels_it():
    # Short-term debt not reported: OL = 1000 - 0 - 150 - 20 - 10 - 300 = 520 -> (900 - 520) / 1000
    value,basis=snoa_with_basis(sample(short_term_debt=np.nan))
    assert value==pytest.approx(.38)
    assert basis.startswith("Partial inputs") and "short-term debt" in basis

def test_snoa_without_total_liabilities_leaves_out_unreported_equity_lines():
    # Course form with MI left out: OL = 1000 - 50 - 150 - 10 - 300 = 490 -> (900 - 490) / 1000
    value,basis=snoa_with_basis(sample(minority_interest=np.nan,total_liabilities=np.nan))
    assert value==pytest.approx(.41)
    assert "minority interest" in basis

def test_snoa_still_requires_assets_cash_and_liabilities_or_equity():
    assert math.isnan(snoa(sample(cash_and_equivalents=np.nan)))
    assert math.isnan(snoa(sample(total_assets=np.nan)))
    assert math.isnan(snoa(sample(minority_interest=np.nan,total_liabilities=np.nan,common_equity=np.nan)))

def test_snoa_identity_matches_course_formula_on_consistent_balance_sheet():
    # A balance sheet where TL = TA - CE - MI - PE (Yahoo's liabilities net of MI).
    consistent=dict(total_liabilities=1000.-300.-20.-10.)
    course,course_basis=snoa_with_basis(sample(**consistent))
    ident,ident_basis=snoa_with_basis(sample(minority_interest=np.nan,preferred_equity=np.nan,**consistent))
    assert course_basis==SNOA_COURSE and ident_basis==SNOA_IDENTITY
    assert ident==pytest.approx(course)

def test_altman_uses_exact_subindustry_list():
    assert is_manufacturer("Technology Hardware, Storage & Peripherals")
    assert is_manufacturer("Health Care Equipment")
    assert not is_manufacturer("Automotive Retail")
    _,model,_=altman(sample(industry="Automotive Retail"))
    assert "double-prime" in model
    _,model,_=altman(sample(industry="Steel"))
    assert "manufacturer" in model and "double-prime" not in model

def test_altman_not_applied_to_utilities():
    z,model,zone=altman(sample(sector="Utilities",industry="Electric Utilities"))
    assert math.isnan(z) and zone=="Not applicable" and "utility" in model

def test_altman_negative_equity_is_review_not_distress():
    z,model,zone=altman(sample(industry="Restaurants",common_equity=-200.))
    assert np.isfinite(z)
    assert zone.startswith("Review")
    d=classify(prepare(pd.DataFrame([sample(industry="Restaurants",common_equity=-200.)]),min_sector_n=1),ScreenSettings())
    assert not bool(d.iloc[0].Altman_risk_flag)

def test_beneish_outlier_is_reported_but_not_flagged():
    # Prior receivables near zero -> DSRI far outside its plausible band.
    m,terms,status=beneish(sample(prior_receivables=1.))
    assert terms["DSRI"]>4 and np.isfinite(m)
    assert status.startswith("Component outlier") and "DSRI" in status
    d=classify(prepare(pd.DataFrame([sample(prior_receivables=1.)]),min_sector_n=1),
               ScreenSettings(sell_beneish_cutoff=-10))
    assert bool(d.iloc[0].Beneish_outlier)
    assert not bool(d.iloc[0].Beneish_risk_flag)

def test_beneish_year_order_with_asymmetric_inputs():
    # Every index differs from 1, so swapping current and prior years would fail.
    m,terms,_=beneish(sample(accounts_receivable=150.,cost_of_revenue=660.,depreciation=45.,sga=150.))
    assert terms["DSRI"]==pytest.approx((150/1200)/(100/1000))
    assert terms["GMI"]==pytest.approx(((1000-500)/1000)/((1200-660)/1200))
    assert terms["DEPI"]==pytest.approx((50/(50+250))/(45/(45+300)))
    assert terms["SGAI"]==pytest.approx((150/1200)/(100/1000))

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
