"""Documented advanced accounting signals for ACCT 3343.

SNOA implements the supplied course definition, without treating absent data
as zero. Altman variants use different published coefficients/thresholds.
Beneish is an eight-variable M-SCORE, NOT a calibrated probability of fraud.
Incomplete, invalid or sector-incompatible source data produces NaN.
"""
import math
import numpy as np
import pandas as pd

EXCLUDED_SECTORS={"Financials","Real Estate"}
# Unambiguously physical-production GICS-like subindustries; other eligible
# operating companies use Z-double-prime. Classification is reviewable.
MANUFACTURER_TERMS=(
 "manufactur","semiconductor","automobile","automotive",
 "machinery","industrial conglomerat","electronic equipment",
 "electrical equipment","household appliances","consumer electronics",
 "building products","aerospace","defense","construction machinery",
 "food products","beverages","chemicals","paper & forest",
 "personal care products","metal","materials","textiles",
 "tobacco","packaged food","pharmaceuticals","medical equipment",
)

def get(row, key):
    try:
        v=float(row.get(key,np.nan))
        return v if math.isfinite(v) else np.nan
    except (TypeError,ValueError):
        return np.nan

def ratio(n,d,positive=True):
    if not (np.isfinite(n) and np.isfinite(d)):return np.nan
    if (d<=0 if positive else d==0):return np.nan
    v=n/d
    return v if np.isfinite(v) else np.nan

def snoa(row):
    """[(assets - cash) - (assets - ST debt - LT debt - minority
    interest - preferred equity - book common equity)] / assets."""
    a=get(row,"total_assets");cash=get(row,"cash_and_equivalents")
    st=get(row,"short_term_debt");lt=get(row,"long_term_debt")
    mi=get(row,"minority_interest");pref=get(row,"preferred_equity")
    common=get(row,"common_equity")
    if not np.isfinite(a) or a<=0:return np.nan
    if not all(np.isfinite(x) for x in [cash,st,lt,mi,pref,common]):return np.nan
    operating_assets=a-cash
    operating_liabilities=a-st-lt-mi-pref-common
    return ratio(operating_assets-operating_liabilities,a)

def altman(row):
    """Return (score, model, zone); not a definitive bankruptcy probability."""
    sector=str(row.get("sector",""))
    if sector in EXCLUDED_SECTORS:return np.nan,"Not applicable","Not applicable"
    assets=get(row,"total_assets")
    ca=get(row,"current_assets");cl=get(row,"current_liabilities")
    re=get(row,"retained_earnings");ebit=get(row,"ebit")
    tl=get(row,"total_liabilities")
    if not np.isfinite(assets) or assets<=0 or not np.isfinite(tl) or tl<=0:
        return np.nan,"Insufficient input","N/A"
    if not all(np.isfinite(x) for x in (ca,cl,re,ebit)):
        return np.nan,"Insufficient input","N/A"
    x1=(ca-cl)/assets;x2=re/assets;x3=ebit/assets
    industry=str(row.get("industry","")).lower()
    manufacturing=any(k in industry for k in MANUFACTURER_TERMS)
    if manufacturing:
        mv=get(row,"market_cap")
        if str(row.get("market_currency","")).upper()!=str(row.get("financial_currency","")).upper():
            return np.nan,"Original Z (manufacturer)","N/A"
        if not np.isfinite(mv) or mv<=0:return np.nan,"Original Z (manufacturer)","N/A"
        score=1.2*x1+1.4*x2+3.3*x3+.6*mv/tl+get(row,"revenue")/assets
        model="Original Z (public manufacturer)"
        zone="Distress" if score<1.81 else "Grey" if score<=2.99 else "Lower distress signal"
    else:
        equity=get(row,"common_equity")
        if not np.isfinite(equity):return np.nan,"Z double-prime (nonmanufacturer)","N/A"
        score=6.56*x1+3.26*x2+6.72*x3+1.05*equity/tl
        model="Z double-prime (nonmanufacturer)"
        zone="Distress" if score<1.1 else "Grey" if score<=2.6 else "Lower distress signal"
    if not np.isfinite(score):return np.nan,model,"N/A"
    return score,model,zone

def beneish(row):
    """Original eight-variable Beneish M-score; uses net income as TATA proxy
    ONLY when continuing-operations income is not separately available, and
    labels that proxy. No imputation of missing components.
    """
    if str(row.get("sector","")) in EXCLUDED_SECTORS:return np.nan,{}, "Not applicable"
    v=lambda k:get(row,k)
    sales=v("revenue");prev=v("prior_revenue")
    assets=v("total_assets");prev_assets=v("prior_total_assets")
    co=v("cost_of_revenue");prev_co=v("prior_cost_of_revenue")
    ar=v("accounts_receivable");prev_ar=v("prior_receivables")
    ca=v("current_assets");prev_ca=v("prior_current_assets")
    pp=v("net_ppe");prev_pp=v("prior_net_ppe")
    dep=v("depreciation");prev_dep=v("prior_depreciation")
    sga=v("sga");prev_sga=v("prior_sga")
    cl=v("current_liabilities");prev_cl=v("prior_current_liabilities")
    ltd=v("long_term_debt");prev_ltd=v("prior_long_term_debt")
    ni=v("net_income");cfo=v("operating_cash_flow")
    if not all(np.isfinite(x) for x in [sales,prev,assets,prev_assets]) or min(sales,prev,assets,prev_assets)<=0:
        return np.nan,{}, "Insufficient input"
    prior_margin=ratio(prev-prev_co,prev);margin=ratio(sales-co,sales)
    dsri=ratio(ratio(ar,sales),ratio(prev_ar,prev))
    gmi=ratio(prior_margin,margin)
    aqi=ratio(1-ratio(ca+pp,assets),1-ratio(prev_ca+prev_pp,prev_assets))
    sgi=ratio(sales,prev)
    depi=ratio(ratio(prev_dep,prev_dep+prev_pp),ratio(dep,dep+pp))
    sgai=ratio(ratio(sga,sales),ratio(prev_sga,prev))
    lvgi=ratio(ratio(cl+ltd,assets),ratio(prev_cl+prev_ltd,prev_assets))
    tata=ratio(ni-cfo,assets,positive=True)
    terms=dict(DSRI=dsri,GMI=gmi,AQI=aqi,SGI=sgi,
               DEPI=depi,SGAI=sgai,LVGI=lvgi,TATA=tata)
    if not all(np.isfinite(x) for x in terms.values()):return np.nan,terms,"Insufficient input"
    m=(-4.84+.920*dsri+.528*gmi+.404*aqi+.892*sgi+.115*depi
       -.172*sgai+4.679*tata-.327*lvgi)
    return m,terms,"Above screen cutoff" if m>-1.78 else "Below screen cutoff"

def compute_advanced(row):
    score,model,zone=altman(row)
    ms,terms,status=beneish(row)
    return pd.Series({"SNOA":snoa(row),"Altman_Z":score,"Altman_model":model,
      "Altman_zone":zone,"Beneish_M":ms,"Beneish_status":status,
      "Beneish_inputs_used":len(terms)})
