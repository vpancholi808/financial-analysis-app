"""Resumable S&P 500 financial statement collection using yfinance.

Index membership is obtained from the publicly maintained S&P 500 constituent
dataset; ALL FINANCIAL VALUES come only from yfinance statements. No estimates
or synthetic numbers are written. Returns an audit for every attempted symbol.
"""
from __future__ import annotations
import argparse
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from build_universe import extract_company, issuer_key
from finance_models import GICS_SECTORS, UNIVERSE_COLUMNS

ROOT = Path(__file__).resolve().parent
CONSTITUENTS = ROOT / "sp500_constituents.csv"
UNIVERSE = ROOT / "universe.csv"
STATE = ROOT / "collection_state.json"
AUDIT = ROOT / "data_quality.csv"
INDEX_SOURCE = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
QUALITY_COLUMNS = ["ticker", "severity", "issue", "detail"]
INDEX_COLUMNS = ["index_symbol", "ticker", "name", "sector", "sub_industry"]

def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def read_state():
    try:
        return json.loads(STATE.read_text())
    except (OSError, ValueError):
        return {}

def write_state(state):
    STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")

def normalize_symbol(value):
    # S&P/Wikipedia denotes share classes BRK.B, Yahoo Finance expects BRK-B.
    return str(value).strip().upper().replace(".", "-")

def parse_membership(raw):
    df = pd.read_csv(io.StringIO(raw), dtype=str).fillna("")
    mapping = {"Symbol":"index_symbol", "Security":"name", "GICS Sector":"sector",
               "GICS Sub-Industry":"sub_industry",
               "symbol":"index_symbol", "name":"name", "sector":"sector",
               "sub_industry":"sub_industry"}
    df = df.rename(columns=mapping)
    if not {"index_symbol","name","sector"}.issubset(df.columns):
        raise ValueError("Membership feed is missing ticker/name/sector fields")
    if "sub_industry" not in df:
        df["sub_industry"] = ""
    df["ticker"] = df["index_symbol"].map(normalize_symbol)
    df = df[INDEX_COLUMNS].loc[lambda d:d.ticker.ne("")]
    df = df.drop_duplicates("ticker").reset_index(drop=True)
    if not 490 <= len(df) <= 515 or not df.sector.isin(GICS_SECTORS).all():
        raise ValueError(f"Suspicious S&P constituent snapshot: {len(df)} rows, sectors {df.sector.unique().tolist()}")
    if "TSM" in set(df.ticker):
        raise ValueError("TSM unexpectedly present in S&P 500 constituent dataset")
    return df

def load_membership(refresh=True, fetch=None):
    """Use known snapshot if remote fetch fails; do not fake a new membership date."""
    if fetch is None:
        def fetch(url):
            r = requests.get(url, headers={"User-Agent":"ACCT3343 student equity research/1.0"}, timeout=35)
            r.raise_for_status()
            return r.text
    if refresh or not CONSTITUENTS.exists():
        try:
            df = parse_membership(fetch(INDEX_SOURCE))
            df.to_csv(CONSTITUENTS, index=False)
            return df, "fresh"
        except Exception as exc:
            if not CONSTITUENTS.exists():
                raise RuntimeError(f"Cannot establish S&P 500 membership: {type(exc).__name__}: {exc}") from exc
            print("MEMBERSHIP REFRESH FAILED, using last recorded snapshot:", exc, flush=True)
            return parse_membership(CONSTITUENTS.read_text()), "cached"
    return parse_membership(CONSTITUENTS.read_text()), "cached"

def anomaly_flags(r, reference_date=None):
    """Detect suspicious inputs; warnings are never silently overwritten."""
    issues = []
    def flag(level, code, detail):
        issues.append(dict(ticker=str(r.get("ticker","")), severity=level, issue=code, detail=detail))
    def value(col):
        return pd.to_numeric(r.get(col), errors="coerce")
    assets, rev, gp, cash = value("total_assets"), value("revenue"), value("gross_profit_reported"), value("cash_and_equivalents")
    cap = value("market_cap")
    if not np.isfinite(assets) or assets <= 0:
        flag("critical","INVALID_ASSETS","Total assets missing or <= 0")
    if not np.isfinite(cap) or cap <= 0:
        flag("warning","MISSING_MARKET_CAP","Market capitalization missing or <= 0")
    if pd.notna(rev) and rev < 0:
        flag("warning","NEGATIVE_REVENUE","Check revenue statement mapping")
    if pd.notna(cash) and pd.notna(assets) and (cash < 0 or cash > assets * 1.05):
        flag("warning","CASH_OUTSIDE_ASSETS","Cash negative or materially exceeds assets")
    if pd.notna(gp) and pd.notna(rev) and rev > 0 and gp > rev * 1.05:
        flag("warning","GROSS_PROFIT_GT_REVENUE","Reported gross profit exceeds revenue")
    calculated_gp = value("revenue") - value("cost_of_revenue")
    if pd.notna(calculated_gp) and pd.notna(gp) and abs(calculated_gp-gp)>max(abs(rev)*.05,1):
        flag("warning","GROSS_PROFIT_DISCREPANCY","Reported gross profit differs materially from revenue less COGS")
    if str(r.get("financial_currency") or "").upper()!=str(r.get("market_currency") or "").upper():
        flag("warning","CURRENCY_MISMATCH","B/M cannot be calculated without FX conversion")
    for name in ("net_income","operating_cash_flow","total_assets","common_equity"):
        if pd.isna(value(name)):
            flag("warning","MISSING_"+name.upper(),"Required source financial figure unavailable")
    fy = pd.to_datetime(r.get("fiscal_year_end"), errors="coerce")
    if pd.isna(fy):
        flag("critical","MISSING_FISCAL_DATE","No matched annual fiscal date")
    else:
        reference_date = reference_date or datetime.now(timezone.utc)
        age = (pd.Timestamp(reference_date).tz_localize(None) - fy).days
        if age > 650:
            flag("warning","STALE_ANNUAL_STATEMENTS",f"Most recent matched annual figures {age} days old")
        if age < -65:
            flag("critical","FUTURE_FISCAL_DATE","Fiscal year occurs in the future")
    return issues

def complete_fraction(df, keys=("gross_profit_reported","net_income","operating_cash_flow","total_assets","common_equity")):
    if df.empty: return 0.
    return float(df[list(keys)].notna().all(axis=1).mean())

def collect(batch_size=75, retry_cap=3, pause=1.0, refresh_index=True):
    import yfinance as yf
    members, source_status = load_membership(refresh=refresh_index)
    members_map = {x.ticker:x for x in members.itertuples(index=False)}
    current = pd.read_csv(UNIVERSE) if UNIVERSE.exists() else pd.DataFrame(columns=UNIVERSE_COLUMNS)
    if "ticker" not in current: current = pd.DataFrame(columns=UNIVERSE_COLUMNS)
    # Remove older broad-market, non-S&P names; membership classification wins.
    current = current.loc[current.ticker.astype(str).str.upper().isin(members_map)].copy()
    current = current.loc[current.ticker.astype(str).str.upper().ne("TSM")].drop_duplicates("ticker")
    rows = {str(r["ticker"]).upper():r for r in current.to_dict("records")}
    prior = read_state()
    if prior.get("universe") != "SP500" or prior.get("membership_snapshot") != sorted(members_map):
        attempts = {}
    else:
        attempts = {str(k):int(v) for k,v in prior.get("attempts",{}).items()}
    # A prior dataset predates SNOA/Beneish inputs. Re-fetch each existing
    # issuer once for the extended schema, without discarding its old statements
    # if Yahoo temporarily fails.
    refresh_attempts = {str(k):int(v) for k,v in prior.get("schema_refresh_attempts",{}).items()}
    needs_schema_refresh = [s for s in members_map if s in rows and refresh_attempts.get(s,0)==0]
    missing_members = [s for s in members_map if s not in rows and attempts.get(s,0)<retry_cap]
    pending = missing_members + needs_schema_refresh
    print(f"S&P constituents: {len(members)}; fetched: {len(rows)}; missing: {len(missing_members)}; scheduled refresh: {len(needs_schema_refresh)}", flush=True)
    new_count = 0
    for symbol in pending[:batch_size]:
        if symbol in rows:
            refresh_attempts[symbol] = refresh_attempts.get(symbol,0) + 1
        else:
            attempts[symbol] = attempts.get(symbol,0) + 1
        member = members_map[symbol]
        try:
            r = extract_company(yf.Ticker(symbol), {"symbol":symbol, "quoteType":"EQUITY"})
            if r is None:
                print("MISSING",symbol,flush=True)
            else:
                # Membership metadata is independently sourced; all monetary figures are yfinance.
                r["sector"] = member.sector
                r["industry"] = member.sub_industry or r.get("industry","Unknown")
                r["data_source"] = "Financials: Yahoo Finance (yfinance); membership: S&P 500 constituent list"
                if "symbol" in r and not r["ticker"]: r["ticker"] = symbol
                r["ticker"] = symbol
                if pd.notna(pd.to_numeric(r.get("total_assets"),errors="coerce")) and float(r["total_assets"])>0:
                    rows[symbol] = r
                    new_count += 1
                    print("ACCEPT",symbol,"total",len(rows),flush=True)
                else:
                    print("INVALID ASSETS",symbol,flush=True)
        except Exception as exc:
            print("FETCH FAILED",symbol,type(exc).__name__,str(exc)[:150],flush=True)
        if pause: time.sleep(pause)
    ordered = [rows[s] for s in members_map if s in rows]
    good = pd.DataFrame(ordered,columns=UNIVERSE_COLUMNS)
    good.to_csv(UNIVERSE,index=False)
    audit = []
    for r in ordered:
        audit.extend(anomaly_flags(r))
    for s in members_map:
        if s not in rows:
            audit.append(dict(ticker=s,severity="missing",issue="FINANCIALS_NOT_FETCHED",
                              detail=f"Yahoo statement attempt count: {attempts.get(s,0)} / {retry_cap}"))
    pd.DataFrame(audit,columns=QUALITY_COLUMNS).to_csv(AUDIT,index=False)
    state = {
        "universe":"SP500","membership_source":INDEX_SOURCE,"membership_status":source_status,
        "membership_snapshot":sorted(members_map),"membership_rows":len(members),
        "fetched_records":len(good),"missing_records":len(members)-len(good),
        "new_records_last_batch":new_count,"attempts":attempts,
        "schema_refresh_attempts":refresh_attempts,
        "schema_refresh_pending":sum(s in rows and refresh_attempts.get(s,0)==0 for s in members_map),
        "attempted_total":sum(attempts.values()),"batch_size":batch_size,
        "quality_warning_count":sum(a["severity"]=="warning" for a in audit),
        "quality_critical_count":sum(a["severity"]=="critical" for a in audit),
        "statement_completeness":complete_fraction(good),
        "updated_at":now(),
        "status":"complete" if len(good)==len(members) else
                 "needs_manual_review" if not any(s not in rows and attempts.get(s,0)<retry_cap for s in members_map)
                 else "collecting",
    }
    write_state(state)
    print(json.dumps({k:v for k,v in state.items() if k not in ("membership_snapshot","attempts","schema_refresh_attempts")},indent=2),flush=True)
    # Never fail away real records because a subset of Yahoo retrievals were incomplete.
    return state

if __name__=="__main__":
    cli=argparse.ArgumentParser()
    cli.add_argument("--batch-size",type=int,default=75)
    cli.add_argument("--retry-cap",type=int,default=3)
    cli.add_argument("--pause",type=float,default=0.7)
    cli.add_argument("--offline-membership",action="store_true")
    x=cli.parse_args()
    if not 1<=x.batch_size<=510 or x.retry_cap<1 or x.pause<0: cli.error("Invalid collection settings")
    collect(x.batch_size,x.retry_cap,x.pause,not x.offline_membership)
