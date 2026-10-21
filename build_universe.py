"""Independently construct `universe.csv` in Colab or locally using yfinance.

Usage:
  pip install -r requirements.txt
  python build_universe.py --tickers AAPL MSFT GOOGL --output universe.csv
  python build_universe.py --limit 1000 --exclude PITCH_TICKER --pause 0.8

This can take considerable time and may be rate-limited. Re-run with --resume.
Quotes may be delayed, and Yahoo field definitions should be audited against filings.
"""
from __future__ import annotations
import argparse
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from finance_models import UNIVERSE_COLUMNS

SECTOR_MAP = {
    "Technology": "Information Technology",
    "Healthcare": "Health Care",
    "Financial Services": "Financials",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Defensive": "Consumer Staples",
    "Industrials": "Industrials",
    "Energy": "Energy",
    "Basic Materials": "Materials",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
    "Communication Services": "Communication Services",
}

FIELD_ALIASES = {
    "revenue": ["Total Revenue", "Operating Revenue"],
    "cost_of_revenue": ["Cost Of Revenue", "Cost Of Goods Sold"],
    "gross_profit_reported": ["Gross Profit"],
    "net_income": ["Net Income", "Net Income Common Stockholders"],
    "operating_cash_flow": ["Operating Cash Flow", "Cash Flow From Continuing Operating Activities"],
    "total_assets": ["Total Assets"],
    "common_equity": ["Common Stock Equity"],
    "current_assets": ["Current Assets"],
    "current_liabilities": ["Current Liabilities"],
    "total_liabilities": ["Total Liabilities Net Minority Interest", "Total Liabilities"],
    "retained_earnings": ["Retained Earnings"],
    "ebit": ["EBIT"],
    "cash_and_equivalents": ["Cash And Cash Equivalents"],
    "accounts_receivable": ["Accounts Receivable", "Net Receivables"],
    "net_ppe": ["Net PPE", "Property Plant Equipment Net"],
    "depreciation": ["Depreciation And Amortization In Cash Flow", "Depreciation Amortization Depletion"],
    "sga": ["Selling General And Administration"],
    "total_debt": ["Total Debt"],
    "short_term_debt": ["Current Debt", "Current Debt And Capital Lease Obligation"],
    "long_term_debt": ["Long Term Debt", "Long Term Debt And Capital Lease Obligation"],
    "minority_interest": ["Minority Interest", "Minority Interest In Consolidated Entity"],
    "preferred_equity": ["Preferred Stock Equity"],
}
INCOME = {"revenue", "cost_of_revenue", "gross_profit_reported", "net_income", "ebit", "sga"}
CASHFLOW = {"operating_cash_flow", "depreciation"}


def series_at(statement: pd.DataFrame, date: pd.Timestamp, names: list[str]) -> float:
    if statement is None or statement.empty or date is None:
        return np.nan
    for label in names:
        if label in statement.index and date in statement.columns:
            v = statement.at[label, date]
            n = pd.to_numeric(v, errors="coerce")
            if pd.notna(n) and np.isfinite(n):
                return float(n)
    return np.nan


def shared_fiscal_dates(income: pd.DataFrame, balance: pd.DataFrame,
                        cash: pd.DataFrame) -> list[pd.Timestamp]:
    """Require all three statement tables to refer to the SAME fiscal-year end."""
    if any(t is None or t.empty for t in (income, balance, cash)):
        return []
    return sorted(set(income.columns) & set(balance.columns) & set(cash.columns), reverse=True)


def extract_company(ticker: object, quote: dict) -> dict | None:
    """Keep original fields and NA values; never replace missing with 0."""
    try:
        info = ticker.info or {}
        income, balance, cash = ticker.income_stmt, ticker.balance_sheet, ticker.cashflow
        dates = shared_fiscal_dates(income, balance, cash)
        if not dates:
            return None
        fy = dates[0]
        prev_fy = dates[1] if len(dates) > 1 else None
        sector = SECTOR_MAP.get(info.get("sector", ""), "")
        if not sector:
            return None
        quote_type = (info.get("quoteType") or quote.get("quoteType") or "").upper()
        if quote_type and quote_type != "EQUITY":
            return None

        market_cap = quote.get("intradaymarketcap") or quote.get("marketCap") or info.get("marketCap")
        row = {field: np.nan for field in UNIVERSE_COLUMNS}
        row.update({
            "ticker": str(quote.get("symbol", "")).strip().upper(),
            "name": info.get("longName") or info.get("shortName") or quote.get("shortName") or "",
            "sector": sector,
            "industry": info.get("industry") or "Unknown",
            "exchange": info.get("exchange") or quote.get("exchange") or "",
            "market_cap": market_cap,
            "market_currency": info.get("currency") or quote.get("currency") or "",
            "financial_currency": info.get("financialCurrency") or "",
            "price_asof": datetime.now(timezone.utc).date().isoformat(),
            "fiscal_year_end": pd.Timestamp(fy).date().isoformat(),
            "prior_fiscal_year_end": pd.Timestamp(prev_fy).date().isoformat() if prev_fy is not None else "",
            "refreshed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "data_source": "Yahoo Finance via yfinance (verify with filings)",
        })
        for field, labels in FIELD_ALIASES.items():
            frame = income if field in INCOME else cash if field in CASHFLOW else balance
            row[field] = series_at(frame, fy, labels)
            prior_field = "prior_" + ("receivables" if field == "accounts_receivable" else field)
            if prior_field in UNIVERSE_COLUMNS:
                row[prior_field] = series_at(frame, prev_fy, labels)
        # Financial reporting must have actual assets, not merely a company quote.
        if pd.isna(row["total_assets"]) or row["total_assets"] <= 0 or not row["ticker"]:
            return None
        return row
    except Exception as exc:
        print(f"  Statement extraction error: {type(exc).__name__}: {exc}")
        return None


def issuer_key(row: dict) -> str:
    """Best-effort common-stock share-class dedupe; requires spot checks."""
    name = str(row.get("name") or "").lower()
    name = re.sub(r"\bclass\s+[a-z]\b", "", name)
    name = re.sub(r"\W+", "", name)
    return name or str(row.get("ticker", ""))


def discover_quotes(yf, target: int, buffer: int = 350) -> list[dict]:
    from yfinance import EquityQuery
    q = EquityQuery("and", [
        EquityQuery("eq", ["region", "us"]),
        EquityQuery("is-in", ["exchange", "NMS", "NYQ"]),
        EquityQuery("gte", ["intradaymarketcap", 3e8]),
    ])
    wanted = target + buffer
    quotes: list[dict] = []
    seen: set[str] = set()
    for start in range(0, wanted, 250):
        size = min(250, wanted - start)
        batch = yf.screen(q, offset=start, size=size,
                          sortField="intradaymarketcap", sortAsc=False)
        got = batch.get("quotes", [])
        if not got:
            break
        for quote in got:
            symbol = quote.get("symbol", "").upper()
            if symbol and symbol not in seen:
                seen.add(symbol)
                quotes.append(quote)
    return quotes


def write_output(path: Path, rows: list[dict]) -> None:
    pd.DataFrame(rows, columns=UNIVERSE_COLUMNS).to_csv(path, index=False)


def build(limit: int, output: Path, exclude: set[str], pause: float,
          resume: bool, buffer: int, tickers: list[str] | None = None) -> None:
    import yfinance as yf  # dependency only required for actual data retrieval
    quotes = ([{"symbol": s.strip().upper()} for s in tickers if s.strip()]
              if tickers else discover_quotes(yf, limit, buffer=buffer))
    print(f"Discovered {len(quotes)} tickers. Fetching financial statements...")
    existing: list[dict] = []
    if resume and output.exists():
        existing = pd.read_csv(output).to_dict(orient="records")
    rows = [r for r in existing if str(r.get("ticker", "")).upper() not in exclude]
    seen_symbols = {str(r["ticker"]).upper() for r in rows}
    seen_issuers = {issuer_key(r) for r in rows}
    for i, q in enumerate(quotes, start=1):
        symbol = str(q.get("symbol", "")).upper()
        if not symbol or symbol in exclude or symbol in seen_symbols:
            continue
        print(f"{i}/{len(quotes)} {symbol}: ", end="", flush=True)
        try:
            row = extract_company(yf.Ticker(symbol), q)
            if row and issuer_key(row) not in seen_issuers:
                rows.append(row)
                seen_issuers.add(issuer_key(row))
                seen_symbols.add(symbol)
                print(f"OK | {len(rows)} rows")
            else:
                print("missing/ineligible/duplicate")
        except Exception as exc:
            print(f"SKIP {type(exc).__name__}: {exc}")
        if i % 25 == 0:
            write_output(output, rows)
            print(f"Checkpoint: {len(rows)} rows -> {output}")
        if len(rows) >= limit:
            break
        if pause:
            time.sleep(pause)
    write_output(output, rows[:limit])
    print(f"Saved {min(len(rows),limit)} REAL sourced rows to {output}")
    if len(rows) < 500:
        print("WARNING: Below the professor's 500-company requirement. Expand/retry collection.")
    if not exclude:
        print("WARNING: Pitch ticker not specified. Add --exclude PITCH_TICKER before submission.")
    print("IMPORTANT: Spot-check data, currency and financial line-item mappings against annual filings.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--tickers", nargs="*", default=None, help="Fetch explicit real tickers instead of discovery (smoke test)")
    ap.add_argument("--buffer", type=int, default=350)
    ap.add_argument("--output", default="universe.csv")
    ap.add_argument("--exclude", nargs="*", default=[], help="Stock-pitch ticker(s) to REMOVE")
    ap.add_argument("--pause", type=float, default=0.6)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if args.limit < 1 or args.pause < 0 or args.buffer < 0:
        ap.error("limit >=1, pause >=0 and buffer >=0 are required")
    build(args.limit, Path(args.output), {x.strip().upper() for x in args.exclude},
          args.pause, args.resume, args.buffer, args.tickers)


if __name__ == "__main__":
    main()
