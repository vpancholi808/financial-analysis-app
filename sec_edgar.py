"""Second statement source: SEC EDGAR XBRL "companyfacts" (10-K annual figures).

Yahoo Finance stays the primary source. SEC figures are used only to
  1. fill statement lines Yahoo left blank for the SAME fiscal year, and
  2. supply a newer fiscal year when the 10-K is filed but Yahoo lags.
Both are allowed only after the SEC figures agree with Yahoo's on the core
lines (total assets, revenue, operating cash flow) for an overlapping year,
which guards against a wrong tag mapping or a mismatched period. Nothing is
estimated: a line with no matching 10-K fact stays missing, and every filled
field is recorded in `sec_filled_fields`.

API: https://www.sec.gov/edgar/sec-api-documentation (no key; SEC asks for a
descriptive User-Agent with a contact address and at most 10 requests/second).
"""
from __future__ import annotations

import os
import re
import time

import numpy as np
import pandas as pd

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
DEFAULT_USER_AGENT = "financial-analysis-app research project (github.com/vpancholi808/financial-analysis-app)"
ANNUAL_FORMS = {"10-K", "10-K/A", "10-KT", "10-KT/A"}
DATE_TOLERANCE_DAYS = 15          # Yahoo reports fiscal year-ends rounded to month-end
CORE_TOLERANCE = 0.02             # SEC vs Yahoo agreement required on core lines
CORE_FIELDS = ("total_assets", "revenue", "operating_cash_flow")
SEC_SOURCE = "SEC EDGAR 10-K (XBRL companyfacts)"

# Each universe field -> us-gaap tags in priority order. "duration" facts cover a
# fiscal year (start..end); "instant" facts are balance-sheet values at year-end.
FIELD_TAGS: dict[str, tuple[str, tuple[str, ...]]] = {
    "revenue": ("duration", ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                             "RevenueFromContractWithCustomerIncludingAssessedTax", "SalesRevenueNet")),
    "cost_of_revenue": ("duration", ("CostOfRevenue", "CostOfGoodsAndServicesSold",
                                     "CostOfGoodsSold", "CostOfServices")),
    "gross_profit_reported": ("duration", ("GrossProfit",)),
    "net_income": ("duration", ("NetIncomeLoss",)),
    "operating_cash_flow": ("duration", ("NetCashProvidedByUsedInOperatingActivities",
                                         "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")),
    "ebit": ("duration", ("OperatingIncomeLoss",)),
    "depreciation": ("duration", ("DepreciationDepletionAndAmortization", "DepreciationAndAmortization",
                                  "DepreciationAmortizationAndAccretionNet", "Depreciation")),
    "sga": ("duration", ("SellingGeneralAndAdministrativeExpense",)),
    "total_assets": ("instant", ("Assets",)),
    "current_assets": ("instant", ("AssetsCurrent",)),
    "current_liabilities": ("instant", ("LiabilitiesCurrent",)),
    "total_liabilities": ("instant", ("Liabilities",)),
    "retained_earnings": ("instant", ("RetainedEarningsAccumulatedDeficit",)),
    "cash_and_equivalents": ("instant", ("CashAndCashEquivalentsAtCarryingValue",)),
    "accounts_receivable": ("instant", ("AccountsReceivableNetCurrent", "ReceivablesNetCurrent")),
    "net_ppe": ("instant", ("PropertyPlantAndEquipmentNet",)),
    "short_term_debt": ("instant", ("DebtCurrent",)),
    "long_term_debt": ("instant", ("LongTermDebtNoncurrent",)),
    "minority_interest": ("instant", ("MinorityInterest",)),
    "preferred_equity": ("instant", ("PreferredStockValue",)),
    "stockholders_equity": ("instant", ("StockholdersEquity",)),
}
# Fields carried for both the current and the prior fiscal year.
PRIOR_FIELDS = ("revenue", "cost_of_revenue", "net_income", "operating_cash_flow", "total_assets",
                "current_assets", "current_liabilities", "total_liabilities", "common_equity",
                "net_ppe", "depreciation", "sga", "short_term_debt", "long_term_debt",
                "minority_interest", "preferred_equity")
PRIOR_NAME = {"accounts_receivable": "prior_receivables"}
CURRENT_FIELDS = tuple(k for k in FIELD_TAGS if k != "stockholders_equity") + ("common_equity",)
FILLABLE = CURRENT_FIELDS + tuple("prior_" + f for f in PRIOR_FIELDS) + ("prior_receivables",)
# Blanks that justify a SEC lookup: inputs the screening models actually use.
# (Minority-interest, preferred and reported gross-profit lines are often
# legitimately absent, so they are filled when found but never trigger a fetch.)
TRIGGER_FIELDS = ("revenue", "cost_of_revenue", "net_income", "operating_cash_flow", "total_assets",
                  "common_equity", "current_assets", "current_liabilities", "total_liabilities",
                  "retained_earnings", "ebit", "cash_and_equivalents", "accounts_receivable", "net_ppe",
                  "depreciation", "sga", "short_term_debt", "long_term_debt",
                  "prior_revenue", "prior_cost_of_revenue", "prior_total_assets", "prior_receivables",
                  "prior_current_assets", "prior_current_liabilities", "prior_net_ppe",
                  "prior_depreciation", "prior_sga", "prior_long_term_debt")
SPECIALIST_SECTORS = {"Financials", "Real Estate"}


def _finite(v) -> bool:
    try:
        return np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def _ts(x):
    t = pd.to_datetime(x, errors="coerce")
    return None if pd.isna(t) else t


def _usd_facts(facts: dict, tag: str) -> list[dict]:
    node = facts.get("facts", {}).get("us-gaap", {}).get(tag, {})
    return node.get("units", {}).get("USD", []) or []


def annual_value(facts: dict, tag: str, end, kind: str):
    """Value of one tag for the fiscal year ending near `end`, from an annual
    report; the most recently filed figure wins (it includes restatements)."""
    target = _ts(end)
    if target is None:
        return np.nan
    best = None
    for f in _usd_facts(facts, tag):
        if f.get("form") not in ANNUAL_FORMS or not _finite(f.get("val")):
            continue
        fend = _ts(f.get("end"))
        if fend is None or abs((fend - target).days) > DATE_TOLERANCE_DAYS:
            continue
        if kind == "duration":
            fstart = _ts(f.get("start"))
            if fstart is None or not 300 <= (fend - fstart).days <= 400:
                continue
        elif f.get("start"):
            continue
        key = (str(f.get("filed", "")), str(f.get("accn", "")))
        if best is None or key > best[0]:
            best = (key, float(f["val"]))
    return best[1] if best else np.nan


def _first(facts, tags, end, kind):
    for tag in tags:
        v = annual_value(facts, tag, end, kind)
        if _finite(v):
            return v
    return np.nan


def _sum_if_any(values):
    present = [v for v in values if _finite(v)]
    return float(sum(present)) if present else np.nan


def extract_year(facts: dict, end) -> dict:
    """Every mapped field for one fiscal year-end; unmatched lines stay NaN."""
    out = {f: _first(facts, tags, end, kind) for f, (kind, tags) in FIELD_TAGS.items()}
    # Short-term debt: total current debt, else its reported components.
    if not _finite(out["short_term_debt"]):
        out["short_term_debt"] = _sum_if_any([annual_value(facts, t, end, "instant") for t in
                                              ("LongTermDebtCurrent", "ShortTermBorrowings", "CommercialPaper")])
    # Long-term debt: non-current portion, else total long-term debt less its current part.
    if not _finite(out["long_term_debt"]):
        total = annual_value(facts, "LongTermDebt", end, "instant")
        current = annual_value(facts, "LongTermDebtCurrent", end, "instant")
        if _finite(total):
            out["long_term_debt"] = total - (current if _finite(current) else 0.0)
    # SG&A reported as two lines.
    if not _finite(out["sga"]):
        g = annual_value(facts, "GeneralAndAdministrativeExpense", end, "duration")
        s = annual_value(facts, "SellingAndMarketingExpense", end, "duration")
        if _finite(g) and _finite(s):
            out["sga"] = g + s
    # Total liabilities net of minority interest (Yahoo's definition) when the
    # company does not tag a "Liabilities" total.
    if not _finite(out["total_liabilities"]):
        lse = annual_value(facts, "LiabilitiesAndStockholdersEquity", end, "instant")
        eq_nci = annual_value(facts, "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", end, "instant")
        se, mi = out["stockholders_equity"], out["minority_interest"]
        if _finite(lse) and _finite(eq_nci):
            out["total_liabilities"] = lse - eq_nci
        elif _finite(lse) and _finite(se):
            out["total_liabilities"] = lse - se - (mi if _finite(mi) else 0.0)
    # Common equity = stockholders' equity less preferred stock (Yahoo "Common Stock Equity").
    se = out.pop("stockholders_equity")
    if not _finite(se):
        # Some filers tag only total equity including noncontrolling interest;
        # usable when the noncontrolling interest is tagged (it may be zero).
        eq_nci = annual_value(facts, "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", end, "instant")
        if _finite(eq_nci) and _finite(out["minority_interest"]):
            se = eq_nci - out["minority_interest"]
    if _finite(se):
        out["common_equity"] = se - (out["preferred_equity"] if _finite(out["preferred_equity"]) else 0.0)
    else:
        out["common_equity"] = np.nan
    return out


def annual_period_ends(facts: dict) -> list[pd.Timestamp]:
    """Fiscal year-ends with an annual-report balance sheet, oldest first."""
    ends = {_ts(f.get("end")) for f in _usd_facts(facts, "Assets")
            if f.get("form") in ANNUAL_FORMS and not f.get("start")}
    ends.discard(None)
    # Comparative balance sheets repeat older dates; keep distinct fiscal years.
    out = []
    for e in sorted(ends):
        if not out or (e - out[-1]).days > 300:
            out.append(e)
        else:
            out[-1] = e
    return out


def _near(ends, target):
    t = _ts(target)
    if t is None:
        return None
    hits = [e for e in ends if abs((e - t).days) <= DATE_TOLERANCE_DAYS]
    return hits[-1] if hits else None


def core_agreement(yahoo: dict, sec: dict, tolerance: float = CORE_TOLERANCE) -> tuple[bool, str]:
    """SEC and Yahoo must agree on at least two core lines and on every core
    line both report. Returns (ok, explanation)."""
    compared, bad = [], []
    for f in CORE_FIELDS:
        y, s = yahoo.get(f), sec.get(f)
        if _finite(y) and _finite(s) and abs(float(y)) > 0:
            compared.append(f)
            if abs(float(s) - float(y)) / abs(float(y)) > tolerance:
                bad.append(f"{f} SEC {float(s):,.0f} vs Yahoo {float(y):,.0f}")
    if len(compared) < 2:
        return False, "fewer than two core lines to compare"
    if bad:
        return False, "; ".join(bad)
    return True, "core lines agree: " + ", ".join(compared)


def _current_view(row: dict) -> dict:
    return {f: row.get(f) for f in CURRENT_FIELDS}


def fill_from_sec(row: dict, facts: dict) -> tuple[dict, list[str], str]:
    """Return (updated row, filled field names, note). Never overwrites a
    reported Yahoo value; a newer fiscal year replaces the statements whole."""
    row = dict(row)
    ends = annual_period_ends(facts)
    if not ends:
        return row, [], "no annual 10-K balance sheet in SEC data"
    yahoo_end = _ts(row.get("fiscal_year_end"))
    latest = ends[-1]

    # 1) Newer fiscal year filed with the SEC than the one Yahoo provided.
    if yahoo_end is not None and (latest - yahoo_end).days > 300:
        overlap = _near(ends, yahoo_end)
        if overlap is None:
            return row, [], "newer 10-K exists but no overlapping year to validate against"
        ok, why = core_agreement(_current_view(row), extract_year(facts, overlap), tolerance=0.03)
        if not ok:
            return row, [], f"newer 10-K not used: {why}"
        prior_end = ends[-2] if len(ends) >= 2 else None
        cur = extract_year(facts, latest)
        if not _finite(cur.get("total_assets")) or cur["total_assets"] <= 0:
            return row, [], "newer 10-K lacks total assets"
        # A full fiscal year is required: after a merger or fiscal-year change the
        # 10-K can cover a stub period, which leaves the income statement blank.
        missing = [f for f in ("revenue", "net_income", "operating_cash_flow", "common_equity") if not _finite(cur.get(f))]
        if missing:
            return row, [], "newer 10-K not used: no full-year " + ", ".join(missing)
        prev = extract_year(facts, prior_end) if prior_end is not None else {}
        for f in CURRENT_FIELDS:
            row[f] = cur.get(f, np.nan)
        for f in PRIOR_FIELDS:
            row["prior_" + f] = prev.get(f, np.nan)
        row["prior_receivables"] = prev.get("accounts_receivable", np.nan)
        row["total_debt"] = np.nan
        row["prior_total_debt"] = np.nan
        row["fiscal_year_end"] = str(latest.date())
        row["prior_fiscal_year_end"] = str(prior_end.date()) if prior_end is not None else ""
        row["data_source"] = f"Financials: {SEC_SOURCE}; market cap: Yahoo Finance; membership: S&P 500 constituent list"
        row["sec_filled_fields"] = "all statement fields (newer 10-K)"
        return row, ["all"], f"replaced with FY ending {latest.date()} from 10-K; {why}"

    # 2) Same fiscal year: fill only the blanks.
    cur_end = _near(ends, yahoo_end)
    if cur_end is None:
        return row, [], "no SEC annual period matches Yahoo's fiscal year-end"
    cur = extract_year(facts, cur_end)
    ok, why = core_agreement(_current_view(row), cur)
    if not ok:
        return row, [], f"not filled: {why}"
    filled = []
    for f in CURRENT_FIELDS:
        if not _finite(row.get(f)) and _finite(cur.get(f)):
            row[f] = cur[f]
            filled.append(f)
    prior_target = row.get("prior_fiscal_year_end") or None
    prior_end = _near(ends, prior_target) if prior_target else None
    if prior_end is None:
        earlier = [e for e in ends if 300 <= (cur_end - e).days <= 400]
        prior_end = earlier[-1] if earlier else None
    if prior_end is not None:
        prev = extract_year(facts, prior_end)
        mapping = {"prior_" + f: f for f in PRIOR_FIELDS}
        mapping["prior_receivables"] = "accounts_receivable"
        for target, src in mapping.items():
            if not _finite(row.get(target)) and _finite(prev.get(src)):
                row[target] = prev[src]
                filled.append(target)
    if filled:
        existing = [x for x in str(row.get("sec_filled_fields") or "").split(";") if x and x != "nan"]
        row["sec_filled_fields"] = ";".join(sorted(set(existing) | set(filled)))
    return row, filled, why


def needs_sec(row: dict, today=None, stale_after_days: int = 405) -> bool:
    """A fillable line is blank, or a newer annual report should exist."""
    today = pd.Timestamp(today or pd.Timestamp.now(tz="UTC").date())
    fye = _ts(row.get("fiscal_year_end"))
    if fye is not None and (today - fye).days >= stale_after_days:
        return True
    if str(row.get("sector", "")) in SPECIALIST_SECTORS:
        return False
    return any(not _finite(row.get(f)) for f in TRIGGER_FIELDS)


def clean_user_agent(value: str) -> str:
    """One-line header value: a pasted contact often contains a line break
    (e.g. "Name\r\nemail"), which HTTP headers cannot carry."""
    return " ".join(str(value or "").split())


_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(\.[\w-]+)+")


def redact(text: str) -> str:
    """Error text is saved to a public repository: never keep the contact."""
    text = re.sub(r"header value: .*", "header value: <SEC_USER_AGENT>", str(text))
    return _EMAIL.sub("<email>", text)


def to_sec_symbol(ticker: str) -> str:
    return str(ticker).upper().replace(".", "-")


class SecClient:
    """Minimal polite client: identifies itself and stays under 10 requests/s."""

    def __init__(self, user_agent: str | None = None, pause: float = 0.15, session=None):
        import requests
        self.session = session or requests.Session()
        ua = clean_user_agent(user_agent or os.environ.get("SEC_USER_AGENT") or "") or DEFAULT_USER_AGENT
        self.contact_set = ua != DEFAULT_USER_AGENT
        self.session.headers.update({"User-Agent": ua,
                                     "Accept-Encoding": "gzip, deflate"})
        self.pause = pause
        self._ciks = None

    def _get(self, url):
        resp = self.session.get(url, timeout=30)
        if self.pause:
            time.sleep(self.pause)
        resp.raise_for_status()
        return resp.json()

    def cik(self, ticker: str):
        if self._ciks is None:
            data = self._get(TICKER_MAP_URL)
            self._ciks = {str(v["ticker"]).upper(): int(v["cik_str"]) for v in data.values()}
        return self._ciks.get(to_sec_symbol(ticker))

    def company_facts(self, ticker: str):
        cik = self.cik(ticker)
        return None if cik is None else self._get(FACTS_URL.format(cik=cik))
