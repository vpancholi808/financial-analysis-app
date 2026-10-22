"""SEC EDGAR companyfacts parsing and the guarded fill rules (offline fixtures)."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sec_edgar import (annual_value, annual_period_ends, extract_year, core_agreement,
                       fill_from_sec, needs_sec, to_sec_symbol)

FY24, FY25, FY26 = "2024-08-29", "2025-08-28", "2026-08-27"   # 52/53-week year-ends
STARTS = {FY24: "2023-09-01", FY25: "2024-08-30", FY26: "2025-08-29"}


def fact(end, val, form="10-K", filed=None, start=None):
    f = {"end": end, "val": val, "form": form, "fp": "FY", "accn": "0000-" + end,
         "filed": filed or end.replace("-08-", "-10-")}
    if start:
        f["start"] = start
    return f


def facts_for(years: dict) -> dict:
    """years: {end: {tag: value}}; duration tags get a one-year start date."""
    durations = {"Revenues", "CostOfRevenue", "NetIncomeLoss", "NetCashProvidedByUsedInOperatingActivities",
                 "OperatingIncomeLoss", "SellingGeneralAndAdministrativeExpense",
                 "DepreciationDepletionAndAmortization", "GeneralAndAdministrativeExpense",
                 "SellingAndMarketingExpense"}
    tags = {}
    for end, values in years.items():
        for tag, val in values.items():
            start = STARTS[end] if tag in durations else None
            tags.setdefault(tag, {"units": {"USD": []}})["units"]["USD"].append(fact(end, val, start=start))
    return {"facts": {"us-gaap": tags}}


def year(scale=1.0, **over):
    v = dict(Revenues=1000, CostOfRevenue=600, NetIncomeLoss=100, NetCashProvidedByUsedInOperatingActivities=150,
             OperatingIncomeLoss=140, SellingGeneralAndAdministrativeExpense=120, DepreciationDepletionAndAmortization=40,
             Assets=2000, AssetsCurrent=800, LiabilitiesCurrent=400, Liabilities=1200, StockholdersEquity=800,
             RetainedEarningsAccumulatedDeficit=500, CashAndCashEquivalentsAtCarryingValue=300,
             AccountsReceivableNetCurrent=150, PropertyPlantAndEquipmentNet=700, DebtCurrent=50,
             LongTermDebtNoncurrent=600)
    v = {k: x * scale for k, x in v.items()}
    v.update(over)
    return v


def yahoo_row(**over):
    r = dict(ticker="MU", fiscal_year_end="2025-08-31", prior_fiscal_year_end="2024-08-31",
             revenue=1100., operating_cash_flow=165., total_assets=2200., net_income=110.,
             common_equity=880., sga=np.nan, prior_sga=np.nan, short_term_debt=np.nan, market_cap=5000.)
    r.update(over)
    return r


def test_annual_value_matches_month_end_rounding_and_ignores_quarters():
    f = facts_for({FY25: {"Assets": 2200}})
    f["facts"]["us-gaap"]["Assets"]["units"]["USD"].append(fact("2025-08-28", 9999, form="10-Q"))
    assert annual_value(f, "Assets", "2025-08-31", "instant") == 2200
    assert np.isnan(annual_value(f, "Assets", "2025-06-30", "instant"))


def test_duration_requires_a_full_year_and_latest_filing_wins():
    f = facts_for({FY25: {"Revenues": 1100}})
    usd = f["facts"]["us-gaap"]["Revenues"]["units"]["USD"]
    usd.append(fact(FY25, 275, start="2025-05-30"))                       # a quarter inside the 10-K
    usd.append(fact(FY25, 1105, start=STARTS[FY25], filed="2026-10-08"))  # restated in the next 10-K
    assert annual_value(f, "Revenues", FY25, "duration") == 1105


def test_extract_year_derives_common_equity_and_debt_fallbacks():
    f = facts_for({FY25: year(1.1, DebtCurrent=None, PreferredStockValue=30,
                             LongTermDebtCurrent=20, ShortTermBorrowings=15)})
    # Remove the None placeholder so DebtCurrent is genuinely absent.
    f["facts"]["us-gaap"].pop("DebtCurrent")
    y = extract_year(f, FY25)
    assert y["common_equity"] == pytest.approx(880 - 30)
    assert y["short_term_debt"] == pytest.approx(35)
    assert y["sga"] == pytest.approx(132)


def test_total_liabilities_from_balance_sheet_when_not_tagged():
    v = year(1.0)
    v.pop("Liabilities")
    v.update(LiabilitiesAndStockholdersEquity=2000,
             StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest=820)
    y = extract_year(facts_for({FY25: v}), FY25)
    assert y["total_liabilities"] == pytest.approx(1180)


def test_sga_from_components_only_when_both_reported():
    v = year(1.0)
    v.pop("SellingGeneralAndAdministrativeExpense")
    v["GeneralAndAdministrativeExpense"] = 70
    assert np.isnan(extract_year(facts_for({FY25: v}), FY25)["sga"])
    v["SellingAndMarketingExpense"] = 50
    assert extract_year(facts_for({FY25: v}), FY25)["sga"] == pytest.approx(120)


def test_core_agreement_needs_two_matching_lines():
    assert core_agreement({"total_assets": 100, "revenue": 50}, {"total_assets": 101, "revenue": 50})[0]
    assert not core_agreement({"total_assets": 100, "revenue": 50}, {"total_assets": 110, "revenue": 50})[0]
    assert not core_agreement({"total_assets": 100}, {"total_assets": 100})[0]


def test_gap_fill_fills_only_blanks_for_the_same_year():
    f = facts_for({FY24: year(1.0), FY25: year(1.1)})
    row, filled, note = fill_from_sec(yahoo_row(), f)
    assert row["sga"] == pytest.approx(132) and row["prior_sga"] == pytest.approx(120)
    assert row["short_term_debt"] == pytest.approx(55)
    assert row["revenue"] == 1100.                      # reported Yahoo value untouched
    assert "sga" in filled and "prior_sga" in filled
    assert "sga" in row["sec_filled_fields"]


def test_gap_fill_refused_when_sec_and_yahoo_disagree():
    f = facts_for({FY24: year(1.0), FY25: year(1.5)})   # 36% apart -> wrong company or period
    row, filled, note = fill_from_sec(yahoo_row(), f)
    assert filled == [] and np.isnan(row["sga"]) and "not filled" in note


def test_newer_10k_replaces_statements_after_validation():
    f = facts_for({FY24: year(1.0), FY25: year(1.1), FY26: year(1.6)})
    row, filled, note = fill_from_sec(yahoo_row(), f)
    assert filled == ["all"]
    assert row["fiscal_year_end"] == FY26 and row["prior_fiscal_year_end"] == FY25
    assert row["revenue"] == pytest.approx(1600) and row["prior_revenue"] == pytest.approx(1100)
    assert row["market_cap"] == 5000.                   # quote data untouched
    assert "SEC EDGAR" in row["data_source"]


def test_newer_10k_ignored_when_overlap_year_disagrees():
    f = facts_for({FY24: year(1.0), FY25: year(2.0), FY26: year(2.5)})
    row, filled, note = fill_from_sec(yahoo_row(), f)
    assert filled == [] and row["fiscal_year_end"] == "2025-08-31"


def test_period_ends_and_symbols():
    f = facts_for({FY24: year(), FY25: year(), FY26: year()})
    assert [str(e.date()) for e in annual_period_ends(f)] == [FY24, FY25, FY26]
    assert to_sec_symbol("BRK.B") == "BRK-B"


def test_needs_sec_for_blanks_or_due_annual_report():
    full = {k: 1.0 for k in __import__("sec_edgar").TRIGGER_FIELDS}
    assert not needs_sec(dict(full, fiscal_year_end="2026-06-30"), today="2026-10-21")
    assert needs_sec(dict(full, fiscal_year_end="2025-08-31"), today="2026-10-21")
    assert needs_sec(dict(full, sga=np.nan, fiscal_year_end="2026-06-30"), today="2026-10-21")


def test_user_agent_with_line_break_becomes_one_line():
    from sec_edgar import clean_user_agent, SecClient
    assert clean_user_agent("Jane Doe\r\njane@example.com ") == "Jane Doe jane@example.com"
    import requests
    c = SecClient(user_agent="Jane Doe\r\njane@example.com", pause=0)
    assert c.session.headers["User-Agent"] == "Jane Doe jane@example.com" and c.contact_set
    requests.Request("GET", "https://example.com", headers=dict(c.session.headers)).prepare()  # no InvalidHeader


def test_saved_errors_never_contain_the_contact():
    from sec_edgar import redact
    msg = ("InvalidHeader: Invalid leading whitespace, reserved character(s), or return character(s) in header value: "
           "'Jane Doe\\r\\njane@example.com'")
    out = redact(msg)
    assert "jane@example.com" not in out and "Jane Doe" not in out and "<SEC_USER_AGENT>" in out
    assert redact("403 Forbidden for bob@x.org") == "403 Forbidden for <email>"
