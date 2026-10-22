"""Plain-language labels, reasons and evidence rows read the computed screen; they never change it."""
from pathlib import Path
import math
import sys
import pandas as pd
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import presentation as pr
from finance_models import prepare, classify, ScreenSettings
from research_workbench import recommendation_table

DATA = Path(__file__).resolve().parents[1] / "universe.csv"


@pytest.fixture(scope="module")
def screen():
    s = ScreenSettings(sell_min_flags=2)
    base = prepare(pd.read_csv(DATA))
    v = classify(base, s)
    return base, v.join(recommendation_table(v, base, s)), s


def test_call_and_evidence_labels():
    assert pr.call_label("Long") == "▲ Buy"
    assert pr.call_label("Short / avoid") == "▼ Avoid"
    assert pr.call_label("No position") == "○ No clear signal"
    assert pr.evidence_label("Long", "High") == "Strong evidence"
    assert pr.evidence_label("No position", "") == ""


def test_top_share_and_minus():
    assert pr.top_share(98.3) == "top 2%"
    assert pr.top_share(100) == "top 1%"
    assert pr.minus(-2.45) == "−2.45"


def test_every_company_gets_a_why_line(screen):
    base, v, s = screen
    lines = [pr.why_line(r, s) for _, r in v.iterrows()]
    assert all(isinstance(x, str) and len(x) > 10 for x in lines)
    for _, r in v.loc[v.rec_position.eq("Short / avoid")].iterrows():
        assert pr.why_line(r, s).startswith(f"{len(pr.warnings_fired(r))} warning signs")
    for _, r in v.loc[~v.industrial_applicable].head(5).iterrows():
        assert "don't apply" in pr.why_line(r, s)


def test_near_miss_only_for_no_call_without_warnings(screen):
    base, v, s = screen
    near = v.loc[[pr.is_near_miss(r, s) for _, r in v.iterrows()]]
    assert near.rec_position.eq("No position").all()
    assert near.SELL_flag_count.eq(0).all()
    assert (near.GPA_sector_pct >= s.buy_gpa_pct - 10).all() and (near.BM_sector_pct >= s.buy_bm_pct - 10).all()


def test_reasons_and_evidence_for_a_buy(screen):
    base, v, s = screen
    r = v.loc[v.rec_position.eq("Long")].iloc[0]
    pro, con = pr.reasons(r, None, s)
    assert 1 <= len(pro) <= 3 and len(con) <= 3
    rows = pr.evidence_rows(r, base, s)
    assert [x["title"] for x in rows][0] == "How much gross profit does it make from its assets?"
    assert len(rows) == 6
    # The status of a failed test is always a concern, a passed one never is.
    for x, (key, flag) in zip(rows[2:5], [("STA", "STA_risk_flag"), ("SNOA", "SNOA_risk_flag"), ("Beneish_M", "Beneish_risk_flag")]):
        if bool(r[flag]):
            assert x["status"] == ("Concern", pr.BAD)
        else:
            assert x["status"][0] != "Concern"
    html = pr.evidence_html(rows, pr.palette("dark"))
    assert "Are profits backed by real cash?" in html and "<script" not in html


def test_missing_values_are_never_shown_as_zero(screen):
    base, v, s = screen
    r = v.loc[v.ticker.eq("BKNG")].iloc[0] if v.ticker.eq("BKNG").any() else v.loc[v.GPA.isna()].iloc[0]
    rows = pr.evidence_rows(r, base, s)
    assert "can't be measured" in rows[0]["lines"][0][0]
    assert rows[0]["strip"] is None and rows[0]["status"] == ("n/a", pr.NA)


def test_data_status_labels():
    assert pr.data_status({"stale_statements": True})[0] == "Out of date"
    assert pr.data_status({"sec_filled_fields": "all statement fields (newer 10-K)", "fiscal_year_end": "2026-09-03"})[0] == "From FY2026 10-K"
    assert pr.data_status({"sec_filled_fields": "sga;prior_sga"})[0] == "Partly from 10-K"
    assert pr.data_status({"sec_filled_fields": float("nan")})[0] == "Complete"


def test_html_escapes_company_text():
    out = pr.bottom_line_html({"rec_position": "No position", "industrial_applicable": False,
                               "sector": "<b>x</b>"}, {}, None, pr.palette("light"))
    assert "<b>x</b>" not in out


def test_split_bar_and_sector_table(screen):
    base, v, s = screen
    counts = {k: int(v.rec_position.eq(k).sum()) for k in ["Long", "Short / avoid", "No position", "Not rated"]}
    bar = pr.split_bar_html(counts, pr.palette("dark"))
    assert f"{counts['Long']} buy" in bar
    tbl = pr.sector_table_html(v, pr.palette("light"))
    assert "Health Care" in tbl and "Typical price" in tbl
