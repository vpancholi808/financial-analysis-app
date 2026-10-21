"""Display helpers: units, missing values never shown as zero, grade direction."""
import math
import numpy as np
import pandas as pd
import pytest
from ui_format import (MISSING, GOOD, BAD, CAUTION, NA, METRICS, fmt_money, fmt_pct,
                       fmt_multiple, fmt_number, ordinal, standing, grade_from_percentile,
                       altman_status, beneish_status, check_status, screen_status,
                       risk_checks, tone_of_label, badge)


@pytest.mark.parametrize("value", [None, np.nan, float("inf"), "abc", pd.NA])
def test_missing_values_render_as_dash_never_zero(value):
    for f in (fmt_money, fmt_pct, fmt_multiple, fmt_number, ordinal):
        assert f(value) == MISSING


def test_money_units():
    assert fmt_money(49_770_000_000) == "$49.8B"
    assert fmt_money(5_320_550_000_000) == "$5.32T"
    assert fmt_money(950_000_000) == "$950M"
    assert fmt_money(-3_200_000_000) == "-$3.2B"
    assert fmt_money(0) == "$0"           # a real zero stays zero
    assert fmt_money(2e9, "EUR") == "EUR 2.0B"


def test_percent_multiple_and_ordinals():
    assert fmt_pct(0.2861) == "28.6%"
    assert fmt_pct(0.067, signed=True) == "+6.7%"
    assert fmt_pct(-0.02, signed=True) == "-2.0%"
    assert fmt_multiple(7.163) == "7.2×"
    assert [ordinal(x) for x in (1, 2, 3, 11, 12, 13, 22, 73.4, 100)] == \
        ["1st", "2nd", "3rd", "11th", "12th", "13th", "22nd", "73rd", "100th"]


def test_grades_respect_metric_direction():
    # High profitability percentile is good; high accruals percentile is bad.
    assert grade_from_percentile(90, True) == ("▲ Strong", GOOD)
    assert grade_from_percentile(90, False) == ("▼ Weak", BAD)
    assert grade_from_percentile(50, True)[1] == CAUTION
    assert grade_from_percentile(np.nan)[1] == NA
    assert standing(20, higher_is_better=False) == 80
    assert math.isnan(standing(None))
    assert METRICS["STA"].higher_is_better is False and METRICS["GPA"].higher_is_better is True
    assert "Lower is better" in METRICS["STA"].tooltip


def test_model_statuses():
    assert altman_status("Distress")[1] == BAD
    assert altman_status("Lower distress signal")[1] == GOOD
    assert altman_status("N/A")[1] == NA
    assert beneish_status(-1.5)[1] == BAD and beneish_status(-2.5)[1] == GOOD
    assert beneish_status(np.nan)[1] == NA
    # A missing input is "n/a", not a pass.
    assert check_status(False, np.nan)[1] == NA
    assert check_status(True, 0.1)[1] == BAD
    assert check_status(True, 0.1, applicable=False)[1] == NA


def test_screen_status_and_risk_checks_read_existing_flags():
    row = pd.Series(dict(industrial_applicable=True, BUY_candidate=True, SELL_candidate=True,
                         conflicting_signals=True, STA=0.05, STA_sector_pct=91, STA_risk_flag=True,
                         SNOA=np.nan, SNOA_sector_pct=np.nan, SNOA_risk_flag=False,
                         Beneish_M=-2.5, Beneish_risk_flag=False, Altman_Z=np.nan,
                         Altman_risk_flag=False, Altman_zone="N/A", Altman_model="Insufficient input"))
    assert screen_status(row)[0] == "◆ Mixed signals"
    checks = risk_checks(row)
    assert [c["tone"] for c in checks] == [BAD, NA, GOOD, NA]
    assert "91st" in checks[0]["detail"]
    assert screen_status(pd.Series(dict(industrial_applicable=False)))[1] == NA


def test_label_tones_and_badges():
    assert tone_of_label("▲ Strong") == GOOD
    assert tone_of_label("⚠ Flag") == BAD
    assert tone_of_label("● Neutral") == CAUTION
    assert tone_of_label("● No signal") == NA
    assert badge("▲ Long [x]", GOOD) == ":green-badge[▲ Long (x)]"
