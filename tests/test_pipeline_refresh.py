"""Scheduling rules for re-fetching companies whose newer annual report is due."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sp500_pipeline import due_for_statement_refresh, is_newer_statement
sys.path.insert(0, str(Path(__file__).resolve().parent))


def test_refresh_due_only_after_next_annual_report_window():
    rows={"OLD":{"fiscal_year_end":"2025-08-31"},"NEW":{"fiscal_year_end":"2026-06-30"},
          "BAD":{"fiscal_year_end":None}}
    assert due_for_statement_refresh(rows,today="2026-10-21")==["OLD"]
    assert due_for_statement_refresh(rows,today="2026-09-30")==[]


def test_refresh_retried_at_most_once_per_day():
    rows={"OLD":{"fiscal_year_end":"2025-08-31"}}
    assert due_for_statement_refresh(rows,today="2026-10-21",last_attempts={"OLD":"2026-10-21"})==[]
    assert due_for_statement_refresh(rows,today="2026-10-22",last_attempts={"OLD":"2026-10-21"})==["OLD"]


def test_refreshed_record_must_be_newer():
    old={"fiscal_year_end":"2025-08-31"}
    assert is_newer_statement(old,{"fiscal_year_end":"2026-08-31"})
    assert not is_newer_statement(old,{"fiscal_year_end":"2025-08-31"})
    assert not is_newer_statement(old,{"fiscal_year_end":None})


def test_sec_pass_fills_and_records_check_dates():
    import numpy as np
    from sp500_pipeline import sec_pass, sec_due
    from test_sec_edgar import facts_for, year, yahoo_row, FY24, FY25

    class FakeClient:
        def __init__(self): self.calls = []
        def company_facts(self, symbol):
            self.calls.append(symbol)
            return facts_for({FY24: year(1.0), FY25: year(1.1)}) if symbol == "MU" else None

    rows = {"MU": yahoo_row(), "ZZZ": yahoo_row(ticker="ZZZ")}
    checks, client = {}, FakeClient()
    notes = {}
    log = sec_pass(rows, checks, today="2026-10-21", client=client, notes=notes)
    assert rows["MU"]["sga"] == 132 and "SEC FILLED MU" in log[0]
    assert notes["MU"].startswith("2026-10-21: filled") and "not in the SEC ticker list" in notes["ZZZ"]
    assert any("SEC NO CIK ZZZ" in line for line in log)
    assert checks["MU"].startswith("2026-10-21|")
    # Checked today: not fetched again the same day.
    assert not sec_due("ZZZ", rows["ZZZ"], checks, "2026-10-21")
    sec_pass(rows, checks, today="2026-10-21", client=client)
    assert client.calls.count("ZZZ") == 1


def test_sec_recheck_weekly_when_only_blanks_remain():
    import numpy as np
    from sp500_pipeline import sec_due
    from sec_edgar import TRIGGER_FIELDS
    row = {k: 1.0 for k in TRIGGER_FIELDS} | {"sga": np.nan, "fiscal_year_end": "2026-06-30"}
    checks = {"X": "2026-10-20|2026-06-30"}
    assert not sec_due("X", row, checks, "2026-10-21")
    assert sec_due("X", row, checks, "2026-10-27")


def test_sec_pass_stops_when_sec_refuses_and_retries_next_day():
    from sp500_pipeline import sec_pass, sec_due
    from test_sec_edgar import yahoo_row

    class Refusing:
        calls = 0
        def company_facts(self, symbol):
            Refusing.calls += 1
            raise RuntimeError("403 Client Error: Forbidden")

    rows = {f"T{i}": yahoo_row(ticker=f"T{i}") for i in range(20)}
    checks, status = {}, {}
    log = sec_pass(rows, checks, today="2026-10-21", client=Refusing(), status=status)
    assert Refusing.calls == 5 and status["aborted"] and "403" in status["last_error"]
    assert "SEC PASS STOPPED" in log[-1]
    assert not sec_due("T0", rows["T0"], checks, "2026-10-21")
    assert sec_due("T0", rows["T0"], checks, "2026-10-22")
