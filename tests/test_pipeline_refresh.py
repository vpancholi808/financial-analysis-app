"""Scheduling rules for re-fetching companies whose newer annual report is due."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sp500_pipeline import due_for_statement_refresh, is_newer_statement


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
