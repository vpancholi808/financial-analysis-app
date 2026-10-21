"""Exercise real-data Streamlit pages without contacting Yahoo Finance."""
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "universe.csv"

@pytest.mark.parametrize("page", [
    "S&P 500 Screener", "Company Profile", "Research Desk", "Glossary & Data",
])
def test_streamlit_render_with_saved_data(page):
    from streamlit.testing.v1 import AppTest
    data = pd.read_csv(DATA)
    if data.empty:
        pytest.skip("Real-company universe collection has not completed its first batch")
    at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=90)
    assert not at.exception, [str(x.message) for x in at.exception]
    options = at.sidebar.radio
    assert options, "Sidebar navigation missing"
    options[0].set_value(page).run(timeout=90)
    assert not at.exception, [str(x.message) for x in at.exception]


@pytest.mark.parametrize("ticker", ["LOW", "SKYD", "GOOG", "NEE", "F"])
def test_profile_renders_review_stale_and_share_class_cases(ticker):
    from streamlit.testing.v1 import AppTest
    if ticker not in set(pd.read_csv(DATA).ticker):
        pytest.skip(f"{ticker} not in the saved universe")
    at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=90)
    at.sidebar.radio[0].set_value("Company Profile").run(timeout=90)
    at.selectbox(key="overview_select").set_value(ticker).run(timeout=90)
    assert not at.exception, [str(x.message) for x in at.exception]


def test_research_draft_does_not_carry_over_between_companies():
    from streamlit.testing.v1 import AppTest
    tickers = set(pd.read_csv(DATA).ticker)
    if not {"AAPL", "MSFT"} <= tickers:
        pytest.skip("Fixture companies missing from the saved universe")
    at = AppTest.from_file(str(ROOT / "app.py")).run(timeout=90)
    at.sidebar.radio[0].set_value("Research Desk").run(timeout=90)
    at.selectbox(key="research_select").set_value("AAPL").run(timeout=90)
    thesis = lambda: next(t for t in at.text_area if t.label == "Investment thesis")
    generated_aapl = thesis().value
    assert generated_aapl and "Apple" in generated_aapl          # pre-filled from the analysis
    thesis().set_value("Apple-specific thesis").run(timeout=90)
    at.selectbox(key="research_select").set_value("MSFT").run(timeout=90)
    assert thesis().value != "Apple-specific thesis" and "Microsoft" in thesis().value
    at.selectbox(key="research_select").set_value("AAPL").run(timeout=90)
    assert thesis().value == "Apple-specific thesis"
    assert not at.exception, [str(x.message) for x in at.exception]
