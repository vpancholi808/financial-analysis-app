"""Exercise real-data Streamlit pages without contacting Yahoo Finance."""
from pathlib import Path
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "universe.csv"

@pytest.mark.parametrize("page", [
    "Stock Screener", "Company Fundamentals", "Investment Research",
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
