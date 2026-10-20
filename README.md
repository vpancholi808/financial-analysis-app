# Financial Analysis Terminal — ACCT 3343

A Bloomberg-inspired **Streamlit equity research platform** with **real Yahoo Finance financial statements**, sector-relative screening, transparent accounting formulas, and an evidence-based investment research desk. This is a **work in progress**, not a complete ACCT 3343 submission.

## Live website

https://financial-analysis-app-cpf2jt6lfpqxjfpik7a7bs.streamlit.app/

The Phase-1 redesign **removes the fictitious DEMO company generator entirely**. An empty dashboard is preferable to presenting invented financial figures as real.

## Quick test with real companies

1. In the sidebar, choose **Live Yahoo test**.
2. Enter tickers such as `AAPL, MSFT, NVDA` (maximum 15 per request).
3. Click **Fetch real financials**. The app uses `yfinance.Ticker(...).income_stmt`, `.balance_sheet`, `.cashflow` and `.info` for the latest **matched fiscal year**.
4. Open **Company Fundamentals** and **Investment Research** to examine annual figures, source metadata, calculated GPA / B/M / STA, risks and a downloadable investment-research draft.
5. If Yahoo blocks/rate-limits the request or financial statements are incomplete, the app shows missing-data diagnostics instead of substituting dummy values. Source queries are cached for six hours within the app service.

**Interpretation:** An initial three-stock test cannot produce defensible sector percentiles: the default is a minimum of 10 eligible companies per sector. No BUY/SELL flag is not the same as a negative investment recommendation.

## Build the 500–1,000-company universe

Install Python dependencies in Google Colab or locally:

```bash
pip install -r requirements.txt
python build_universe.py --tickers AAPL MSFT NVDA --limit 3 --output real_test.csv
python build_universe.py --limit 1000 --buffer 350 --pause 0.8 --resume --exclude YOUR_PITCH_TICKER --output universe.csv
```

The script independently searches a US-region/NYSE/NASDAQ equity universe (proxy, **not** the complete US-listed market), collects annual statements, matches fiscal dates, retains missing inputs and saves a snapshot. Collection of 500–1,000 individual reports may take considerable time; Yahoo requests may throttle or fail, so use `--resume`. Check large sample coverage, company types, duplications, currencies, and all formula inputs against original filings. **Do not push partial/inaccurate `universe.csv` as the finished assignment**.

After committing an adequately validated `universe.csv` to GitHub, the app will automatically default to **Saved real universe.csv**, avoiding hundreds of Yahoo requests during every visitor session.

## Calculated models (Phase 1)

- **GPA:** `(revenue - cost of revenue) / total assets`
- **B/M:** `positive common equity / market cap` only if market/statement currencies match
- **STA:** `(net income - operating cash flow) / total assets`
- Supporting **ROA**, **CFOA**, current vs. matched prior-year revenue and cash-flow charts
- Sector-relative median/mean/percentile statistics from full eligible universe; filters do not rebase rankings
- **BUY** = sector percentiles of GPA and B/M meeting separately adjusted thresholds plus a disclosed weighted composite
- **SELL** = independent positive-STA percentile risk flag; not the inverse of BUY
- Financials and Real Estate are present but deliberately excluded from incompatible industrial-company ratios

Not yet implemented: **SNOA, Beneish PMAN, Altman Z**. The course's definitions, coefficients and variant rules are necessary before reliable implementation. Never silently substitute variants.

## Investment Research Desk

Choose a company to examine accounting evidence, cash-flow warnings, sector comparatives, missing inputs and an editable human-reviewed investment case. Download the report as Markdown. Template text is **not an approved recommendation**: manually verify SEC filings, valuation, forward estimates, catalysts, risks and management disclosures before submission.

## Testing and deployment

```bash
pip install -r requirements.txt pytest
pytest -q
streamlit run app.py
```

The GitHub workflow `.github/workflows/validation.yml` runs offline accounting tests **and separately attempts a live Yahoo Finance retrieval** for `AAPL`, `MSFT`, and `NVDA`, saving an artifact if possible. The real-data step can fail due to Yahoo restrictions even when the code is correct; inspect workflow logs before treating it as verified.

Streamlit Community Cloud deployment: repository `vpancholi808/financial-analysis-app`, branch `main`, entrypoint `app.py`.

## Assignment remaining

1. Collect and validate **500+ real securities** (or approved single-industry alternative).
2. Add your 60-second pitch company ticker to `excluded_tickers` in `config.json`.
3. Upload Chapter 3–4 notes to confirm SNOA / Beneish / Altman exact specifications.
4. Manually cross-check one company for each mandatory model.
5. Write genuine company-specific investment cases; use research desk downloads as working papers.
