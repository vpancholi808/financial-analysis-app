# Financial Analysis Terminal | ACCT 3343

A Bloomberg-inspired financial accounting screener for the current **S&P 500 constituent listings**. The public Streamlit interface draws saved, actual company financial-statement figures from **Yahoo Finance through yfinance** and tracks missing values and anomalies. This remains a **research and university-assignment project**, not an independently audited financial-data vendor or verified trading model.

**Website:** https://financial-analysis-app-cpf2jt6lfpqxjfpik7a7bs.streamlit.app/

## Assignment grading alignment (20 points)

| Category | Points | How the application supports it |
| --- | ---: | --- |
| Financial analysis and methodology | 10 | Reproducible ratios, independent BUY versus SELL criteria, sector-appropriate models, accounting anomalies, source disclosures, and separately reviewed company recommendations |
| Interactivity | 6 | S&P sector and industry filters, minimum capitalization, GPA/B-M thresholds and weights, risk flag thresholds, adjustable minimum risk signal count |
| Functionality and presentation | 4 | Streamlit multipage navigation, Plotly sector and factor charts, financial trends, research report downloads, audit evidence |

**Important:** A BUY/SELL *screen candidate* is not automatically a well-supported investment recommendation. The research desk requires a written case with actual filings, valuation, catalysts and risks before submission.

## Data provenance and scope

- S&P 500 ticker membership: https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv
- Actual annual statement fields and market cap: yfinance Ticker `income_stmt`, `balance_sheet`, `cashflow`, and `info`.
- A record requires a common latest fiscal-year date across all three statements. The previous annual period is also saved when available. Period-end dates returned by Yahoo are sometimes rounded to calendar month-end.
- `universe.csv` is a frozen snapshot; no bulk Yahoo refresh is triggered when visitors open the app.
- **Second source: SEC EDGAR 10-K data** (`sec_edgar.py`, https://www.sec.gov/edgar/sec-api-documentation). The collector looks up the SEC's XBRL "companyfacts" for companies with a blank model input or a newer annual report due. It (1) fills only lines Yahoo left blank for the *same* fiscal year and (2) replaces the statements with a newer fiscal year when the 10-K is filed but Yahoo lags (market cap stays Yahoo's). Either step runs only when the SEC and Yahoo figures agree within 2% (3% for a newer year's comparatives) on at least two of total assets, revenue and operating cash flow. Filled lines are listed per company in `sec_filled_fields`, on the company profile and as `SEC_FILLED_FIELDS` rows in `data_quality.csv`. Tag mapping: revenue (Revenues / RevenueFromContractWithCustomer…), cost of revenue, net income (NetIncomeLoss), operating cash flow, EBIT (OperatingIncomeLoss), D&A, SG&A (or G&A + selling & marketing when both are tagged), assets, current assets/liabilities, total liabilities (Liabilities, else liabilities-and-equity less total equity), retained earnings, cash, receivables, net PP&E, current debt (DebtCurrent, else its tagged components), non-current long-term debt, minority interest, preferred stock; common equity = stockholders' equity − preferred stock. Set a repository variable or secret `SEC_USER_AGENT` (name and contact email) as the SEC requests. Each run records its result in `collection_state.json` → `sec_status` (companies due, filled, failed, last error, and whether the contact was set); after 5 failed lookups in a row the pass stops and failed companies are retried the next day.
- `.github/workflows/collect-universe.yml` refreshes missing companies, can update fields for advanced models, and re-fetches a company once a day from 40 days after its next fiscal year-end until a newer annual report is available (a refreshed record is accepted only if its fiscal year is later). `sp500_constituents.csv`, `data_quality.csv`, and `collection_state.json` record coverage and discrepancies.
- TSM (TSMC ADR), the separate 60-second pitch company, is excluded from recommendations. TSM is not a constituent of the US S&P 500 in the current constituent snapshot.
- Absent financial inputs remain missing. **Never substitute zero for an undisclosed preference share, minority interest, debt, or earnings figure.**

Market cap is current as of the quote snapshot; book equity and earnings are latest available historical annual statements. This **timing mismatch means B/M, some Altman variants and screen rankings are cross-sectional snapshots, not historical backtests**.

## Accounting formulas and eligibility

| Metric | Implementation and boundaries |
| --- | --- |
| GPA | Gross profit / assets. Gross profit = annual revenue − annual cost of revenue when both exist; otherwise directly reported gross profit with source basis recorded |
| B/M | Common book equity / market capitalization; requires positive equity/cap and matching financial/market currencies |
| STA | (Net income − operating cash flow) / assets |
| ROA | Net income / assets |
| CFOA | Operating cash flow / assets |
| SNOA | [(Assets − cash) − (Assets − short-term debt − long-term debt − minority interest − preferred equity − common equity)] / assets, per course definition. Yahoo omits the minority-interest and preferred-equity lines when they are zero, so when either is absent their combined value is taken from the balance sheet itself: MI + PE = assets − total liabilities (net of minority interest) − common equity. This is algebraically the same formula (operating liabilities = total liabilities − short-term debt − long-term debt) and substitutes no value; the basis used is recorded per company in `SNOA_basis`. If a debt line (or, without total liabilities, the minority-interest/preferred line) is not reported, it is left out of the formula and the value is labelled "Partial inputs: … not reported, left out". Total assets, cash and total liabilities or common equity are still required |
| Altman Z | Original public-manufacturer: 1.2 WC/A + 1.4 RE/A + 3.3 EBIT/A + 0.6 Market equity/L + Revenue/A. Thresholds: distress below 1.81, grey 1.81–2.99, lower distress signal above 2.99 |
| Altman Z double-prime | Non-manufacturer: 6.56 WC/A + 3.26 RE/A + 6.72 EBIT/A + 1.05 Book equity/L. Thresholds: distress below 1.1, grey 1.1–2.6, lower distress signal above 2.6 |
| Beneish M-score | Published eight-variable model using DSRI, GMI, AQI, SGI, DEPI, SGAI, TATA, LVGI. Default warning threshold > −1.78; not a calibrated probability of fraud. Net income is a documented proxy where continuing-operations net income is unavailable. An index outside its plausible band (DSRI/GMI/AQI 0.2–4, SGI/DEPI/SGAI/LVGI 0.3–3, TATA −0.5 to 0.5) — usually a near-zero prior-year figure — marks the score "Component outlier – review": it is shown, excluded from sector ranks and never counted as a SELL flag |

Financials and REITs are excluded from the industrial screening methods. The original Altman Z is used for an explicit, reviewable list of GICS manufacturing sub-industries (`MANUFACTURER_SUBINDUSTRIES` in `advanced_models.py`, exact match — e.g. "Automotive Retail" is not a manufacturer); every other eligible company uses Z″. Altman is not applied to regulated utilities, and a Z″ result for a company with negative book equity (often from buybacks) is reported as "Review – negative book equity" rather than distress. Companies whose annual statements are more than ~15 months older than the market-cap quote are shown but not ranked or screened, and dual share classes (GOOG/GOOGL, FOX/FOXA, NWS/NWSA) count once in sector benchmarks. Full Beneish eight-variable calculation is **not** asserted if any source field is absent; a Beneish score does not prove wrongdoing.

- Altman variants: https://en.wikipedia.org/wiki/Altman_Z-score
- Beneish explanation and coefficients: https://www.aaii.com/journal/article/83450-detecting-earnings-manipulation-with-the-m-score

## Screening and research

BUY candidates require sufficient and above-threshold **GPA and B/M sector percentiles**, plus the adjusted weighted BUY score. SELL risk candidates are derived independently from: high positive STA percentile; high SNOA sector percentile; Beneish M-score above the model threshold; or Altman model-specific distress. Minimum distinct SELL signals is adjustable. Overlapping signals remain visible as conflicts.

The investment research desk distinguishes actual financial facts, reported abnormalities, unavailable model inputs and source fiscal periods, and **pre-fills a rule-based recommendation** (`investment_case` in `research_workbench.py`): **Long** when the company passes the long screen with positive, cash-backed earnings and no accounting-risk check; **Short / avoid** when enough independent risk checks trigger and it is not a long candidate; otherwise **No position** with the reason. Conviction (High/Moderate/Low) rises when valuation (P/S vs sector median), growth and accruals agree. The thesis, catalyst (next annual report), strongest counterargument, measurable invalidation tests and SEC 10-K link are generated from the same evidence and can be edited; the working paper records which sections the analyst edited. No price target or return forecast is produced.

## Reproducible validation

```bash
pip install -r requirements.txt pytest
python -m pytest -q
python audit_data.py
streamlit run app.py
```

- `.github/workflows/financial-audit.yml`: automated unit/integration tests plus a universe-wide independent arithmetic and completeness audit, keeping an `audit_summary.json` and `metric_audit.csv` artifact.
- `tests/test_advanced_models.py`: hand-reconciled, original definitions and missing-data guards.
- `tests/test_sec_spotchecks.py`: anchored annual FY2025 Apple and FY2026 NVIDIA numbers compared with actual annual SEC 10-K figures.
- `tests/test_streamlit_app.py`: page-load checks for the Screener, Company Profile, Research Desk and Glossary & Data pages.
- `tests/test_ui_format.py`: display helpers in `ui_format.py` (units, grades, missing values shown as "—" and never as zero).
- `audit_data.py`: coverage counts, individual missing fields, currency mismatches, statement staleness, negative/inconsistent assets and common-versus-total-equity reconciliation flags.

**Limitations:** Automatic ratio verification is necessary but not equivalent to independent external audit of 500+ sets of source financial statements. Flags may reflect legitimate minority interests or issuer-specific reporting conventions. Mark material discrepancies for filing review instead of silently correcting them.

## Before grading submission

1. Confirm latest S&P 500 membership and coverage, with all uncollected/invalid tickers identified.
2. Review anomalies and sample filings from each sector; resolve source differences.
3. Validate any course-specific definition differences for Altman, Beneish and SNOA.
4. Write, source and defend at least one strong long and one short/avoid case, including valuation and alternative explanations.
5. Ensure charts, controls, downloadable research and GitHub Actions work on the live Streamlit URL.
