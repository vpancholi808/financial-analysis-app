# Financial Analysis App — ACCT 3343

Streamlit starter application for US equity accounting analysis. The first launch uses clearly labeled **synthetic demonstration data**, not actual stock fundamentals.

## Streamlit Community Cloud

Set the repository to `vpancholi808/financial-analysis-app`, branch `main`, main file `app.py`. Dependencies are in `requirements.txt`.

## Development status

Phase 1: GPA (gross profitability), book-to-market, and scaled total accruals; sector-relative benchmarking; independent BUY and SELL candidate screening. SNOA, Beneish PMAN and Altman Z are pending validation against ACCT 3343 course formulas. Real company data for 500–1,000 firms and exclusion of the 60-second pitch ticker remain required before submission.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
