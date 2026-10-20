"""Evidence-led research notes. Quant signals are not human-approved recommendations."""
from __future__ import annotations
import math
import pandas as pd


def metric(val, percent=False):
    if pd.isna(val):
        return 'Unavailable'
    return f'{val * 100:,.2f}%' if percent else f'{val:,.3f}'


def growth(current, prior):
    """Avoid misleading percent changes from nonpositive prior-year bases."""
    if pd.isna(current) or pd.isna(prior) or prior <= 0:
        return float('nan')
    return (current / prior) - 1


def evidence(row):
    """Only statements grounded in existing row values; qualify interpretation."""
    r = row
    good, risks, gaps = [], [], []
    if pd.notna(r.get('GPA')):
        good.append(f"Gross profitability of {metric(r['GPA'], True)} of assets "
                    f"(gross profit {metric(r.get('gross_profit'))} in reporting-currency units).")
    else:
        gaps.append('Gross profit or assets needed for GPA.')
    if pd.notna(r.get('BM')):
        good.append(f"Book-to-market ratio {metric(r['BM'])}; not a stand-alone undervaluation conclusion.")
    else:
        gaps.append('Comparable positive common equity and same-currency market capitalization for B/M.')
    if pd.notna(r.get('CFOA')):
        good.append(f"Operating cash flow/assets {metric(r['CFOA'], True)}.")
    else:
        gaps.append('Operating cash flow data for CFOA and STA.')
    if pd.notna(r.get('STA')):
        if r['STA'] > 0:
            risks.append(f"Positive scaled total accruals of {metric(r['STA'], True)}; profits exceed operating cash flow relative to assets.")
        else:
            good.append(f"Non-positive scaled total accruals of {metric(r['STA'], True)}; not evidence by itself of strong earnings quality.")
    if pd.notna(r.get('STA_sector_pct')) and r['STA_sector_pct'] >= 80:
        risks.append(f"STA sits at the {r['STA_sector_pct']:.0f}th sector percentile (higher accruals imply more concern).")
    if pd.notna(r.get('BM_sector_pct')):
        good.append(f"B/M sector percentile: {r['BM_sector_pct']:.0f}th (higher means greater book value relative to market cap).")
    if pd.notna(r.get('GPA_sector_pct')):
        good.append(f"GPA sector percentile: {r['GPA_sector_pct']:.0f}th.")
    if not bool(r.get('industrial_applicable', False)):
        risks.append('Standard industrial-company screening is not applicable to this sector; specialist analysis is required.')
    gr = growth(r.get('revenue'), r.get('prior_revenue'))
    if pd.notna(gr):
        good.append(f"Reported fiscal-year revenue change: {metric(gr, True)} versus prior matched annual statement.")
    else:
        gaps.append('Matched prior-year revenue for a reliable growth comparison.')
    gaps.append('Course-verified SNOA, Beneish PMAN, Altman Z and filing-based diligence.')
    gaps.append('Price target, forward earnings and catalysts are not modeled; investment return cannot be inferred.')
    return {'facts':good, 'risks':risks, 'missing':gaps, 'revenue_growth':gr}


def research_report(row, analyst_note=''):
    r = row
    ev = evidence(r)
    ticker = str(r.get('ticker', '')).upper()
    src = str(r.get('data_source', 'Unavailable'))
    asof = str(r.get('price_asof', 'Unavailable'))
    fy = str(r.get('fiscal_year_end', 'Unavailable'))
    signal = ('BUY and SELL signals (conflict)' if r.get('conflicting_signals', False) else
              'BUY screen candidate' if r.get('BUY_candidate', False) else
              'SELL accounting-risk candidate' if r.get('SELL_candidate', False) else
              'No confirmed screening signal')
    sections = [f'# {ticker} — Evidence-based equity research draft', '',
                f'**Company:** {r.get("name", ticker)}',
                f'**Sector / industry:** {r.get("sector", "")} / {r.get("industry", "")}',
                f'**Screening classification:** {signal}',
                f'**Fiscal year:** {fy} | **Market snapshot date:** {asof}',
                f'**Source:** {src}',
                '**All amounts in source financial currency unless noted; data must be checked against filings.**',
                '', '## Financial facts']
    sections += [f'- {v}' for v in ev['facts']] or ['- Required data unavailable.']
    sections += ['', '## Source financial inputs (reported units)',
                 f'- Revenue: {metric(r.get("revenue"))}',
                 f'- Gross profit: {metric(r.get("gross_profit"))}',
                 f'- Net income: {metric(r.get("net_income"))}',
                 f'- Operating cash flow: {metric(r.get("operating_cash_flow"))}',
                 f'- Total assets: {metric(r.get("total_assets"))}',
                 f'- Common equity: {metric(r.get("common_equity"))}',
                 f'- Market capitalization: {metric(r.get("market_cap"))}',
                 '', '## Accounting risks']
    sections += [f'- {v}' for v in ev['risks']] or ['- No quantified STA red flag in the available indicators; this does not prove absence of risk.']
    sections += ['', '## Missing evidence and model limitations']
    sections += [f'- {v}' for v in ev['missing']]
    sections += ['', '## Analyst investment conclusion (human review required)',
                 analyst_note.strip() if analyst_note.strip() else
                 'Not completed. Review original SEC filings, financial health, valuation, industry context, and limitations before writing a BUY or SELL thesis.',
                 '', '**Academic screening research only — not investment advice.**']
    return '\n'.join(sections)
