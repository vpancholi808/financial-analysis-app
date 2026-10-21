"""Evidence-led research notes. Quant signals are not human-approved recommendations."""
from __future__ import annotations
import math
import pandas as pd


def metric(val, percent=False):
    if pd.isna(val):
        return 'Unavailable'
    return f'{val * 100:,.2f}%' if percent else f'{val:,.3f}'


def readable_amount(value, currency='USD'):
    if pd.isna(value): return 'Unavailable'
    amount=abs(float(value))
    formatted=(f'{amount/1e12:,.2f}T' if amount>=1e12 else
               f'{amount/1e9:,.2f}B' if amount>=1e9 else
               f'{amount/1e6:,.2f}M' if amount>=1e6 else
               f'{amount/1e3:,.1f}K' if amount>=1e3 else f'{amount:,.0f}')
    prefix='$' if currency=='USD' else currency+' '
    return ('-' if value<0 else '')+prefix+formatted

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
                    f"(gross profit {readable_amount(r.get('gross_profit'),str(r.get('financial_currency') or 'USD'))} in reporting-currency units).")
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
    if pd.notna(r.get('SNOA')):
        if pd.notna(r.get('SNOA_sector_pct')) and r['SNOA_sector_pct'] >= 80:
            risks.append(f"Scaled net operating assets {metric(r['SNOA'],True)} ({r['SNOA_sector_pct']:.0f}th percentile). Investigate persistent high net operating assets.")
        else:
            good.append(f"Scaled net operating assets {metric(r['SNOA'],True)} with complete operating-asset and liability inputs.")
    else:
        gaps.append('SNOA unavailable: requires assets, cash, short/long debt and total liabilities (or the explicit minority-interest, preferred and common equity lines); absent values are not assumed zero.')
    if pd.notna(r.get('Beneish_M')):
        if r['Beneish_M'] > -1.78:
            risks.append(f"Beneish eight-variable M-score {metric(r['Beneish_M'])} exceeds -1.78 screening cutoff; this does NOT demonstrate fraud.")
        else:
            good.append(f"Beneish M-score {metric(r['Beneish_M'])} below -1.78 screening cutoff; this does not guarantee accurate reporting.")
    else:
        gaps.append('Eight-variable Beneish M-score unavailable because one or more two-year financial inputs are missing or inapplicable.')
    if pd.notna(r.get('Altman_Z')):
        label=str(r.get('Altman_model',''))
        zone=str(r.get('Altman_zone',''))
        item=f"Altman {label} score {metric(r['Altman_Z'])}: {zone}. Business-type-dependent distress screening only."
        (risks if zone == 'Distress' else good).append(item)
    else:
        gaps.append('Altman Z/Z double-prime unavailable: missing model-specific inputs or an excluded financial/REIT sector.')
    gaps.append('SEC/issuer filing checks, earnings quality, valuation and catalyst diligence are still required.')
    gaps.append('Price target, forward earnings and catalysts are not modeled; investment return cannot be inferred.')
    return {'facts':good, 'risks':risks, 'missing':gaps, 'revenue_growth':gr}


def research_report(row, analyst_note=''):
    r = row
    ev = evidence(r)
    currency=str(r.get('financial_currency') or 'USD')
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
                 f'- Revenue: {readable_amount(r.get("revenue"),currency)}',
                 f'- Gross profit: {readable_amount(r.get("gross_profit"),currency)}',
                 f'- Net income: {readable_amount(r.get("net_income"),currency)}',
                 f'- Operating cash flow: {readable_amount(r.get("operating_cash_flow"),currency)}',
                 f'- Total assets: {readable_amount(r.get("total_assets"),currency)}',
                 f'- Common equity: {readable_amount(r.get("common_equity"),currency)}',
                 f'- Market capitalization: {readable_amount(r.get("market_cap"),str(r.get("market_currency") or "USD"))}',
                 '', '## Accounting risks']
    sections += [f'- {v}' for v in ev['risks']] or ['- No quantified accounting red flag in the available indicators; this does not prove absence of risk.']
    sections += ['', '## Missing evidence and model limitations']
    sections += [f'- {v}' for v in ev['missing']]
    sections += ['', '## Analyst investment conclusion (human review required)',
                 analyst_note.strip() if analyst_note.strip() else
                 'Not completed. Review original SEC filings, financial health, valuation, industry context, and limitations before writing a BUY or SELL thesis.',
                 '', '**Academic screening research only — not investment advice.**']
    return '\n'.join(sections)
