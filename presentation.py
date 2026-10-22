"""Plain-language presentation of the screen results.

Turns the numbers already computed by finance_models / research_workbench into
the words, labels and small HTML pieces the app shows: call names, one-line
reasons, data status, reasons for and against, and the per-question evidence
rows with peer strips. Nothing here changes a calculation; every value shown
is read from the prepared/classified frame. Missing inputs stay missing.
"""
from __future__ import annotations

import html
import math

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- basics
GOOD, BAD, CAUTION, NEUTRAL, NA = "good", "bad", "caution", "neutral", "na"

CALL_NAME = {"Long": "Buy candidate", "Short / avoid": "Avoid",
             "No position": "No clear signal", "Not rated": "Not rated"}
CALL_SHORT = {"Long": "Buy", "Short / avoid": "Avoid",
              "No position": "No clear signal", "Not rated": "Not rated"}
CALL_ICON = {"Long": "▲", "Short / avoid": "▼", "No position": "○", "Not rated": "–"}
CALL_TONE = {"Long": GOOD, "Short / avoid": BAD, "No position": NEUTRAL, "Not rated": NA}
EVIDENCE = {"High": "Strong evidence", "Moderate": "Moderate evidence", "Low": "Limited evidence"}
WARNING_NAMES = (("STA_risk_flag", "profit running ahead of cash"),
                 ("SNOA_risk_flag", "heavy balance sheet"),
                 ("Beneish_risk_flag", "high manipulation score"),
                 ("Altman_risk_flag", "distress-zone score"))


def num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return math.nan
    return v if math.isfinite(v) else math.nan


def ok(x) -> bool:
    return not math.isnan(num(x))


def get(row, key, default=None):
    try:
        v = row.get(key, default)
    except AttributeError:
        return default
    return default if v is None else v


def money(x) -> str:
    v = num(x)
    if math.isnan(v):
        return "—"
    a = abs(v)
    s = (f"${a/1e12:,.2f}T" if a >= 1e12 else f"${a/1e9:,.1f}B" if a >= 1e9
         else f"${a/1e6:,.0f}M" if a >= 1e6 else f"${a:,.0f}")
    return ("−" if v < 0 else "") + s


def minus(x, digits=2) -> str:
    """Number with a typographic minus sign."""
    v = num(x)
    return "—" if math.isnan(v) else f"{v:.{digits}f}".replace("-", "−")


def top_share(pct) -> str:
    """Sector percentile 98 -> 'top 2%'."""
    p = num(pct)
    return "" if math.isnan(p) else f"top {max(1, int(round(100 - p)))}%"


def cheaper_than(pct, sector) -> str:
    p = num(pct)
    return "the cheapest in its sector" if p >= 99.5 else f"cheaper than {int(round(p))}% of peers"


def sector_word(sector) -> str:
    return str(sector or "sector")


# ---------------------------------------------------------------- calls
def call_label(position, conviction="", with_icon=True) -> str:
    name = CALL_SHORT.get(position, str(position))
    lab = (CALL_ICON.get(position, "") + " " + name).strip() if with_icon else name
    return lab


def evidence_label(position, conviction) -> str:
    if position in ("Long", "Short / avoid"):
        return EVIDENCE.get(str(conviction), "")
    return ""


def warnings_fired(row) -> list[str]:
    return [name for key, name in WARNING_NAMES if bool(get(row, key, False))]


def is_near_miss(row, settings=None) -> bool:
    """No call, but within 10 percentile points of both buy thresholds and no warnings."""
    if get(row, "rec_position") != "No position" or not bool(get(row, "scoreable", True)):
        return False
    g, b = num(get(row, "GPA_sector_pct")), num(get(row, "BM_sector_pct"))
    gmin = getattr(settings, "buy_gpa_pct", 60.)
    bmin = getattr(settings, "buy_bm_pct", 60.)
    return (ok(g) and ok(b) and g >= gmin - 10 and b >= bmin - 10
            and int(num(get(row, "SELL_flag_count", 0)) or 0) == 0)


def data_status(row) -> tuple[str, str]:
    """(label, tone) describing where the statements came from and how fresh they are."""
    if bool(get(row, "stale_statements", False)):
        return "Out of date", CAUTION
    filled = str(get(row, "sec_filled_fields", "") or "")
    if filled in ("", "nan"):
        return "Complete", NEUTRAL
    if filled.startswith("all"):
        fy = pd.to_datetime(get(row, "fiscal_year_end"), errors="coerce")
        return (f"From FY{fy.year} 10-K" if pd.notna(fy) else "From 10-K"), NEUTRAL
    return "Partly from 10-K", NEUTRAL


def _level(p) -> str:
    p = num(p)
    if math.isnan(p):
        return ""
    return ("Very high" if p >= 80 else "Above-average" if p >= 60 else "Average" if p >= 40
            else "Below-average" if p >= 20 else "Low")


def why_line(row, settings=None) -> str:
    """One plain sentence explaining the call (used in cards, tables and exports)."""
    pos = get(row, "rec_position")
    sector = sector_word(get(row, "sector"))
    g, b = num(get(row, "GPA_sector_pct")), num(get(row, "BM_sector_pct"))
    fired = warnings_fired(row)
    if not bool(get(row, "industrial_applicable", True)):
        return "Bank, insurer or REIT: these accounting screens are built for operating companies and don't apply."
    if bool(get(row, "stale_statements", False)):
        return "Latest annual figures are out of date, so it isn't scored until they refresh."
    if pos == "Long":
        s = f"{top_share(g).capitalize()} profitability in {sector}, {cheaper_than(b, sector)}."
        return s + (" No red flags." if not fired else f" One concern: {fired[0]}." if len(fired) == 1
                    else f" {len(fired)} concerns to watch.")
    if pos == "Short / avoid":
        s = f"{len(fired)} warning signs: " + ", ".join(fired) + "."
        pb = 1 / num(get(row, "BM")) if num(get(row, "BM")) > 0 else math.nan
        if ok(pb) and pb >= 10:
            s += f" Priced at {pb:.0f}× book."
        return s[0].upper() + s[1:]
    # No clear signal
    if is_near_miss(row, settings):
        gmin = getattr(settings, "buy_gpa_pct", 60.)
        bmin = getattr(settings, "buy_bm_pct", 60.)
        if g < gmin:
            return (f"Near miss: profitability rank {int(round(g))} (a buy needs {int(gmin)}), "
                    f"value rank {int(round(b))}. No warning signs.")
        if b < bmin:
            return (f"Near miss: {top_share(g)} profitability, but value rank {int(round(b))} "
                    f"(a buy needs {int(bmin)}). No warning signs.")
        return "Near miss: passes both ranks but not the combined score. No warning signs."
    parts = []
    if not ok(g):
        parts.append("No cost of revenue reported, so profitability can't be ranked")
    else:
        parts.append(f"{_level(g)} profitability")
    if not ok(b):
        eq = num(get(row, "common_equity"))
        parts.append("book equity is negative, so value can't be ranked" if ok(eq) and eq <= 0
                     else "value can't be ranked")
    else:
        parts.append("cheap against book value" if b >= 60 else "mid-pack on value" if b >= 40
                     else "expensive against book value")
    s = "; ".join(parts) + "."
    s = s[0].upper() + s[1:]
    if fired:
        s += f" One warning sign: {fired[0]}." if len(fired) == 1 else f" {len(fired)} warning signs."
    else:
        s += " No warning signs."
    return s


# ---------------------------------------------------------------- bottom line
def reasons(row, findings=None, settings=None) -> tuple[list[str], list[str]]:
    """Up to three reasons for and three against, in plain words (markdown bold allowed)."""
    findings = findings or {}
    sector = sector_word(get(row, "sector"))
    pos = get(row, "rec_position")
    g, b = num(get(row, "GPA_sector_pct")), num(get(row, "BM_sector_pct"))
    gpa, bm = num(get(row, "GPA")), num(get(row, "BM"))
    mbm = num(get(row, "BM_sector_median"))
    ni, cfo = num(get(row, "net_income")), num(get(row, "operating_cash_flow"))
    rev, prev = num(get(row, "revenue")), num(get(row, "prior_revenue"))
    growth = rev / prev - 1 if ok(rev) and ok(prev) and prev > 0 else math.nan
    fired = warnings_fired(row)
    pro, con = [], []
    if ok(g) and g >= 60:
        pro.append(f"Earns **${gpa:.2f} of gross profit per $1 of assets**, more than {int(round(g))}% of {sector} companies")
    if ok(b) and b >= 60 and bm > 0:
        pro.append(f"Priced at **{1/bm:.1f}× book value** against a sector typical of {1/mbm:.1f}×, {cheaper_than(b, sector)}"
                   if ok(mbm) and mbm > 0 else f"{cheaper_than(b, sector).capitalize()} against book value")
    if ok(ni) and ok(cfo) and cfo > ni > 0:
        s = f"Operating cash flow of **{money(cfo)} beats profit of {money(ni)}**"
        pro.append(s + (f"; revenue grew {growth*100:.1f}%" if ok(growth) and growth > 0 else ""))
    if not fired and bool(get(row, "scoreable", True)):
        pro.append("No accounting warning signs")
    if str(get(row, "Altman_zone", "")) == "Lower distress signal":
        pro.append(f"Well clear of financial distress (Altman score {num(get(row, 'Altman_Z')):.2f})")
    for key, name in WARNING_NAMES:
        if not bool(get(row, key, False)):
            continue
        if key == "STA_risk_flag":
            con.append(f"Profit is running ahead of cash: accruals are {num(get(row,'STA'))*100:.1f}% of assets, "
                       f"higher than {int(round(num(get(row,'STA_sector_pct'))))}% of peers")
        elif key == "SNOA_risk_flag":
            con.append(f"Operating assets are **{num(get(row,'SNOA'))*100:.0f}% of total assets**, heavier than "
                       f"{int(round(num(get(row,'SNOA_sector_pct'))))}% of peers; bloated balance sheets tend to precede weaker earnings")
        elif key == "Beneish_risk_flag":
            con.append(f"Manipulation score of {minus(get(row,'Beneish_M'))} is above the warning line; worth a closer look at the filings")
        else:
            con.append(f"Altman distress score of {num(get(row,'Altman_Z')):.2f} is in the distress zone")
    if ok(g) and g < 40:
        con.append(f"Less profitable than {int(round(100-g))}% of {sector} peers")
    if ok(b) and b < 40 and bm > 0:
        con.append(f"More expensive than {int(round(100-b))}% of peers against book value")
    eq = num(get(row, "common_equity"))
    if not ok(b) and ok(eq) and eq <= 0:
        con.append("Book equity is negative (often from buybacks), so cheapness can't be judged")
    if ok(ni) and ni <= 0:
        con.append(f"Lost money last year ({money(ni)})")
    if ok(growth) and growth < 0:
        con.append(f"Revenue fell {abs(growth)*100:.1f}% last year")
    if bool(get(row, "Beneish_outlier", False)):
        con.append("Manipulation score is distorted by an extreme input, so it isn't used")
    if pos == "Long":
        con.append("A low price against book value can mean the market expects these assets to earn poorly")
    if pos == "Short / avoid":
        con.append("Warning signs flag risk, not certainty; check the 10-K before acting")
    return pro[:3], con[:3]


def bottom_line(row, settings=None) -> str:
    pos = get(row, "rec_position")
    sector = sector_word(get(row, "sector"))
    fired = warnings_fired(row)
    ni, cfo = num(get(row, "net_income")), num(get(row, "operating_cash_flow"))
    if pos == "Long":
        g = num(get(row, "GPA_sector_pct"))
        s = (f"{'Very profitable' if ok(g) and g >= 80 else 'Profitable'} and cheap for a {sector} company"
             + (", with profits backed by cash." if ok(ni) and ok(cfo) and cfo > ni > 0 else "."))
        if fired:
            s += f" One concern to watch: {fired[0]}." if len(fired) == 1 else f" {len(fired)} concerns to watch."
        return s
    if pos == "Short / avoid":
        return f"{len(fired)} independent accounting warning signs: {', '.join(fired)}."
    return why_line(row, settings)


# ---------------------------------------------------------------- palette
def palette(theme: str | None) -> dict:
    if str(theme).lower() == "light":
        return dict(text="#1f2430", muted="#5b6475", card="#ffffff", border="#dde2ea", track="#e3e7ee",
                    dot="#8f99ab", median="#3b4252", accent="#2f6fd6",
                    good="#1e7a46", good_bg="#e3f4ea", good_shade="#cdebd8",
                    bad="#b42318", bad_bg="#fde8e7", bad_zone="#f7d4d1",
                    caution="#8a5a00", caution_bg="#fdf1d8", na="#5b6475", na_bg="#eef1f5",
                    box_good="#eef8f2", box_good_border="#9fd4b4", box_bad="#fdf1f0", box_bad_border="#f0b5af",
                    box_neutral="#f6f7f9")
    return dict(text="#e6e8ee", muted="#9aa3b2", card="#161b26", border="#262d3d", track="#2a3142",
                dot="#6b7690", median="#c3cad6", accent="#5b9cf6",
                good="#7fe0a6", good_bg="#17321f", good_shade="#1e4a31",
                bad="#f6a0a0", bad_bg="#3b1d1d", bad_zone="#4a2323",
                caution="#f3c27a", caution_bg="#3a2e14", na="#9aa3b2", na_bg="#1c2230",
                box_good="#13261c", box_good_border="#2e6b47", box_bad="#2a1717", box_bad_border="#6b2e2e",
                box_neutral="#161b26")


def _e(s) -> str:
    return html.escape(str(s), quote=True)


def _md_bold(s) -> str:
    """Escape, then turn **x** into <b>x</b> (the only markup reasons use)."""
    parts = _e(s).split("**")
    return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))


def pill(label, tone, P) -> str:
    fg = {GOOD: P["good"], BAD: P["bad"], CAUTION: P["caution"]}.get(tone, P["na"])
    bg = {GOOD: P["good_bg"], BAD: P["bad_bg"], CAUTION: P["caution_bg"]}.get(tone, P["na_bg"])
    return (f"<span style='display:inline-block;padding:3px 11px;border-radius:999px;background:{bg};"
            f"color:{fg};font-weight:600;font-size:14px;white-space:nowrap'>{_e(label)}</span>")


# ---------------------------------------------------------------- bottom-line box
def bottom_line_html(row, findings, settings, P) -> str:
    pos = get(row, "rec_position")
    tone = CALL_TONE.get(pos, NA)
    box = {GOOD: (P["box_good"], P["box_good_border"]), BAD: (P["box_bad"], P["box_bad_border"])}.get(
        tone, (P["box_neutral"], P["border"]))
    fg = {GOOD: P["good"], BAD: P["bad"]}.get(tone, P["text"])
    ev = evidence_label(pos, get(row, "rec_conviction", ""))
    if is_near_miss(row, settings):
        ev = "Near miss"
    pro, con = reasons(row, findings, settings)
    li = lambda items, mark, col: "".join(
        f"<div style='display:flex;gap:10px'><span style='color:{col}' aria-hidden='true'>{mark}</span>"
        f"<span>{_md_bold(t)}</span></div>" for t in items)
    lists = ""
    if pos in ("Long", "Short / avoid", "No position") and bool(get(row, "scoreable", True)):
        lists = (f"<div style='display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px 32px'>"
                 f"<div style='display:flex;flex-direction:column;gap:8px'><div style='font-weight:600;color:{P['good']}'>"
                 f"{'Why buy' if pos == 'Long' else 'In its favour'}</div>"
                 + (li(pro, "✓", P["good"]) or f"<div style='color:{P['muted']}'>Nothing stands out.</div>") +
                 f"</div><div style='display:flex;flex-direction:column;gap:8px'><div style='font-weight:600;color:{P['caution']}'>"
                 f"{'Why avoid' if pos == 'Short / avoid' else 'What could go wrong'}</div>"
                 + (li(con, "!", P["caution"]) or f"<div style='color:{P['muted']}'>Nothing stands out.</div>") +
                 "</div></div>")
    return (f"<div style='background:{box[0]};border:1px solid {box[1]};border-radius:14px;padding:22px 26px;"
            f"display:flex;flex-direction:column;gap:16px;color:{P['text']};font-size:16px;line-height:1.5'>"
            f"<div style='display:flex;flex-wrap:wrap;align-items:center;gap:10px 16px'>"
            f"<span style='font-size:13px;letter-spacing:.1em;text-transform:uppercase;color:{P['muted']}'>Bottom line</span>"
            f"<span style='font-size:22px;font-weight:700;color:{fg}'>{_e(CALL_ICON.get(pos,'') + ' ' + CALL_NAME.get(pos, str(pos)))}</span>"
            + (pill(ev, CAUTION if ev == "Near miss" else tone, P) if ev else "") +
            f"</div><p style='margin:0;font-size:19px;line-height:1.45;max-width:900px'>{_md_bold(bottom_line(row, settings))}</p>"
            f"{lists}</div>")


# ---------------------------------------------------------------- evidence rows
def _peers(base, row, key):
    if base is None or key not in base:
        return pd.Series(dtype=float)
    m = (base["sector"].eq(get(row, "sector")) & base[key].notna())
    if "scoreable" in base:
        m &= base["scoreable"].astype(bool)
    if "issuer_primary" in base:
        m &= base["issuer_primary"].astype(bool)
    if key == "Beneish_M" and "Beneish_outlier" in base:
        m &= ~base["Beneish_outlier"].fillna(False).astype(bool)
    return pd.to_numeric(base.loc[m, key], errors="coerce").dropna()


def strip(peers, value, median, higher_better, extra=()):
    """Positions (0-100) for a peer strip; range spans the 2nd-98th percentile,
    the company and any marker in `extra`."""
    if len(peers) < 3 or not ok(value):
        return None
    lo, hi = float(np.nanpercentile(peers, 2)), float(np.nanpercentile(peers, 98))
    pts = [lo, hi, value] + [x for x in [median, *extra] if ok(x)]
    lo, hi = min(pts), max(pts)
    pad = (hi - lo) * 0.04 or 1.0
    lo, hi = lo - pad, hi + pad
    pos = lambda v: float(min(100, max(0, (v - lo) / (hi - lo) * 100)))
    m = pos(median) if ok(median) else None
    shade = None if m is None else ((m, 100 - m) if higher_better else (0, m))
    return dict(dots=[pos(v) for v in peers.clip(np.nanpercentile(peers, 2), np.nanpercentile(peers, 98))],
                value=pos(value), median=m, shade=shade, pos=pos)


def _standing(pct, higher_better):
    p = num(pct)
    return math.nan if math.isnan(p) else (p if higher_better else 100 - p)


def evidence_rows(row, base, settings=None) -> list[dict]:
    """One dict per question: title, lines [(text, tone)], status (label, tone),
    strip (positions or None), markers and axis labels."""
    sector = sector_word(get(row, "sector"))
    rows = []
    applicable = bool(get(row, "scoreable", True))

    def status_ranked(s):
        if math.isnan(s):
            return ("n/a", NA)
        return ("Good", GOOD) if s >= 67 else ("In line", NEUTRAL) if s >= 33 else ("Below peers", CAUTION)

    def status_tested(failed, s, review=False, missing=False):
        if missing:
            return ("n/a", NA)
        if review:
            return ("Review", CAUTION)
        if failed:
            return ("Concern", BAD)
        return ("Good", GOOD) if (math.isnan(s) or s >= 50) else ("OK, below peers", CAUTION)

    # 1 Gross profitability
    gpa, p = num(get(row, "GPA")), num(get(row, "GPA_sector_pct"))
    med = num(get(row, "GPA_sector_median"))
    lines = [(f"<b>${gpa:.2f} per $1 of assets</b> · typical peer ${med:.2f}" if ok(gpa) and ok(med)
              else f"<b>${gpa:.2f} per $1 of assets</b>" if ok(gpa)
              else "No cost of revenue reported, so it can't be measured", NEUTRAL)]
    if ok(p):
        lines.append((f"▲ More profitable than {int(round(p))}% of {sector} peers" if p >= 50
                      else f"▼ Less profitable than {int(round(100-p))}% of {sector} peers", GOOD if p >= 50 else CAUTION))
    rows.append(dict(title="How much gross profit does it make from its assets?", lines=lines,
                     status=status_ranked(_standing(p, True)) if applicable else ("n/a", NA),
                     strip=strip(_peers(base, row, "GPA"), gpa, med, True) if applicable else None,
                     axis=("Less", "More is better →")))

    # 2 Value (shown as price-to-book; ranked on book-to-market)
    bm, p = num(get(row, "BM")), num(get(row, "BM_sector_pct"))
    med = num(get(row, "BM_sector_median"))
    eq = num(get(row, "common_equity"))
    if ok(bm) and bm > 0:
        lines = [(f"<b>Price {1/bm:.1f}× book value</b>" + (f" · typical peer {1/med:.1f}×" if ok(med) and med > 0 else ""), NEUTRAL)]
    elif ok(eq) and eq <= 0:
        lines = [("Book equity is negative (often from buybacks), so price against book can't be measured", NEUTRAL)]
    else:
        lines = [("Not available", NEUTRAL)]
    if ok(p):
        lines.append((f"▲ Cheaper than {int(round(p))}% of {sector} peers" if p >= 50
                      else f"▼ More expensive than {int(round(100-p))}% of {sector} peers", GOOD if p >= 50 else CAUTION))
    rows.append(dict(title="Is it cheap compared with its book value?", lines=lines,
                     status=status_ranked(_standing(p, True)) if applicable else ("n/a", NA),
                     strip=strip(_peers(base, row, "BM"), bm, med, True) if applicable and ok(bm) else None,
                     axis=("Expensive", "Cheaper is better →")))

    # 3 Accruals
    sta, p = num(get(row, "STA")), num(get(row, "STA_sector_pct"))
    med = num(get(row, "STA_sector_median"))
    flag = bool(get(row, "STA_risk_flag", False))
    if not ok(sta):
        lines = [("Needs net income and operating cash flow", NEUTRAL)]
    elif flag:
        lines = [(f"<b>⚠ No:</b> profit is running ahead of cash (accruals {sta*100:.1f}% of assets)", BAD)]
    elif sta <= 0:
        lines = [("<b>✓ Yes:</b> cash from operations exceeds profit", GOOD)]
    else:
        lines = [(f"<b>✓ Mostly:</b> profit slightly ahead of cash ({sta*100:.1f}% of assets), within sector norms", GOOD)]
    if ok(p):
        lines.append((f"▼ But less cash-backed than {int(round(p))}% of peers" if p >= 50 and not flag
                      else f"▼ Less cash-backed than {int(round(p))}% of peers" if p >= 50
                      else f"▲ More cash-backed than {int(round(100-p))}% of peers", CAUTION if p >= 50 else GOOD))
    rows.append(dict(title="Are profits backed by real cash?", lines=lines,
                     status=status_tested(flag, _standing(p, False), missing=not ok(sta)) if applicable else ("n/a", NA),
                     strip=strip(_peers(base, row, "STA"), sta, med, False, extra=(0.0,)) if applicable else None,
                     zero=0.0, axis=("← More cash-backed is better", "Profit ahead of cash")))

    # 4 Net operating assets
    snoa, p = num(get(row, "SNOA")), num(get(row, "SNOA_sector_pct"))
    med = num(get(row, "SNOA_sector_median"))
    flag = bool(get(row, "SNOA_risk_flag", False))
    basis = str(get(row, "SNOA_basis", "") or "")
    if not ok(snoa):
        lines = [("Needs total assets, cash and total liabilities", NEUTRAL)]
    else:
        lines = [((f"<b>⚠ Yes:</b> operating assets are {snoa*100:.0f}% of total assets" if flag
                   else f"<b>✓ No:</b> operating assets are {snoa*100:.0f}% of total assets"), BAD if flag else GOOD)]
    if ok(p):
        threshold = getattr(settings, "sell_snoa_pct", 80.)
        lines.append((f"▼ Heavier than {int(round(p))}% of peers (warning at the {int(threshold)}th percentile)" if p >= 50
                      else f"▲ Leaner than {int(round(100-p))}% of peers", (BAD if flag else CAUTION) if p >= 50 else GOOD))
    if basis.startswith("Partial"):
        lines.append(("Partial inputs: " + basis.split(": ", 1)[1], NEUTRAL))
    rows.append(dict(title="Is the balance sheet bloated?", lines=lines,
                     status=status_tested(flag, _standing(p, False), missing=not ok(snoa)) if applicable else ("n/a", NA),
                     strip=strip(_peers(base, row, "SNOA"), snoa, med, False) if applicable else None,
                     axis=("← Leaner is better", "Heavier")))

    # 5 Beneish
    m, p = num(get(row, "Beneish_M")), num(get(row, "Beneish_M_sector_pct"))
    med = num(get(row, "Beneish_M_sector_median"))
    cutoff = getattr(settings, "sell_beneish_cutoff", -1.78)
    flag = bool(get(row, "Beneish_risk_flag", False))
    outlier = bool(get(row, "Beneish_outlier", False))
    if not ok(m):
        lines = [("Needs two years of eight statement lines", NEUTRAL)]
    elif outlier:
        detail = str(get(row, "Beneish_status", "")).split(":", 1)[-1].strip()
        lines = [(f"<b>● Unclear:</b> score {minus(m)} is distorted by an extreme input ({_e(detail)}), so it isn't counted", CAUTION)]
    elif flag:
        lines = [(f"<b>⚠ Yes:</b> score {minus(m)}, above the {minus(cutoff)} warning line", BAD)]
    else:
        lines = [(f"<b>✓ No:</b> score {minus(m)}; warning starts above {minus(cutoff)}", GOOD)]
    if ok(p) and not outlier:
        lines.append((f"▼ Higher than {int(round(p))}% of peers" if p >= 50
                      else f"▲ Lower than {int(round(100-p))}% of peers", CAUTION if p >= 50 else GOOD))
    rows.append(dict(title="Any warning signs of earnings manipulation?", lines=lines,
                     status=status_tested(flag, _standing(p, False), review=outlier, missing=not ok(m)) if applicable else ("n/a", NA),
                     strip=strip(_peers(base, row, "Beneish_M"), m, med, False, extra=(cutoff,)) if applicable and not outlier else None,
                     cutoff=cutoff, axis=("← Lower is better", "Warning zone")))

    # 6 Altman
    z = num(get(row, "Altman_Z"))
    zone = str(get(row, "Altman_zone", "") or "")
    model = str(get(row, "Altman_model", "") or "")
    manu = model.startswith("Original")
    d_cut, g_cut = (1.81, 2.99) if manu else (1.1, 2.6)
    if "utility" in model:
        lines, status = [("Not applied to regulated utilities", NEUTRAL)], ("n/a", NA)
    elif zone.startswith("Review"):
        lines = [(f"<b>● Not judged:</b> score {z:.2f}, but book equity is negative (often from buybacks), which the model doesn't fit", CAUTION)]
        status = ("Review", CAUTION)
    elif not ok(z):
        lines, status = [("Model inputs missing", NEUTRAL)], ("n/a", NA)
    else:
        label = {"Distress": ("⚠ Distress zone", BAD), "Grey": ("● Grey zone", CAUTION),
                 "Lower distress signal": ("✓ Safe zone", GOOD)}.get(zone, (zone, NEUTRAL))
        lines = [(f"<b>{label[0]}:</b> Altman score {z:.2f} ({'manufacturer' if manu else 'non-manufacturer'} model)", label[1]),
                 (f"Distress below {d_cut} · grey zone {d_cut} to {g_cut}", NEUTRAL)]
        status = {"Distress": ("Concern", BAD), "Grey": ("Watch", CAUTION),
                  "Lower distress signal": ("Good", GOOD)}.get(zone, ("n/a", NA))
    gauge = None
    if ok(z) and not zone.startswith("Review") and "utility" not in model:
        top = max(g_cut * 1.9, z * 1.1)
        gauge = dict(d=d_cut / top * 100, g=g_cut / top * 100, value=min(100, max(0, z / top * 100)))
    rows.append(dict(title="How close is it to financial distress?", lines=lines,
                     status=status if applicable else ("n/a", NA), strip=None, gauge=gauge if applicable else None,
                     axis=("Distress", "Safe →")))
    return rows


def _strip_html(sd, P, row) -> str:
    if not sd:
        return ""
    dots = "".join(f"<span style='position:absolute;top:12px;width:6px;height:6px;margin-left:-3px;border-radius:50%;"
                   f"background:{P['dot']};left:{x:.2f}%'></span>" for x in sd["dots"])
    shade = ""
    if sd["shade"]:
        shade = (f"<div style='position:absolute;top:9px;height:12px;border-radius:3px;background:{P['good_shade']};"
                 f"left:{sd['shade'][0]:.2f}%;width:{sd['shade'][1]:.2f}%'></div>")
    marks = ""
    if sd["median"] is not None:
        marks += f"<span style='position:absolute;top:4px;width:2px;height:22px;background:{P['median']};left:{sd['median']:.2f}%'></span>"
    if row.get("zero") is not None and "pos" in sd:
        marks += f"<span style='position:absolute;top:2px;width:2px;height:26px;background:{P['bad']};left:{sd['pos'](row['zero']):.2f}%'></span>"
    if row.get("cutoff") is not None and "pos" in sd:
        c = sd["pos"](row["cutoff"])
        marks += (f"<div style='position:absolute;top:11px;height:8px;border-radius:3px;background:{P['bad_zone']};left:{c:.2f}%;right:0'></div>"
                  f"<span style='position:absolute;top:2px;width:2px;height:26px;background:{P['bad']};left:{c:.2f}%'></span>")
    you = (f"<span style='position:absolute;top:7px;width:16px;height:16px;margin-left:-8px;border-radius:50%;"
           f"background:{P['accent']};border:2px solid {P['card']};left:{sd['value']:.2f}%'></span>")
    return (f"<div style='position:relative;height:30px'>"
            f"<div style='position:absolute;left:0;right:0;top:13px;height:4px;background:{P['track']};border-radius:2px'></div>"
            f"{shade}{dots}{marks}{you}</div>")


def _gauge_html(g, P) -> str:
    if not g:
        return ""
    return (f"<div style='position:relative;height:30px'>"
            f"<div style='position:absolute;top:9px;left:0;width:{g['d']:.1f}%;height:12px;background:{P['bad_zone']};border-radius:3px 0 0 3px'></div>"
            f"<div style='position:absolute;top:9px;left:{g['d']:.1f}%;width:{g['g']-g['d']:.1f}%;height:12px;background:{P['caution_bg']}'></div>"
            f"<div style='position:absolute;top:9px;left:{g['g']:.1f}%;right:0;height:12px;background:{P['good_shade']};border-radius:0 3px 3px 0'></div>"
            f"<span style='position:absolute;top:7px;width:16px;height:16px;margin-left:-8px;border-radius:50%;background:{P['accent']};"
            f"border:2px solid {P['card']};left:{g['value']:.1f}%'></span></div>")


def evidence_html(rows, P) -> str:
    tone_col = {GOOD: P["good"], BAD: P["bad"], CAUTION: P["caution"]}
    out = []
    for i, r in enumerate(rows):
        text = "".join(f"<div style='color:{tone_col.get(t, P['muted'] if j else P['text'])}'>{l}</div>"
                       for j, (l, t) in enumerate(r["lines"]))
        vis = _strip_html(r.get("strip"), P, r) or _gauge_html(r.get("gauge"), P)
        legend = ""
        if vis:
            legend = (f"<div style='display:flex;justify-content:space-between;font-size:13px;color:{P['muted']}'>"
                      f"<span>{_e(r['axis'][0])}</span>"
                      + ("" if r.get("gauge") else "<span>│ typical peer · <span style='color:" + P['accent'] + "'>●</span> this company</span>")
                      + f"<span>{_e(r['axis'][1])}</span></div>")
        border = f"border-bottom:1px solid {P['border']};" if i < len(rows) - 1 else ""
        out.append(
            f"<div style='display:flex;flex-wrap:wrap;gap:14px 28px;padding:18px 22px;{border}align-items:center'>"
            f"<div style='flex:1 1 320px;display:flex;flex-direction:column;gap:3px'>"
            f"<div style='font-weight:600;font-size:17px;color:{P['text']}'>{_e(r['title'])}</div>{text}</div>"
            f"<div style='flex:1 1 340px;display:flex;flex-direction:column;gap:6px'>{vis}{legend}</div>"
            f"<div style='flex:0 0 140px'>{pill(r['status'][0], r['status'][1], P)}</div></div>")
    return (f"<div style='border:1px solid {P['border']};border-radius:14px;overflow:hidden;background:{P['card']};"
            f"font-size:16px;line-height:1.5'>" + "".join(out) + "</div>")


# ---------------------------------------------------------------- screener pieces
def split_bar_html(counts: dict, P) -> str:
    total = max(1, sum(counts.values()))
    parts = [("Long", P["good"]), ("Short / avoid", P["bad"]), ("No position", P["dot"]), ("Not rated", P["track"])]
    bar = "".join(f"<div style='width:{counts.get(k,0)/total*100:.2f}%;background:{c}'></div>" for k, c in parts)
    label = ", ".join(f"{counts.get(k,0)} {CALL_SHORT[k].lower()}" for k, _ in parts)
    legend = (f"<span><b style='color:{P['good']}'>{counts.get('Long',0)}</b> buy</span>"
              f"<span><b style='color:{P['bad']}'>{counts.get('Short / avoid',0)}</b> avoid</span>"
              f"<span><b>{counts.get('No position',0)}</b> no clear signal</span>"
              f"<span><b>{counts.get('Not rated',0)}</b> not rated (banks, insurers, REITs, out-of-date data)</span>")
    return (f"<div style='display:flex;flex-direction:column;gap:8px;color:{P['text']};font-size:15px'>"
            f"<div role='img' aria-label='{_e(label)}' style='display:flex;height:14px;border-radius:7px;overflow:hidden;background:{P['na_bg']}'>{bar}</div>"
            f"<div style='display:flex;flex-wrap:wrap;gap:6px 22px;color:{P['muted']}'>{legend}</div></div>")


def sector_table_html(view, P) -> str:
    """One row per sector: ideas found and the sector's typical values (its own yardstick)."""
    sc = view
    if "scoreable" in sc:
        sc = sc.loc[sc.scoreable.astype(bool)]
    if "issuer_primary" in sc:
        sc = sc.loc[sc.issuer_primary.astype(bool)]
    if sc.empty:
        return ""
    rows = []
    for sector, g in sc.groupby("sector"):
        n_all = len(g)
        nb = int(g.rec_position.eq("Long").sum())
        na_ = int(g.rec_position.eq("Short / avoid").sum())
        gpa = num(g.GPA_sector_median.dropna().iloc[0]) if g.GPA_sector_median.notna().any() else math.nan
        bm = num(g.BM_sector_median.dropna().iloc[0]) if g.BM_sector_median.notna().any() else math.nan
        sta = num(g.STA_sector_median.dropna().iloc[0]) if g.STA_sector_median.notna().any() else math.nan
        bar = (f"<span style='display:flex;height:10px;border-radius:5px;overflow:hidden;background:{P['track']};margin-bottom:6px'>"
               f"<span style='width:{nb/n_all*100:.1f}%;background:{P['good']}'></span>"
               f"<span style='width:{na_/n_all*100:.1f}%;background:{P['bad']}'></span></span>")
        td = f"padding:12px 16px;border-bottom:1px solid {P['border']};"
        rows.append(
            f"<tr><td style='{td}'><b>{_e(sector)}</b><br><span style='color:{P['muted']};font-size:14px'>{n_all} scored</span></td>"
            f"<td style='{td}'>{bar}<span style='color:{P['good'] if nb else P['muted']}'>{nb} buy</span> · "
            f"<span style='color:{P['bad'] if na_ else P['muted']}'>{na_} avoid</span></td>"
            f"<td style='{td}text-align:right'>{'—' if math.isnan(gpa) else f'${gpa:.2f}'}</td>"
            f"<td style='{td}text-align:right'>{'—' if math.isnan(bm) or bm <= 0 else f'{1/bm:.1f}×'}</td>"
            f"<td style='{td}text-align:right'>{'—' if math.isnan(sta) else f'{sta*100:.1f}%'}</td></tr>")
    th = f"padding:12px 16px;border-bottom:1px solid {P['border']};font-weight:600;"
    return (f"<div style='overflow-x:auto;border:1px solid {P['border']};border-radius:14px;background:{P['card']}'>"
            f"<table style='width:100%;min-width:720px;border-collapse:collapse;font-size:15px;color:{P['text']}'>"
            f"<thead><tr style='text-align:left;color:{P['muted']};font-size:13px;text-transform:uppercase;letter-spacing:.06em'>"
            f"<th style='{th}'>Sector</th><th style='{th}width:280px'>Ideas in this sector</th>"
            f"<th style='{th}text-align:right'>Typical gross profit<br>per $1 of assets</th>"
            f"<th style='{th}text-align:right'>Typical price<br>vs book</th><th style='{th}text-align:right'>Typical accruals</th>"
            f"</tr></thead><tbody>" + "".join(rows) + "</tbody></table></div>")
