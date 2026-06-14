"""Validate the revised volatility score against real customer demand and
emit a PDF report (spec + 4-expert review + empirical evidence).

Run:
    cd /Users/manoharchidambaram/projects/DemandProject
    .venv/bin/python analysis_volatility/build_report.py
"""

from __future__ import annotations

import os
import sys
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak,
)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from volatility_score import (  # noqa: E402
    volatility_score, original_score, W_INTERMITTENCY, W_CV2, W_SPIKE,
    ADI_LO, ADI_HI, CV2_LO, CV2_HI, SPIKE_LO, SPIKE_HI, MIN_NONZERO,
)

DATA = os.path.join(HERE, "..", "data", "input")
YEARS = [2023, 2024, 2025]
MAX_DFUS = 12000   # cap for runtime; sampled deterministically from full set


# ---------------------------------------------------------------------------
# 1. Load + aggregate real demand into per-DFU monthly series
# ---------------------------------------------------------------------------
def load_monthly_series() -> dict[tuple, np.ndarray]:
    print("Loading customer demand (chunked aggregation)...")
    frames = []
    for y in YEARS:
        path = os.path.join(DATA, f"{y}_customer_demand.csv")
        if not os.path.exists(path):
            continue
        # DFU = site + warehouse + item (customers aggregated up)
        df = pd.read_csv(
            path,
            usecols=["site", "warehouse_no", "item_no", "posting_prd", "demand_cases"],
            dtype={"site": "int32", "warehouse_no": "int32",
                   "item_no": "int32", "posting_prd": "int32",
                   "demand_cases": "float32"},
        )
        g = df.groupby(["site", "warehouse_no", "item_no", "posting_prd"],
                       sort=False)["demand_cases"].sum().reset_index()
        frames.append(g)
        print(f"  {y}: {len(df):,} rows -> {len(g):,} DFU-months")
    allm = pd.concat(frames, ignore_index=True)
    allm = (allm.groupby(["site", "warehouse_no", "item_no", "posting_prd"],
                         sort=False)["demand_cases"].sum().reset_index())

    # full monthly calendar across observed range, so zero months are explicit
    prds = sorted(allm["posting_prd"].unique())
    cal = []
    for p in range(prds[0], prds[-1] + 1):
        if p % 100 >= 1 and p % 100 <= 12:
            cal.append(p)
    cal_idx = {p: i for i, p in enumerate(cal)}
    print(f"  calendar: {len(cal)} months {cal[0]}..{cal[-1]}")

    series: dict[tuple, np.ndarray] = {}
    for (s, w, it), grp in allm.groupby(["site", "warehouse_no", "item_no"], sort=False):
        arr = np.zeros(len(cal), dtype=float)
        for prd, q in zip(grp["posting_prd"], grp["demand_cases"]):
            arr[cal_idx[prd]] = q
        series[(s, w, it)] = arr
    print(f"  DFUs: {len(series):,}")

    if len(series) > MAX_DFUS:
        rng = np.random.default_rng(42)
        keys = list(series.keys())
        pick = rng.choice(len(keys), MAX_DFUS, replace=False)
        series = {keys[i]: series[i] if False else series[keys[i]] for i in pick}
        print(f"  sampled to {len(series):,} DFUs")
    return series


# ---------------------------------------------------------------------------
# 2. Score every DFU, collect a tidy frame
# ---------------------------------------------------------------------------
def score_all(series: dict[tuple, np.ndarray]) -> pd.DataFrame:
    rows = []
    for key, arr in series.items():
        r = volatility_score(arr).as_dict()
        r["orig_score"] = original_score(arr)
        rows.append(r)
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# 3. Charts
# ---------------------------------------------------------------------------
def fig_redundancy(df: pd.DataFrame) -> str:
    d = df[(df["n_nonzero"] >= 1)].copy()
    corr = d["adi"].corr(d["zero_share"], method="spearman")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.scatter(d["zero_share"], d["adi"], s=4, alpha=0.15, color="#2b6cb0")
    xs = np.linspace(0.001, d["zero_share"].max(), 200)
    ax.plot(xs, 1 / (1 - xs), color="#c53030", lw=2, label="ADI = 1/(1-ZeroShare)")
    ax.set_xlabel("ZeroShare (fraction of zero months)")
    ax.set_ylabel("ADI (avg demand interval)")
    ax.set_ylim(0, min(20, d["adi"].quantile(0.99) * 1.1))
    ax.set_title(f"ADI vs ZeroShare on real DFUs — Spearman r = {corr:.3f}\n"
                 "(points hug the identity curve = same signal)")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    p = os.path.join(HERE, "_fig_redundancy.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p, corr


def fig_corr_matrix(df: pd.DataFrame) -> str:
    cols = ["adi", "zero_share", "cv2", "spike_ratio"]
    d = df[df["scorable"]][cols].dropna()
    c = d.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(c.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right")
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(cols)
    for i in range(len(cols)):
        for j in range(len(cols)):
            ax.text(j, i, f"{c.values[i, j]:.2f}", ha="center", va="center",
                    color="white" if abs(c.values[i, j]) > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("Spearman correlation of the 4 raw metrics")
    fig.tight_layout()
    p = os.path.join(HERE, "_fig_corr.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p, c


def fig_bands(df: pd.DataFrame) -> str:
    # original bands
    o = df["orig_score"].dropna()
    o_band = pd.cut(o, [-0.01, 0.33, 0.66, 1.01], labels=["low", "medium", "high"])
    orig_counts = o_band.value_counts().reindex(["low", "medium", "high"]).fillna(0)
    rev_counts = (df["risk_band"].value_counts()
                  .reindex(["low", "medium", "high", "manual_review"]).fillna(0))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6))
    axes[0].bar(orig_counts.index, orig_counts.values, color=["#38a169", "#dd6b20", "#c53030"])
    axes[0].set_title("Original spec — risk bands")
    axes[1].bar(rev_counts.index, rev_counts.values,
                color=["#38a169", "#dd6b20", "#c53030", "#718096"])
    axes[1].set_title("Revised spec — risk bands")
    for ax in axes:
        ax.set_ylabel("# DFUs"); ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    p = os.path.join(HERE, "_fig_bands.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p, orig_counts, rev_counts


def fig_score_hist(df: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(df["orig_score"].dropna(), bins=40, alpha=0.5, label="original", color="#a0aec0")
    ax.hist(df["score"].dropna(), bins=40, alpha=0.6, label="revised", color="#2b6cb0")
    ax.axvline(0.33, ls="--", color="#666"); ax.axvline(0.66, ls="--", color="#666")
    ax.set_xlabel("Volatility score"); ax.set_ylabel("# DFUs")
    ax.set_title("Score distribution: original vs revised")
    ax.legend()
    fig.tight_layout()
    p = os.path.join(HERE, "_fig_hist.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p


# ---------------------------------------------------------------------------
# 4. PDF
# ---------------------------------------------------------------------------
def build_pdf(df: pd.DataFrame, figs: dict, stats: dict):
    out = os.path.join(HERE, "Demand_Sensing_Volatility_Review.pdf")
    doc = SimpleDocTemplate(out, pagesize=LETTER,
                            leftMargin=0.8 * inch, rightMargin=0.8 * inch,
                            topMargin=0.7 * inch, bottomMargin=0.7 * inch)
    ss = getSampleStyleSheet()
    H1 = ParagraphStyle("H1", parent=ss["Title"], fontSize=20, spaceAfter=6)
    H2 = ParagraphStyle("H2", parent=ss["Heading2"], textColor=colors.HexColor("#2b6cb0"))
    H3 = ParagraphStyle("H3", parent=ss["Heading3"], textColor=colors.HexColor("#1a202c"))
    body = ParagraphStyle("body", parent=ss["BodyText"], fontSize=10, leading=14)
    mono = ParagraphStyle("mono", parent=ss["Code"], fontSize=8.5, leading=11,
                          backColor=colors.HexColor("#f5f7fa"), borderPadding=6)
    small = ParagraphStyle("small", parent=ss["BodyText"], fontSize=8.5,
                           textColor=colors.HexColor("#555"))
    E = []

    def para(t, st=body): E.append(Paragraph(t, st))
    def gap(h=8): E.append(Spacer(1, h))

    # ---- Title ----
    para("Demand-Sensing Volatility Score", H1)
    para("Four-expert review, revised specification, and empirical validation "
         "against 2023–2025 customer demand", small)
    gap(4)
    para(f"DFUs analysed: <b>{stats['n_dfu']:,}</b> &nbsp;|&nbsp; "
         f"calendar months: <b>{stats['n_months']}</b> &nbsp;|&nbsp; "
         f"scorable: <b>{stats['n_scorable']:,}</b> &nbsp;|&nbsp; "
         f"routed to manual review (small sample): <b>{stats['n_manual']:,}</b>", small)
    gap(10)

    para("Executive summary", H2)
    para(
        "The proposed design is well grounded — the ADI=1.32 and CV²=0.49 cutoffs are "
        "the Syntetos–Boylan–Croston demand-classification boundaries, and the repo's "
        "<i>demand_classifier.py</i> already uses them. The issues below are about how the "
        "four metrics are <b>combined</b>, not whether they are the right metrics.", body)
    gap(4)
    items = [
        ("Redundancy (highest impact)", f"ADI and ZeroShare are the same variable: "
         f"ADI = 1/(1−ZeroShare). On the real data their Spearman correlation is "
         f"<b>{stats['adi_zero_corr']:.3f}</b>. Equal 25% weights therefore give intermittency "
         "~50% and quantity-variability ~50% — not four independent quarters. The revised "
         "score keeps ADI and drops ZeroShare from the weighted sum (still reported)."),
        ("Spike fragility", "max/mean is sensitive to one outlier and grows with history "
         "length. Replaced with robust p95/median."),
        ("CV² saturation", "the 0.25→0.49 band is only 0.24 wide and pegs at 1 almost "
         "immediately; widened to 0.25→1.0 so it discriminates across the erratic range."),
        ("Seasonality", "predictable seasonal SKUs (zero in the off-season) score high on "
         "ADI+ZeroShare and get wrongly flagged high-risk. Added a seasonal-strength guard "
         "that exempts/deseasonalises them."),
        ("Small samples", f"with &lt;{MIN_NONZERO} non-zero months CV²/spike are unstable; "
         "those DFUs are routed to manual review rather than given a false score."),
    ]
    for h, t in items:
        para(f"• <b>{h}.</b> {t}", body)
    gap(2)

    E.append(PageBreak())

    # ---- 4 experts ----
    para("The four-expert review", H2)
    experts = [
        ("1 — Intermittent-demand statistician",
         ["ADI ≡ 1/(1−ZeroShare): same signal, two scales. Confirmed on data "
          f"(Spearman {stats['adi_zero_corr']:.3f}).",
          f"CV² and SpikeRatio also overlap (both measure non-zero dispersion; "
          f"Spearman {stats['cv2_spike_corr']:.2f}). Four 'independent' metrics are really ~2 factors.",
          "Fix: one intermittency term (ADI), keep CV² + a de-correlated spike term; state the real weighting."]),
        ("2 — Forecasting practitioner",
         ["Seasonality is read as volatility; the most forecastable seasonal SKUs get pushed to manual review.",
          "MTD-vs-LY must be like-for-like: this year's elapsed days vs the SAME elapsed days last "
          "year (MTD-to-MTD), not partial month vs full prior-year month, or early-month signals read low and over-trigger."]),
        ("3 — Normalization & robustness reviewer",
         ["Make every subscore an explicit clamped linear ramp.",
          "Widen CV² top to 1.0; replace max/mean spike with p95/median.",
          f"Guard: &lt;{MIN_NONZERO} non-zero months → not scorable. Fix typos: header 'ASI'→ADI; "
          "Spike '≥2 = 2' → '= 1'."]),
        ("4 — Calibration & productionization",
         ["0.33/0.66 are placeholders — re-derive the low-risk band as the score range where "
          "automated adjustment historically matched/beat baseline (backtest by score decile).",
          "Separate 'volatility' from 'confidence' (data sufficiency + signal size) as two gates.",
          "Log all four subscores per DFU in UAT, not just the total."]),
    ]
    for name, pts in experts:
        para(name, H3)
        for p in pts:
            para(f"• {p}", body)
        gap(4)

    E.append(PageBreak())

    # ---- Revised spec ----
    para("Revised specification", H2)
    para("Subscore convention unchanged: 0 = stable, 1 = highly volatile; final score in [0,1].", body)
    gap(4)
    spec = (
        "GUARDS\n"
        f"  n_nonzero &lt; {MIN_NONZERO}        -> not scorable, route to manual_review\n"
        "  strong seasonality      -> deseasonalise / exempt (band = low)\n\n"
        "RAW METRICS (on trailing monthly series incl. zero months)\n"
        "  ADI         = n_periods / n_nonzero\n"
        "  ZeroShare   = n_zero / n_periods            # reported only, NOT scored\n"
        "  CV2         = (std_nz / mean_nz)^2          # ddof=1, non-zero months\n"
        "  SpikeRatio  = p95(nonzero) / median(nonzero)\n\n"
        "SUBSCORES (clamped linear ramp 0..1)\n"
        f"  intermittency_sub = clamp((ADI - {ADI_LO}) / ({ADI_HI} - {ADI_LO}), 0, 1)\n"
        f"  cv2_sub           = clamp((CV2 - {CV2_LO}) / ({CV2_HI} - {CV2_LO}), 0, 1)\n"
        f"  spike_sub         = clamp((SpikeRatio - {SPIKE_LO}) / ({SPIKE_HI} - {SPIKE_LO}), 0, 1)\n\n"
        "SCORE (ZeroShare dropped; weights rebalanced)\n"
        f"  score = {W_INTERMITTENCY}*intermittency_sub + {W_CV2}*cv2_sub + {W_SPIKE}*spike_sub\n\n"
        "BANDS (UAT default — calibrate against backtested error)\n"
        "  0.00-0.33 low  (automate)   0.34-0.66 medium   0.67-1.0 high"
    )
    E.append(Paragraph(spec.replace(" ", "&nbsp;").replace("\n", "<br/>"), mono))
    gap(6)
    para("MTD-vs-LY decision logic (like-for-like):", H3)
    dec = (
        "ratio = MTD_consumption_this_year / MTD_consumption_same_elapsed_days_last_year\n"
        "if band == 'low':      auto-apply forecast adjustment from ratio\n"
        "elif band == 'medium': adjust but flag for review / damp the change\n"
        "else (high/manual):    no auto-change; planner review"
    )
    E.append(Paragraph(dec.replace(" ", "&nbsp;").replace("\n", "<br/>"), mono))

    E.append(PageBreak())

    # ---- Empirical evidence ----
    para("Empirical validation (real 2023–2025 demand)", H2)
    para(f"<b>Finding 1 — ADI and ZeroShare are redundant.</b> Spearman "
         f"correlation {stats['adi_zero_corr']:.3f}; points sit on the algebraic curve "
         "ADI = 1/(1−ZeroShare). Including both double-counts intermittency.", body)
    E.append(Image(figs["redundancy"], width=5.6 * inch, height=3.65 * inch))
    gap(6)
    E.append(Image(figs["corr"], width=4.5 * inch, height=3.8 * inch))
    para("CV² and SpikeRatio also correlate, confirming the four metrics collapse to "
         "~two underlying factors.", small)

    E.append(PageBreak())
    para("Impact on risk bands & automation rate", H2)
    para("Re-weighting and the seasonality / small-sample guards shift how many DFUs are "
         "eligible for automation (the 'low' band) and isolate the truly hard ones.", body)
    E.append(Image(figs["bands"], width=6.6 * inch, height=3.1 * inch))
    gap(6)
    E.append(Image(figs["hist"], width=5.6 * inch, height=3.15 * inch))

    # band table
    gap(8)
    tdata = [["Band", "Original", "Revised"]]
    for b in ["low", "medium", "high", "manual_review"]:
        o = int(stats["orig_counts"].get(b, 0)) if b != "manual_review" else "—"
        r = int(stats["rev_counts"].get(b, 0))
        tdata.append([b, o, r])
    t = Table(tdata, colWidths=[1.8 * inch, 1.4 * inch, 1.4 * inch])
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2b6cb0")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
    ]))
    E.append(t)

    E.append(PageBreak())
    para("Recommendations & open questions", H2)
    recs = [
        "Drop ZeroShare from the weighted sum (keep ADI); report ZeroShare for transparency.",
        "Use p95/median for spike intensity; widen the CV² ramp to 0.25→1.0.",
        "Compare MTD-to-MTD on equal elapsed days, not partial-vs-full-month.",
        "Add the seasonality guard so predictable seasonal SKUs stay automatable.",
        "Route &lt;3-non-zero-month DFUs to manual review (low confidence ≠ low volatility).",
        "Backtest-calibrate the 0.33/0.66 cut points against realized forecast error per decile.",
        "Treat 'confidence' (history length + signal size) as a second gate alongside volatility.",
    ]
    for i, r in enumerate(recs, 1):
        para(f"{i}. {r}", body)
    gap(6)
    para("Open questions for the team", H3)
    qs = [
        "Lookback window for the score — exactly 12 months, or trailing 24–36 for stability?",
        "DFU grain: is site+warehouse+item correct, or should customer be included/excluded?",
        "Should medium-risk auto-adjust with damping, or always go to a planner?",
        "Final weights: keep 0.40/0.35/0.25, or fit them to minimize backtested error?",
    ]
    for q in qs:
        para(f"• {q}", body)
    gap(8)
    para("Generated from analysis_volatility/build_report.py · volatility_score.py is the "
         "reusable scoring module.", small)

    doc.build(E)
    return out


def main():
    series = load_monthly_series()
    df = score_all(series)

    fig_red, adi_zero_corr = fig_redundancy(df)
    fig_cm, cmat = fig_corr_matrix(df)
    fig_bd, orig_counts, rev_counts = fig_bands(df)
    fig_h = fig_score_hist(df)

    stats = {
        "n_dfu": len(df),
        "n_months": int(df["n_periods"].max()),
        "n_scorable": int(df["scorable"].sum()),
        "n_manual": int((df["risk_band"] == "manual_review").sum()),
        "adi_zero_corr": adi_zero_corr,
        "cv2_spike_corr": cmat.loc["cv2", "spike_ratio"],
        "orig_counts": orig_counts,
        "rev_counts": rev_counts,
    }
    figs = {"redundancy": fig_red, "corr": fig_cm, "bands": fig_bd, "hist": fig_h}
    out = build_pdf(df, figs, stats)

    # also drop a CSV of scored DFUs for inspection
    df.to_csv(os.path.join(HERE, "scored_dfus_sample.csv"), index=False)
    for f in figs.values():
        try: os.remove(f)
        except OSError: pass
    print(f"\nPDF: {out}")
    print(f"ADI~ZeroShare Spearman: {adi_zero_corr:.3f}  "
          f"CV2~Spike: {stats['cv2_spike_corr']:.2f}")
    print(f"Bands revised: {dict(rev_counts)}")


if __name__ == "__main__":
    main()
