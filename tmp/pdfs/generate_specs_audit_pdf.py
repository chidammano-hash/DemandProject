#!/usr/bin/env python3
"""Generate PDF for specs vs codebase audit report."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "output/pdf/specs-codebase-audit.pdf"


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "AuditTitle",
            parent=base["Title"],
            fontSize=22,
            leading=26,
            spaceAfter=6,
            textColor=colors.HexColor("#0f172a"),
        ),
        "subtitle": ParagraphStyle(
            "AuditSubtitle",
            parent=base["Normal"],
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#475569"),
            spaceAfter=18,
        ),
        "h1": ParagraphStyle(
            "AuditH1",
            parent=base["Heading1"],
            fontSize=15,
            leading=18,
            spaceBefore=14,
            spaceAfter=8,
            textColor=colors.HexColor("#1e3a5f"),
        ),
        "h2": ParagraphStyle(
            "AuditH2",
            parent=base["Heading2"],
            fontSize=12,
            leading=15,
            spaceBefore=10,
            spaceAfter=6,
            textColor=colors.HexColor("#334155"),
        ),
        "body": ParagraphStyle(
            "AuditBody",
            parent=base["BodyText"],
            fontSize=10,
            leading=14,
            spaceAfter=6,
            alignment=TA_LEFT,
        ),
        "bullet": ParagraphStyle(
            "AuditBullet",
            parent=base["BodyText"],
            fontSize=10,
            leading=14,
            leftIndent=14,
            bulletIndent=0,
            spaceAfter=4,
        ),
        "small": ParagraphStyle(
            "AuditSmall",
            parent=base["Normal"],
            fontSize=9,
            leading=12,
            textColor=colors.HexColor("#64748b"),
        ),
    }


def _table(data: list[list[str]], col_widths: list[float] | None = None) -> Table:
    tbl = Table(data, colWidths=col_widths, repeatRows=1)
    tbl.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a5f")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8.5),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#cbd5e1")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return tbl


def _bullets(story: list, style: ParagraphStyle, items: list[str]) -> None:
    for item in items:
        story.append(Paragraph(f"&bull; {item}", style))


def build_story() -> list:
    s = _styles()
    story: list = []
    today = date.today().strftime("%B %d, %Y")

    story.append(Paragraph("Specs vs Codebase Audit", s["title"]))
    story.append(
        Paragraph(
            f"Supply Chain Command Center &mdash; Updated audit after doc cleanup<br/>"
            f"Generated {today}",
            s["subtitle"],
        )
    )
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1")))
    story.append(Spacer(1, 0.15 * inch))

    # Executive summary
    story.append(Paragraph("Executive Summary", s["h1"]))
    _bullets(
        story,
        s["bullet"],
        [
            "82 markdown files under docs/specs/ (81 feature specs + README.md), organized across 8 domains.",
            "Removed since prior audit: Gen-4 roadmap, PRD folder, and README cross-domain/PRD sections.",
            "No broken links to removed docs remain in the repository.",
            "Most numbered specs are Implemented with matching backend and UI surfaces.",
            "Primary gaps: README index orphans, stale path references, AI Champion (backend-only), partial FVA ladder and decision-ledger policy.",
        ],
    )

    story.append(Paragraph("What Was Removed", s["h2"]))
    story.append(
        _table(
            [
                ["Removed", "Impact"],
                ["docs/specs/10-gen4-roadmap.md", "No forward-looking backlog doc in specs"],
                ["docs/specs/PRD/*", "No draft/planned-work section in specs index"],
                ["README cross-domain + PRD sections", "Index now covers implemented features only (domains 01-08)"],
            ],
            [2.2 * inch, 4.3 * inch],
        )
    )
    story.append(Spacer(1, 0.12 * inch))

    # Spec cleanup
    story.append(Paragraph("Spec Cleanup Still Needed", s["h1"]))

    story.append(Paragraph("1. Orphan Specs (not in README index)", s["h2"]))
    story.append(
        _table(
            [
                ["Spec", "Status", "Notes"],
                ["02-forecasting/16-expert-system-backtest.md", "Implemented", "expsys_accuracy router"],
                ["02-forecasting/17-ext-ml-forecast-load.md", "Implemented", "External ML ETL"],
                ["02-forecasting/23-feature-selection-pipeline.md", "Implemented", "Collides with 23-lgbm-accuracy-tuning"],
                ["02-forecasting/24-candidate-forecast-promotion.md", "Implemented", "Candidate to production promotion"],
                ["04-inventory/12-service-level-unification.md", "Implemented", "common/core/service_levels.py"],
                ["06-ai-platform/05-decision-ledger-and-policy.md", "Partial", "Ledger yes; policy engine no"],
            ],
            [2.5 * inch, 1.1 * inch, 2.9 * inch],
        )
    )
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("2. Index and Numbering Issues", s["h2"]))
    _bullets(
        story,
        s["bullet"],
        [
            "Duplicate 23: 23-lgbm-accuracy-tuning (indexed) vs 23-feature-selection-pipeline (orphan).",
            "Forecasting index gaps: lists 15 then 18; skips 16, 17, 24.",
            "Demand intel gap: index 01, 03, 05-07 (no 04 file exists).",
            "Wrong spec ID: 07-customer-analytics.md header says Spec ID 03-04.",
            "Misplaced file: 01-foundation/02-sku-feature-engineering.md indexed under Demand Intelligence.",
        ],
    )

    story.append(Paragraph("3. Stale Content Inside Specs", s["h2"]))
    story.append(
        _table(
            [
                ["Spec", "Problem"],
                ["08-production-forecast.md", "Deleted production_forecast_config.yaml; wrong script path"],
                ["26-operational-reference.md", "Treats algorithm_config.yaml as active"],
                ["13-production-baseline-seeding.md", "load_config('model_competition') example"],
                ["05-champion-experimentation-studio.md", "model_competition_config source"],
                ["01-infrastructure.md", "Says 5 champion strategies (spec 07 says 8); Chatbot + Market Intel label"],
                ["07-customer-analytics.md", "Status still 'Implementation'"],
            ],
            [2.6 * inch, 3.9 * inch],
        )
    )
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("4. Status Label Mismatches", s["h2"]))
    story.append(
        _table(
            [
                ["Spec", "Says", "Reality"],
                ["08-integration/07-fva.md", "Implemented", "Core tab yes; ai_adjusted / planner_adjusted planned"],
                ["02-forecasting/27-ai-champion-forecast.md", "Implemented", "Backend + job + SQL yes; no frontend"],
                ["06-ai-platform/05-decision-ledger-and-policy.md", "Partial", "Policy engine removed as unwired"],
            ],
            [2.0 * inch, 1.3 * inch, 3.2 * inch],
        )
    )

    story.append(PageBreak())

    # Feature matrix
    story.append(Paragraph("Feature Matrix", s["h1"]))
    story.append(Paragraph("Legend: Implemented = full stack; Partial = scaffold or backend-only; Not built = no code", s["small"]))
    story.append(Spacer(1, 0.08 * inch))

    story.append(
        _table(
            [
                ["Domain", "Specs", "Status", "Notes"],
                ["01 Foundation", "9", "Implemented", "Data layer, DQ, planning date, profiling, customer demand"],
                ["02 Forecasting", "29", "Mostly Implemented", "AI Champion (27) is backend-only"],
                ["03 Demand Intelligence", "5", "Implemented", "Clusters tab URL-only"],
                ["04 Inventory", "12", "Implemented", "Includes orphan 12-service-level-unification"],
                ["05 Operations", "4", "Implemented", "S&OP, finance, events, scenarios"],
                ["06 AI Platform", "5", "4 Implemented, 1 Partial", "Decision ledger partial"],
                ["07 User Experience", "8", "Implemented", "08-sku-validation-plan is QA playbook"],
                ["08 Integration", "10", "9 Implemented, 1 Partial", "FVA upper ladder stages planned"],
                ["Removed from specs", "-", "Not tracked", "Gen-4 roadmap and PRD removed"],
            ],
            [1.35 * inch, 0.65 * inch, 1.35 * inch, 3.15 * inch],
        )
    )
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("02 Forecasting Detail", s["h2"]))
    _bullets(
        story,
        s["bullet"],
        [
            "Core pipeline (accuracy, backtest, trees, champion, production, CI, bias, tuning studio): Implemented.",
            "Orphans 16, 17, 23b feature selection, 24 candidate promotion: code exists.",
            "Workflow docs 25, 26 and diagram 22: operational reference.",
            "AI Champion (27): API, job, SQL complete; no frontend tab or query module.",
        ],
    )

    story.append(Paragraph("06 AI Platform Detail", s["h2"]))
    story.append(
        _table(
            [
                ["Feature", "Status", "Surface"],
                ["AI Planning Agent", "Implemented", "URL /ai-planner, not sidebar"],
                ["Market Intel", "Implemented", "URL-only"],
                ["Control Tower", "Implemented", "CommandCenterTab"],
                ["Storyboard", "Implemented", "URL-only"],
                ["Decision ledger + policy", "Partial", "Ledger only; policy not wired"],
            ],
            [1.8 * inch, 1.2 * inch, 3.5 * inch],
        )
    )
    story.append(Spacer(1, 0.12 * inch))

    # Code without specs
    story.append(Paragraph("Code Without Dedicated Specs", s["h1"]))
    story.append(
        _table(
            [
                ["Feature", "Evidence"],
                ["Consensus plan + overrides", "consensus_plan.py, OverrideQueuePanel, /forecast/consensus-plan"],
                ["Integrated targets", "integrated_targets.py"],
                ["Working capital analytics", "working_capital.py, FinancialPlanPanel"],
                ["Algorithm comparison", "inv_planning_algorithm_comparison.py"],
                ["Inv planning insights (async)", "inv_planning_insights.py"],
                ["Explain API", "intelligence/explain.py"],
                ["Accuracy budget", "accuracy_budget.py"],
                ["Feature lab / sampled backtest / tuning chat", "Routers exist; only in tuning sub-specs"],
                ["PO / sourcing, config manager, integration chain", "Platform routers, no spec"],
            ],
            [2.2 * inch, 4.3 * inch],
        )
    )

    story.append(PageBreak())

    # Product suggestions
    story.append(Paragraph("Product Suggestions", s["h1"]))
    story.append(Paragraph("Prioritized from code/spec gaps (no roadmap doc in specs).", s["small"]))
    story.append(Spacer(1, 0.08 * inch))

    suggestions = [
        (
            "1. README index pass",
            "Add 6 orphan specs; renumber 23-feature-selection to 23b or 28; fix stale paths in "
            "08-production-forecast and 26-operational-reference. Low effort, high onboarding value.",
        ),
        (
            "2. AI Champion UI",
            "Biggest product gap: spec says Implemented but zero frontend usage. Surface run status, "
            "champion vs ai_champion comparison, and a generate action (Portfolio or FVA tab).",
        ),
        (
            "3. Override audit + planner-FVA",
            "Consensus overrides exist; FVA planner_adjusted stage is still planned. Connect overrides "
            "to decision ledger and measured ROI.",
        ),
        (
            "4. Lag-decomposed accuracy API",
            "Data already in backtest_lag_archive; no spec. High-value analytics with minimal new pipeline work.",
        ),
        (
            "5. Decision ledger policy engine",
            "Rebuild against real agent write sites before expanding AI autonomy "
            "(agent_autonomy.yaml exists; engine was removed as unwired).",
        ),
        (
            "6. Lightweight backlog doc (optional)",
            "If planned work should be tracked again, a single docs/specs/09-roadmap/README.md with "
            "P0/P1 bullets would fill the gap left by removing Gen-4.",
        ),
    ]
    for title, body in suggestions:
        story.append(Paragraph(title, s["h2"]))
        story.append(Paragraph(body, s["body"]))

    story.append(Spacer(1, 0.15 * inch))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#cbd5e1")))
    story.append(Spacer(1, 0.1 * inch))

    # Bottom line
    story.append(Paragraph("Bottom Line", s["h1"]))
    story.append(
        _table(
            [
                ["Metric", "Count"],
                ["Spec files (current)", "82 (down from ~86)"],
                ["Domains in index", "8"],
                ["Orphan specs (not indexed)", "6"],
                ["Stale path references", "~5 specs"],
                ["Backend-only vs spec", "AI Champion (primary)"],
                ["Draft PRD / roadmap docs", "Removed"],
            ],
            [3.5 * inch, 3.0 * inch],
        )
    )
    story.append(Spacer(1, 0.12 * inch))
    story.append(
        Paragraph(
            "The doc trim helped: specs are closer to what ships. Remaining work is mostly index hygiene, "
            "stale path fixes, and closing backend/UI gaps (AI Champion first).",
            s["body"],
        )
    )

    return story


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
        title="Specs vs Codebase Audit",
        author="Supply Chain Command Center",
    )

    def footer(canvas, doc_obj):  # noqa: ANN001
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#94a3b8"))
        canvas.drawString(0.75 * inch, 0.5 * inch, "Supply Chain Command Center - Specs vs Codebase Audit")
        canvas.drawRightString(7.75 * inch, 0.5 * inch, f"Page {doc_obj.page}")
        canvas.restoreState()

    doc.build(build_story(), onFirstPage=footer, onLaterPages=footer)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
