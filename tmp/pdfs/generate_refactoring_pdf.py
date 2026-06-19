#!/usr/bin/env python3
"""Generate the refactoring / simplification ideas PDF."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Flowable,
    HRFlowable,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "output/pdf/refactoring-simplification-ideas.pdf"

PAGE_W = letter[0]
MARGIN = 0.75 * inch
CONTENT_W = PAGE_W - 2 * MARGIN

NAVY = colors.HexColor("#1e3a5f")
SLATE = colors.HexColor("#334155")
MUTED = colors.HexColor("#64748b")
BORDER = colors.HexColor("#cbd5e1")
GREEN = colors.HexColor("#166534")


# ---------------------------------------------------------------------------
# Content (edit here; rendering logic stays below)
# ---------------------------------------------------------------------------

EXECUTIVE_SUMMARY = (
    "The codebase is well-structured at the domain level: clear router folders, "
    "centralized model_registry, make_pool in API tests, and shared_constants.yaml "
    "inheritance. Main debt is oversized monolith files, parallel legacy + new "
    "implementations (especially tuning), and mechanical rule drift in scripts."
)

STRENGTHS = [
    "Domain router layout under api/routers/{inventory, forecasting, operations, platform, intelligence, core}/",
    "Reference split patterns: forecasting/tuning/ (13 sub-routers), customer_analytics/ (6 sub-routers)",
    "Test infrastructure: make_pool in tests/api/conftest.py",
    "ML centralization: tree estimators only in common/ml/model_registry.py",
    "Config inheritance via shared_constants.yaml _includes",
    "Frontend queries use fetchJson; zero : any in queries/",
    "Planning date centralized; no backward-compat config shims",
]

@dataclass(frozen=True)
class PriorityItem:
    title: str
    evidence: str
    action: str


P0_ITEMS = [
    PriorityItem(
        "Split inv_planning_insights.py",
        "1,807 LoC, 11 endpoints - largest router",
        "Create api/routers/inventory/insights/ subpackage",
    ),
    PriorityItem(
        "Migrate query modules to fetchJson",
        "8 modules still use raw fetch()",
        "Route through queries/core.ts fetchJson",
    ),
    PriorityItem(
        "Extend raw-fetch guard",
        "no-raw-fetch.test.ts watches only 4 files",
        "Add 4 model-tuning tab files to WATCHED list",
    ),
    PriorityItem(
        "PROJECT_ROOT migration in scripts",
        "~60 scripts use Path(__file__).parents[N]",
        "Use from common.core.paths import PROJECT_ROOT",
    ),
    PriorityItem(
        "Barrel-mount inventory routers",
        "18 separate include_router calls in main.py",
        "Aggregate like customer_analytics/__init__.py",
    ),
    PriorityItem(
        "Barrel-mount domain routers",
        "~80 manual imports in main.py (377 LoC)",
        "Group by domain __init__.py - no URL changes",
    ),
]

P1_ITEMS = [
    PriorityItem(
        "Retire triple tuning stack",
        "lgbm_tuning, model_tuning, forecasting/tuning/ in parallel",
        "Deprecate legacy routers; UI uses unified-model-tuning.ts",
    ),
    PriorityItem(
        "Split routers over 800 LoC",
        "9 routers exceed limit (see reference table)",
        "Split by feature area; share experiment CRUD base",
    ),
    PriorityItem(
        "Frontend tab size violations",
        "8+ tabs/panels over 600 LoC",
        "Extract orchestration into existing subpanels",
    ),
    PriorityItem(
        "Deduplicate comparison UI",
        "6 comparison panels with shared layout",
        "Single parameterized ExperimentComparisonPanel",
    ),
    PriorityItem(
        "inv_planning prefix cleanup",
        "Full paths embedded in decorators",
        "Use APIRouter(prefix=...) per CLAUDE.md",
    ),
    PriorityItem(
        "Inline hex chart colors",
        "~170 hex literals across 40+ tab files",
        "Migrate to useChartColors() / useThemeContext()",
    ),
    PriorityItem(
        "Chunked fact-table reads",
        "Bare pd.read_sql in several scripts",
        "Use read_sql_chunked() for fact tables",
    ),
    PriorityItem(
        "Consolidate inv-planning queries",
        "14 separate frontend query files",
        "Merge into inv-planning/ subfolder with barrel export",
    ),
]

P2_ITEMS = [
    PriorityItem("Split backtest_framework.py", "1,776 LoC", "Extract sampling, embargo, persistence"),
    PriorityItem("Split job_state + job_registry", "3,162 LoC combined", "Clarify read vs write surfaces"),
    PriorityItem("Consolidate run_backtest_*.py", "10+ thin wrappers", "Single --algorithm=id CLI"),
    PriorityItem("Decompose domains.py", "813 LoC catch-all", "Migrate stable routes; shrink catch-all"),
    PriorityItem("Split api/core.py", "653 LoC", "Move domain query builders out"),
    PriorityItem("Split largest scripts", "run_backtest, generate_production_forecasts, load", "Stage-based modules"),
    PriorityItem("SHAP panel duplication", "ShapPanel vs DfuShapPanel", "Shared chart + thin wrappers"),
]

ANTIPATTERNS = [
    ("Routers >800 LoC", "9 files", "inv_planning_insights (1,807) is worst offender"),
    ("Tabs/panels >600 LoC", "8+ files", "EnhancedComparisonPanel (991) is worst offender"),
    ("date.today() in production", "1 file", "dashboard.py:52 - document as exception"),
    ("Path(__file__).parents", "~60 scripts", "Use common.core.paths.PROJECT_ROOT"),
    ("except Exception volume", "~70 files", "Heaviest in experiment routers"),
    ("_row_to_dict duplicates", "4 routers", "Use common/core/sql_helpers.py"),
    ("Raw fetch in tabs", "4 files", "model-tuning/ panels bypass fetchJson"),
]

OVER_UNDER = [
    ("3 parallel tuning API stacks", "scripts_base.py underused (~7/93 scripts)"),
    ("6 comparison panel components", "inv_planning_insights monolith (11 endpoints)"),
    ("job_state + job_registry (3,162 LoC)", "60 scripts reinvent ROOT paths"),
    ("14 inv-planning query modules", "no-raw-fetch guard covers 4 files only"),
]

LARGEST_FILES = [
    ("Router", "inv_planning_insights.py", "1,807"),
    ("Router", "champion_experiments.py", "1,272"),
    ("Router", "backtest_management.py", "1,061"),
    ("common/", "backtest_framework.py", "1,776"),
    ("common/", "job_state.py", "1,734"),
    ("Script", "run_backtest.py", "1,573"),
    ("Script", "generate_production_forecasts.py", "1,541"),
    ("Frontend", "EnhancedComparisonPanel.tsx", "991"),
]

QUICK_WINS = [
    ("Replace raw fetch() with fetchJson", "8 query modules - ~1 hour, no URL changes"),
    ("Extend no-raw-fetch.test.ts", "Cover 4 model-tuning tab files"),
    ("Split inv_planning_insights.py", "Highest LoC violation - split at endpoint boundaries"),
    ("Barrel inventory routers", "Remove 18 mount lines from main.py"),
    ("PROJECT_ROOT in top scripts", "run_backtest, load, generate_production_forecasts"),
    ("Template inv_planning prefix", "Prove pattern on one router, replicate to 17"),
    ("Deprecate legacy tuning routers", "New UI only through /model-tuning"),
    ("Extract action-feed endpoint", "Largest block inside insights router"),
    ("Migrate FeatureLabPanel colors", "Highest hex literal density (13)"),
    ("Parameterize tuning API tests", "Merge shared cases into test_unified_model_tuning"),
]

RECOMMENDED_START = (
    "Highest ROI: split inv_planning_insights.py and migrate query modules to fetchJson. "
    "Both are mechanical, do not change URLs, and address the most violated CLAUDE.md rules. "
    "Second wave: retire the triple tuning stack and barrel-mount routers in main.py."
)


# ---------------------------------------------------------------------------
# PDF builder
# ---------------------------------------------------------------------------

class RefactoringPdf:
    """Build a readable refactoring report PDF."""

    def __init__(self) -> None:
        base = getSampleStyleSheet()
        self.cell = ParagraphStyle(
            "Cell",
            parent=base["Normal"],
            fontSize=8.5,
            leading=11,
            alignment=TA_LEFT,
        )
        self.cell_header = ParagraphStyle(
            "CellHeader",
            parent=self.cell,
            fontName="Helvetica-Bold",
            textColor=colors.white,
        )
        self.title = ParagraphStyle(
            "Title",
            parent=base["Title"],
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0f172a"),
        )
        self.subtitle = ParagraphStyle(
            "Subtitle",
            parent=base["Normal"],
            fontSize=11,
            leading=14,
            textColor=MUTED,
            spaceAfter=16,
        )
        self.h1 = ParagraphStyle(
            "H1",
            parent=base["Heading1"],
            fontSize=15,
            leading=18,
            spaceBefore=12,
            spaceAfter=8,
            textColor=NAVY,
        )
        self.h2 = ParagraphStyle(
            "H2",
            parent=base["Heading2"],
            fontSize=11,
            leading=14,
            spaceBefore=8,
            spaceAfter=4,
            textColor=SLATE,
        )
        self.body = ParagraphStyle(
            "Body",
            parent=base["BodyText"],
            fontSize=10,
            leading=14,
            spaceAfter=6,
        )
        self.bullet = ParagraphStyle(
            "Bullet",
            parent=self.body,
            leftIndent=12,
            spaceAfter=3,
        )
        self.good = ParagraphStyle(
            "Good",
            parent=self.bullet,
            textColor=GREEN,
        )

    def _p(self, text: str, header: bool = False) -> Paragraph:
        style = self.cell_header if header else self.cell
        return Paragraph(text.replace("\n", "<br/>"), style)

    def _table(self, headers: list[str], rows: list[list[str]], widths: list[float]) -> Table:
        data = [[self._p(h, header=True) for h in headers]]
        data += [[self._p(cell) for cell in row] for row in rows]
        tbl = Table(data, colWidths=widths, repeatRows=1)
        tbl.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), NAVY),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    ("GRID", (0, 0), (-1, -1), 0.25, BORDER),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                ]
            )
        )
        return tbl

    def _priority_table(self, items: list[PriorityItem]) -> Table:
        widths = [1.35 * inch, 2.05 * inch, 2.85 * inch]
        rows = [[i.title, i.evidence, i.action] for i in items]
        return self._table(["Opportunity", "Evidence", "Action"], rows, widths)

    def _bullets(self, items: list[str], *, positive: bool = False) -> list[Flowable]:
        style = self.good if positive else self.bullet
        return [Paragraph(f"&bull; {item}", style) for item in items]

    def _section_rule(self) -> list[Flowable]:
        return [Spacer(1, 0.08 * inch), HRFlowable(width="100%", thickness=0.5, color=BORDER)]

    def build(self) -> list[Flowable]:
        today = date.today().strftime("%B %d, %Y")
        story: list[Flowable] = []

        # Cover
        story.append(Paragraph("Simplification &amp; Refactoring Ideas", self.title))
        story.append(Paragraph(f"Supply Chain Command Center &mdash; {today}", self.subtitle))
        story.extend(self._section_rule())

        story.append(Paragraph("Executive Summary", self.h1))
        story.append(Paragraph(EXECUTIVE_SUMMARY, self.body))

        story.append(Paragraph("What Is Already Good", self.h2))
        story.extend(self._bullets(STRENGTHS, positive=True))

        story.append(PageBreak())

        # Priorities
        story.append(Paragraph("P0 - High Impact, Low Risk", self.h1))
        story.append(self._priority_table(P0_ITEMS))
        story.append(Spacer(1, 0.15 * inch))

        story.append(Paragraph("P1 - Medium Effort, Good Payoff", self.h1))
        story.append(self._priority_table(P1_ITEMS))
        story.append(PageBreak())

        story.append(Paragraph("P2 - Larger Refactors", self.h1))
        story.append(self._priority_table(P2_ITEMS))
        story.append(Spacer(1, 0.15 * inch))

        # Anti-patterns
        story.append(Paragraph("Anti-Patterns Found", self.h1))
        story.append(
            self._table(
                ["Rule", "Scope", "Notes"],
                ANTIPATTERNS,
                [1.4 * inch, 1.1 * inch, CONTENT_W - 2.5 * inch],
            )
        )

        story.append(PageBreak())

        # Balance + reference
        story.append(Paragraph("Over- vs Under-Engineering", self.h1))
        story.append(
            self._table(
                ["Over-Engineered", "Under-Engineered"],
                OVER_UNDER,
                [CONTENT_W / 2, CONTENT_W / 2],
            )
        )
        story.append(Spacer(1, 0.12 * inch))

        story.append(Paragraph("Largest Files", self.h1))
        story.append(
            self._table(
                ["Layer", "File", "Lines"],
                LARGEST_FILES,
                [0.9 * inch, CONTENT_W - 1.5 * inch, 0.6 * inch],
            )
        )

        story.append(PageBreak())

        # Quick wins
        story.append(Paragraph("Top 10 Quick Wins", self.h1))
        for idx, (title, detail) in enumerate(QUICK_WINS, start=1):
            story.append(Paragraph(f"{idx}. {title}", self.h2))
            story.append(Paragraph(detail, self.body))

        story.extend(self._section_rule())
        story.append(Paragraph("Recommended Starting Point", self.h1))
        story.append(Paragraph(RECOMMENDED_START, self.body))

        return story


def _footer(canvas, doc) -> None:  # noqa: ANN001
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MUTED)
    canvas.drawString(MARGIN, 0.5 * inch, "Supply Chain Command Center - Refactoring Ideas")
    canvas.drawRightString(PAGE_W - MARGIN, 0.5 * inch, f"Page {doc.page}")
    canvas.restoreState()


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=letter,
        leftMargin=MARGIN,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
        title="Simplification and Refactoring Ideas",
        author="Supply Chain Command Center",
    )
    doc.build(RefactoringPdf().build(), onFirstPage=_footer, onLaterPages=_footer)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
