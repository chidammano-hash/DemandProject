import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

// U2.3: the Customer-Analytics filter dropdown, Clear button and the Customer
// Ranking sticky header used bare `bg-white` / `bg-gray-*` with no `dark:`
// variant, so in Dark theme they render opaque light panels (white-on-white
// text risk). Surfaces must use theme tokens (bg-popover / bg-card / bg-muted /
// bg-accent) that adapt to Light/Soft/Dark. These guards read the source and
// fail on any bare light-mode-only background class on those surfaces.
const TAB = resolve(process.cwd(), "src/tabs/CustomerAnalyticsTab.tsx");
const RANKING = resolve(process.cwd(), "src/tabs/customer-analytics/CustomerRanking.tsx");
const KPI_CARDS = resolve(process.cwd(), "src/tabs/customer-analytics/KpiSummaryCards.tsx");

// U3.1 — the metric/grain/group-by toggle pills inside the six CA chart panels
// were left on bare `bg-gray-100 text-gray-600` with no `dark:` variant, so in
// Dark theme the inactive pills render gray-on-gray (illegible). They must use
// the same theme tokens the ranking header already uses.
const CHART_PANELS = [
  "src/tabs/customer-analytics/CustomerDemandMap.tsx",
  "src/tabs/customer-analytics/CustomerHeatmap.tsx",
  "src/tabs/customer-analytics/ChannelSunburst.tsx",
  "src/tabs/customer-analytics/OosImpactBubble.tsx",
  "src/tabs/customer-analytics/SegmentSparklines.tsx",
].map((p) => resolve(process.cwd(), p));

// Matches text-gray-{400,500,600,...} not preceded by `dark:`.
const BARE_GRAY_TEXT = /(?<!dark:)\btext-gray-\d{2,3}\b/g;

// Matches `bg-white` / `bg-gray-100` / `bg-gray-200` only when NOT immediately
// preceded by `dark:` (a dark-variant override is acceptable).
const BARE_LIGHT_BG = /(?<!dark:)\bbg-(?:white|gray-\d{2,3})\b/g;

describe("CustomerAnalytics dark-mode legibility (U2.3)", () => {
  it("CustomerAnalyticsTab has no bare bg-white/bg-gray-* surface class", () => {
    const source = readFileSync(TAB, "utf8");
    const matches = source.match(BARE_LIGHT_BG) ?? [];
    expect(matches).toEqual([]);
  });

  it("CustomerRanking sticky header has no bare bg-white surface class", () => {
    const source = readFileSync(RANKING, "utf8");
    const matches = source.match(BARE_LIGHT_BG) ?? [];
    expect(matches).toEqual([]);
  });

  // U2.4 — the KPI loading skeleton used hardcoded bg-gray-200, near-invisible on
  // a dark card. It must use a theme-aware token (bg-muted) instead.
  it("KpiSummaryCards skeleton uses a theme token, not bg-gray-200", () => {
    const source = readFileSync(KPI_CARDS, "utf8");
    const matches = source.match(BARE_LIGHT_BG) ?? [];
    expect(matches).toEqual([]);
  });

  // U3.1 — chart-panel toggle pills + the tab search clear-× must not use bare
  // light-only grays for the inactive/idle state.
  it.each(CHART_PANELS)("%s toggle pills use theme tokens, not bare grays", (path) => {
    const source = readFileSync(path, "utf8");
    expect(source.match(BARE_LIGHT_BG) ?? []).toEqual([]);
    expect(source.match(BARE_GRAY_TEXT) ?? []).toEqual([]);
  });

  it("CustomerAnalyticsTab search clear-× has no bare text-gray-* class", () => {
    const source = readFileSync(TAB, "utf8");
    expect(source.match(BARE_GRAY_TEXT) ?? []).toEqual([]);
  });
});
