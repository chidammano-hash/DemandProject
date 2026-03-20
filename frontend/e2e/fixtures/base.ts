import { test as base, expect } from "@playwright/test";

/**
 * Shared fixtures for Supply Chain Command Center E2E tests.
 *
 * DOM structure reference:
 *  - Sidebar: <aside role="navigation"> containing <nav> with buttons
 *  - Content: <div id="tab-content" role="tabpanel">
 *  - Theme: <html class="light|dark" data-theme="general">
 *  - Nav buttons: <button title="Label"> (collapsed) or <button>...<span>Label</span></button> (expanded)
 */
export const test = base.extend<{ appPage: typeof base }>({
  // eslint-disable-next-line no-empty-pattern
  appPage: async ({ page }, use) => {
    await page.goto("/");
    await waitForAppReady(page);
    await use(page as never);
  },
});

export { expect };

/** Wait for the app shell to be ready (sidebar rendered) */
export async function waitForAppReady(page: import("@playwright/test").Page) {
  await page.waitForSelector('[role="navigation"]', { timeout: 15_000 });
}

/** Navigate to a specific tab via URL param and wait for ready */
export async function navigateToTab(
  page: import("@playwright/test").Page,
  tabKey: string
) {
  await page.goto(`/?tab=${tabKey}`);
  await waitForAppReady(page);
  // Allow lazy component + data fetch to settle
  await page.waitForTimeout(1000);
}

/** Get the main content area locator */
export function getContentArea(page: import("@playwright/test").Page) {
  return page.locator("#tab-content");
}

/**
 * Click a sidebar nav button by its label.
 * Works in both collapsed (title attr) and expanded (text label) modes.
 */
export async function clickNavItem(
  page: import("@playwright/test").Page,
  label: string
) {
  const btn = page.locator(
    `aside[role="navigation"] button:has-text("${label}"), aside[role="navigation"] button[title="${label}"]`
  ).first();
  await btn.click();
}

/** Sidebar nav item labels (in order — operations-first layout) */
export const SIDEBAR_LABELS = [
  "Command Center",
  "S&OP",
  "Jobs",
  "Data Quality",
  "Inv. Planning",
  "Clusters",
  "Inv. Backtest",
  "Portfolio",
  "Item Analysis",
  "FVA & ROI",
  "Customer Map",
  "Explorer",
] as const;

/** Tabs that show the global filter bar */
/** Tabs that previously showed the global filter bar — now filters are local to aggregateAnalysis */
export const FILTER_BAR_TABS = [
  "aggregateAnalysis",
  "itemAnalysis",
  "invPlanning",
  "explorer",
  "intel",
  "sop",
  "controlTower",
  "fva",
  "dataQuality",
] as const;

/** Tabs that hide the global filter bar */
/** All tabs that do NOT have a filter bar */
export const NO_FILTER_BAR_TABS = [
  "aiPlanner",
  "jobs",
  "chat",
  "clusters",
  "invBacktest",
  "exceptions",
  "storyboard",
  "itemAnalysis",
  "invPlanning",
  "explorer",
  "intel",
  "sop",
  "controlTower",
  "fva",
  "dataQuality",
] as const;
