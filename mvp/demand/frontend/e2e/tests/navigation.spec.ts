import { test, expect } from "@playwright/test";
import {
  SIDEBAR_LABELS,
  navigateToTab,
  waitForAppReady,
  clickNavItem,
} from "../fixtures/base";

test.describe("Sidebar Navigation", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForAppReady(page);
  });

  test("renders all sidebar nav buttons", async ({ page }) => {
    // Each nav item has a button with title=label (collapsed) or text span (expanded)
    for (const label of SIDEBAR_LABELS) {
      const btn = page.locator(
        `aside[role="navigation"] button[title="${label}"], aside[role="navigation"] button:has-text("${label}")`
      );
      await expect(btn.first()).toBeVisible();
    }
  });

  test("clicking Overview navigates to overview tab", async ({ page }) => {
    await clickNavItem(page, "Overview");
    await expect(page).toHaveURL(/tab=overview/);
  });

  test("clicking Accuracy navigates to accuracy tab", async ({ page }) => {
    await clickNavItem(page, "Accuracy");
    await expect(page).toHaveURL(/tab=accuracy/);
  });

  test("clicking Inv. Planning navigates to invPlanning tab", async ({
    page,
  }) => {
    await clickNavItem(page, "Inv. Planning");
    await expect(page).toHaveURL(/tab=invPlanning/);
  });

  test("clicking AI Planner navigates to aiPlanner tab", async ({ page }) => {
    await clickNavItem(page, "AI Planner");
    await expect(page).toHaveURL(/tab=aiPlanner/);
  });

  test("clicking FVA & ROI navigates to fva tab", async ({ page }) => {
    await clickNavItem(page, "FVA & ROI");
    await expect(page).toHaveURL(/tab=fva/);
  });

  test("clicking Data Quality navigates to dataQuality tab", async ({ page }) => {
    await clickNavItem(page, "Data Quality");
    await expect(page).toHaveURL(/tab=dataQuality/);
  });

  test("no error boundary shown on key tabs", async ({ page }) => {
    const tabs = [
      "overview",
      "accuracy",
      "itemAnalysis",
      "invPlanning",
      "aiPlanner",
      "controlTower",
      "jobs",
      "exceptions",
      "fva",
      "dataQuality",
    ];
    for (const tab of tabs) {
      await navigateToTab(page, tab);
      const errorBoundary = page.getByText("Something went wrong");
      await expect(errorBoundary).not.toBeVisible();
    }
  });
});

test.describe("Keyboard Shortcuts", () => {
  test.beforeEach(async ({ page }) => {
    await page.goto("/");
    await waitForAppReady(page);
    // Click content area to ensure keyboard events target the app
    await page.locator("#tab-content").click();
  });

  test("pressing 3 navigates to Overview", async ({ page }) => {
    await page.keyboard.press("3");
    await expect(page).toHaveURL(/tab=overview/);
  });

  test("pressing 4 navigates to Accuracy", async ({ page }) => {
    await page.keyboard.press("4");
    await expect(page).toHaveURL(/tab=accuracy/);
  });

  test("pressing 1 navigates to AI Planner", async ({ page }) => {
    await page.keyboard.press("1");
    await expect(page).toHaveURL(/tab=aiPlanner/);
  });
});

test.describe("URL State", () => {
  test("direct URL navigation loads correct tab", async ({ page }) => {
    await navigateToTab(page, "accuracy");
    // The Accuracy tab button should be highlighted as active (aria-current="page")
    const activeBtn = page.locator(
      'aside[role="navigation"] button[aria-current="page"]'
    );
    await expect(activeBtn).toBeVisible();
  });

  test("invalid tab falls back to overview", async ({ page }) => {
    await page.goto("/?tab=nonexistent");
    await waitForAppReady(page);
    await expect(page).toHaveURL(/tab=overview/);
  });
});
