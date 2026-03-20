import { test, expect } from "@playwright/test";
import { navigateToTab, getContentArea } from "../fixtures/base";

test.describe("Aggregate Analysis — Accuracy Section", () => {
  test.beforeEach(async ({ page }) => {
    // accuracy redirects to aggregateAnalysis
    await navigateToTab(page, "aggregateAnalysis");
  });

  test("renders without error boundary", async ({ page }) => {
    const errorBoundary = page.getByText("Something went wrong");
    await expect(errorBoundary).not.toBeVisible();
  });

  test("content area is visible with data", async ({ page }) => {
    const content = getContentArea(page);
    await expect(content).toBeVisible();

    const text = await content.textContent();
    expect(text?.length).toBeGreaterThan(0);
  });

  test("shows accuracy-related content", async ({ page }) => {
    const content = getContentArea(page);
    const text = await content.textContent();
    // Should contain at least some accuracy-related text
    const hasAccuracyContent =
      text?.includes("WAPE") ||
      text?.includes("Accuracy") ||
      text?.includes("Bias") ||
      text?.includes("Portfolio Analysis") ||
      text?.includes("Cluster");
    expect(hasAccuracyContent).toBeTruthy();
  });
});
