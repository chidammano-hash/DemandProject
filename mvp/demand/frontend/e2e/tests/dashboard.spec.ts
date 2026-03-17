import { test, expect } from "@playwright/test";
import { navigateToTab, getContentArea } from "../fixtures/base";

test.describe("Aggregate Analysis Tab", () => {
  test.beforeEach(async ({ page }) => {
    await navigateToTab(page, "aggregateAnalysis");
  });

  test("content area renders", async ({ page }) => {
    const content = getContentArea(page);
    await expect(content).toBeVisible();
  });

  test("renders without error boundary", async ({ page }) => {
    const errorBoundary = page.getByText("Something went wrong");
    await expect(errorBoundary).not.toBeVisible();
  });

  test("shows content (not blank)", async ({ page }) => {
    const content = getContentArea(page);
    await expect(content).toBeVisible();

    const text = await content.textContent();
    expect(text?.length).toBeGreaterThan(0);
  });
});
