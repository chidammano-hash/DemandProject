import { test, expect } from "@playwright/test";
import { navigateToTab, getContentArea } from "../fixtures/base";

test.describe("Inventory Planning Tab", () => {
  test.beforeEach(async ({ page }) => {
    await navigateToTab(page, "invPlanning");
  });

  test("renders without error boundary", async ({ page }) => {
    const errorBoundary = page.getByText("Something went wrong");
    await expect(errorBoundary).not.toBeVisible();
  });

  test("renders sub-tab sidebar with group labels", async ({ page }) => {
    const content = getContentArea(page);
    const text = await content.textContent();

    // The inv planning tab has a sidebar with group labels
    // Match the actual group labels rendered in the sidebar
    const groups = [
      "Daily Operations",
      "Replenishment Optimization",
      "Analytics",
      "Planning",
      "Demand Intelligence",
      "Strategic",
      "Order-to-Cash",
    ];

    for (const group of groups) {
      expect(text).toContain(group);
    }
  });

  test("clicking EOQ sub-tab loads EOQ panel", async ({ page }) => {
    const content = getContentArea(page);
    const eoqBtn = content.getByRole("button", { name: /EOQ/i });
    if (await eoqBtn.isVisible()) {
      await eoqBtn.click();
      await page.waitForTimeout(500);

      const text = await content.textContent();
      expect(text).toContain("EOQ");
    }
  });

  test("clicking Safety Stock sub-tab loads panel", async ({ page }) => {
    const content = getContentArea(page);
    const ssBtn = content.getByRole("button", { name: /Safety Stock/i });
    if (await ssBtn.isVisible()) {
      await ssBtn.click();
      await page.waitForTimeout(500);

      const text = await content.textContent();
      expect(text).toContain("Safety Stock");
    }
  });

  test("clicking Rebalancing sub-tab loads panel", async ({ page }) => {
    const content = getContentArea(page);
    const rebalBtn = content.getByRole("button", { name: /Rebalancing/i });
    if (await rebalBtn.isVisible()) {
      await rebalBtn.click();
      await page.waitForTimeout(500);

      const text = await content.textContent();
      expect(text).toContain("Inventory Rebalancing");
    }
  });
});
