import { test, expect } from "@playwright/test";
import { navigateToTab } from "../fixtures/base";

test.describe("Aggregate Analysis — Local Filter Bar", () => {
  test("visible on Aggregate Analysis tab", async ({ page }) => {
    await navigateToTab(page, "aggregateAnalysis");

    // Filter bar contains Brand, Category, Item, Location buttons
    for (const label of ["Brand", "Category", "Item", "Location"]) {
      const btn = page.getByRole("button", { name: label, exact: true });
      await expect(btn).toBeVisible();
    }
  });

  test("not visible on AI Planner tab", async ({ page }) => {
    await navigateToTab(page, "aiPlanner");

    const brandBtn = page.getByRole("button", { name: "Brand", exact: true });
    await expect(brandBtn).not.toBeVisible();
  });

  test("not visible on Jobs tab", async ({ page }) => {
    await navigateToTab(page, "jobs");

    const brandBtn = page.getByRole("button", { name: "Brand", exact: true });
    await expect(brandBtn).not.toBeVisible();
  });

  test("Brand dropdown opens on click", async ({ page }) => {
    await navigateToTab(page, "aggregateAnalysis");

    const brandBtn = page.getByRole("button", { name: "Brand", exact: true });
    await brandBtn.click();

    // Dropdown renders as a div with shadow-lg class containing brand value buttons
    const dropdown = page.locator(".shadow-lg");
    await expect(dropdown.first()).toBeVisible({ timeout: 5_000 });
  });
});
