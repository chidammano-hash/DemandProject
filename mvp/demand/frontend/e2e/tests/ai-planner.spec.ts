import { test, expect } from "@playwright/test";
import { navigateToTab, getContentArea } from "../fixtures/base";

test.describe("AI Planner Tab", () => {
  test.beforeEach(async ({ page }) => {
    await navigateToTab(page, "aiPlanner");
  });

  test("renders without error boundary", async ({ page }) => {
    const errorBoundary = page.getByText("Something went wrong");
    await expect(errorBoundary).not.toBeVisible();
  });

  test("shows AI Planner content", async ({ page }) => {
    const content = getContentArea(page);
    await expect(content).toBeVisible();

    const text = await content.textContent();

    // Should show either insight cards, portfolio health, command center, or empty state
    const hasContent =
      text?.includes("Command Center") ||
      text?.includes("AI Planner") ||
      text?.includes("Insight") ||
      text?.includes("Portfolio") ||
      text?.includes("healthy") ||
      text?.includes("Generate") ||
      text?.includes("CRITICAL") ||
      text?.includes("Work Queue");
    expect(hasContent).toBeTruthy();
  });

  test("global filter bar is hidden", async ({ page }) => {
    const brandBtn = page.getByRole("button", { name: "Brand", exact: true });
    await expect(brandBtn).not.toBeVisible();
  });
});
