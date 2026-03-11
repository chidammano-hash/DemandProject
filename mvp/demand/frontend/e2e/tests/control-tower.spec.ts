import { test, expect } from "@playwright/test";
import { navigateToTab, getContentArea } from "../fixtures/base";

test.describe("Control Tower Tab", () => {
  test.beforeEach(async ({ page }) => {
    await navigateToTab(page, "controlTower");
  });

  test("renders without error boundary", async ({ page }) => {
    const errorBoundary = page.getByText("Something went wrong");
    await expect(errorBoundary).not.toBeVisible();
  });

  test("shows Control Tower content", async ({ page }) => {
    const content = getContentArea(page);
    await expect(content).toBeVisible();

    const text = await content.textContent();

    const hasContent =
      text?.includes("Control Tower") ||
      text?.includes("Alert") ||
      text?.includes("KPI") ||
      text?.includes("Critical") ||
      text?.includes("Health") ||
      text?.includes("Supply Chain");
    expect(hasContent).toBeTruthy();
  });
});
