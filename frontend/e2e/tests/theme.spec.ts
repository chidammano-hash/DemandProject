import { test, expect } from "@playwright/test";
import { waitForAppReady } from "../fixtures/base";

test.describe("Theme Toggle", () => {
  test("app starts in light mode by default", async ({ page }) => {
    await page.addInitScript(() => localStorage.removeItem("ds-color-mode"));
    await page.goto("/");
    await waitForAppReady(page);

    const html = page.locator("html");
    await expect(html).toHaveClass(/light/);
  });

  test("pressing d cycles through color modes to dark", async ({ page }) => {
    // Start in light mode
    await page.addInitScript(() =>
      localStorage.setItem("ds-color-mode", "light")
    );
    await page.goto("/");
    await waitForAppReady(page);

    const html = page.locator("html");
    await expect(html).toHaveClass(/light/);

    // Focus on body (not an input) so keyboard shortcut works
    await page.locator("body").click({ position: { x: 400, y: 300 } });

    // d cycles: light → soft → dark (soft still uses CSS class "light")
    await page.keyboard.press("d"); // light → soft
    await page.keyboard.press("d"); // soft → dark
    await expect(html).toHaveClass(/dark/);
  });

  test("theme persists across page reload", async ({ page }) => {
    await page.addInitScript(() =>
      localStorage.setItem("ds-color-mode", "dark")
    );
    await page.goto("/");
    await waitForAppReady(page);

    const html = page.locator("html");
    await expect(html).toHaveClass(/dark/);
  });
});
