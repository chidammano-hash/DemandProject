import { describe, it, expect } from "vitest";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

/**
 * U3.10 — guard against re-introducing the retired tab components.
 *
 * `ControlTowerTab` and `AIPlannerTab` were consolidated into `CommandCenterTab`
 * (TAB_REDIRECTS in useUrlState.ts routes the old `controlTower`/`aiPlanner` URL
 * keys to `commandCenter`). The dead components were imported nowhere and never
 * rendered — ~1,000 LoC of unreachable UI plus their test suites carried in the
 * bundle/refactor surface, prone to drift (theme/colour fixes applied to the live
 * CommandCenterTab but not the dead twins).
 *
 * This test fails if either retired tab module reappears under src/tabs, so the
 * removal cannot silently regress.
 */
const tabsDir = resolve(dirname(fileURLToPath(import.meta.url)), "..");

describe("retired tab components stay deleted (U3.10)", () => {
  it("ControlTowerTab.tsx is not present in src/tabs", () => {
    expect(existsSync(resolve(tabsDir, "ControlTowerTab.tsx"))).toBe(false);
  });

  it("AIPlannerTab.tsx is not present in src/tabs", () => {
    expect(existsSync(resolve(tabsDir, "AIPlannerTab.tsx"))).toBe(false);
  });
});
