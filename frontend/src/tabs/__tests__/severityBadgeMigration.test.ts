import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const TABS = resolve(__dirname, "..");

// U5.1 — source guard: the migrated status/health-pill surfaces must not carry a
// bare `bg-{color}-100` (or `text-{color}-800`) tint with NO `dark:` companion on
// the same line. Such Light-only chips render gray-on-near-black-illegible in Dark.
// Either route through `severityBadgeClass()` / `SEVERITY_CONFIG`, or hand-pair a
// `dark:` tint inline.
const MIGRATED_FILES = ["inv-planning/PortfolioHealthPanel.tsx", "SopTab.tsx"];

const BARE_TINT = /\bbg-(red|green|amber|orange|blue|yellow)-100\b/;

describe("U5.1 severity-badge dark-variant guard", () => {
  for (const rel of MIGRATED_FILES) {
    it(`${rel} has no bare bg-*-100 without a dark: sibling`, () => {
      const src = readFileSync(resolve(TABS, rel), "utf8");
      const offenders = src
        .split("\n")
        .map((line, i) => ({ line, n: i + 1 }))
        .filter(({ line }) => BARE_TINT.test(line) && !line.includes("dark:"));
      expect(
        offenders.map((o) => `${o.n}: ${o.line.trim()}`),
        `${rel} still has Light-only status chips`,
      ).toEqual([]);
    });
  }
});
