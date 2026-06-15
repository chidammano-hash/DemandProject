import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

// U4.1 — the cycle-3 U3.1 dark-theme fix swapped Customer-Analytics off bare
// grays, but the same regression lived untouched in the Demand-History panels
// (Workbench is the default-visible view). 22 `text-gray-400/500` label/empty
// sites with no `dark:` companion render gray-on-near-black, well below WCAG AA.
// These guards read the source and fail on any bare light-mode-only gray label
// or bg-gray chip on the Demand-History surfaces.
const PANELS = [
  "src/tabs/demand-history/WorkbenchPanel.tsx",
  "src/tabs/demand-history/MatrixPanel.tsx",
  "src/tabs/demand-history/DecompositionPanel.tsx",
  "src/tabs/demand-history/ComparisonPanel.tsx",
].map((p) => resolve(process.cwd(), p));

// Matches text-gray-{400,500,600} not preceded by `dark:`.
const BARE_GRAY_TEXT = /(?<!dark:)\btext-gray-[456]00\b/g;
// Matches bg-gray-{100,200} not preceded by `dark:`.
const BARE_LIGHT_BG = /(?<!dark:)\bbg-gray-[12]00\b/g;

describe("Demand-History dark-mode legibility (U4.1)", () => {
  it.each(PANELS)("%s uses theme tokens, not bare grays", (path) => {
    const source = readFileSync(path, "utf8");
    expect(source.match(BARE_GRAY_TEXT) ?? []).toEqual([]);
    expect(source.match(BARE_LIGHT_BG) ?? []).toEqual([]);
  });
});
