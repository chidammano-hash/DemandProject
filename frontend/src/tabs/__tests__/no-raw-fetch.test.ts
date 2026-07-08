import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const __dirname = dirname(fileURLToPath(import.meta.url));
const TABS_DIR = resolve(__dirname, "..");

/**
 * U3.1 guard — CLAUDE.md: "All HTTP from frontend goes through
 * src/api/queries/<module>.ts using fetchJson. Never raw fetch( in
 * tabs/components." Raw fetch bypasses the error-sanitization layer
 * (status attach + FastAPI detail parse) shipped in cycle 2, so a 404/500
 * either silently swallows or surfaces a raw error.
 *
 * This guard fails if a watched tab file calls the global `fetch(` directly
 * (it ignores fetchJson / prefetch / refetch, which legitimately end in
 * "fetch(").
 */
// U6.10 — extend the guard to the two lowest-risk single-GET offenders called
// out this cycle (a profile GET and a core-features GET), now migrated to
// fetchJson query fetchers.
const WATCHED = [
  "ItemAnalysisTab.tsx",
  "SopTab.tsx",
  "lgbm-tuning/FeatureLabPanel.tsx",
  "clusters/ClusterExperimentBuilder.tsx",
  "model-tuning/ExperimentBuilder.tsx",
  "model-tuning/LogViewer.tsx",
  "model-tuning/EnhancedComparisonPanel.tsx",
  "model-tuning/EnhancedPromoteModal.tsx",
];

// Matches a bare `fetch(` that is NOT preceded by an identifier char or `.`
// (so fetchJson, prefetch, refetch, queryClient.fetch... are allowed).
const RAW_FETCH = /(^|[^.\w])fetch\(/;

describe("U3.1 — tabs must not call raw fetch()", () => {
  for (const file of WATCHED) {
    it(`${file} uses fetchJson query modules, not raw fetch()`, () => {
      const src = readFileSync(resolve(TABS_DIR, file), "utf8");
      const offending = src
        .split("\n")
        .map((line, i) => ({ line: line.trim(), n: i + 1 }))
        .filter(({ line }) => RAW_FETCH.test(line));
      expect(
        offending,
        `raw fetch( found in ${file}: ${JSON.stringify(offending)}`,
      ).toEqual([]);
    });
  }
});
