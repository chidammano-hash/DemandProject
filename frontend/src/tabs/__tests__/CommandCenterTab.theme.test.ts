import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

// U1.2: CommandCenterTab must NOT hardcode hex chart colors. CLAUDE.md forbids
// inline hex in tabs/ — chart series colors must come from useChartColors()/
// useThemeContext() so they adapt to Light/Soft/Dark themes. This guard reads
// the source and fails on any 6-digit hex literal (e.g. the Portfolio Trend
// `stroke="#3b82f6"` / `stroke="#10b981"` lines).
// vitest runs from the frontend/ root, so resolve relative to cwd.
const SRC = resolve(process.cwd(), "src/tabs/CommandCenterTab.tsx");

describe("CommandCenterTab theme compliance (U1.2)", () => {
  it("contains no inline 6-digit hex chart-color literals", () => {
    const source = readFileSync(SRC, "utf8");
    const matches = source.match(/#[0-9a-fA-F]{6}\b/g) ?? [];
    expect(matches).toEqual([]);
  });

  it("reads chart line colors from useChartColors", () => {
    const source = readFileSync(SRC, "utf8");
    expect(source).toContain("useChartColors");
  });
});
