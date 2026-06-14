import { describe, it, expect } from "vitest";
import { formatHeatmapAccuracy } from "../aggregateShared";

/**
 * F3.2 / U3.6 — the accuracy heatmap rendered unbounded negative accuracy
 * (BEER -263.9%) which reads as a bug rather than "WAPE > 100% on a tiny base".
 * formatHeatmapAccuracy floors the *displayed* value at 0% and marks it so a
 * planner can distinguish a low-base artifact from a healthy number.
 */
describe("formatHeatmapAccuracy (F3.2 / U3.6)", () => {
  it("renders healthy accuracy verbatim with one decimal", () => {
    expect(formatHeatmapAccuracy(73.9)).toBe("73.9%");
    expect(formatHeatmapAccuracy(100)).toBe("100.0%");
    expect(formatHeatmapAccuracy(0)).toBe("0.0%");
  });

  it("floors negative accuracy to <0% with a low-base marker", () => {
    expect(formatHeatmapAccuracy(-263.9)).toBe("<0%*");
    expect(formatHeatmapAccuracy(-12.89)).toBe("<0%*");
    expect(formatHeatmapAccuracy(-0.1)).toBe("<0%*");
  });
});
