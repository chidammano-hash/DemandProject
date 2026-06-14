import { describe, it, expect } from "vitest";
import { formatSliceCell } from "../SliceTablePanel";

/**
 * F4.4 — the cluster-assignment comparison table rendered raw negative
 * accuracy (`-12.89%`, `-128.04%`) for low-base buckets, inconsistent with the
 * accuracy heatmap which floors them to `<0%*` (F3.2). formatSliceCell must
 * apply the same flooring to the `accuracy_pct` column only — WAPE and bias
 * keep their raw (meaningful) values.
 */
describe("formatSliceCell (F4.4)", () => {
  it("floors negative accuracy_pct to <0%* (consistent with heatmap)", () => {
    expect(formatSliceCell("accuracy_pct", "pct", -12.89)).toBe("<0%*");
    expect(formatSliceCell("accuracy_pct", "pct", -128.04)).toBe("<0%*");
    expect(formatSliceCell("accuracy_pct", "pct", -0.1)).toBe("<0%*");
  });

  it("renders non-negative accuracy_pct normally", () => {
    expect(formatSliceCell("accuracy_pct", "pct", 73.9)).toBe("73.9%");
    expect(formatSliceCell("accuracy_pct", "pct", 0)).toBe("0.0%");
  });

  it("does NOT floor WAPE — a high WAPE is meaningful, not a bug", () => {
    expect(formatSliceCell("wape", "pct", 228.04)).toBe("228.04%");
    expect(formatSliceCell("wape", "pct", 112.89)).toBe("112.89%");
  });

  it("renders bias as a signed percent (negative allowed)", () => {
    expect(formatSliceCell("bias", "bias", -0.692)).toBe("-69.2%");
  });

  it("returns '-' for null/undefined", () => {
    expect(formatSliceCell("accuracy_pct", "pct", null)).toBe("-");
    expect(formatSliceCell("accuracy_pct", "pct", undefined)).toBe("-");
  });
});
