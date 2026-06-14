import { describe, it, expect } from "vitest";
import {
  formatCompactCurrency,
  shouldRenderStat,
} from "../todaysPlanFormat";

describe("formatCompactCurrency (U8.1)", () => {
  it("renders sub-$10K values with one decimal so the banner matches the Action Feed", () => {
    // financial_at_risk = 3598.89 must read "$3.6K" (not "$4K") to match the
    // feed KPI which shows $3.6K — same metric, identical string.
    expect(formatCompactCurrency(3598.89)).toBe("$3.6K");
  });

  it("renders >=$10K values without spurious decimals", () => {
    expect(formatCompactCurrency(246723)).toBe("$247K");
  });

  it("returns the placeholder for null/zero", () => {
    expect(formatCompactCurrency(null)).toBe("--");
    expect(formatCompactCurrency(0)).toBe("--");
  });
});

describe("shouldRenderStat (U8.2)", () => {
  it("treats a zero total_skus as no-data when another stat is non-zero", () => {
    // The live briefing returns total_skus:0 while below_ss_count:3152, so a
    // literal "0 SKUs" alongside "3,152 at risk" is self-contradictory.
    expect(shouldRenderStat(0)).toBe(false);
  });

  it("renders a populated stat", () => {
    expect(shouldRenderStat(3152)).toBe(true);
  });

  it("treats null as no-data", () => {
    expect(shouldRenderStat(null)).toBe(false);
  });
});
