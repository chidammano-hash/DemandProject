/**
 * Tests (U2.3 / U2.4) for KPI delta color/arrow semantics.
 *
 * U2.3: color must reflect whether the metric moved in its *good* direction,
 *       not merely the sign. A rising OOS/Lost-Sales metric (goodDirection
 *       "down") must render red, not green.
 * U2.4: a near-zero delta must render flat (neutral, no up/down arrow).
 */
import { describe, it, expect } from "vitest";
import { deltaPresentation } from "@/tabs/customer-analytics/KpiSummaryCards";

describe("deltaPresentation (U2.3 inverted-direction metrics)", () => {
  it("positive delta on a good='down' metric is red with a down-meaning bad arrow", () => {
    const p = deltaPresentation(42.9, "down");
    expect(p.color).toBe("text-red-600");
    expect(p.arrow).toBe("↑");
    expect(p.flat).toBe(false);
  });

  it("positive delta on a good='up' metric is green", () => {
    const p = deltaPresentation(5.0, "up");
    expect(p.color).toBe("text-green-600");
    expect(p.arrow).toBe("↑");
  });

  it("negative delta on a good='down' metric is green (improvement)", () => {
    const p = deltaPresentation(-3.0, "down");
    expect(p.color).toBe("text-green-600");
    expect(p.arrow).toBe("↓");
  });

  it("negative delta on a good='up' metric is red", () => {
    const p = deltaPresentation(-0.3, "up");
    expect(p.color).toBe("text-red-600");
    expect(p.arrow).toBe("↓");
  });
});

describe("deltaPresentation (U2.4 zero / near-zero delta is flat)", () => {
  it("exact-zero delta renders neutral with no directional arrow", () => {
    const p = deltaPresentation(0, "up");
    expect(p.flat).toBe(true);
    expect(p.color).toBe("text-muted-foreground");
    expect(p.arrow).not.toBe("↑");
    expect(p.arrow).not.toBe("↓");
  });

  it("a delta below the flat threshold is neutral regardless of direction", () => {
    const p = deltaPresentation(0.04, "down");
    expect(p.flat).toBe(true);
    expect(p.color).toBe("text-muted-foreground");
  });
});
