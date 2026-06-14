import { describe, it, expect } from "vitest";
import { formatSeriesTitle } from "../WorkbenchPanel";
import type { WorkbenchSeries } from "@/api/queries/demand-history";

function series(key: string, label: string): WorkbenchSeries {
  return { key, label, total_demand: 0, months: [] };
}

describe("formatSeriesTitle (U6.4 — duplicate item descriptions must be disambiguated)", () => {
  it("appends the key/id so two same-named items get distinct labels", () => {
    const a = formatSeriesTitle(series("105430", "TITOS HANDMADE VODKA 80"));
    const b = formatSeriesTitle(series("221877", "TITOS HANDMADE VODKA 80"));
    expect(a).not.toBe(b);
    expect(a).toContain("105430");
    expect(b).toContain("221877");
  });

  it("renders item||loc keys with a readable separator", () => {
    expect(formatSeriesTitle(series("105430||1401-BULK", "TITOS HANDMADE VODKA 80"))).toContain(
      "105430 - 1401-BULK",
    );
  });

  it("does not duplicate the key when label already equals the key", () => {
    expect(formatSeriesTitle(series("105430", "105430"))).toBe("105430");
  });
});
