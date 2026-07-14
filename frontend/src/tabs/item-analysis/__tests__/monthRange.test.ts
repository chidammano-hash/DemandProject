import { describe, it, expect } from "vitest";
import {
  clampFutureMonths,
  formatMonthLabel,
  isToDisabled,
  isFromDisabled,
} from "../monthRange";

describe("formatMonthLabel (U2.20)", () => {
  it("renders a YYYY-MM-01 string as 'Mon YYYY'", () => {
    expect(formatMonthLabel("2023-04-01")).toBe("Apr 2023");
    expect(formatMonthLabel("2026-03-01")).toBe("Mar 2026");
    expect(formatMonthLabel("2028-12-01")).toBe("Dec 2028");
  });

  it("falls back to the raw string for an unparseable value", () => {
    expect(formatMonthLabel("not-a-date")).toBe("not-a-date");
    expect(formatMonthLabel("")).toBe("");
  });
});

describe("inverted-range guards (U2.20)", () => {
  const months = ["2023-04-01", "2023-05-01", "2023-06-01", "2023-07-01"];

  it("disables TO options earlier than the selected FROM", () => {
    const from = "2023-06-01";
    expect(isToDisabled("2023-04-01", from)).toBe(true);
    expect(isToDisabled("2023-05-01", from)).toBe(true);
    expect(isToDisabled("2023-06-01", from)).toBe(false); // equal is allowed
    expect(isToDisabled("2023-07-01", from)).toBe(false);
  });

  it("disables FROM options later than the selected TO", () => {
    const to = "2023-05-01";
    expect(isFromDisabled("2023-04-01", to)).toBe(false);
    expect(isFromDisabled("2023-05-01", to)).toBe(false); // equal allowed
    expect(isFromDisabled("2023-06-01", to)).toBe(true);
    expect(isFromDisabled("2023-07-01", to)).toBe(true);
  });

  it("never disables anything when the opposing bound is empty (All)", () => {
    for (const m of months) {
      expect(isToDisabled(m, "")).toBe(false);
      expect(isFromDisabled(m, "")).toBe(false);
    }
  });
});

describe("clampFutureMonths", () => {
  const future = ["2026-07-01", "2026-08-01", "2027-06-01", "2028-06-01"];

  it("returns all future months when no TO bound is set", () => {
    expect(clampFutureMonths(future, "")).toEqual(future);
  });

  it("drops forecast months beyond an explicit TO bound", () => {
    expect(clampFutureMonths(future, "2026-08-01")).toEqual(["2026-07-01", "2026-08-01"]);
  });

  it("drops every future month when TO ends inside history", () => {
    expect(clampFutureMonths(future, "2026-06-01")).toEqual([]);
  });
});
