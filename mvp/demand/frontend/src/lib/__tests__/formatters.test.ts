import { describe, it, expect } from "vitest";
import {
  formatCompactNumber,
  formatNumber,
  formatPercent,
  formatCell,
  titleCase,
} from "@/lib/formatters";

describe("formatNumber", () => {
  it("formats an integer with comma separator", () => {
    expect(formatNumber(1000)).toBe("1,000");
  });

  it("formats a large number with commas", () => {
    expect(formatNumber(1234567)).toBe("1,234,567");
  });

  it("returns '-' for null", () => {
    expect(formatNumber(null)).toBe("-");
  });

  it("returns '-' for undefined", () => {
    expect(formatNumber(undefined)).toBe("-");
  });

  it("formats zero as '0'", () => {
    expect(formatNumber(0)).toBe("0");
  });

  it("formats a decimal number with up to 2 fraction digits", () => {
    const result = formatNumber(1234.567);
    // Intl.NumberFormat rounds to 2 decimal places
    expect(result).toBe("1,234.57");
  });

  it("formats negative numbers", () => {
    expect(formatNumber(-5000)).toBe("-5,000");
  });
});

describe("formatPercent", () => {
  it("formats a positive number with '%' suffix", () => {
    expect(formatPercent(95)).toBe("95%");
  });

  it("formats a decimal number appended with '%'", () => {
    // formatPercent uses numberFmt which has maximumFractionDigits:2
    expect(formatPercent(23.456)).toBe("23.46%");
  });

  it("returns '-' for null", () => {
    expect(formatPercent(null)).toBe("-");
  });

  it("returns '-' for undefined", () => {
    expect(formatPercent(undefined)).toBe("-");
  });

  it("returns '-' for Infinity", () => {
    expect(formatPercent(Infinity)).toBe("-");
  });

  it("returns '-' for NaN", () => {
    expect(formatPercent(NaN)).toBe("-");
  });

  it("formats zero as '0%'", () => {
    expect(formatPercent(0)).toBe("0%");
  });
});

describe("formatCompactNumber", () => {
  it("formats a number in compact notation", () => {
    // 1,000,000 → "1M" in compact notation
    expect(formatCompactNumber(1_000_000)).toBe("1M");
  });

  it("formats thousands in compact notation", () => {
    expect(formatCompactNumber(1000)).toBe("1K");
  });

  it("accepts a numeric string and parses it", () => {
    expect(formatCompactNumber("2000")).toBe("2K");
  });

  it("returns the string as-is for non-numeric strings", () => {
    expect(formatCompactNumber("abc")).toBe("abc");
  });

  it("formats zero", () => {
    expect(formatCompactNumber(0)).toBe("0");
  });
});

describe("formatCell", () => {
  it("returns '-' for null", () => {
    expect(formatCell(null)).toBe("-");
  });

  it("returns '-' for undefined", () => {
    expect(formatCell(undefined)).toBe("-");
  });

  it("returns '-' for empty string", () => {
    expect(formatCell("")).toBe("-");
  });

  it("formats numbers using formatNumber", () => {
    expect(formatCell(1000)).toBe("1,000");
  });

  it("converts non-numeric non-null values to string", () => {
    expect(formatCell("hello")).toBe("hello");
  });

  it("formats boolean false (non-null) as a string", () => {
    expect(formatCell(false)).toBe("false");
  });

  it("formats zero as '0'", () => {
    expect(formatCell(0)).toBe("0");
  });
});

describe("titleCase", () => {
  it("converts underscore-separated string to title case", () => {
    expect(titleCase("hello_world")).toBe("Hello World");
  });

  it("handles a single word without underscores", () => {
    expect(titleCase("hello")).toBe("Hello");
  });

  it("handles multiple underscores", () => {
    expect(titleCase("abc_def_ghi")).toBe("Abc Def Ghi");
  });

  it("filters out empty parts from leading/trailing underscores", () => {
    expect(titleCase("_hello_")).toBe("Hello");
  });

  it("returns empty string for empty input", () => {
    expect(titleCase("")).toBe("");
  });

  it("handles all caps correctly (only first letter uppercased)", () => {
    // The function uppercases only the first char and leaves the rest
    expect(titleCase("ABC_DEF")).toBe("ABC DEF");
  });
});
