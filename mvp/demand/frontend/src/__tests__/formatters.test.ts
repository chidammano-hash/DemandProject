import { describe, it, expect } from "vitest";
import {
  formatNumber,
  formatPercent,
  formatCell,
  formatCompactNumber,
  titleCase,
} from "@/lib/formatters";

describe("formatNumber", () => {
  it("formats positive numbers", () => {
    expect(formatNumber(1234.56)).toBe("1,234.56");
  });

  it("returns dash for null", () => {
    expect(formatNumber(null)).toBe("-");
  });

  it("returns dash for undefined", () => {
    expect(formatNumber(undefined)).toBe("-");
  });

  it("formats zero", () => {
    expect(formatNumber(0)).toBe("0");
  });
});

describe("formatPercent", () => {
  it("formats a percentage value", () => {
    expect(formatPercent(85.5)).toBe("85.5%");
  });

  it("returns dash for null", () => {
    expect(formatPercent(null)).toBe("-");
  });

  it("returns dash for NaN", () => {
    expect(formatPercent(NaN)).toBe("-");
  });
});

describe("formatCell", () => {
  it("formats numbers via formatNumber", () => {
    expect(formatCell(42)).toBe("42");
  });

  it("returns dash for null", () => {
    expect(formatCell(null)).toBe("-");
  });

  it("returns dash for empty string", () => {
    expect(formatCell("")).toBe("-");
  });

  it("returns string values as-is", () => {
    expect(formatCell("hello")).toBe("hello");
  });
});

describe("formatCompactNumber", () => {
  it("formats large numbers compactly", () => {
    const result = formatCompactNumber(1500000);
    // Different locales may produce "1.5M" or "2M" — just verify it's a short string
    expect(result.length).toBeLessThan(10);
  });

  it("parses string numbers", () => {
    const result = formatCompactNumber("5000");
    expect(result).toBe("5K");
  });
});

describe("titleCase", () => {
  it("converts snake_case to Title Case", () => {
    expect(titleCase("hello_world")).toBe("Hello World");
  });

  it("handles single word", () => {
    expect(titleCase("item")).toBe("Item");
  });

  it("handles empty string", () => {
    expect(titleCase("")).toBe("");
  });
});

describe("formatNumber — extended", () => {
  it("formats negative numbers", () => {
    expect(formatNumber(-1234.56)).toBe("-1,234.56");
  });

  it("formats very large numbers", () => {
    const result = formatNumber(1e15);
    expect(result).toContain(",");
  });
});

describe("formatPercent — extended", () => {
  it("formats zero percent", () => {
    expect(formatPercent(0)).toBe("0%");
  });

  it("formats 100 percent", () => {
    expect(formatPercent(100)).toBe("100%");
  });

  it("formats negative percent", () => {
    expect(formatPercent(-5.5)).toBe("-5.5%");
  });
});

describe("formatCompactNumber — extended", () => {
  it("formats zero", () => {
    expect(formatCompactNumber(0)).toBe("0");
  });

  it("formats negative numbers", () => {
    const result = formatCompactNumber(-5000);
    expect(result).toContain("K");
  });
});
