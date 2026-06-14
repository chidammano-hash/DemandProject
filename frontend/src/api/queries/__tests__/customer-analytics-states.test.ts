import { describe, it, expect } from "vitest";
import { normalizeStateOptions } from "../customer-analytics";

/**
 * U3.3 — the State filter dropdown was full of garbage codes (`.`, `00`, `0D`,
 * `XX`, `null`, numeric/junk). normalizeStateOptions keeps only canonical
 * 2-letter alpha codes (US states + CA provinces), uppercased, de-duped, sorted.
 */
describe("normalizeStateOptions (U3.3)", () => {
  it("drops null/empty/numeric/junk and keeps 2-letter alpha codes", () => {
    const raw = [
      ".", "0", "00", "01", "0D", "XX", "99", "null", "", "  ",
      "fl", "FL", "ca", "TX", "AB", "ON", "123",
    ];
    const out = normalizeStateOptions(raw);
    expect(out).toEqual(["AB", "CA", "FL", "ON", "TX"]);
  });

  it("returns an empty array for nullish input", () => {
    expect(normalizeStateOptions(undefined)).toEqual([]);
    expect(normalizeStateOptions(null)).toEqual([]);
  });

  it("rejects mixed alphanumeric like 0D and 1A", () => {
    expect(normalizeStateOptions(["0D", "1A", "A1", "GA"])).toEqual(["GA"]);
  });
});
