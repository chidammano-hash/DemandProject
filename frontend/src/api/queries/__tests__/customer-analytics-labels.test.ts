import { describe, it, expect } from "vitest";
import { normalizeLabelOptions } from "../customer-analytics";

/**
 * F4.5 / U4.2 — the Channel and Store Type filter dropdowns were rendered
 * verbatim from the MV: trailing-whitespace duplicates, case-variant
 * duplicates, `null`/`undefined`/empty entries. normalizeLabelOptions trims,
 * drops nullish/empty/"null"/"undefined", de-dupes case-insensitively keeping
 * the FIRST canonical casing seen (so the WHERE clause still matches an
 * emitted original), and sorts case-insensitively.
 */
describe("normalizeLabelOptions (F4.5 / U4.2)", () => {
  it("trims and collapses whitespace/case duplicates keeping first casing", () => {
    const raw = [
      "Off Premise Chains",
      "OFF PREMISE CHAINS",
      "Off Premise Chains            ",
      "On Premise Accounts",
      "on premise accounts",
    ];
    const out = normalizeLabelOptions(raw);
    expect(out).toEqual(["Off Premise Chains", "On Premise Accounts"]);
  });

  it("drops null / undefined / empty / 'null' / 'undefined'", () => {
    const raw = ["null", "undefined", "", "   ", "Bar", "Undefined", "NULL"];
    const out = normalizeLabelOptions(raw as string[]);
    expect(out).toEqual(["Bar"]);
  });

  it("sorts case-insensitively", () => {
    const out = normalizeLabelOptions(["zebra", "Apple", "mango"]);
    expect(out).toEqual(["Apple", "mango", "zebra"]);
  });

  it("returns an empty array for nullish input", () => {
    expect(normalizeLabelOptions(undefined)).toEqual([]);
    expect(normalizeLabelOptions(null)).toEqual([]);
  });

  it("skips non-string entries defensively", () => {
    const raw = ["Bar", 5 as unknown as string, null as unknown as string, "Casino"];
    expect(normalizeLabelOptions(raw)).toEqual(["Bar", "Casino"]);
  });
});
