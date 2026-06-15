import { describe, it, expect } from "vitest";

import { normalizeStoryboardException } from "../command-center/exceptions";
import type { StoryboardException } from "@/types/storyboard";

// U4.1: the Command Center storyboard card must NOT leak the raw replenishment
// enum into its chip. Replenishment exceptions (below_ss/stockout/below_rop/
// excess/zero_velocity) emitted by the cycle-4 _replenishment_fallback must
// render the same friendly labels the Inv Planning action feed uses.

function makeExc(
  type: string,
  headline: string,
  itemDesc?: string,
): StoryboardException {
  return {
    exception_id: "x1",
    exception_type: type,
    item_id: "664631",
    loc: "1401-BULK",
    item_desc: itemDesc ?? null,
    severity: 0.95,
    financial_impact: 292.39,
    headline,
    supporting_data: {},
    status: "open",
    assigned_to: null,
    generated_at: "2026-04-02",
    expires_at: null,
    month_start: "2026-04-02",
    source: "fact_replenishment_exceptions",
  } as unknown as StoryboardException;
}

describe("normalizeStoryboardException replenishment labels (U4.1)", () => {
  it("maps below_ss to 'Below Safety Stock' (not the raw enum)", () => {
    const out = normalizeStoryboardException(
      makeExc("below_ss", "Below Safety Stock — 664631 @ 1401-BULK"),
    );
    expect(out.typeLabel).toBe("Below Safety Stock");
    expect(out.typeLabel).not.toContain("below_ss");
  });

  it("maps the other replenishment enums to friendly labels", () => {
    const cases: Array<[string, string]> = [
      ["stockout", "Stockout"],
      ["below_rop", "Below Reorder Point"],
      ["excess", "Excess Inventory"],
      ["zero_velocity", "Zero Velocity"],
    ];
    for (const [enumVal, label] of cases) {
      const out = normalizeStoryboardException(makeExc(enumVal, `${label} — x`));
      expect(out.typeLabel).toBe(label);
    }
  });
});

describe("normalizeStoryboardException item description (U2.1)", () => {
  it("surfaces item_desc so Command Center rows show the product name", () => {
    const out = normalizeStoryboardException(
      makeExc(
        "stockout",
        "Stockout — 627099 @ 1401-BULK",
        "MENAGE A TROIS A(D/R/S)3P PAD(44",
      ),
    );
    expect(out.itemDesc).toBe("MENAGE A TROIS A(D/R/S)3P PAD(44");
  });

  it("leaves itemDesc undefined when the backend has no description", () => {
    const out = normalizeStoryboardException(makeExc("stockout", "Stockout — x"));
    expect(out.itemDesc).toBeUndefined();
  });
});

describe("normalizeStoryboardException summary de-dup (U2.2)", () => {
  it("strips the redundant '— {item} @ {loc}' suffix so the code is not shown twice", () => {
    // makeExc fixes item_id=664631 @ 1401-BULK; the backend builds the headline
    // from that same identity. The identity line renders "664631 @ 1401-BULK";
    // the summary must not restate it — only the type label remains.
    const out = normalizeStoryboardException(
      makeExc("stockout", "Stockout — 664631 @ 1401-BULK"),
    );
    expect(out.summary).toBe("Stockout");
    expect(out.summary).not.toContain("664631 @ 1401-BULK");
  });

  it("keeps a headline that does not restate the identity untouched", () => {
    const out = normalizeStoryboardException(
      makeExc("stockout_risk", "Stockout Risk: 8.0 days of supply"),
    );
    expect(out.summary).toBe("Stockout Risk: 8.0 days of supply");
  });
});
