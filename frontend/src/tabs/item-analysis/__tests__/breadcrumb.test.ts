import { describe, it, expect } from "vitest";
import { itemBreadcrumbLabel } from "../breadcrumb";

describe("itemBreadcrumbLabel (U3.5 — show product name, not a bare code)", () => {
  it("renders 'Item <id> — <desc>' when the description is loaded", () => {
    expect(itemBreadcrumbLabel("185690", "DAMMANN JARDIN BLEU TEA(96CT)")).toBe(
      "Item 185690 — DAMMANN JARDIN BLEU TEA(96CT)",
    );
  });

  it("falls back to the bare 'Item <id>' while the description is loading", () => {
    expect(itemBreadcrumbLabel("185690", null)).toBe("Item 185690");
    expect(itemBreadcrumbLabel("185690", undefined)).toBe("Item 185690");
    expect(itemBreadcrumbLabel("185690", "")).toBe("Item 185690");
  });

  it("does not duplicate the id when the description already starts with it", () => {
    // Some feeds embed the id in the description; avoid 'Item 15502 — 15502 ...'.
    expect(itemBreadcrumbLabel("15502", "15502 TITOS HANDMADE VODKA")).toBe(
      "Item 15502 — TITOS HANDMADE VODKA",
    );
  });
});
