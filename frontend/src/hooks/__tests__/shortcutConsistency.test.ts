import { describe, it, expect } from "vitest";
import { NAV_ITEMS, NUMERIC_SHORTCUTS } from "@/components/AppSidebar";
import { TAB_MAP } from "@/hooks/useKeyboardShortcuts";

/**
 * U6.1 — the numeric tab shortcuts must have a SINGLE source of truth.
 *
 * Three surfaces previously disagreed:
 *   - sidebar <kbd> hints (NAV_ITEMS[].shortcut)
 *   - the key handler (TAB_MAP)
 *   - the help modal (KeyboardShortcutHelp SHORTCUTS)
 *
 * These tests pin them all to NAV_ITEMS so they cannot drift.
 */
describe("numeric shortcut consistency (U6.1)", () => {
  it("TAB_MAP[d] equals the NavItem.key whose shortcut===d, for every advertised digit", () => {
    const advertised = NAV_ITEMS.filter((i) => i.shortcut);
    expect(advertised.length).toBeGreaterThan(0);
    for (const item of advertised) {
      expect(TAB_MAP[item.shortcut!]).toBe(item.key);
    }
  });

  it("every digit TAB_MAP routes to is a real NAV_ITEMS key (no phantom 'clusters'/8/9)", () => {
    const navKeys = new Set(NAV_ITEMS.map((i) => i.key));
    for (const [digit, key] of Object.entries(TAB_MAP)) {
      expect(navKeys.has(key), `digit ${digit} routes to non-existent tab '${key}'`).toBe(true);
    }
  });

  it("'2' routes to Portfolio (aggregateAnalysis), matching the sidebar kbd hint", () => {
    expect(TAB_MAP["2"]).toBe("aggregateAnalysis");
  });

  it("NUMERIC_SHORTCUTS is derived from NAV_ITEMS and carries live labels", () => {
    for (const s of NUMERIC_SHORTCUTS) {
      const nav = NAV_ITEMS.find((i) => i.key === s.key);
      expect(nav, `shortcut ${s.digit} -> missing nav item`).toBeTruthy();
      expect(s.label).toBe(nav!.label);
      expect(TAB_MAP[s.digit]).toBe(s.key);
    }
    // labels must be the live IA, not pre-restructure strings
    const labels = NUMERIC_SHORTCUTS.map((s) => s.label);
    expect(labels).not.toContain("Control Tower");
    expect(labels).not.toContain("AI Planner");
    expect(labels).not.toContain("DFU Analysis");
  });
});
