import { describe, it, expect } from "vitest";
import { severityBadgeClass } from "@/lib/severityBadge";

// U5.1 — there was no shared themed severity badge, so 30+ tabs hand-rolled
// `bg-{color}-100 text-{color}-700` chips with NO `dark:` companion, rendering
// a pale pastel-on-near-black pill that barely separates from the page in Dark.
// The shared helper must emit a Light + dark: tint pair for each severity.
describe("severityBadgeClass", () => {
  const severities = ["critical", "high", "medium", "low", "info", "warning", "success"] as const;

  it("returns a dark: variant for every known severity", () => {
    for (const sev of severities) {
      const cls = severityBadgeClass(sev);
      expect(cls, `${sev} must carry a dark: tint`).toMatch(/dark:/);
    }
  });

  it("maps critical to a red tint with a dark companion", () => {
    const cls = severityBadgeClass("critical");
    expect(cls).toContain("bg-red-100");
    expect(cls).toContain("text-red-700");
    expect(cls).toMatch(/dark:bg-red-/);
    expect(cls).toMatch(/dark:text-red-/);
  });

  it("maps success to a green tint with a dark companion", () => {
    const cls = severityBadgeClass("success");
    expect(cls).toMatch(/dark:bg-(green|emerald)-/);
  });

  it("falls back to a neutral themed tint for an unknown severity", () => {
    const cls = severityBadgeClass("totally-unknown");
    // Must not silently emit a bare bg-*-100 with no dark sibling.
    expect(cls).toMatch(/(muted|dark:)/);
  });
});
