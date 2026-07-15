import { describe, it, expect } from "vitest";
import { severityBadgeClass } from "@/lib/severityBadge";
import { SEVERITY_CONFIG, STATUS_TONE_BADGE } from "@/constants/severity";

// severityBadgeClass is a thin @deprecated wrapper over SEVERITY_CONFIG /
// STATUS_TONE_BADGE (constants/severity.ts) — assert against those exports,
// not re-typed literals, so the two never drift.
describe("severityBadgeClass", () => {
  const severities = ["critical", "high", "medium", "low", "info", "warning", "success"] as const;

  it("returns a token utility (no dark: sibling needed) for every known severity", () => {
    for (const sev of severities) {
      const cls = severityBadgeClass(sev);
      expect(cls.length).toBeGreaterThan(0);
      expect(cls, `${sev} must not carry a dark: sibling`).not.toMatch(/dark:/);
    }
  });

  it("delegates critical/high/medium/low to SEVERITY_CONFIG", () => {
    expect(severityBadgeClass("critical")).toBe(SEVERITY_CONFIG.critical.badge);
    expect(severityBadgeClass("high")).toBe(SEVERITY_CONFIG.high.badge);
    expect(severityBadgeClass("medium")).toBe(SEVERITY_CONFIG.medium.badge);
    expect(severityBadgeClass("low")).toBe(SEVERITY_CONFIG.low.badge);
  });

  it("maps 'warning' to the same tone as 'medium'", () => {
    expect(severityBadgeClass("warning")).toBe(SEVERITY_CONFIG.medium.badge);
  });

  it("delegates info/success to STATUS_TONE_BADGE", () => {
    expect(severityBadgeClass("info")).toBe(STATUS_TONE_BADGE.info);
    expect(severityBadgeClass("success")).toBe(STATUS_TONE_BADGE.success);
  });

  it("is case-insensitive", () => {
    expect(severityBadgeClass("CRITICAL")).toBe(SEVERITY_CONFIG.critical.badge);
  });

  it("falls back to the neutral/muted tone for null, undefined, or unknown values", () => {
    expect(severityBadgeClass(null)).toBe(SEVERITY_CONFIG.low.badge);
    expect(severityBadgeClass(undefined)).toBe(SEVERITY_CONFIG.low.badge);
    expect(severityBadgeClass("totally-unknown")).toBe(SEVERITY_CONFIG.low.badge);
  });
});
