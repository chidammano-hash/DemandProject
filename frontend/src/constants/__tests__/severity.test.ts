import { describe, it, expect } from "vitest";
import {
  SEVERITY_CONFIG,
  SEVERITY_ORDER,
  STATUS_TONE_BADGE,
  getSeverityConfig,
  compareSeverity,
  type Severity,
} from "../severity";

const ALL_SEVERITIES: Severity[] = ["critical", "high", "medium", "low"];
const REQUIRED_KEYS = [
  "label",
  "badge",
  "border",
  "dot",
  "bg",
  "text",
  "ring",
  "icon",
  "rowBg",
] as const;

describe("SEVERITY_CONFIG", () => {
  it.each(ALL_SEVERITIES)("has all required keys for '%s'", (sev) => {
    const config = SEVERITY_CONFIG[sev];
    for (const key of REQUIRED_KEYS) {
      expect(config).toHaveProperty(key);
      expect(typeof config[key]).toBe("string");
      expect(config[key].length).toBeGreaterThan(0);
    }
  });

  it("has a capitalized label for each severity", () => {
    expect(SEVERITY_CONFIG.critical.label).toBe("Critical");
    expect(SEVERITY_CONFIG.high.label).toBe("High");
    expect(SEVERITY_CONFIG.medium.label).toBe("Medium");
    expect(SEVERITY_CONFIG.low.label).toBe("Low");
  });

  // Token-only contract — every class string must resolve through a CSS var
  // (`bg-destructive/10`, not `bg-red-100`), so one class works across light,
  // soft, and dark mode with no `dark:` sibling required.
  it.each(ALL_SEVERITIES)("carries no raw Tailwind palette color for '%s'", (sev) => {
    const config = SEVERITY_CONFIG[sev];
    for (const key of REQUIRED_KEYS) {
      if (key === "label") continue;
      expect(config[key], `${sev}.${key} must not carry a dark: sibling`).not.toMatch(/dark:/);
      expect(
        config[key],
        `${sev}.${key} must not use a raw palette color (red/orange/yellow/amber/gray/etc.)`,
      ).not.toMatch(/\b(red|orange|yellow|amber|gray|slate|zinc|green|blue)-\d/);
    }
  });

  it("maps critical to destructive tokens", () => {
    expect(SEVERITY_CONFIG.critical.badge).toBe("border-destructive/25 bg-destructive/10 text-destructive");
    expect(SEVERITY_CONFIG.critical.text).toBe("text-destructive");
    expect(SEVERITY_CONFIG.critical.dot).toBe("bg-destructive");
  });

  it("maps high to severity-high tokens", () => {
    expect(SEVERITY_CONFIG.high.badge).toBe("border-severity-high/25 bg-severity-high/10 text-severity-high");
    expect(SEVERITY_CONFIG.high.text).toBe("text-severity-high");
    expect(SEVERITY_CONFIG.high.dot).toBe("bg-severity-high");
  });

  it("maps medium to warning tokens", () => {
    expect(SEVERITY_CONFIG.medium.badge).toBe("border-warning/25 bg-warning/10 text-warning");
    expect(SEVERITY_CONFIG.medium.text).toBe("text-warning");
    expect(SEVERITY_CONFIG.medium.dot).toBe("bg-warning");
  });

  it("maps low to muted tokens", () => {
    expect(SEVERITY_CONFIG.low.badge).toBe("border-border bg-muted text-muted-foreground");
    expect(SEVERITY_CONFIG.low.text).toBe("text-muted-foreground");
  });
});

describe("STATUS_TONE_BADGE", () => {
  it("maps info and success to alpha-tinted tokens", () => {
    expect(STATUS_TONE_BADGE.info).toBe("border-info/25 bg-info/10 text-info");
    expect(STATUS_TONE_BADGE.success).toBe("border-success/25 bg-success/10 text-success");
  });

  it("carries no raw Tailwind palette color or dark: sibling", () => {
    for (const cls of Object.values(STATUS_TONE_BADGE)) {
      expect(cls).not.toMatch(/dark:/);
      expect(cls).not.toMatch(/\b(red|orange|yellow|amber|gray|slate|zinc|green|blue)-\d/);
    }
  });
});

describe("SEVERITY_ORDER", () => {
  it("assigns critical=0, high=1, medium=2, low=3", () => {
    expect(SEVERITY_ORDER.critical).toBe(0);
    expect(SEVERITY_ORDER.high).toBe(1);
    expect(SEVERITY_ORDER.medium).toBe(2);
    expect(SEVERITY_ORDER.low).toBe(3);
  });
});

describe("getSeverityConfig", () => {
  it.each(ALL_SEVERITIES)("returns correct config for '%s'", (sev) => {
    const config = getSeverityConfig(sev);
    expect(config).toBe(SEVERITY_CONFIG[sev]);
  });

  it("returns 'low' config for unknown severity values", () => {
    expect(getSeverityConfig("unknown")).toBe(SEVERITY_CONFIG.low);
    expect(getSeverityConfig("")).toBe(SEVERITY_CONFIG.low);
    expect(getSeverityConfig("CRITICAL")).toBe(SEVERITY_CONFIG.low);
  });
});

describe("compareSeverity", () => {
  it("sorts critical before high", () => {
    expect(compareSeverity("critical", "high")).toBeLessThan(0);
  });

  it("sorts high before medium", () => {
    expect(compareSeverity("high", "medium")).toBeLessThan(0);
  });

  it("sorts medium before low", () => {
    expect(compareSeverity("medium", "low")).toBeLessThan(0);
  });

  it("returns 0 for equal severities", () => {
    expect(compareSeverity("critical", "critical")).toBe(0);
    expect(compareSeverity("low", "low")).toBe(0);
  });

  it("sorts low before unknown values", () => {
    expect(compareSeverity("low", "bogus")).toBeLessThan(0);
  });

  it("sorts an array of severity strings correctly", () => {
    const input = ["low", "critical", "medium", "high"];
    const sorted = [...input].sort(compareSeverity);
    expect(sorted).toEqual(["critical", "high", "medium", "low"]);
  });
});
