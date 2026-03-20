import { describe, it, expect } from "vitest";
import {
  SEVERITY_CONFIG,
  SEVERITY_ORDER,
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
