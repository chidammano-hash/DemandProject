import { describe, it, expect } from "vitest";
import { formatClusterLabel } from "../../lib/formatters";

describe("formatClusterLabel 4-letter SC velocity codes", () => {
  const cases: [string, string][] = [
    // Single pattern
    ["very_high_volume_periodic", "FAST.CYCL"],
    ["high_volume_seasonal", "MOVR.SEAS"],
    ["medium_volume_steady", "BASE.CALM"],
    ["low_volume_volatile", "SLOW.WILD"],
    ["very_low_volume_intermittent", "TAIL.RARE"],
    ["medium_volume_moderate", "BASE.EVEN"],
    ["very_high_volume_seasonal", "FAST.SEAS"],
    ["high_volume_very_steady", "MOVR.CALM"],
    ["very_low_volume_sparse", "TAIL.RARE"],
    ["medium_volume_accelerating", "BASE.RISE"],
    // 2-code compound
    ["high_volume_seasonal_growing", "MOVR.SEAS.RISE"],
    ["high_volume_declining", "MOVR.FALL"],
    ["low_volume_seasonal_volatile", "SLOW.SEAS.WILD"],
    // 3-code compound
    ["low_volume_seasonal_growing_volatile", "SLOW.SEAS.RISE.WILD"],
    ["medium_volume_seasonal_declining_noisy", "BASE.SEAS.FALL.WILD"],
  ];

  it.each(cases)("%s → %s", (input, expected) => {
    expect(formatClusterLabel(input)).toBe(expected);
  });
});
