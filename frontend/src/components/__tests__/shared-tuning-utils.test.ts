import { describe, expect, it } from "vitest";

import { formatDuration } from "../shared-tuning-utils";

const START = "2026-07-12T00:00:00Z";

function endAfter(seconds: number): string {
  return new Date(Date.parse(START) + seconds * 1000).toISOString();
}

describe("formatDuration", () => {
  it("returns -- without a start timestamp", () => {
    expect(formatDuration(null, null)).toBe("--");
  });

  it("renders sub-minute durations as seconds", () => {
    expect(formatDuration(START, endAfter(42))).toBe("42s");
  });

  it("renders sub-hour durations as minutes and seconds", () => {
    expect(formatDuration(START, endAfter(5 * 60 + 20))).toBe("5m 20s");
  });

  it("renders sub-day durations as hours and minutes", () => {
    expect(formatDuration(START, endAfter(2 * 3600 + 15 * 60 + 9))).toBe("2h 15m");
  });

  it("renders multi-day durations as days and hours, never raw minutes", () => {
    // 4217m 31s — previously rendered literally.
    expect(formatDuration(START, endAfter(4217 * 60 + 31))).toBe("2d 22h");
    // 35914m 51s — a ~25-day runaway run.
    expect(formatDuration(START, endAfter(35_914 * 60 + 51))).toBe("24d 22h");
  });
});
