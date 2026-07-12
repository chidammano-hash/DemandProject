import { describe, expect, it } from "vitest";

import { isExpectedStagingRunReady } from "../ForecastPanel";

describe("isExpectedStagingRunReady", () => {
  it("is false while submission has no expected run id or staging row", () => {
    expect(isExpectedStagingRunReady(undefined, undefined)).toBe(false);
  });

  it("requires the exact submitted run to be ready", () => {
    const candidate = {
      source_run_id: "run-2",
      run_status: "ready" as const,
    };
    expect(isExpectedStagingRunReady(candidate, "run-1")).toBe(false);
    expect(isExpectedStagingRunReady(candidate, "run-2")).toBe(true);
  });
});
