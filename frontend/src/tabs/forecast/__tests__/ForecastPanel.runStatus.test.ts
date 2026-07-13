import { describe, expect, it } from "vitest";

import type { Job } from "@/types/jobs";
import {
  buildForecastGenerationOptions,
  findFailedGenerationModels,
  generationFailureMessage,
  isExpectedStagingRunReady,
  productionPromotionBlockedReason,
  resolveConfidenceIntervals,
} from "../ForecastPanel";
import { requiresTraining } from "../forecastPanelShared";

function generationJob(overrides: Partial<Job> = {}): Job {
  return {
    job_id: "job-1",
    job_type: "generate_production_forecast",
    job_label: "Generate Forecast",
    status: "failed",
    params: { run_id: "run-1" },
    result: null,
    error: "generation failed",
    submitted_at: "2026-07-12T10:00:00Z",
    started_at: "2026-07-12T10:00:01Z",
    completed_at: "2026-07-12T10:00:02Z",
    progress_pct: 100,
    progress_msg: null,
    pid: null,
    ...overrides,
  };
}

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

describe("forecast generation options", () => {
  it("defaults the UI to configured confidence intervals without overriding the API", () => {
    expect(resolveConfidenceIntervals(true, undefined)).toBe(true);
    expect(resolveConfidenceIntervals(undefined, undefined)).toBe(true);
    expect(buildForecastGenerationOptions(24, undefined)).toEqual({ horizon: 24 });
  });

  it("sends an explicit override only after the user changes the control", () => {
    expect(buildForecastGenerationOptions(24, false)).toEqual({
      horizon: 24,
      confidenceIntervals: false,
    });
  });
});

describe("production artifact requirements", () => {
  it("requires artifacts for LightGBM and neural models only", () => {
    expect(requiresTraining("tree")).toBe(true);
    expect(requiresTraining("deep_learning")).toBe(true);
    expect(requiresTraining("statistical")).toBe(false);
    expect(requiresTraining("foundation")).toBe(false);
  });
});

describe("production promotion availability", () => {
  it("allows any staged candidate regardless of Period Roll snapshot state", () => {
    expect(productionPromotionBlockedReason(false, true)).toBeUndefined();
  });

  it("blocks only while generation is active or the candidate is not staged", () => {
    expect(productionPromotionBlockedReason(true, true)).toMatch(/active forecast generation/i);
    expect(productionPromotionBlockedReason(false, false)).toMatch(/staging first/i);
  });
});

describe("findFailedGenerationModels", () => {
  it("matches only terminal failures for the exact submitted run", () => {
    const jobs = [
      generationJob({ job_id: "old", params: { run_id: "old-run" } }),
      generationJob({ job_id: "active", status: "running" }),
      generationJob({ job_id: "cancelled", status: "cancelled" }),
    ];

    const failures = findFailedGenerationModels(jobs, { lgbm_cluster: "run-1", mstl: "run-2" }, [
      "lgbm_cluster",
      "mstl",
    ]);

    expect(failures).toHaveLength(1);
    expect(failures[0]?.modelId).toBe("lgbm_cluster");
    expect(failures[0]?.job.job_id).toBe("cancelled");
  });

  it("keeps internal job errors out of user-facing failure copy", () => {
    const message = generationFailureMessage("lgbm_cluster");
    expect(message).toBe("LightGBM generation failed. Open Jobs for details.");
    expect(message).not.toContain("secret");
  });
});
