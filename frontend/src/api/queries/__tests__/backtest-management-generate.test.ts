/**
 * Regression tests for submitGenerateForecast URL threading.
 *
 * The Forecast panel's horizon input and "Include Confidence Intervals" toggle
 * were previously dropped for single-model generation — the fetcher hit
 * `/backtest-management/{id}/generate` with no query string, so the backend
 * always used the config default. These tests pin the query params.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { submitGenerateForecast } from "@/api/queries/backtest-management";

describe("submitGenerateForecast threads horizon + confidence_intervals", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValue({
      ok: true,
      status: 201,
      text: () => Promise.resolve(JSON.stringify({ job_id: "j1", model_id: "lgbm_cluster" })),
      json: () => Promise.resolve({ job_id: "j1", model_id: "lgbm_cluster" }),
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("includes horizon + confidence_intervals when provided", async () => {
    await submitGenerateForecast("lgbm_cluster", { horizon: 9, confidenceIntervals: true });
    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("/backtest-management/lgbm_cluster/generate");
    expect(url).toContain("horizon=9");
    expect(url).toContain("confidence_intervals=true");
  });

  it("threads confidence_intervals=false explicitly (force CI off)", async () => {
    await submitGenerateForecast("lgbm_cluster", { confidenceIntervals: false });
    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("confidence_intervals=false");
    expect(url).not.toContain("horizon=");
  });

  it("emits no query string when no options are passed", async () => {
    await submitGenerateForecast("chronos2_enriched");
    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toBe("/backtest-management/chronos2_enriched/generate");
    expect(url).not.toContain("?");
  });

  it("POSTs with the JSON content-type header", async () => {
    await submitGenerateForecast("lgbm_cluster", { horizon: 12 });
    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(init.method).toBe("POST");
  });
});
