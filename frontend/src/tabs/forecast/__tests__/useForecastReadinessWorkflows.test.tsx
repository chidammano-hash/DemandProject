import type { ReactNode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { submitTraining } from "@/api/queries/backtest-management";
import { fetchJobDetail, fetchJobs, runNamedPipeline } from "@/api/queries/jobs";
import { useForecastPublishPreparation } from "../useForecastPublishPreparation";
import { useForecastTraining } from "../useForecastTraining";

vi.mock("@/api/queries/backtest-management", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/api/queries/backtest-management")>();
  return { ...original, submitTraining: vi.fn() };
});

vi.mock("@/api/queries/jobs", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/api/queries/jobs")>();
  return {
    ...original,
    fetchJobDetail: vi.fn(),
    fetchJobs: vi.fn(),
    runNamedPipeline: vi.fn(),
  };
});

vi.mock("@/components/Toaster", () => ({
  toast: { info: vi.fn(), success: vi.fn(), error: vi.fn() },
}));

function createWrapper() {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  return function Wrapper({ children }: { children: ReactNode }) {
    return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
  };
}

describe("forecast readiness workflow hooks", () => {
  afterEach(() => vi.unstubAllGlobals());

  beforeEach(() => {
    const values = new Map<string, string>();
    vi.stubGlobal("localStorage", {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => values.set(key, value),
      removeItem: (key: string) => values.delete(key),
    });
    vi.clearAllMocks();
    vi.mocked(fetchJobs).mockResolvedValue({ jobs: [], total: 0, limit: 100, offset: 0 });
    vi.mocked(fetchJobDetail).mockResolvedValue({
      job_id: "job-train",
      job_type: "train_production_model",
      job_label: "Train Production Models",
      status: "running",
      params: {},
      result: null,
      error: null,
      submitted_at: "2026-07-12T10:00:00Z",
      started_at: "2026-07-12T10:00:01Z",
      completed_at: null,
      progress_pct: 10,
      progress_msg: "Training",
      pid: null,
    });
  });

  it("launches the canonical publish pipeline once and enters polling state", async () => {
    vi.mocked(runNamedPipeline).mockResolvedValue({
      pipeline_id: "pipe-1",
      name: "forecast-publish",
      status: "running",
      steps: 3,
    });
    const { result } = renderHook(() => useForecastPublishPreparation(), {
      wrapper: createWrapper(),
    });

    await act(async () => result.current.preparePublish());

    expect(runNamedPipeline).toHaveBeenCalledWith("forecast-publish");
    expect(result.current.isPreparingPublish).toBe(true);
  });

  it("launches champion-refresh when champion lineage is the readiness blocker", async () => {
    vi.mocked(runNamedPipeline).mockResolvedValue({
      pipeline_id: "pipe-champion",
      name: "champion-refresh",
      status: "running",
      steps: 1,
    });
    const { result } = renderHook(() => useForecastPublishPreparation(), {
      wrapper: createWrapper(),
    });

    await act(async () => result.current.preparePublish("champion-refresh"));

    expect(runNamedPipeline).toHaveBeenCalledWith("champion-refresh");
    expect(result.current.isPreparingPublish).toBe(true);
  });

  it("tracks the exact bulk training job instead of trusting old metadata", async () => {
    vi.mocked(submitTraining).mockResolvedValue({ job_id: "job-train" });
    const { result } = renderHook(() => useForecastTraining(), { wrapper: createWrapper() });

    await act(async () => result.current.trainAll());

    await waitFor(() => expect(result.current.isTraining).toBe(true));
    expect(result.current.trainingModelId).toBe("__all__");
    expect(submitTraining).toHaveBeenCalledWith("all");
  });
});
