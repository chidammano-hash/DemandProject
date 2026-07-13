import type { ReactNode } from "react";
import { act, renderHook, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { submitTraining } from "@/api/queries/backtest-management";
import { fetchJobDetail } from "@/api/queries/jobs";
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

  it("tracks the exact bulk training job instead of trusting old metadata", async () => {
    vi.mocked(submitTraining).mockResolvedValue({ job_id: "job-train" });
    const { result } = renderHook(() => useForecastTraining(), { wrapper: createWrapper() });

    await act(async () => result.current.trainAll());

    await waitFor(() => expect(result.current.isTraining).toBe(true));
    expect(result.current.trainingModelId).toBe("__all__");
    expect(submitTraining).toHaveBeenCalledWith("all");
  });
});
