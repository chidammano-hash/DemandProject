import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import type { ReactNode } from "react";
import {
  JobNotificationProvider,
  useJobNotification,
} from "@/context/JobNotificationContext";

function Wrapper({ children }: { children: ReactNode }) {
  return <JobNotificationProvider>{children}</JobNotificationProvider>;
}

describe("JobNotificationContext", () => {
  it("provides default values", () => {
    const { result } = renderHook(() => useJobNotification(), { wrapper: Wrapper });
    expect(result.current.activeJobs.size).toBe(0);
    expect(result.current.recentCompletions).toHaveLength(0);
    expect(result.current.activeJobCount).toBe(0);
  });

  it("startJob adds to activeJobs", () => {
    const { result } = renderHook(() => useJobNotification(), { wrapper: Wrapper });
    act(() => result.current.startJob("j1", "backtest_lgbm", "LGBM Backtest"));
    expect(result.current.activeJobs.size).toBe(1);
    expect(result.current.activeJobCount).toBe(1);
    expect(result.current.activeJobs.get("j1")?.label).toBe("LGBM Backtest");
  });

  it("completeJob moves from active to completions", () => {
    const { result } = renderHook(() => useJobNotification(), { wrapper: Wrapper });
    act(() => result.current.startJob("j1", "backtest_lgbm", "LGBM Backtest"));
    act(() =>
      result.current.completeJob({
        id: "j1",
        type: "backtest_lgbm",
        label: "LGBM Backtest",
        runtimeSeconds: 120,
        status: "completed",
      }),
    );
    expect(result.current.activeJobs.size).toBe(0);
    expect(result.current.recentCompletions).toHaveLength(1);
    expect(result.current.recentCompletions[0].runtimeSeconds).toBe(120);
  });

  it("failJob removes from activeJobs", () => {
    const { result } = renderHook(() => useJobNotification(), { wrapper: Wrapper });
    act(() => result.current.startJob("j1", "cluster_pipeline", "Pipeline"));
    act(() => result.current.failJob("j1"));
    expect(result.current.activeJobs.size).toBe(0);
  });

  it("dismissCompletion removes from recentCompletions", () => {
    const { result } = renderHook(() => useJobNotification(), { wrapper: Wrapper });
    act(() =>
      result.current.completeJob({
        id: "j1",
        type: "backtest_lgbm",
        label: "LGBM",
        runtimeSeconds: 60,
        status: "completed",
      }),
    );
    expect(result.current.recentCompletions).toHaveLength(1);
    act(() => result.current.dismissCompletion("j1"));
    expect(result.current.recentCompletions).toHaveLength(0);
  });
});
