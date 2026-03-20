import { describe, it, expect } from "vitest";
import { renderHook, act } from "@testing-library/react";
import type { ReactNode } from "react";
import {
  ScenarioNotificationProvider,
  useScenarioNotification,
} from "@/context/ScenarioNotificationContext";

function Wrapper({ children }: { children: ReactNode }) {
  return <ScenarioNotificationProvider>{children}</ScenarioNotificationProvider>;
}

describe("ScenarioNotificationContext", () => {
  it("provides default values", () => {
    const { result } = renderHook(() => useScenarioNotification(), { wrapper: Wrapper });
    expect(result.current.runningScenarioId).toBeNull();
    expect(result.current.completedScenario).toBeNull();
    expect(typeof result.current.dismissNotification).toBe("function");
  });

  it("startScenario sets running ID and label", () => {
    const { result } = renderHook(() => useScenarioNotification(), { wrapper: Wrapper });
    act(() => result.current.startScenario("sc_123", "A"));
    expect(result.current.runningScenarioId).toBe("sc_123");
    expect(result.current.runningScenarioLabel).toBe("A");
  });

  it("completeScenario sets notification and clears running", () => {
    const { result } = renderHook(() => useScenarioNotification(), { wrapper: Wrapper });
    act(() => result.current.startScenario("sc_123", "A"));
    act(() =>
      result.current.completeScenario({
        id: "sc_123",
        label: "A",
        runtimeSeconds: 45,
        result: { scenario_id: "sc_123", status: "completed", runtime_seconds: 45, params: {}, result: null },
      }),
    );
    expect(result.current.runningScenarioId).toBeNull();
    expect(result.current.completedScenario?.id).toBe("sc_123");
    expect(result.current.completedScenario?.runtimeSeconds).toBe(45);
  });

  it("dismissNotification clears completed scenario", () => {
    const { result } = renderHook(() => useScenarioNotification(), { wrapper: Wrapper });
    act(() =>
      result.current.completeScenario({
        id: "sc_123",
        label: "A",
        runtimeSeconds: 45,
        result: { scenario_id: "sc_123", status: "completed", runtime_seconds: 45, params: {}, result: null },
      }),
    );
    expect(result.current.completedScenario).not.toBeNull();
    act(() => result.current.dismissNotification());
    expect(result.current.completedScenario).toBeNull();
  });
});
