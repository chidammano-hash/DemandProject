import { createContext, useCallback, useContext, useState } from "react";
import type { ClusteringScenarioResult } from "@/api/queries";

export interface CompletedScenario {
  id: string;
  label: string;
  runtimeSeconds: number;
  result: ClusteringScenarioResult;
}

export interface ScenarioNotificationContextValue {
  runningScenarioId: string | null;
  runningScenarioLabel: string | null;
  completedScenario: CompletedScenario | null;
  startScenario: (id: string, label: string) => void;
  completeScenario: (scenario: CompletedScenario) => void;
  failScenario: () => void;
  dismissNotification: () => void;
}

const ScenarioNotificationContext = createContext<ScenarioNotificationContextValue | null>(null);

export function ScenarioNotificationProvider({ children }: { children: React.ReactNode }) {
  const [runningScenarioId, setRunningScenarioId] = useState<string | null>(null);
  const [runningScenarioLabel, setRunningScenarioLabel] = useState<string | null>(null);
  const [completedScenario, setCompletedScenario] = useState<CompletedScenario | null>(null);

  const startScenario = useCallback((id: string, label: string) => {
    setRunningScenarioId(id);
    setRunningScenarioLabel(label);
    setCompletedScenario(null);
  }, []);

  const completeScenario = useCallback((scenario: CompletedScenario) => {
    setRunningScenarioId(null);
    setRunningScenarioLabel(null);
    setCompletedScenario(scenario);
  }, []);

  const failScenario = useCallback(() => {
    setRunningScenarioId(null);
    setRunningScenarioLabel(null);
  }, []);

  const dismissNotification = useCallback(() => {
    setCompletedScenario(null);
  }, []);

  return (
    <ScenarioNotificationContext.Provider
      value={{
        runningScenarioId,
        runningScenarioLabel,
        completedScenario,
        startScenario,
        completeScenario,
        failScenario,
        dismissNotification,
      }}
    >
      {children}
    </ScenarioNotificationContext.Provider>
  );
}

export function useScenarioNotification(): ScenarioNotificationContextValue {
  const ctx = useContext(ScenarioNotificationContext);
  if (!ctx) throw new Error("useScenarioNotification must be used within ScenarioNotificationProvider");
  return ctx;
}
