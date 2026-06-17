/**
 * Backtest stage panel for the Model Experimentation Studio.
 * Shows ALL models with run/load actions and a detail panel for the selected model.
 */
import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import {
  backtestMgmtKeys,
  fetchBacktestSummary,
  submitBacktestRun,
  submitBacktestLoad,
} from "@/api/queries/backtest-management";

import { BacktestDetailPanel } from "./BacktestDetailPanel";
import { TYPE_COLORS } from "./_helpers";
import type { ModelInfo } from "./_types";

interface Props {
  models: ModelInfo[];
  selectedModelId: string;
  selectedModelInfo: ModelInfo;
  onSelectModel: (modelId: string) => void;
}

export function BacktestStagePanel({
  models,
  selectedModelId,
  selectedModelInfo,
  onSelectModel,
}: Props) {
  const queryClient = useQueryClient();
  const [runningModels, setRunningModels] = useState<Set<string>>(new Set());
  const [loadingModels, setLoadingModels] = useState<Set<string>>(new Set());
  const [parallel, setParallel] = useState(false);
  const [runError, setRunError] = useState<string | null>(null);

  const { data: backtestSummary } = useQuery({
    queryKey: backtestMgmtKeys.summary,
    queryFn: fetchBacktestSummary,
    staleTime: 30_000,
    refetchInterval: loadingModels.size > 0 ? 5_000 : false,
  });

  const handleRunBacktest = async (modelId: string) => {
    setRunError(null);
    setRunningModels((prev) => new Set(prev).add(modelId));
    try {
      await submitBacktestRun(modelId, parallel);
      queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.summary });
    } catch (err) {
      // Surface 409 "already running/queued" and other errors to the user.
      setRunError(err instanceof Error ? err.message : "Failed to submit backtest");
    } finally {
      // Keep the running indicator for a bit so the user sees feedback
      setTimeout(() => {
        setRunningModels((prev) => {
          const next = new Set(prev);
          next.delete(modelId);
          return next;
        });
      }, 5_000);
    }
  };

  const handleLoadBacktest = async (modelId: string) => {
    setLoadingModels((prev) => new Set(prev).add(modelId));
    // Pass the latest run ID so the backend can mark it as loaded
    const runId = backtestSummary?.[modelId]?.latest_run?.id;
    try {
      await submitBacktestLoad(modelId, runId ?? undefined);
      queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.summary });
      // Keep loadingModels populated — cleared by polling effect below
    } catch (err) {
      console.error("Failed to load backtest:", err);
      setLoadingModels((prev) => {
        const next = new Set(prev);
        next.delete(modelId);
        return next;
      });
    }
  };

  // Clear loading state when polled summary shows the model is loaded
  useEffect(() => {
    if (loadingModels.size === 0 || !backtestSummary) return;
    const done = new Set<string>();
    for (const modelId of loadingModels) {
      const summary = backtestSummary[modelId];
      if (summary?.latest_run?.is_loaded_to_db) {
        done.add(modelId);
      }
    }
    if (done.size > 0) {
      setLoadingModels((prev) => {
        const next = new Set(prev);
        for (const m of done) next.delete(m);
        return next;
      });
    }
  }, [backtestSummary, loadingModels]);

  return (
    <>
      {/* ---- Model Grid (all models) ---- */}
      <div className="space-y-3">
        <div>
          <div className="mb-2 flex items-center justify-between gap-3">
            <h3 className="text-sm font-medium text-muted-foreground">All Models</h3>
            <label
              className="flex items-center gap-1.5 text-xs text-muted-foreground select-none cursor-pointer"
              title="When on, different model families run at the same time. When off, backtests run one at a time (extra runs queue). The same family never runs twice at once."
            >
              <input
                type="checkbox"
                checked={parallel}
                onChange={(e) => setParallel(e.target.checked)}
                className="h-3.5 w-3.5"
              />
              Run in parallel
            </label>
          </div>
          {runError && (
            <p className="mb-2 rounded-md border border-amber-300/60 bg-amber-50 px-3 py-2 text-xs text-amber-900 dark:border-amber-700/60 dark:bg-amber-950/30 dark:text-amber-200">
              {runError}
            </p>
          )}
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-2">
            {models.map((m) => {
              const btSummary = backtestSummary?.[m.id];
              const isSelected = selectedModelId === m.id;
              return (
                <div
                  key={m.id}
                  className={cn(
                    "rounded-lg border p-3 text-left transition-all",
                    isSelected
                      ? "border-primary bg-primary/5 ring-1 ring-primary shadow-sm"
                      : "hover:bg-muted/50",
                  )}
                >
                  <button onClick={() => onSelectModel(m.id)} className="w-full text-left">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-semibold truncate">{m.label}</span>
                      {btSummary?.latest_run?.is_loaded_to_db && (
                        <span
                          className="h-2 w-2 rounded-full bg-green-500 shrink-0"
                          title="Loaded to DB"
                        />
                      )}
                    </div>
                    <Badge
                      variant="outline"
                      className={`text-[9px] px-1.5 py-0 ${TYPE_COLORS[m.type] ?? ""}`}
                    >
                      {m.type.replace("_", " ")}
                    </Badge>
                    {btSummary?.current_accuracy != null ? (
                      <div className="mt-2 text-xs tabular-nums">
                        <span className="font-semibold">
                          {btSummary.current_accuracy.toFixed(1)}%
                        </span>
                        <span className="text-muted-foreground ml-1 text-[10px]">accuracy</span>
                      </div>
                    ) : (
                      <div className="mt-2 text-[10px] text-muted-foreground">
                        No backtest yet
                      </div>
                    )}
                  </button>
                  {/* Action buttons */}
                  <div className="flex gap-1 mt-2">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRunBacktest(m.id);
                      }}
                      className="flex-1 rounded bg-primary/10 px-2 py-1 text-[10px] font-medium text-primary hover:bg-primary/20 disabled:opacity-40"
                      disabled={runningModels.has(m.id)}
                    >
                      {runningModels.has(m.id) ? "Running..." : "Run"}
                    </button>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleLoadBacktest(m.id);
                      }}
                      className="flex-1 rounded bg-emerald-500/10 px-2 py-1 text-[10px] font-medium text-emerald-600 hover:bg-emerald-500/20 disabled:opacity-30"
                      disabled={!btSummary?.has_predictions || loadingModels.has(m.id)}
                    >
                      {loadingModels.has(m.id) ? "Loading..." : "Load"}
                    </button>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* ---- Selected Model Backtest Detail ---- */}
      <Card>
        <CardHeader className="pb-3">
          <div className="flex items-center gap-2">
            <CardTitle className="text-base">{selectedModelInfo.label}</CardTitle>
            <Badge
              variant="outline"
              className={`text-[10px] ${TYPE_COLORS[selectedModelInfo.type] ?? ""}`}
            >
              {selectedModelInfo.type.replace("_", " ")}
            </Badge>
          </div>
        </CardHeader>
        <CardContent>
          <BacktestDetailPanel modelId={selectedModelId} />
        </CardContent>
      </Card>
    </>
  );
}
