/**
 * Backtest stage panel for the Model Experimentation Studio.
 * Shows ALL models with run/load actions and a detail panel for the selected model.
 */
import { useEffect, useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { cn } from "@/lib/utils";
import { toast } from "@/components/Toaster";
import { formatApiError } from "@/lib/formatApiError";
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

  const { data: backtestSummary } = useQuery({
    queryKey: backtestMgmtKeys.summary,
    queryFn: fetchBacktestSummary,
    staleTime: 30_000,
    // Poll while anything is in flight (a load, an optimistic click, or a run the
    // server reports as queued/running) so the grid reflects real status.
    refetchInterval: (query) => {
      const summary = query.state.data;
      const anyRunInFlight = summary
        ? Object.values(summary).some(
            (s) => s.latest_run?.status === "queued" || s.latest_run?.status === "running",
          )
        : false;
      return loadingModels.size > 0 || runningModels.size > 0 || anyRunInFlight ? 5_000 : false;
    },
  });

  // A model is "active" if the user just clicked it (optimistic) or the server
  // reports its latest run as queued/running. Driving the button off real status
  // — not just a timer — keeps the label honest for the whole run.
  const runLabel = (id: string): string => {
    const status = backtestSummary?.[id]?.latest_run?.status;
    if (status === "queued") return "Queued...";
    if (status === "running" || runningModels.has(id)) return "Running...";
    return "Run";
  };
  const isModelActive = (id: string): boolean => {
    const status = backtestSummary?.[id]?.latest_run?.status;
    return runningModels.has(id) || status === "queued" || status === "running";
  };

  const handleRunBacktest = async (modelId: string) => {
    const label = models.find((m) => m.id === modelId)?.label ?? modelId;
    setRunningModels((prev) => new Set(prev).add(modelId));
    try {
      const res = await submitBacktestRun(modelId, parallel);
      queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.summary });
      // Calm, informative feedback — never a blocking error for concurrency.
      if (res.status === "already_running") {
        toast.info(res.message ?? `A ${label} backtest is already in progress.`);
      } else {
        toast.success(`${label} backtest queued — follow progress below.`);
      }
    } catch (err) {
      toast.error(formatApiError(err));
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
              title="When on, different model types run at the same time. When off, backtests run one at a time — extra runs queue automatically. Re-running a model that's already in progress is skipped."
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
                      disabled={isModelActive(m.id)}
                    >
                      {runLabel(m.id)}
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
