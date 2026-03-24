/**
 * IPAIfeature1 — AI Planning Agent
 *
 * Exception work-queue: structured, ranked, actionable insights generated
 * by an AI agent that reads across all data layers and traces causal chains.
 * NOT a chatbot — a proactive scan-and-triage system for planners.
 *
 * Sub-components live in ./ai-planner/:
 *   InsightCard, CausalChainCard, ConfirmModal, AutoAcceptModal,
 *   BulkActionBar, CopyButton, aiPlannerShared (constants + helpers)
 */
import { useState, useEffect, useRef, useCallback } from "react";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  queryKeys,
  fetchAiInsights,
  fetchAiMemos,
  triggerPortfolioScan,
  updateInsightStatus,
  triggerAutoAccept,
  snoozeInsight,
  STALE,
} from "@/api/queries";
import type { AutoAcceptResponse } from "@/api/queries/ai-planner";
import type { AiInsight, InsightSeverity, InsightStatus, InsightType } from "@/types/ai-planner";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { cn } from "@/lib/utils";
import { formatCurrency as fmtCurrency } from "@/lib/formatters";
import { RefreshCw, Loader2, Brain, Zap } from "lucide-react";

import { SEVERITY_ORDER } from "./ai-planner/aiPlannerShared";
import { InsightCard } from "./ai-planner/InsightCard";
import { ConfirmModal } from "./ai-planner/ConfirmModal";
import { AutoAcceptModal } from "./ai-planner/AutoAcceptModal";
import { BulkActionBar } from "./ai-planner/BulkActionBar";
import { CopyButton } from "./ai-planner/CopyButton";
import type { ConfirmActionState } from "./ai-planner/aiPlannerShared";

// ---------------------------------------------------------------------------
// Main tab component
// ---------------------------------------------------------------------------
export default function AIPlannerTab() {
  const qc = useQueryClient();
  const { filters } = useGlobalFilterContext();

  const globalFilterParams = {
    brand: filters.brand.length > 0 ? filters.brand.join(",") : undefined,
    category: filters.category.length > 0 ? filters.category.join(",") : undefined,
    market: filters.market.length > 0 ? filters.market.join(",") : undefined,
    channel: filters.channel.length > 0 ? filters.channel.join(",") : undefined,
    item_id: filters.item.length === 1 ? filters.item[0] : undefined,
    loc: filters.location.length === 1 ? filters.location[0] : undefined,
  };

  // ── Persistent URL filter state ──────────────────────────────────────────
  const getUrlParam = (key: string, fallback: string) => {
    try { return new URLSearchParams(window.location.search).get(key) ?? fallback; } catch { return fallback; }
  };
  const [severityFilter, setSeverityFilter] = useState(() => getUrlParam("ai_severity", "all"));
  const [statusFilter, setStatusFilter] = useState(() => getUrlParam("ai_status", "open"));
  const [typeFilter, setTypeFilter] = useState(() => getUrlParam("ai_type", "all"));

  // Sync filter changes to URL
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (severityFilter === "all") params.delete("ai_severity"); else params.set("ai_severity", severityFilter);
    if (statusFilter === "open") params.delete("ai_status"); else params.set("ai_status", statusFilter);
    if (typeFilter === "all") params.delete("ai_type"); else params.set("ai_type", typeFilter);
    const newSearch = params.toString();
    const newUrl = `${window.location.pathname}${newSearch ? "?" + newSearch : ""}`;
    window.history.replaceState(null, "", newUrl);
  }, [severityFilter, statusFilter, typeFilter]);

  // ── Bulk selection state ──────────────────────────────────────────────────
  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());

  const handleSelect = useCallback((id: number) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) { next.delete(id); } else { next.add(id); }
      return next;
    });
  }, []);

  const clearSelection = useCallback(() => setSelectedIds(new Set()), []);

  // Clear selection when filters change
  useEffect(() => { setSelectedIds(new Set()); }, [severityFilter, statusFilter, typeFilter]);

  // Confirm modal state
  const [confirmAction, setConfirmAction] = useState<ConfirmActionState | null>(null);

  // Auto-accept modal state
  const [showAutoAccept, setShowAutoAccept] = useState(false);
  const [autoAcceptResult, setAutoAcceptResult] = useState<AutoAcceptResponse | null>(null);

  const autoAcceptMutation = useMutation({
    mutationFn: triggerAutoAccept,
    onSuccess: (data) => {
      setAutoAcceptResult(data);
      if (!data.dry_run) {
        qc.invalidateQueries({ queryKey: queryKeys.aiInsights({}) });
      }
    },
  });

  const insightParams = {
    ...(severityFilter !== "all" && { severity: severityFilter as InsightSeverity }),
    ...(statusFilter !== "all" && { status: statusFilter as InsightStatus }),
    ...(typeFilter !== "all" && { insight_type: typeFilter as InsightType }),
    ...globalFilterParams,
    page_size: 50,
  };

  const insightsQ = useQuery({
    queryKey: queryKeys.aiInsights(insightParams),
    queryFn: () => fetchAiInsights(insightParams),
    staleTime: STALE.THIRTY_SEC,
    refetchInterval: 30_000,
  });

  const memosQ = useQuery({
    queryKey: queryKeys.aiMemos({ scope: "portfolio", limit: 5 }),
    queryFn: () => fetchAiMemos({ scope: "portfolio", limit: 5 }),
    staleTime: STALE.FIVE_MIN,
  });
  const [memoIndex, setMemoIndex] = useState(0);

  const [showScanSuccess, setShowScanSuccess] = useState(false);
  const [scanQueuedAt, setScanQueuedAt] = useState<Date | null>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // PL-007: Poll for new insights for up to 5 minutes after a scan is queued
  useEffect(() => {
    if (!scanQueuedAt) return;
    const POLL_INTERVAL = 30_000;
    const TIMEOUT = 5 * 60_000;
    pollRef.current = setInterval(() => {
      qc.invalidateQueries({ queryKey: queryKeys.aiInsights({}) });
      qc.invalidateQueries({ queryKey: queryKeys.aiMemos({}) });
      if (Date.now() - scanQueuedAt.getTime() > TIMEOUT) {
        clearInterval(pollRef.current!);
        setScanQueuedAt(null);
      }
    }, POLL_INTERVAL);
    return () => { if (pollRef.current) clearInterval(pollRef.current); };
  }, [scanQueuedAt, qc]);

  const scanMutation = useMutation({
    mutationFn: triggerPortfolioScan,
    onSuccess: () => {
      setShowScanSuccess(true);
      setScanQueuedAt(new Date());
      setTimeout(() => setShowScanSuccess(false), 8000);
    },
  });

  const statusMutation = useMutation({
    mutationFn: ({ id, status }: { id: number; status: InsightStatus }) =>
      updateInsightStatus(id, status),
    onSuccess: () => {
      setConfirmAction(null);
      qc.invalidateQueries({ queryKey: queryKeys.aiInsights({}) });
    },
  });

  function handleAcknowledge(insight: AiInsight) {
    setConfirmAction({
      insight,
      status: "acknowledged",
      label: "Accept Recommendation",
      verb: "Accept",
    });
  }

  function handleResolve(insight: AiInsight) {
    setConfirmAction({
      insight,
      status: "resolved",
      label: "Mark as Resolved",
      verb: "Resolve",
    });
  }

  const snoozeMutation = useMutation({
    mutationFn: ({ id, days }: { id: number; days: number }) => snoozeInsight(id, days),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: queryKeys.aiInsights({}) });
    },
  });

  function handleSnooze(insight: AiInsight, days: number) {
    snoozeMutation.mutate({ id: insight.insight_id, days });
  }

  // Bulk acknowledge
  const [bulkPending, setBulkPending] = useState(false);
  async function handleBulkAcknowledge() {
    if (selectedIds.size === 0) return;
    setBulkPending(true);
    try {
      for (const id of selectedIds) {
        await updateInsightStatus(id, "acknowledged");
      }
      qc.invalidateQueries({ queryKey: queryKeys.aiInsights({}) });
      setSelectedIds(new Set());
    } finally {
      setBulkPending(false);
    }
  }

  function executeConfirmedAction() {
    if (!confirmAction) return;
    statusMutation.mutate({ id: confirmAction.insight.insight_id, status: confirmAction.status });
  }

  const insights = insightsQ.data?.insights ?? [];
  const total = insightsQ.data?.total ?? 0;
  const allMemos = memosQ.data?.memos ?? [];
  const latestMemo = allMemos[memoIndex];

  // Last-scan timestamp
  const lastScanAt = latestMemo?.created_at ?? (insights.length > 0 ? insights[0].created_at : null);
  const lastScanLabel = lastScanAt
    ? (() => {
        const diffMs = Date.now() - new Date(lastScanAt).getTime();
        const diffMin = Math.floor(diffMs / 60_000);
        const diffHr = Math.floor(diffMin / 60);
        const diffDay = Math.floor(diffHr / 24);
        if (diffMin < 2) return "just now";
        if (diffMin < 60) return `${diffMin} min ago`;
        if (diffHr < 24) return `${diffHr}h ago`;
        return `${diffDay}d ago`;
      })()
    : null;

  const openInsights = insights.filter((i) => i.status !== "resolved");
  const criticalCount = insights.filter((i) => i.severity === "critical").length;

  const sorted = [...insights].sort((a, b) => {
    const so = SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity];
    if (so !== 0) return so;
    return (b.financial_impact_estimate ?? 0) - (a.financial_impact_estimate ?? 0);
  });

  return (
    <>
      {/* Confirm modal */}
      {confirmAction && (
        <ConfirmModal
          action={confirmAction}
          onConfirm={executeConfirmedAction}
          onCancel={() => setConfirmAction(null)}
          isPending={statusMutation.isPending}
        />
      )}

      {/* Auto-accept modal */}
      {showAutoAccept && (
        <AutoAcceptModal
          onConfirm={(minSeverity, dryRun) =>
            autoAcceptMutation.mutate({ min_severity: minSeverity, insight_types: [], dry_run: dryRun })
          }
          onCancel={() => {
            setShowAutoAccept(false);
            setAutoAcceptResult(null);
            autoAcceptMutation.reset();
          }}
          isPending={autoAcceptMutation.isPending}
          result={autoAcceptResult}
        />
      )}

      <div className="space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Brain className="h-6 w-6 text-teal-600 dark:text-teal-400" />
            <div>
              <h2 className="text-xl font-semibold">AI Planner</h2>
              <p className="text-sm text-muted-foreground max-w-2xl">
                Proactive exception work-queue powered by AI. The agent scans your entire portfolio,
                traces causal chains across forecast accuracy, inventory levels, replenishment policies,
                and financial exposure — then generates prioritized, actionable insights ranked by risk.
                {lastScanLabel && (
                  <span className="ml-2 text-xs text-muted-foreground/70">
                    · Last scan: {lastScanLabel}
                  </span>
                )}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="outline"
              onClick={() => { setAutoAcceptResult(null); setShowAutoAccept(true); }}
              disabled={showAutoAccept}
              className="gap-2"
            >
              <Zap className="h-4 w-4" />
              Auto-Accept
            </Button>
            <Button
              onClick={() => scanMutation.mutate()}
              disabled={scanMutation.isPending}
              className="gap-2"
            >
              {scanMutation.isPending ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Scanning portfolio…
                </>
              ) : (
                <>
                  <RefreshCw className="h-4 w-4" />
                  Generate Now
                </>
              )}
            </Button>
          </div>
        </div>

        {/* Distinction banner (PL-004) */}
        <div className="rounded-md border border-sky-200 bg-sky-50 px-4 py-3 text-xs text-sky-700 dark:border-sky-800 dark:bg-sky-950/30 dark:text-sky-300 space-y-1">
          <p><strong>How it works:</strong> The AI agent reads across all data layers — sales history, forecast models, inventory snapshots, EOQ targets, and replenishment policies — to identify exceptions that require planner attention. Each insight traces a causal chain: <em>forecast inaccuracy → inventory consequence → policy mismatch → financial exposure</em>.</p>
          <p><strong>AI Planner vs Exceptions:</strong> This tab shows ML-generated insights that consider cross-dimensional relationships. The <strong>Exceptions</strong> tab shows rule-based threshold alerts from replenishment policies. Both are important — use AI Planner for complex, multi-factor issues and Exceptions for straightforward policy violations.</p>
        </div>

        {/* Portfolio Health Bar */}
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            {
              label: "Open Insights",
              value: statusFilter === "open" ? String(total) : String(openInsights.length),
              color: "text-foreground",
            },
            {
              label: "Critical",
              value: String(criticalCount),
              color: criticalCount > 0 ? "text-red-600 dark:text-red-400" : "text-foreground",
            },
            {
              label: "High Priority",
              value: String(insights.filter((i) => i.severity === "high").length),
              color: "text-orange-600 dark:text-orange-400",
            },
            {
              label: "Total Financial Risk",
              value: fmtCurrency(
                insights.reduce((s, i) => s + (i.financial_impact_estimate ?? 0), 0),
              ),
              color: "text-amber-600 dark:text-amber-400",
            },
          ].map((kpi) => (
            <Card key={kpi.label} className="py-3">
              <CardContent className="px-4">
                <p className="text-xs text-muted-foreground">{kpi.label}</p>
                <p className={cn("text-2xl font-bold", kpi.color)}>{kpi.value}</p>
              </CardContent>
            </Card>
          ))}
        </div>

        {/* Scan success/error banners */}
        {showScanSuccess && (
          <div className="rounded-md bg-green-50 px-4 py-2 text-sm text-green-800 dark:bg-green-900/20 dark:text-green-300 flex items-center gap-2">
            <Loader2 className="h-3.5 w-3.5 animate-spin flex-shrink-0" />
            Portfolio scan in progress — insights will refresh automatically. Track in the <strong>Jobs</strong> tab.
          </div>
        )}
        {!showScanSuccess && scanQueuedAt && (
          <div className="rounded-md border border-blue-200 bg-blue-50 px-4 py-2 text-sm text-blue-800 dark:border-blue-800 dark:bg-blue-950/30 dark:text-blue-300 flex items-center gap-2">
            <Loader2 className="h-3.5 w-3.5 animate-spin flex-shrink-0" />
            Background scan running — checking for new insights every 30s.
            <button className="ml-auto text-xs underline" onClick={() => { setScanQueuedAt(null); if (pollRef.current) clearInterval(pollRef.current); }}>Dismiss</button>
          </div>
        )}
        {scanMutation.isError && (
          <div className="rounded-md bg-red-50 px-4 py-2 text-sm text-red-800 dark:bg-red-900/20 dark:text-red-300">
            Scan failed: {(scanMutation.error as Error)?.message ?? "Unknown error"}
          </div>
        )}

        {/* Filter bar */}
        <div className="flex flex-wrap gap-3">
          <Select value={severityFilter} onValueChange={setSeverityFilter}>
            <SelectTrigger className="w-36">
              <SelectValue placeholder="Severity" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Severities</SelectItem>
              <SelectItem value="critical">Critical</SelectItem>
              <SelectItem value="high">High</SelectItem>
              <SelectItem value="medium">Medium</SelectItem>
              <SelectItem value="low">Low</SelectItem>
            </SelectContent>
          </Select>

          <Select value={statusFilter} onValueChange={setStatusFilter}>
            <SelectTrigger className="w-36">
              <SelectValue placeholder="Status" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="open">Open</SelectItem>
              <SelectItem value="acknowledged">Acknowledged</SelectItem>
              <SelectItem value="resolved">Resolved</SelectItem>
              <SelectItem value="all">All Statuses</SelectItem>
            </SelectContent>
          </Select>

          <Select value={typeFilter} onValueChange={setTypeFilter}>
            <SelectTrigger className="w-44">
              <SelectValue placeholder="Insight Type" />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="all">All Types</SelectItem>
              <SelectItem value="stockout_risk">Stockout Risk</SelectItem>
              <SelectItem value="excess_inventory">Excess Inventory</SelectItem>
              <SelectItem value="forecast_bias">Forecast Bias</SelectItem>
              <SelectItem value="policy_gap">Policy Gap</SelectItem>
              <SelectItem value="champion_degradation">Model Degradation</SelectItem>
            </SelectContent>
          </Select>

          {insightsQ.isFetching && (
            <Loader2 className="h-5 w-5 animate-spin self-center text-muted-foreground" />
          )}
        </div>

        {/* Insight card list */}
        {insightsQ.isLoading ? (
          <div className="flex items-center justify-center py-16 text-muted-foreground">
            <Loader2 className="mr-2 h-5 w-5 animate-spin" />
            Loading insights…
          </div>
        ) : insightsQ.isError ? (
          <div className="rounded-md border border-destructive/30 bg-destructive/10 p-6 text-center text-sm text-destructive">
            Failed to load insights: {(insightsQ.error as Error)?.message}
          </div>
        ) : sorted.length === 0 ? (
          <Card>
            <CardContent className="py-16 text-center">
              <Brain className="mx-auto mb-3 h-10 w-10 text-muted-foreground/40" />
              {statusFilter === "open" && severityFilter === "all" && typeFilter === "all" ? (
                <>
                  <p className="text-sm font-medium text-green-700 dark:text-green-400">
                    Portfolio looks healthy!
                  </p>
                  <p className="mt-1 max-w-md mx-auto text-xs text-muted-foreground">
                    No open exceptions found. The AI agent scanned your portfolio across forecast accuracy,
                    inventory levels, policy assignments, and financial exposure — and found no items requiring
                    immediate planner attention.
                    {lastScanLabel && ` Last scan: ${lastScanLabel}.`}
                    {" "}Click "Generate Now" to run a fresh scan against the latest data.
                  </p>
                </>
              ) : (
                <>
                  <p className="text-sm font-medium text-muted-foreground">No insights match your filters</p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    Adjust the severity, status, or type filters to see more results.
                  </p>
                </>
              )}
            </CardContent>
          </Card>
        ) : (
          <div className="space-y-3">
            {sorted.map((insight) => (
              <InsightCard
                key={insight.insight_id}
                insight={insight}
                selected={selectedIds.has(insight.insight_id)}
                onSelect={handleSelect}
                onAcknowledge={handleAcknowledge}
                onResolve={handleResolve}
                onSnooze={handleSnooze}
              />
            ))}
            {total > sorted.length && (
              <p className="text-center text-xs text-muted-foreground">
                Showing {sorted.length} of {total} insights
              </p>
            )}
          </div>
        )}

        {/* Bulk action bar */}
        {selectedIds.size > 0 && (
          <BulkActionBar
            count={selectedIds.size}
            onAcknowledgeAll={handleBulkAcknowledge}
            onClear={clearSelection}
            isPending={bulkPending}
          />
        )}

        {/* Planning Memo panel */}
        {latestMemo && (
          <Card>
            <CardHeader className="pb-2">
              <div className="flex items-center justify-between">
                <CardTitle className="text-base">Planning Memo</CardTitle>
                <div className="flex items-center gap-2">
                  <CopyButton text={latestMemo.narrative_text} label="Copy markdown" />
                  {latestMemo.model_version && (
                    <Badge variant="outline" className="text-xs">
                      {latestMemo.model_version}
                    </Badge>
                  )}
                  <span className="text-xs text-muted-foreground">
                    {new Date(latestMemo.period).toLocaleDateString("en-US", {
                      year: "numeric",
                      month: "long",
                    })}
                  </span>
                  {/* PL-017: History navigation */}
                  {allMemos.length > 1 && (
                    <div className="flex items-center gap-1 ml-1">
                      <button
                        onClick={() => setMemoIndex((i) => Math.min(i + 1, allMemos.length - 1))}
                        disabled={memoIndex >= allMemos.length - 1}
                        className="rounded p-0.5 text-muted-foreground hover:text-foreground disabled:opacity-30"
                        title="Older memo"
                      >
                        ‹
                      </button>
                      <span className="text-[10px] text-muted-foreground tabular-nums">
                        {memoIndex + 1}/{allMemos.length}
                      </span>
                      <button
                        onClick={() => setMemoIndex((i) => Math.max(i - 1, 0))}
                        disabled={memoIndex === 0}
                        className="rounded p-0.5 text-muted-foreground hover:text-foreground disabled:opacity-30"
                        title="Newer memo"
                      >
                        ›
                      </button>
                    </div>
                  )}
                </div>
              </div>
            </CardHeader>
            <CardContent>
              <div className="prose prose-sm dark:prose-invert max-w-none">
                <pre className="whitespace-pre-wrap rounded bg-muted/50 p-4 text-xs leading-relaxed text-muted-foreground">
                  {latestMemo.narrative_text}
                </pre>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </>
  );
}
