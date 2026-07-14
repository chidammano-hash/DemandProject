import { useEffect, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { GitCompare, Loader2, Play, Search } from "lucide-react";
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  customerForecastKeys,
  fetchCustomerBlendReadiness,
  fetchCustomerBlendSeries,
  fetchLatestCustomerBlend,
  fetchLatestCustomerForecastBacktest,
  generateCustomerBlend,
  generateCustomerForecastBacktest,
  type CustomerBlendSeriesFilters,
} from "@/api/queries/customerForecast";
import { backtestMgmtKeys } from "@/api/queries/backtest-management";
import { fetchJobDetail, jobKeys } from "@/api/queries/jobs";
import { toast } from "@/components/Toaster";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useChartColors } from "@/hooks/useChartColors";
import { formatApiError } from "@/lib/formatApiError";
import { CustomerBlendBacktestPanel } from "./CustomerBlendBacktestPanel";
import { CustomerBlendSeriesTable } from "./CustomerBlendSeriesTable";

const ACTIVE_BACKTEST_STATUSES = new Set(["queued", "generating"]);
const TERMINAL_BACKTEST_STATUSES = new Set(["completed", "failed", "cancelled"]);

function formatMonth(value: string): string {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(value));
}

function formatPercent(value: number | null): string {
  return value == null ? "—" : `${value.toFixed(1)}%`;
}

function formatQuantity(value: number | null): string {
  return value == null
    ? "—"
    : new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(value);
}

function statusVariant(status: string | undefined) {
  if (status === "completed" || status === "ready" || status === "promoted") {
    return "success" as const;
  }
  if (status === "failed" || status === "cancelled" || status === "invalid") {
    return "critical" as const;
  }
  if (status === "queued" || status === "generating") return "info" as const;
  return "outline" as const;
}

export function CustomerBottomUpBlendPanel({ customerRunId }: { customerRunId: string | null }) {
  const queryClient = useQueryClient();
  const { chartColors, okabeIto } = useChartColors();
  const [expectedBacktestRunId, setExpectedBacktestRunId] = useState<string | null>(null);
  const [expectedBlendRunId, setExpectedBlendRunId] = useState<string | null>(null);
  const [blendJobId, setBlendJobId] = useState<string | null>(null);
  const [blendJobFailure, setBlendJobFailure] = useState<string | null>(null);
  const handledTerminalBacktestRef = useRef<string | null>(null);
  const [filters, setFilters] = useState({ item_id: "", location_id: "" });
  const [submittedFilters, setSubmittedFilters] = useState<CustomerBlendSeriesFilters | null>(null);

  const backtestQuery = useQuery({
    queryKey: customerForecastKeys.latestBacktest,
    queryFn: fetchLatestCustomerForecastBacktest,
    refetchInterval: (query) => {
      const latest = query.state.data;
      const awaitingSubmittedRun = Boolean(
        expectedBacktestRunId && latest?.run_id !== expectedBacktestRunId
      );
      return awaitingSubmittedRun || ACTIVE_BACKTEST_STATUSES.has(latest?.status ?? "")
        ? 5_000
        : false;
    },
  });
  const blendQuery = useQuery({
    queryKey: customerForecastKeys.latestBlend,
    queryFn: fetchLatestCustomerBlend,
    refetchInterval: (query) => {
      const latest = query.state.data;
      const awaitingSubmittedRun = Boolean(
        expectedBlendRunId && latest?.run_id !== expectedBlendRunId
      );
      return awaitingSubmittedRun || latest?.status === "generating" ? 5_000 : false;
    },
  });
  const blendReadinessQuery = useQuery({
    queryKey: customerForecastKeys.blendReadiness(customerRunId ?? undefined),
    queryFn: () => fetchCustomerBlendReadiness(customerRunId ?? undefined),
    enabled: Boolean(customerRunId),
  });
  const { refetch: refetchBlendReadiness } = blendReadinessQuery;
  const blendJobQuery = useQuery({
    queryKey: jobKeys.detail(blendJobId),
    queryFn: () => fetchJobDetail(blendJobId as string),
    enabled: Boolean(blendJobId),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "completed" || status === "failed" || status === "cancelled"
        ? false
        : 3_000;
    },
  });

  const backtestMutation = useMutation({
    mutationFn: generateCustomerForecastBacktest,
    onSuccess: async (result) => {
      setExpectedBacktestRunId(result.run_id);
      toast.info("Customer bottom-up backtest queued.");
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: customerForecastKeys.latestBacktest }),
        queryClient.invalidateQueries({
          queryKey: customerForecastKeys.blendReadiness(customerRunId ?? undefined),
        }),
      ]);
    },
    onError: (error) => toast.error(formatApiError(error)),
  });
  const blendMutation = useMutation({
    mutationFn: () => generateCustomerBlend(customerRunId ?? undefined),
    onSuccess: async (result) => {
      setExpectedBlendRunId(result.run_id);
      setBlendJobId(result.job_id);
      setBlendJobFailure(null);
      setSubmittedFilters(null);
      toast.info("Customer blend draft queued.");
      await queryClient.invalidateQueries({ queryKey: customerForecastKeys.latestBlend });
    },
    onError: (error) => toast.error(formatApiError(error)),
  });

  useEffect(() => {
    if (blendJobQuery.isError) {
      const message = formatApiError(blendJobQuery.error);
      setBlendJobFailure(message);
      setExpectedBlendRunId(null);
      setBlendJobId(null);
      toast.error(message);
      return;
    }
    const status = blendJobQuery.data?.status;
    if (status === "failed" || status === "cancelled") {
      const message =
        blendJobQuery.data?.error ||
        `Customer blend draft ${status === "failed" ? "failed" : "was cancelled"}.`;
      setBlendJobFailure(message);
      setExpectedBlendRunId(null);
      setBlendJobId(null);
      toast.error(message);
      return;
    }
    if (status === "completed") {
      void queryClient.invalidateQueries({ queryKey: customerForecastKeys.latestBlend });
      void queryClient.invalidateQueries({
        queryKey: customerForecastKeys.blendReadiness(customerRunId ?? undefined),
      });
      void queryClient.invalidateQueries({ queryKey: customerForecastKeys.blendSeriesAll });
      void queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.stagingSummary });
    }
  }, [
    blendJobQuery.data?.error,
    blendJobQuery.data?.status,
    blendJobQuery.error,
    blendJobQuery.isError,
    customerRunId,
    queryClient,
  ]);

  const seriesQuery = useQuery({
    queryKey: customerForecastKeys.blendSeries(
      submittedFilters ?? { item_id: "", location_id: "" }
    ),
    queryFn: () => fetchCustomerBlendSeries(submittedFilters as CustomerBlendSeriesFilters),
    enabled: submittedFilters !== null,
  });

  const backtest = backtestQuery.data;
  const blend = blendQuery.data;
  useEffect(() => {
    if (!backtest || !customerRunId || backtest.customer_run_id !== customerRunId) {
      handledTerminalBacktestRef.current = null;
      return;
    }
    if (!TERMINAL_BACKTEST_STATUSES.has(backtest.status)) {
      handledTerminalBacktestRef.current = null;
      return;
    }
    const terminalObservation = `${backtest.run_id}:${backtest.status}`;
    if (handledTerminalBacktestRef.current === terminalObservation) return;
    handledTerminalBacktestRef.current = terminalObservation;
    if (expectedBacktestRunId === backtest.run_id) setExpectedBacktestRunId(null);
    void refetchBlendReadiness();
  }, [backtest, customerRunId, expectedBacktestRunId, refetchBlendReadiness]);
  useEffect(() => {
    if (blend?.status === "generating" && blend.job_id && !blendJobId) {
      setBlendJobId(blend.job_id);
    }
  }, [blend?.job_id, blend?.status, blendJobId]);
  useEffect(() => {
    if (expectedBlendRunId && blend?.run_id === expectedBlendRunId) {
      setExpectedBlendRunId(null);
      setBlendJobId(null);
      void queryClient.invalidateQueries({ queryKey: customerForecastKeys.blendSeriesAll });
      void queryClient.invalidateQueries({ queryKey: backtestMgmtKeys.stagingSummary });
    }
  }, [blend?.run_id, expectedBlendRunId, queryClient]);
  const isBlendActive =
    Boolean(expectedBlendRunId && blend?.run_id !== expectedBlendRunId) ||
    blend?.status === "generating";
  const isAwaitingSubmittedBacktest = Boolean(
    expectedBacktestRunId && backtest?.run_id !== expectedBacktestRunId
  );
  const isBacktestActive =
    isAwaitingSubmittedBacktest || ACTIVE_BACKTEST_STATUSES.has(backtest?.status ?? "");
  const hasCurrentCompletedBacktest = Boolean(
    !isAwaitingSubmittedBacktest &&
    customerRunId &&
    backtest?.status === "completed" &&
    backtest.customer_run_id === customerRunId
  );
  const blendReadiness = blendReadinessQuery.data;
  const hasCurrentBlendReadiness = Boolean(
    customerRunId && blendReadiness?.ready && blendReadiness.customer_run_id === customerRunId
  );
  const isBlendViewable = Boolean(
    blend &&
    (blend.status === "ready" || blend.status === "promoted" || blend.status === "archived")
  );
  const filtersComplete = Boolean(filters.item_id.trim() && filters.location_id.trim());
  const fallbackMonths =
    seriesQuery.data?.months.filter((month) => month.coverage_status === "champion_fallback")
      .length ?? 0;
  const tooltipContentStyle = {
    backgroundColor: chartColors.tooltip_bg,
    borderColor: chartColors.tooltip_border,
  };

  function loadComparison() {
    if (!filtersComplete || !blend || !isBlendViewable) return;
    setSubmittedFilters({
      item_id: filters.item_id.trim(),
      location_id: filters.location_id.trim(),
      run_id: blend.run_id,
    });
  }

  return (
    <div className="space-y-4">
      <CustomerBlendBacktestPanel
        backtest={backtest}
        canRun={Boolean(customerRunId)}
        error={backtestQuery.error}
        hasCurrentCompletedBacktest={hasCurrentCompletedBacktest}
        isError={backtestQuery.isError}
        isLoading={backtestQuery.isLoading}
        isRunning={isBacktestActive || backtestMutation.isPending}
        onRun={() => backtestMutation.mutate()}
      />

      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <CardTitle className="flex items-center gap-2">
                <GitCompare className="h-5 w-5" /> Customer Bottom-Up Blend Draft
              </CardTitle>
              <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
                Generate a governed item-location draft from raw customer demand, fulfillment-
                normalized bottom-up demand, and the source champion. Missing customer evidence uses
                champion fallback and customer-only DFUs remain excluded.
              </p>
            </div>
            <Button
              disabled={!hasCurrentBlendReadiness || isBlendActive || blendMutation.isPending}
              onClick={() => blendMutation.mutate()}
            >
              {isBlendActive || blendMutation.isPending ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Play className="mr-2 h-4 w-4" />
              )}
              {isBlendActive ? "Draft Generating" : "Generate Draft"}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {isBacktestActive && !hasCurrentBlendReadiness && (
            <p
              role="status"
              className="rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-sm text-foreground"
            >
              Waiting for the submitted customer backtest to complete.
            </p>
          )}
          {!isBacktestActive && blendReadinessQuery.isLoading && (
            <p role="status" className="text-sm text-muted-foreground">
              Verifying current champion and backtest lineage…
            </p>
          )}
          {blendReadiness && !hasCurrentBlendReadiness && (
            <p
              role="alert"
              className="rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-sm text-foreground"
            >
              {blendReadiness.blockers[0] ?? "Current blend evidence is not ready."}
            </p>
          )}
          {blendReadinessQuery.isError && (
            <p role="alert" className="text-sm text-destructive">
              {formatApiError(blendReadinessQuery.error)}
            </p>
          )}
          {blendJobFailure && (
            <p role="alert" className="text-sm text-destructive">
              {blendJobFailure}
            </p>
          )}
          {blendQuery.isError && (
            <p role="alert" className="text-sm text-destructive">
              {formatApiError(blendQuery.error)}
            </p>
          )}
          {blend?.status === "invalid" && blend.invalid_reason && (
            <p role="alert" className="text-sm text-destructive">
              {blend.invalid_reason}
            </p>
          )}
          {blend && (
            <div className="rounded-md border p-3 text-sm">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant={statusVariant(blend.status)} className="capitalize">
                  {blend.status}
                </Badge>
                <Badge variant={blend.promotion_enabled ? "success" : "outline"}>
                  {blend.promotion_enabled
                    ? "Promotion policy enabled"
                    : "Promotion policy disabled"}
                </Badge>
                <span className="font-medium">
                  {blend.model_id === "customer_bottom_up_blend"
                    ? "Customer Blend"
                    : (blend.model_id ?? "Customer Blend")}
                </span>
              </div>
              <p className="mt-2 text-muted-foreground">
                {blend.blended_row_count.toLocaleString()} blended rows ·{" "}
                {blend.champion_fallback_row_count.toLocaleString()} champion fallback rows ·{" "}
                {blend.customer_only_excluded_count.toLocaleString()} customer-only DFUs excluded
              </p>
              <p className="mt-1 break-all text-xs text-muted-foreground">
                Draft {blend.run_id} · Customer {blend.customer_run_id ?? "—"} · Source production{" "}
                {blend.source_production_run_id ?? "—"}
              </p>
            </div>
          )}

          <div className="grid gap-3 md:grid-cols-[1fr_1fr_auto]">
            <label className="space-y-1 text-sm font-medium">
              <span>Item</span>
              <Input
                aria-label="Blend item"
                value={filters.item_id}
                onChange={(event) =>
                  setFilters((current) => ({ ...current, item_id: event.target.value }))
                }
              />
            </label>
            <label className="space-y-1 text-sm font-medium">
              <span>Location</span>
              <Input
                aria-label="Blend location"
                value={filters.location_id}
                onChange={(event) =>
                  setFilters((current) => ({ ...current, location_id: event.target.value }))
                }
              />
            </label>
            <Button
              className="self-end"
              variant="outline"
              disabled={!filtersComplete || !isBlendViewable || isBlendActive}
              onClick={loadComparison}
            >
              <Search className="mr-2 h-4 w-4" /> Load Blend Comparison
            </Button>
          </div>

          {seriesQuery.isError && (
            <p role="alert" className="text-sm text-destructive">
              {formatApiError(seriesQuery.error)}
            </p>
          )}
          {seriesQuery.data && (
            <>
              {fallbackMonths > 0 && (
                <p
                  role="status"
                  className="rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-sm text-foreground"
                >
                  {fallbackMonths.toLocaleString()} month{fallbackMonths === 1 ? "" : "s"} use the
                  source champion only; no customer signal contributes in those months.
                </p>
              )}
              <div className="rounded-md border p-3">
                <p className="text-sm font-medium">Item-Location Blend Comparison</p>
                <p className="text-xs text-muted-foreground">
                  {`${seriesQuery.data.item_id} at ${seriesQuery.data.location_id} · raw demand and sales-target quantity · ${seriesQuery.data.months.length.toLocaleString()} forecast months`}
                </p>
                <div className="mt-2 h-72" aria-hidden="true">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={seriesQuery.data.months}>
                      <CartesianGrid stroke={chartColors.grid} strokeDasharray="3 3" />
                      <XAxis
                        dataKey="forecast_month"
                        tickFormatter={formatMonth}
                        stroke={chartColors.axis}
                      />
                      <YAxis stroke={chartColors.axis} />
                      <Tooltip
                        contentStyle={tooltipContentStyle}
                        labelFormatter={(value) => formatMonth(String(value))}
                        formatter={(value: number) => formatQuantity(Number(value))}
                      />
                      <Legend />
                      <Line
                        type="monotone"
                        dataKey="raw_customer_demand_qty"
                        name="Raw Customer Demand"
                        stroke={okabeIto[0]}
                        strokeDasharray="2 3"
                        connectNulls={false}
                        dot={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="normalized_customer_qty"
                        name="Normalized Customer Bottom-Up"
                        stroke={okabeIto[1]}
                        strokeDasharray="6 3"
                        connectNulls={false}
                        dot={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="champion_qty"
                        name="Source Champion"
                        stroke={okabeIto[4]}
                        strokeDasharray="8 3"
                        strokeWidth={2}
                        dot={false}
                      />
                      <Line
                        type="monotone"
                        dataKey="blended_qty"
                        name="Customer Blend"
                        stroke={okabeIto[2]}
                        strokeWidth={3}
                        dot={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>

              <CustomerBlendSeriesTable
                months={seriesQuery.data.months}
                formatMonth={formatMonth}
                formatPercent={formatPercent}
                formatQuantity={formatQuantity}
              />
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
