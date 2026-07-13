import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import {
  AlertTriangle,
  Ban,
  CheckCircle2,
  Download,
  Loader2,
  Play,
  RefreshCw,
  Search,
  Users,
} from "lucide-react";
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
  cancelCustomerForecastRun,
  customerForecastExportUrl,
  customerForecastKeys,
  fetchCustomerForecastReadiness,
  fetchCustomerForecastSeries,
  fetchLatestCustomerForecastRun,
  generateCustomerForecast,
  type CustomerForecastFilters,
} from "@/api/queries/customerForecast";
import { toast } from "@/components/Toaster";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { useChartColors } from "@/hooks/useChartColors";
import { formatApiError } from "@/lib/formatApiError";

function formatMonth(value: string): string {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(value));
}

function statusVariant(status: string | undefined) {
  if (status === "completed") return "success" as const;
  if (status === "failed" || status === "cancelled") return "critical" as const;
  if (status === "queued" || status === "generating") return "info" as const;
  return "outline" as const;
}

export function CustomerForecastPanel() {
  const queryClient = useQueryClient();
  const { chartColors, trendColors } = useChartColors();
  const [filters, setFilters] = useState<CustomerForecastFilters>({
    item_id: "",
    location_id: "",
    customer_no: "",
  });
  const [submittedFilters, setSubmittedFilters] = useState<CustomerForecastFilters | null>(null);

  const readinessQuery = useQuery({
    queryKey: customerForecastKeys.readiness,
    queryFn: fetchCustomerForecastReadiness,
    staleTime: 30_000,
  });
  const latestRunQuery = useQuery({
    queryKey: customerForecastKeys.latestRun,
    queryFn: () => fetchLatestCustomerForecastRun(),
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      return status === "queued" || status === "generating" ? 5_000 : false;
    },
  });
  const latestRun = latestRunQuery.data;
  const latestCompletedRunQuery = useQuery({
    queryKey: customerForecastKeys.latestCompletedRun,
    queryFn: () => fetchLatestCustomerForecastRun(true),
  });
  const resultRun = latestRun?.status === "completed" ? latestRun : latestCompletedRunQuery.data;
  const isActive = latestRun?.status === "queued" || latestRun?.status === "generating";

  const generateMutation = useMutation({
    mutationFn: generateCustomerForecast,
    onSuccess: async () => {
      toast.info("Customer forecast generation queued.");
      await queryClient.invalidateQueries({ queryKey: customerForecastKeys.latestRun });
    },
    onError: (error) => toast.error(formatApiError(error)),
  });
  const cancelMutation = useMutation({
    mutationFn: cancelCustomerForecastRun,
    onSuccess: async () => {
      toast.info("Customer forecast cancellation requested.");
      await queryClient.invalidateQueries({ queryKey: customerForecastKeys.latestRun });
    },
    onError: (error) => toast.error(formatApiError(error)),
  });

  const seriesQuery = useQuery({
    queryKey: customerForecastKeys.series(
      submittedFilters ?? { item_id: "", location_id: "", customer_no: "" }
    ),
    queryFn: () => fetchCustomerForecastSeries(submittedFilters as CustomerForecastFilters),
    enabled: submittedFilters !== null,
  });

  const chartData = useMemo(() => {
    const points = new Map<string, { month: string; actual_qty?: number; forecast_qty?: number }>();
    for (const row of seriesQuery.data?.history ?? []) {
      points.set(row.month, { month: row.month, actual_qty: row.actual_qty });
    }
    for (const row of seriesQuery.data?.forecast ?? []) {
      points.set(row.month, {
        ...(points.get(row.month) ?? { month: row.month }),
        forecast_qty: row.forecast_qty,
      });
    }
    return [...points.values()].sort((left, right) => left.month.localeCompare(right.month));
  }, [seriesQuery.data]);

  const readiness = readinessQuery.data;
  const filtersComplete = Boolean(filters.item_id && filters.location_id && filters.customer_no);
  const exportFilters = resultRun ? { ...filters, run_id: resultRun.run_id } : filters;

  return (
    <div className="space-y-4">
      <Card>
        <CardHeader>
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div>
              <CardTitle className="flex items-center gap-2">
                <Users className="h-5 w-5" /> Customer Forecast Generation
              </CardTitle>
              <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
                Generate customer-level demand with Chronos 2E. Results are read-only and never
                adjust, reconcile, stage, promote, or replace the item-location champion.
              </p>
            </div>
            <div className="flex gap-2">
              {isActive && latestRun?.job_id && (
                <Button
                  variant="outline"
                  disabled={cancelMutation.isPending}
                  onClick={() => cancelMutation.mutate(latestRun.run_id)}
                >
                  <Ban className="mr-2 h-4 w-4" /> Cancel
                </Button>
              )}
              <Button
                disabled={!readiness?.ready || isActive || generateMutation.isPending}
                onClick={() => generateMutation.mutate()}
              >
                {isActive || generateMutation.isPending ? (
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                ) : latestRun?.status === "failed" || latestRun?.status === "cancelled" ? (
                  <RefreshCw className="mr-2 h-4 w-4" />
                ) : (
                  <Play className="mr-2 h-4 w-4" />
                )}
                {isActive || generateMutation.isPending
                  ? "Generation Running"
                  : latestRun?.status === "failed" || latestRun?.status === "cancelled"
                    ? "Retry Generation"
                    : latestRun?.status === "completed"
                      ? "Generate Again"
                      : "Generate Customer Forecasts"}
              </Button>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {readinessQuery.isLoading ? (
            <p className="text-sm text-muted-foreground">Checking customer-demand readiness…</p>
          ) : readinessQuery.isError ? (
            <p className="text-sm text-destructive">
              Readiness could not be checked: {formatApiError(readinessQuery.error)}
            </p>
          ) : readiness ? (
            <>
              <div className="grid gap-3 md:grid-cols-3">
                <WindowCard
                  label="Historical input"
                  value={`${formatMonth(readiness.history_start)} – ${formatMonth(readiness.history_end)}`}
                  detail={`${readiness.history_months} fully closed months`}
                />
                <WindowCard
                  label="Forecast output"
                  value={`${formatMonth(readiness.forecast_start)} – ${formatMonth(readiness.forecast_end)}`}
                  detail={`${readiness.horizon_months} future months`}
                />
                <WindowCard
                  label="Coverage"
                  value={`${readiness.eligible_series.toLocaleString()} eligible series`}
                  detail={`${readiness.skipped_series.toLocaleString()} series currently ineligible`}
                />
              </div>
              <div className="flex items-start gap-2 rounded-md border p-3">
                {readiness.ready ? (
                  <CheckCircle2 className="mt-0.5 h-4 w-4 text-emerald-600" />
                ) : (
                  <AlertTriangle className="mt-0.5 h-4 w-4 text-amber-600" />
                )}
                <div>
                  <p className="text-sm font-medium">
                    {readiness.ready ? "Ready to generate" : "Action required"}
                  </p>
                  {readiness.blockers.map((blocker) => (
                    <p key={blocker} className="text-sm text-muted-foreground">
                      {blocker}
                    </p>
                  ))}
                </div>
              </div>
            </>
          ) : null}
        </CardContent>
      </Card>

      {latestRun && (
        <Card>
          <CardContent className="flex flex-wrap items-center justify-between gap-3 p-4">
            <div>
              <p className="text-sm font-medium">Latest run · {latestRun.run_id}</p>
              <p className="text-xs text-muted-foreground">
                {latestRun.row_count.toLocaleString()} rows ·{" "}
                {latestRun.eligible_series.toLocaleString()} series
              </p>
            </div>
            <Badge variant={statusVariant(latestRun.status)} className="capitalize">
              {latestRun.status}
            </Badge>
            {latestRun.error_summary && (
              <p className="w-full text-sm text-destructive">{latestRun.error_summary}</p>
            )}
            {Object.keys(latestRun.skip_reason_counts).length > 0 && (
              <p className="w-full text-xs text-muted-foreground">
                Skipped:{" "}
                {Object.entries(latestRun.skip_reason_counts)
                  .map(([reason, count]) => `${reason.split("_").join(" ")} (${count})`)
                  .join(", ")}
              </p>
            )}
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>View Customer Forecast</CardTitle>
          <p className="text-sm text-muted-foreground">
            Enter one exact item, location, and customer key from the completed run.
          </p>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid gap-3 md:grid-cols-3">
            <FilterInput
              label="Item"
              value={filters.item_id}
              onChange={(value) => setFilters((current) => ({ ...current, item_id: value }))}
            />
            <FilterInput
              label="Location"
              value={filters.location_id}
              onChange={(value) => setFilters((current) => ({ ...current, location_id: value }))}
            />
            <FilterInput
              label="Customer"
              value={filters.customer_no}
              onChange={(value) => setFilters((current) => ({ ...current, customer_no: value }))}
            />
          </div>
          <div className="flex gap-2">
            <Button
              variant="outline"
              disabled={!filtersComplete || !resultRun}
              onClick={() => setSubmittedFilters({ ...filters, run_id: resultRun?.run_id })}
            >
              <Search className="mr-2 h-4 w-4" /> Load Series
            </Button>
            {resultRun && (
              <Button variant="outline" asChild>
                <a href={customerForecastExportUrl(exportFilters)}>
                  <Download className="mr-2 h-4 w-4" /> Export CSV
                </a>
              </Button>
            )}
          </div>

          {seriesQuery.isError && (
            <p className="text-sm text-destructive">{formatApiError(seriesQuery.error)}</p>
          )}
          {seriesQuery.data && (
            <>
              <div className="h-72 rounded-md border p-3">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={chartData}>
                    <CartesianGrid stroke={chartColors.grid} strokeDasharray="3 3" />
                    <XAxis dataKey="month" tickFormatter={formatMonth} stroke={chartColors.axis} />
                    <YAxis stroke={chartColors.axis} />
                    <Tooltip labelFormatter={(value) => formatMonth(String(value))} />
                    <Legend />
                    <Line
                      type="monotone"
                      dataKey="actual_qty"
                      name="Actual demand"
                      stroke={trendColors[0]}
                      connectNulls
                    />
                    <Line
                      type="monotone"
                      dataKey="forecast_qty"
                      name="Customer forecast"
                      stroke={trendColors[1]}
                      connectNulls
                    />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="overflow-x-auto rounded-md border">
                <table className="w-full text-sm">
                  <thead className="bg-muted/50 text-left">
                    <tr>
                      <th className="px-3 py-2">Month</th>
                      <th className="px-3 py-2 text-right">Actual demand</th>
                      <th className="px-3 py-2 text-right">Customer forecast</th>
                    </tr>
                  </thead>
                  <tbody>
                    {chartData.map((row) => (
                      <tr key={row.month} className="border-t">
                        <td className="px-3 py-2">{formatMonth(row.month)}</td>
                        <td className="px-3 py-2 text-right">
                          {row.actual_qty?.toLocaleString() ?? "—"}
                        </td>
                        <td className="px-3 py-2 text-right">
                          {row.forecast_qty?.toLocaleString() ?? "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          )}
        </CardContent>
      </Card>
    </div>
  );
}

function WindowCard({ label, value, detail }: { label: string; value: string; detail: string }) {
  return (
    <div className="rounded-md border p-3">
      <p className="text-xs uppercase text-muted-foreground">{label}</p>
      <p className="mt-1 font-medium">{value}</p>
      <p className="text-xs text-muted-foreground">{detail}</p>
    </div>
  );
}

function FilterInput({
  label,
  value,
  onChange,
}: {
  label: string;
  value: string;
  onChange: (value: string) => void;
}) {
  return (
    <label className="space-y-1 text-sm font-medium">
      <span>{label}</span>
      <Input aria-label={label} value={value} onChange={(event) => onChange(event.target.value)} />
    </label>
  );
}
