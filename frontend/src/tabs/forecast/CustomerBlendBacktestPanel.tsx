import { useMemo } from "react";
import { BarChart3, Loader2, Play } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type {
  CustomerForecastBacktest,
  CustomerForecastBacktestMetric,
  CustomerForecastBacktestStatus,
  CustomerForecastComparisonModelId,
} from "@/api/queries/customerForecast";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useChartColors } from "@/hooks/useChartColors";
import { formatApiError } from "@/lib/formatApiError";

const MODEL_LABELS: Record<CustomerForecastComparisonModelId, string> = {
  champion: "Source Champion",
  customer_bottom_up: "Customer Bottom-Up",
  customer_bottom_up_blend: "Customer Blend",
};

const CHART_MODEL_LABELS: Record<CustomerForecastComparisonModelId, string> = {
  champion: "Champion",
  customer_bottom_up: "Customer BU",
  customer_bottom_up_blend: "Blend",
};

interface CustomerBlendBacktestPanelProps {
  backtest: CustomerForecastBacktest | null | undefined;
  canRun: boolean;
  error: unknown;
  hasCurrentCompletedBacktest: boolean;
  isError: boolean;
  isLoading: boolean;
  isRunning: boolean;
  onRun: () => void;
}

function formatPercent(value: number | null): string {
  return value == null ? "—" : `${value.toFixed(1)}%`;
}

function formatQuantity(value: number | null): string {
  return value == null
    ? "—"
    : new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(value);
}

function formatPercentagePoints(value: number | null): string {
  return value == null ? "—" : `${value.toFixed(1)} pp`;
}

function statusVariant(status: CustomerForecastBacktestStatus) {
  if (status === "completed") return "success" as const;
  if (status === "failed" || status === "cancelled") return "critical" as const;
  if (status === "queued" || status === "generating") return "info" as const;
  return "outline" as const;
}

function MetricCard({ metric }: { metric: CustomerForecastBacktestMetric }) {
  return (
    <div className="rounded-md border p-3">
      <p className="text-xs font-medium text-muted-foreground">{MODEL_LABELS[metric.model_id]}</p>
      <p className="mt-1 text-lg font-semibold">{formatPercent(metric.accuracy_pct)} accuracy</p>
      <p className="text-xs text-muted-foreground">
        WAPE {formatPercent(metric.wape_pct)} · Bias {formatPercent(metric.bias_pct)}
      </p>
    </div>
  );
}

export function CustomerBlendBacktestPanel({
  backtest,
  canRun,
  error,
  hasCurrentCompletedBacktest,
  isError,
  isLoading,
  isRunning,
  onRun,
}: CustomerBlendBacktestPanelProps) {
  const { chartColors, okabeIto } = useChartColors();
  const accuracyChartData = useMemo(
    () =>
      (backtest?.metrics ?? []).map((metric) => ({
        model: CHART_MODEL_LABELS[metric.model_id],
        accuracy: metric.accuracy_pct,
        wape: metric.wape_pct,
      })),
    [backtest?.metrics]
  );
  const tooltipContentStyle = {
    backgroundColor: chartColors.tooltip_bg,
    borderColor: chartColors.tooltip_border,
  };

  return (
    <Card>
      <CardHeader>
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <CardTitle className="flex items-center gap-2">
              <BarChart3 className="h-5 w-5" /> Bottom-Up Blend Validation
            </CardTitle>
            <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
              Backtest Croston/SBA customer forecasts against the source champion on one common
              cohort. The normalized customer bottom-up signal must be validated before a blend
              draft can be generated.
            </p>
          </div>
          <Button variant="outline" disabled={!canRun || isRunning} onClick={onRun}>
            {isRunning ? (
              <Loader2 className="mr-2 h-4 w-4 animate-spin" />
            ) : (
              <Play className="mr-2 h-4 w-4" />
            )}
            Run Blend Backtest
          </Button>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {isError && (
          <p role="alert" className="text-sm text-destructive">
            {formatApiError(error)}
          </p>
        )}
        {!isLoading && !backtest && (
          <p className="rounded-md border border-dashed p-3 text-sm text-muted-foreground">
            No customer bottom-up backtest exists yet.
          </p>
        )}
        {backtest && (
          <>
            <div className="flex flex-wrap items-center gap-2 text-sm">
              <Badge variant={statusVariant(backtest.status)} className="capitalize">
                {backtest.status}
              </Badge>
              {hasCurrentCompletedBacktest && <Badge variant="success">Current customer run</Badge>}
              {backtest.gate_passed != null && (
                <Badge variant={backtest.gate_passed ? "success" : "critical"}>
                  {backtest.gate_passed ? "Backtest gate passed" : "Backtest gate blocked"}
                </Badge>
              )}
              <span className="text-muted-foreground">
                {backtest.common_months.toLocaleString()} common months ·{" "}
                {backtest.common_dfus.toLocaleString()} DFUs ·{" "}
                {backtest.common_rows.toLocaleString()} observations
              </span>
            </div>

            {backtest.error_summary && (
              <p role="alert" className="text-sm text-destructive">
                {backtest.error_summary}
              </p>
            )}

            {backtest.status === "completed" && (
              <div className="rounded-md border p-3 text-sm">
                <p className="text-muted-foreground">
                  {`Blend WAPE delta ${formatPercentagePoints(
                    backtest.blend_wape_degradation_pct
                  )} · limit ${formatPercentagePoints(
                    backtest.max_wape_degradation_pct
                  )} · minimum ${backtest.min_common_months?.toLocaleString() ?? "—"} months / ${
                    backtest.min_common_dfus?.toLocaleString() ?? "—"
                  } DFUs`}
                </p>
                {backtest.gate_passed === false && backtest.gate_reason && (
                  <p role="alert" className="mt-1 text-destructive">
                    {backtest.gate_reason}
                  </p>
                )}
              </div>
            )}

            {backtest.metrics.length > 0 && (
              <>
                <div className="grid gap-3 md:grid-cols-3">
                  {backtest.metrics.map((metric) => (
                    <MetricCard key={metric.model_id} metric={metric} />
                  ))}
                </div>

                <div className="rounded-md border p-3">
                  <p className="text-sm font-medium">Common-Cohort Accuracy Comparison</p>
                  <p className="text-xs text-muted-foreground">
                    {`Higher accuracy and lower WAPE are better; every model uses the same ${backtest.common_rows.toLocaleString()} observations.`}
                  </p>
                  <div className="mt-2 h-56" aria-hidden="true">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={accuracyChartData}>
                        <CartesianGrid stroke={chartColors.grid} strokeDasharray="3 3" />
                        <XAxis dataKey="model" stroke={chartColors.axis} />
                        <YAxis
                          stroke={chartColors.axis}
                          domain={[0, "auto"]}
                          tickFormatter={(value: number) => `${value}%`}
                        />
                        <Tooltip
                          contentStyle={tooltipContentStyle}
                          formatter={(value: number) => `${Number(value).toFixed(1)}%`}
                        />
                        <Legend />
                        <Bar dataKey="accuracy" name="Accuracy" fill={okabeIto[4]} />
                        <Bar dataKey="wape" name="WAPE" fill={okabeIto[0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="overflow-x-auto rounded-md border">
                  <table className="w-full text-sm">
                    <caption className="sr-only">Common-cohort model accuracy metrics</caption>
                    <thead className="bg-muted/50 text-left">
                      <tr>
                        <th scope="col" className="px-3 py-2">
                          Model
                        </th>
                        <th scope="col" className="px-3 py-2 text-right">
                          Accuracy
                        </th>
                        <th scope="col" className="px-3 py-2 text-right">
                          WAPE
                        </th>
                        <th scope="col" className="px-3 py-2 text-right">
                          Bias
                        </th>
                        <th scope="col" className="px-3 py-2 text-right">
                          MAE
                        </th>
                        <th scope="col" className="px-3 py-2 text-right">
                          Observations
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {backtest.metrics.map((metric) => (
                        <tr key={metric.model_id} className="border-t">
                          <th scope="row" className="px-3 py-2 text-left font-medium">
                            {MODEL_LABELS[metric.model_id]}
                          </th>
                          <td className="px-3 py-2 text-right">
                            {formatPercent(metric.accuracy_pct)}
                          </td>
                          <td className="px-3 py-2 text-right">{formatPercent(metric.wape_pct)}</td>
                          <td className="px-3 py-2 text-right">{formatPercent(metric.bias_pct)}</td>
                          <td className="px-3 py-2 text-right">{formatQuantity(metric.mae)}</td>
                          <td className="px-3 py-2 text-right">
                            {metric.observations.toLocaleString()}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
}
