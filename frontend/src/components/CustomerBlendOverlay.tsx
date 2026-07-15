import { Line } from "recharts";

import type { CustomerBlendSeriesMonth } from "@/api/queries/customerForecast";
import { useChartColors } from "@/hooks/useChartColors";
import { CUSTOMER_BLEND_KEYS, type CustomerBlendOverlayStatus } from "@/lib/customer-blend-overlay";

export function CustomerBlendLines({ yAxisId }: { yAxisId?: string | number }) {
  const { chartColors, okabeIto } = useChartColors();
  return (
    <>
      <Line
        type="monotone"
        dataKey={CUSTOMER_BLEND_KEYS.bottomUp}
        yAxisId={yAxisId}
        name="Customer Bottom-Up"
        stroke={okabeIto[1]}
        strokeWidth={2}
        strokeDasharray="8 3 2 3"
        dot={false}
        connectNulls={false}
        activeDot={{ r: 4 }}
        legendType="none"
      />
      <Line
        type="monotone"
        dataKey={CUSTOMER_BLEND_KEYS.blend}
        yAxisId={yAxisId}
        name="Customer Blend"
        stroke={okabeIto[6]}
        strokeWidth={2}
        dot={false}
        connectNulls
        activeDot={{ r: 4 }}
        legendType="none"
      />
      <Line
        type="monotone"
        dataKey={CUSTOMER_BLEND_KEYS.sourceChampion}
        yAxisId={yAxisId}
        name="Source Champion"
        stroke={chartColors.axis}
        strokeWidth={2}
        strokeDasharray="2 3"
        dot={false}
        connectNulls
        activeDot={{ r: 4 }}
        legendType="none"
      />
    </>
  );
}

interface CustomerBlendLegendProps {
  months: CustomerBlendSeriesMonth[];
  status: CustomerBlendOverlayStatus;
  emptyMessage?: string;
  runId?: string | null;
  planningMonth?: string | null;
  stagedSeries?: boolean;
}

function formatMonth(value: string): string {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(value));
}

function formatQuantity(value: number | null): string {
  return value == null
    ? "—"
    : new Intl.NumberFormat("en-US", { maximumFractionDigits: 1 }).format(value);
}

export function CustomerBlendLegend({
  months,
  status,
  emptyMessage = "No customer blend is available for this item and location.",
  runId,
  planningMonth,
  stagedSeries = false,
}: CustomerBlendLegendProps) {
  const { chartColors, okabeIto, roles } = useChartColors();
  if (status === "idle") return null;

  if (status !== "ready") {
    const message = {
      loading: "Loading customer blend…",
      error: "Customer blend could not be loaded.",
      empty: emptyMessage,
    }[status];
    return (
      <p
        className="min-h-7 rounded-md border border-dashed px-3 py-1.5 text-xs text-muted-foreground"
        role="status"
        aria-live="polite"
      >
        {message}
      </p>
    );
  }

  const fallbackCount = months.filter(
    (month) => month.coverage_status === "champion_fallback"
  ).length;
  const series = [
    {
      label: "Customer Bottom-Up",
      color: stagedSeries ? roles.good : okabeIto[1],
      title: "Normalized customer bottom-up forecast",
      lineClassName: "border-dashed",
    },
    { label: "Source Champion", color: chartColors.axis, lineClassName: "border-dotted" },
    {
      label: "Customer Blend",
      color: stagedSeries ? roles.ai : okabeIto[6],
      lineClassName: stagedSeries ? "border-dashed" : "border-solid",
    },
  ];
  const monthLabel = months.length === 1 ? "month" : "months";
  const vintageLabel = planningMonth ? formatMonth(planningMonth) : null;
  const accessibleTableLabel = [
    "Monthly customer forecast blend quantities",
    vintageLabel,
    runId ? `run ${runId}` : null,
  ]
    .filter(Boolean)
    .join(" · ");

  return (
    <>
      <div
        className="flex min-h-7 flex-wrap items-center gap-x-4 gap-y-1 rounded-md border bg-muted/20 px-3 py-1.5 text-xs"
        role="status"
        aria-live="polite"
      >
        <span className="font-semibold text-foreground">Customer forecast blend</span>
        {stagedSeries && (
          <span className="rounded-full bg-primary/10 px-2 py-0.5 font-medium text-primary">
            Staged draft
          </span>
        )}
        <span className="text-muted-foreground">Customer signal normalized to sales</span>
        {vintageLabel && runId && (
          <span
            className="font-mono text-muted-foreground"
            aria-label={`Blend vintage ${vintageLabel}, run ${runId}`}
            title={runId}
          >
            {vintageLabel} · run {runId.slice(0, 8)}
          </span>
        )}
        {series.map((item) => (
          <span key={item.label} className="inline-flex items-center gap-1.5" title={item.title}>
            <span
              className={`inline-block w-4 border-t-2 ${item.lineClassName}`}
              style={{ borderColor: item.color }}
              aria-hidden="true"
            />
            <span>{item.label}</span>
          </span>
        ))}
        <span className="ml-auto rounded-full bg-muted px-2 py-0.5 font-medium text-muted-foreground">
          {fallbackCount > 0
            ? `Champion fallback · ${fallbackCount} of ${months.length} ${monthLabel}`
            : `All ${months.length} ${monthLabel} blended`}
        </span>
      </div>
      <table className="sr-only">
        <caption>{accessibleTableLabel}</caption>
        <thead>
          <tr>
            <th scope="col">Month</th>
            <th scope="col">Customer Bottom-Up</th>
            <th scope="col">Source Champion</th>
            <th scope="col">Customer Blend</th>
            <th scope="col">Coverage</th>
          </tr>
        </thead>
        <tbody>
          {months.map((month) => (
            <tr key={month.forecast_month}>
              <th scope="row">{formatMonth(month.forecast_month)}</th>
              <td>{formatQuantity(month.normalized_customer_qty)}</td>
              <td>{formatQuantity(month.champion_qty)}</td>
              <td>{formatQuantity(month.blended_qty)}</td>
              <td>{month.coverage_status === "blended" ? "Blended" : "Champion fallback"}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
