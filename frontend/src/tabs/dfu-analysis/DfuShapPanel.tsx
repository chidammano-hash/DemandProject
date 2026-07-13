import { useEffect, useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { BrainCircuit } from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/Skeleton";

import { useChartColors } from "@/hooks/useChartColors";
import { fetchSkuShap, fetchShapSummary } from "@/api/queries/core";
import type { SkuShapPayload } from "@/types/shap";
import type { ShapSummaryPayload } from "@/types/shap";
import type { SkuAnalysisMode } from "@/types";

// ---------------------------------------------------------------------------
// Color palette for features (15 distinct colors)
// ---------------------------------------------------------------------------
const SHAP_FEATURE_COLORS = [
  "#2563EB",
  "#0D9488",
  "#D97706",
  "#0891B2",
  "#DC2626",
  "#0284C7",
  "#7C3AED",
  "#059669",
  "#DB2777",
  "#EA580C",
  "#CA8A04",
  "#0E7490",
  "#4F46E5",
  "#16A34A",
  "#B91C1C",
];

function featureColor(idx: number): string {
  return SHAP_FEATURE_COLORS[idx % SHAP_FEATURE_COLORS.length];
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
interface SkuShapPanelProps {
  selectedModel: string | null;
  itemNo: string;
  loc: string;
  customerGroup?: string;
  skuMode: SkuAnalysisMode;
  visibleMonths: string[];
}

// ---------------------------------------------------------------------------
// Fallback: cluster-level aggregate SHAP bar chart
// ---------------------------------------------------------------------------
function FallbackShapChart({
  modelId,
  chartColors,
}: {
  modelId: string;
  chartColors: ReturnType<typeof useChartColors>["chartColors"];
}) {
  const [summary, setSummary] = useState<ShapSummaryPayload | null>(null);

  useEffect(() => {
    let cancelled = false;
    fetchShapSummary(modelId, 10)
      .then((d) => {
        if (!cancelled) setSummary(d);
      })
      .catch(() => {
        /* no summary available */
      });
    return () => {
      cancelled = true;
    };
  }, [modelId]);

  if (!summary) return null;

  const data = summary.features.map((f, i) => ({
    feature: f.feature,
    importance: f.mean_abs_shap_across_timeframes,
    color: featureColor(i),
  }));

  return (
    <div className="space-y-2">
      <p className="text-xs text-amber-600 dark:text-amber-400">
        Showing cluster-level SHAP — model artifacts not available for per-DFU analysis.
      </p>
      <div className="h-[220px] overflow-x-auto">
        <div style={{ minWidth: `${Math.max(400, data.length * 60)}px`, height: "100%" }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart
              data={data}
              layout="vertical"
              margin={{ top: 4, right: 16, left: 120, bottom: 4 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} horizontal={false} />
              <XAxis type="number" tick={{ fill: chartColors.axis, fontSize: 10 }} />
              <YAxis
                type="category"
                dataKey="feature"
                tick={{ fill: chartColors.axis, fontSize: 10 }}
                width={115}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: chartColors.tooltip_bg,
                  borderColor: chartColors.tooltip_border,
                }}
                formatter={(v: number) => [v.toFixed(4), "mean |SHAP|"]}
              />
              <Bar dataKey="importance" radius={[0, 3, 3, 0]}>
                {data.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export function SkuShapPanel({
  selectedModel,
  itemNo,
  loc,
  customerGroup,
  skuMode,
  visibleMonths,
}: SkuShapPanelProps) {
  const { chartColors } = useChartColors();

  const [shapData, setShapData] = useState<SkuShapPayload | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isFallback, setIsFallback] = useState(false);

  useEffect(() => {
    if (!selectedModel || skuMode !== "item_location") {
      setShapData(null);
      setError(null);
      setIsFallback(false);
      return;
    }
    if (!itemNo.trim() || !loc.trim()) {
      setShapData(null);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setShapData(null);
    setError(null);
    setIsFallback(false);

    // Extract raw model_id from production forecast label "Production Forecast (model_id)"
    const modelIdForApi =
      selectedModel.match(/^Production Forecast \((.+)\)$/)?.[1] ?? selectedModel;
    fetchSkuShap(modelIdForApi, itemNo.trim(), loc.trim(), 10, customerGroup)
      .then((data) => {
        if (!cancelled) {
          setShapData(data);
          setLoading(false);
        }
      })
      .catch((err: Error) => {
        if (!cancelled) {
          // 404 → fall back to cluster-level summary
          if (err.message.includes("404") || err.message.includes("No model")) {
            setIsFallback(true);
          } else {
            setError(err.message);
          }
          setLoading(false);
        }
      });
    return () => {
      cancelled = true;
    };
  }, [selectedModel, itemNo, loc, customerGroup, skuMode]);

  // -- No model selected --
  if (!selectedModel) {
    return (
      <Card className="min-w-0 border-dashed border-muted shadow-none">
        <CardContent className="flex h-[80px] items-center justify-center text-sm text-muted-foreground">
          Click a forecast line above to explore SHAP feature contributions per month.
        </CardContent>
      </Card>
    );
  }

  // -- Wrong mode --
  if (skuMode !== "item_location") {
    return (
      <Card className="min-w-0 border-muted shadow-none">
        <CardContent className="flex h-[80px] items-center justify-center text-sm text-muted-foreground">
          Per-DFU SHAP requires single item + location mode.
        </CardContent>
      </Card>
    );
  }

  return (
    <Card className="min-w-0 border-muted shadow-none">
      <CardHeader className="pb-0">
        <CardTitle className="flex items-center gap-2 text-sm">
          <BrainCircuit className="h-4 w-4" />
          SHAP Feature Contributions — {selectedModel}
          <span className="ml-1 text-xs font-normal text-muted-foreground">
            (signed: positive pushes forecast up)
          </span>
          {shapData && (
            <span className="ml-auto text-xs font-normal text-muted-foreground">
              cluster: {shapData.cluster_id}
            </span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="pt-2">
        {loading && (
          <div className="space-y-2 py-2">
            <Skeleton className="h-4 w-3/4" />
            <Skeleton className="h-[220px] w-full" />
          </div>
        )}
        {error && !loading && (
          <div className="py-4 text-center space-y-1">
            <p className="text-sm text-destructive">SHAP computation failed: {error}</p>
            <p className="text-xs text-muted-foreground">
              Ensure model artifacts exist at{" "}
              <code className="bg-muted px-1 rounded">data/models/{selectedModel}/</code>. Run{" "}
              <code className="bg-muted px-1 rounded">make forecast-generate</code> to persist model
              weights. Verify the <code className="bg-muted px-1 rounded">shap</code> library is
              installed for LightGBM analysis.
            </p>
          </div>
        )}
        {isFallback && !loading && (
          <FallbackShapChart modelId={selectedModel} chartColors={chartColors} />
        )}
        {shapData && !loading && !error && (
          <>
            {shapData.future_lag_model_id && shapData.future_lag_model_id !== shapData.model_id && (
              <p className="mb-2 text-xs text-amber-600 dark:text-amber-400">
                Future-month SHAP is approximate — future lags sourced from{" "}
                <strong>{shapData.future_lag_model_id}</strong> (the production champion for this
                DFU), not from {shapData.model_id}.
              </p>
            )}
            <ShapStackedChart
              shapData={shapData}
              visibleMonths={visibleMonths}
              chartColors={chartColors}
            />
          </>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Custom tooltip for the dual-stack SHAP chart
// ---------------------------------------------------------------------------
interface ShapTooltipProps {
  active?: boolean;
  label?: string;
  chartData: Record<string, unknown>[];
  allFeatNames: string[];
  chartColors: ReturnType<typeof useChartColors>["chartColors"];
}

function ShapTooltip({ active, label, chartData, allFeatNames, chartColors }: ShapTooltipProps) {
  if (!active || !label) return null;
  const row = chartData.find((d) => d.__month__ === label);
  if (!row) return null;

  const total = row.__total__ as number | null;
  const base = row.__base__ as number | null;
  const isFuture = row.is_future as boolean;

  const fmt = (v: number) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(1));

  // Collect all signed values, sort by abs descending, suppress near-zero
  const items = [
    ...allFeatNames.map((feat, i) => ({
      name: feat,
      value: row[feat] as number,
      color: featureColor(i),
    })),
    { name: "Other features", value: row.__other__ as number, color: "#94a3b8" },
  ]
    .filter((it) => Math.abs(it.value) >= 0.001)
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  return (
    <div
      style={{
        backgroundColor: chartColors.tooltip_bg,
        border: `1px solid ${chartColors.tooltip_border}`,
        borderRadius: 6,
        padding: "8px 12px",
        fontSize: 11,
        maxHeight: 340,
        overflowY: "auto",
      }}
    >
      <p
        style={{
          fontWeight: 600,
          marginBottom: 6,
          borderBottom: `1px solid ${chartColors.tooltip_border}`,
          paddingBottom: 4,
        }}
      >
        {label}
        {isFuture ? " (future)" : ""}
        {total != null && (
          <>
            <span style={{ fontWeight: 400, marginLeft: 8 }}>Forecast: </span>
            <span style={{ fontWeight: 700 }}>{fmt(total)}</span>
          </>
        )}
        {base != null && (
          <>
            <span style={{ fontWeight: 400, marginLeft: 6 }}>Base: </span>
            <span style={{ fontWeight: 400 }}>{fmt(base)}</span>
          </>
        )}
      </p>
      {items.map((it) => (
        <div
          key={it.name}
          style={{ display: "flex", justifyContent: "space-between", gap: 12, marginBottom: 2 }}
        >
          <span style={{ color: it.color }}>{it.name}</span>
          <span
            style={{
              fontFamily: "monospace",
              color: it.value >= 0 ? "#16a34a" : "#ef4444",
              fontWeight: 600,
            }}
          >
            {it.value >= 0 ? "+" : ""}
            {it.value.toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Stacked bar chart for per-DFU SHAP
// Positive contributions stack ABOVE 0; negative stack BELOW 0.
// Uses dual stackId ("pos" / "neg") so zero is always the visual center.
// ---------------------------------------------------------------------------
function ShapStackedChart({
  shapData,
  visibleMonths,
  chartColors,
}: {
  shapData: SkuShapPayload;
  visibleMonths: string[];
  chartColors: ReturnType<typeof useChartColors>["chartColors"];
}) {
  const shapPointMap = useMemo(
    () => new Map(shapData.points.map((pt) => [pt.month, pt])),
    [shapData.points]
  );

  // Collect all feature names in top-N order (from all points)
  const allFeatNames = useMemo(() => {
    const seen: string[] = [];
    for (const pt of shapData.points)
      for (const f of pt.features) if (!seen.includes(f.name)) seen.push(f.name);
    return seen;
  }, [shapData.points]);

  // Drive X-axis from visibleMonths, but filter to only months that have SHAP data.
  // Without this filter, months outside the 48-month lookback window would render
  // as zero-height bars — misleadingly implying "no contribution" rather than "no data".
  const axisMonths = useMemo(() => {
    const candidates =
      visibleMonths.length > 0 ? visibleMonths : shapData.points.map((p) => p.month);
    return candidates.filter((m) => shapPointMap.has(m));
  }, [visibleMonths, shapData.points, shapPointMap]);

  // Count how many visible months were dropped (for coverage note)
  const droppedMonths = visibleMonths.length > 0 ? visibleMonths.length - axisMonths.length : 0;

  // Chart rows: dual-stack pos/neg split.
  // Keys: "__month__" (X-axis), orig signed values for tooltip,
  //       "${feat}__p" / "${feat}__n" for the two bar stacks.
  // IMPORTANT: "__month__" avoids collision with the "month" SHAP feature (calendar int 1-12).
  const chartData = useMemo(
    () =>
      axisMonths.map((month) => {
        const pt = shapPointMap.get(month);
        const featMap = pt ? Object.fromEntries(pt.features.map((f) => [f.name, f.value])) : {};
        const otherVal = pt?.other_shap ?? 0;
        const row: Record<string, unknown> = {
          __month__: month,
          is_future: pt?.is_future ?? false,
          __other__: otherVal,
          __other__p: Math.max(0, otherVal),
          __other__n: Math.min(0, otherVal),
          __total__: pt
            ? pt.base_value + pt.other_shap + pt.features.reduce((s, f) => s + f.value, 0)
            : null,
          __base__: pt?.base_value ?? null,
        };
        for (const feat of allFeatNames) {
          const v = featMap[feat] ?? 0;
          row[feat] = v;
          row[`${feat}__p`] = Math.max(0, v); // positive stack (above 0)
          row[`${feat}__n`] = Math.min(0, v); // negative stack (below 0)
        }
        return row;
      }),
    [axisMonths, shapPointMap, allFeatNames]
  );

  // Average base value for the header chip
  const avgBase = useMemo(() => {
    const pts = shapData.points.filter((p) => p.base_value != null);
    return pts.length ? pts.reduce((s, p) => s + p.base_value, 0) / pts.length : null;
  }, [shapData.points]);

  if (chartData.length === 0) {
    return (
      <p className="py-4 text-center text-sm text-muted-foreground">
        No data in the current time range.
      </p>
    );
  }

  const fmt = (v: number) => (Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(1)}k` : v.toFixed(1));

  const minWidth = Math.max(1200, axisMonths.length * 100);

  return (
    <div className="space-y-2">
      {/* Summary strip */}
      <div className="flex flex-wrap items-center gap-3 px-1 text-xs text-muted-foreground">
        {avgBase != null && (
          <span className="rounded bg-slate-100 px-2 py-0.5 font-mono dark:bg-slate-800">
            Base avg: <span className="font-semibold text-foreground">{fmt(avgBase)}</span>
          </span>
        )}
        <span className="text-[11px]">
          Zero line = base prediction. Bars above 0 push forecast up; bars below push it down. Bar
          total + base ≈ model forecast (small gaps due to rounding/reconstruction).
        </span>
        {droppedMonths > 0 && (
          <span className="text-[11px] text-amber-600 dark:text-amber-400">
            {droppedMonths} chart month{droppedMonths > 1 ? "s" : ""} outside SHAP lookback window
            (not shown).
          </span>
        )}
      </div>

      {/* Chart */}
      <div className="overflow-x-auto pb-2 [scrollbar-gutter:stable]">
        <div style={{ minWidth: `${minWidth}px`, height: "280px" }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} margin={{ top: 4, right: 16, left: 18, bottom: 4 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
              <XAxis
                dataKey="__month__"
                tick={{ fill: chartColors.axis, fontSize: 10 }}
                interval="preserveStartEnd"
              />
              <YAxis
                tick={{ fill: chartColors.axis, fontSize: 10 }}
                tickFormatter={(v: number) =>
                  Math.abs(v) >= 1000
                    ? `${(v / 1000).toFixed(1)}k`
                    : String(Math.round(v * 10) / 10)
                }
                width={56}
                label={{
                  value: "Δ from base",
                  angle: -90,
                  position: "insideLeft",
                  offset: 10,
                  style: { fill: chartColors.axis, fontSize: 9 },
                }}
              />
              <ReferenceLine y={0} stroke={chartColors.axis} strokeWidth={2} />
              <Tooltip
                content={
                  <ShapTooltip
                    chartData={chartData}
                    allFeatNames={allFeatNames}
                    chartColors={chartColors}
                  />
                }
              />
              <Legend
                wrapperStyle={{ fontSize: 10, paddingTop: 4 }}
                payload={[
                  ...allFeatNames.map((feat, i) => ({
                    value: feat,
                    type: "square" as const,
                    color: featureColor(i),
                  })),
                  { value: "Other features", type: "square" as const, color: "#94a3b8" },
                ]}
              />
              {/* Positive halves of each feature (above 0) */}
              {allFeatNames.map((feat, i) => (
                <Bar
                  key={`${feat}__p`}
                  dataKey={`${feat}__p`}
                  stackId="pos"
                  fill={featureColor(i)}
                  legendType="none"
                  name={feat}
                >
                  {chartData.map((entry, j) => (
                    <Cell
                      key={j}
                      fill={featureColor(i)}
                      fillOpacity={(entry.is_future as boolean) ? 0.4 : 0.88}
                    />
                  ))}
                </Bar>
              ))}
              <Bar
                key="__other__p"
                dataKey="__other__p"
                stackId="pos"
                fill="#94a3b8"
                legendType="none"
                name="Other features"
              >
                {chartData.map((entry, j) => (
                  <Cell
                    key={j}
                    fill="#94a3b8"
                    fillOpacity={(entry.is_future as boolean) ? 0.3 : 0.5}
                  />
                ))}
              </Bar>
              {/* Negative halves of each feature (below 0) */}
              {allFeatNames.map((feat, i) => (
                <Bar
                  key={`${feat}__n`}
                  dataKey={`${feat}__n`}
                  stackId="neg"
                  fill={featureColor(i)}
                  legendType="none"
                  name={feat}
                >
                  {chartData.map((entry, j) => (
                    <Cell
                      key={j}
                      fill={featureColor(i)}
                      fillOpacity={(entry.is_future as boolean) ? 0.4 : 0.88}
                    />
                  ))}
                </Bar>
              ))}
              <Bar
                key="__other__n"
                dataKey="__other__n"
                stackId="neg"
                fill="#94a3b8"
                legendType="none"
                name="Other features"
              >
                {chartData.map((entry, j) => (
                  <Cell
                    key={j}
                    fill="#94a3b8"
                    fillOpacity={(entry.is_future as boolean) ? 0.3 : 0.5}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {shapData.points.some((p) => p.is_future) && (
        <p className="text-center text-[11px] text-muted-foreground">
          Faded bars = future forecast months · hover for signed contributions
        </p>
      )}
    </div>
  );
}
