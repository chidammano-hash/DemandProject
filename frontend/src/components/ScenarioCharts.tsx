import { memo, useMemo } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  ComposedChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  BarChart,
  Bar,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ReferenceLine,
  Cell,
  PieChart,
  Pie,
  ScatterChart,
  Scatter,
  ZAxis,
} from "recharts";
import { ModularReactECharts } from "@/components/echarts-modular";
import { useChartColors } from "@/hooks/useChartColors";
import type { ClusteringScenarioResult, PCAScatterData } from "@/api/queries";
import { formatNumber, formatCompactNumber, formatClusterLabel } from "@/lib/formatters";

// Maximum number of PCA scatter points rendered on the canvas.
// Beyond this the tooltip hit-test degrades and first-paint stalls.
// TODO(P0-3): push topN/point-cap server-side so the payload itself is bounded.
const PCA_RENDER_CAP = 5_000;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const RADAR_LABELS: Record<string, string> = {
  cv_demand: "CV",
  seasonality_strength: "Seasonal",
  trend_slope: "Trend",
  growth_rate: "Growth",
  zero_demand_pct: "Zero %",
};

export function silhouetteQuality(score: number): string {
  if (score >= 0.71) return "Strong";
  if (score >= 0.51) return "Reasonable";
  if (score >= 0.26) return "Weak";
  return "No structure";
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
interface ScenarioChartsProps {
  result: NonNullable<ClusteringScenarioResult["result"]>;
  pcaScatter?: PCAScatterData;
}

export const ScenarioCharts = memo(function ScenarioCharts({ result, pcaScatter }: ScenarioChartsProps) {
  const { series, fallback, roles, chartColors } = useChartColors();

  // Shared per-cluster categorical ramp (series + fallback = 14 distinguishable
  // hues) so the same cluster index reads as the same color across the PCA
  // scatter, pie, radar, and separation-scatter charts below.
  const categoricalColors = useMemo(() => [...series, ...fallback], [series, fallback]);

  const kSel = result.k_selection_results;
  const kData = kSel.k_values.map((k, i) => ({
    k,
    inertia: kSel.inertias[i],
    silhouette: kSel.silhouette_scores[i],
    feasible: kSel.feasible_mask?.[i] ?? true,
    ...(kSel.ch_scores?.[i] != null ? { ch: kSel.ch_scores[i] } : {}),
    ...(kSel.combined_scores?.[i] != null ? { combined: kSel.combined_scores[i] } : {}),
  }));

  const sizeData = result.profiles.map((p) => ({
    label: p.label,
    count: p.count,
    pct: p.pct_of_total,
  }));

  const radarKeys = ["cv_demand", "seasonality_strength", "trend_slope", "growth_rate", "zero_demand_pct"] as const;
  const radarData = radarKeys.map((key) => {
    const entry: Record<string, string | number> = { feature: key };
    for (const p of result.profiles) {
      entry[p.label] = Math.abs(p[key]);
    }
    return entry;
  });

  const featureImportance = (result.feature_importance ?? [])
    .slice(0, 10)
    .map((f) => ({ ...f, pct: Math.round(f.variance_ratio * 100) }));

  const hasCombinedScores = result.k_selection_results.combined_scores && result.k_selection_results.combined_scores.length > 0;

  const hasCH = kData.some((d) => d.ch != null);

  // ── PCA scatter: canvas ECharts series (one series per cluster) ──────────
  // Points are capped at PCA_RENDER_CAP before building series; excess points
  // are sampled uniformly to keep the distribution representative.
  const { pcaEChartsOption, pcaRenderedCount, pcaTotalCount } = useMemo(() => {
    if (!pcaScatter || pcaScatter.points.length === 0) {
      return { pcaEChartsOption: null, pcaRenderedCount: 0, pcaTotalCount: 0 };
    }

    const allPoints = pcaScatter.points;
    const totalCount = allPoints.length;

    // Uniform sampling when over cap: pick every Nth point.
    const points =
      totalCount <= PCA_RENDER_CAP
        ? allPoints
        : allPoints.filter((_, i) => i % Math.ceil(totalCount / PCA_RENDER_CAP) === 0).slice(0, PCA_RENDER_CAP);

    const renderedCount = points.length;

    // Group by cluster id for separate series (preserves legend + per-cluster color).
    const byCluster = new Map<number, Array<[number, number]>>();
    for (const pt of points) {
      const arr = byCluster.get(pt.cluster) ?? [];
      arr.push([pt.pc1, pt.pc2]);
      byCluster.set(pt.cluster, arr);
    }

    const series = Array.from(byCluster.entries())
      .sort(([a], [b]) => a - b)
      .map(([cluster, pts]) => ({
        name: `Cluster ${cluster}`,
        type: "scatter" as const,
        // Canvas scatter: O(1) hit-test via ECharts spatial index, not O(n) SVG.
        large: true,
        largeThreshold: 2000,
        data: pts,
        symbolSize: 5,
        itemStyle: {
          // Per-cluster color from the shared categorical ramp — no inline hex.
          color: categoricalColors[cluster % categoricalColors.length],
          opacity: 0.65,
        },
      }));

    const option = {
      tooltip: {
        trigger: "item" as const,
        formatter: (p: { seriesName: string; value: [number, number] }) =>
          `<b>${p.seriesName}</b><br/>PC1: ${p.value[0].toFixed(3)}<br/>PC2: ${p.value[1].toFixed(3)}`,
      },
      legend: {
        type: "scroll" as const,
        right: 10,
        top: 0,
        orient: "vertical" as const,
        textStyle: { fontSize: 10 },
      },
      grid: { left: 60, right: 120, top: 20, bottom: 50 },
      xAxis: {
        type: "value" as const,
        name: `PC1 (${pcaScatter.pc1_variance}% variance)`,
        nameLocation: "center" as const,
        nameGap: 30,
        axisLabel: { fontSize: 10 },
        splitLine: { lineStyle: { color: chartColors.grid } },
        axisLine: { lineStyle: { color: chartColors.axis } },
      },
      yAxis: {
        type: "value" as const,
        name: `PC2 (${pcaScatter.pc2_variance}% variance)`,
        nameLocation: "center" as const,
        nameGap: 40,
        axisLabel: { fontSize: 10 },
        splitLine: { lineStyle: { color: chartColors.grid } },
        axisLine: { lineStyle: { color: chartColors.axis } },
      },
      series,
    };

    return { pcaEChartsOption: option, pcaRenderedCount: renderedCount, pcaTotalCount: totalCount };
  }, [pcaScatter, categoricalColors, chartColors]);

  return (
    <div className="mt-4 space-y-6">
      {/* ── K Selection 3-Panel ── */}
      <div>
        <p className="mb-2 text-sm font-semibold text-foreground">K Selection (Elbow / Silhouette / Calinski-Harabasz)</p>
        <div className="grid gap-3 grid-cols-1 md:grid-cols-3">
          {/* Elbow Method */}
          <div className="rounded-md border border-input bg-background p-2">
            <p className="mb-1 text-[11px] font-medium text-muted-foreground text-center">Elbow Method</p>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={kData} margin={{ top: 5, right: 15, bottom: 5, left: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="k" tick={{ fontSize: 10 }} label={{ value: "Number of Clusters (K)", position: "insideBottom", offset: -2, fontSize: 10 }} />
                <YAxis tick={{ fontSize: 9 }} tickFormatter={(v) => formatCompactNumber(v)} label={{ value: "WCSS", angle: -90, position: "insideLeft", fontSize: 10 }} />
                <Tooltip
                  formatter={(value: number) => [formatCompactNumber(value), "WCSS"]}
                  labelFormatter={(k) => `K = ${k}${Number(k) === result.optimal_k ? " (Optimal)" : ""}`}
                />
                <ReferenceLine
                  x={result.optimal_k}
                  stroke={roles.error}
                  strokeDasharray="5 5"
                  label={{ value: `K=${result.optimal_k}`, position: "top", fill: roles.error, fontSize: 10 }}
                />
                <Line type="monotone" dataKey="inertia" stroke={roles.forecast} strokeWidth={2} dot={{ r: 3 }} name="WCSS" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Silhouette Score */}
          <div className="rounded-md border border-input bg-background p-2">
            <p className="mb-1 text-[11px] font-medium text-muted-foreground text-center">
              Silhouette Score <span className="text-[9px] text-destructive">(red=infeasible &lt;5%)</span>
            </p>
            <ResponsiveContainer width="100%" height={200}>
              <ComposedChart data={kData} margin={{ top: 5, right: 15, bottom: 5, left: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="k" tick={{ fontSize: 10 }} label={{ value: "Number of Clusters (K)", position: "insideBottom", offset: -2, fontSize: 10 }} />
                <YAxis tick={{ fontSize: 9 }} label={{ value: "Silhouette Score", angle: -90, position: "insideLeft", fontSize: 10 }} />
                <Tooltip
                  formatter={(value: number) => [value.toFixed(4), "Silhouette"]}
                  labelFormatter={(k) => `K = ${k}${Number(k) === result.optimal_k ? " (Optimal)" : ""}`}
                />
                <ReferenceLine
                  x={result.optimal_k}
                  stroke={roles.error}
                  strokeDasharray="5 5"
                  label={{ value: `K=${result.optimal_k}`, position: "top", fill: roles.error, fontSize: 10 }}
                />
                <Bar dataKey="silhouette" name="Silhouette">
                  {kData.map((entry, i) => (
                    <Cell
                      key={i}
                      fill={entry.feasible ? roles.good : roles.error}
                      fillOpacity={entry.feasible ? 1 : 0.45}
                      stroke={entry.k === result.optimal_k ? roles.forecast : undefined}
                      strokeWidth={entry.k === result.optimal_k ? 2 : 0}
                    />
                  ))}
                </Bar>
                <Line type="monotone" dataKey="silhouette" stroke={roles.forecast} strokeWidth={1.5} dot={{ r: 2 }} name="Trend" />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          {/* Calinski-Harabasz Score */}
          <div className="rounded-md border border-input bg-background p-2">
            <p className="mb-1 text-[11px] font-medium text-muted-foreground text-center">
              Calinski-Harabasz Score <span className="text-[9px]">(higher = better separation)</span>
            </p>
            <ResponsiveContainer width="100%" height={200}>
              <ComposedChart data={kData} margin={{ top: 5, right: 15, bottom: 5, left: 5 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="k" tick={{ fontSize: 10 }} label={{ value: "Number of Clusters (K)", position: "insideBottom", offset: -2, fontSize: 10 }} />
                <YAxis tick={{ fontSize: 9 }} tickFormatter={(v) => formatCompactNumber(v)} label={{ value: "CH Score", angle: -90, position: "insideLeft", fontSize: 10 }} />
                <Tooltip
                  formatter={(value: number) => [formatCompactNumber(value), "CH Score"]}
                  labelFormatter={(k) => `K = ${k}${Number(k) === result.optimal_k ? " (Optimal)" : ""}`}
                />
                <ReferenceLine
                  x={result.optimal_k}
                  stroke={roles.error}
                  strokeDasharray="5 5"
                  label={{ value: `K=${result.optimal_k}`, position: "top", fill: roles.error, fontSize: 10 }}
                />
                {hasCH ? (
                  <>
                    <Bar dataKey="ch" name="CH Score">
                      {kData.map((entry, i) => (
                        <Cell
                          key={i}
                          fill={entry.feasible ? roles.good : roles.error}
                          fillOpacity={entry.feasible ? 1 : 0.45}
                          stroke={entry.k === result.optimal_k ? roles.forecast : undefined}
                          strokeWidth={entry.k === result.optimal_k ? 2 : 0}
                        />
                      ))}
                    </Bar>
                    <Line type="monotone" dataKey="ch" stroke={roles.forecast} strokeWidth={1.5} dot={{ r: 2 }} name="Trend" />
                  </>
                ) : (
                  <Bar dataKey="silhouette" name="N/A" fill={chartColors.grid} />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* ── PCA 2D Scatter — canvas ECharts (replaces recharts SVG, P0-4) ── */}
      {pcaEChartsOption && (
        <div>
          <div className="mb-2 flex items-baseline gap-2">
            <p className="text-sm font-semibold text-foreground">Cluster Visualization (2D PCA)</p>
            {pcaTotalCount > pcaRenderedCount && (
              <span className="text-[10px] text-muted-foreground">
                showing {pcaRenderedCount.toLocaleString()} of {pcaTotalCount.toLocaleString()} points
              </span>
            )}
          </div>
          <div className="rounded-md border border-input bg-background p-3" role="img" aria-label="PCA cluster scatter chart">
            <ModularReactECharts
              option={pcaEChartsOption}
              style={{ height: 400 }}
              notMerge
            />
          </div>
        </div>
      )}

      {/* ── Existing Charts (unchanged) ── */}
      <div className="grid gap-4 md:grid-cols-2">
      {/* Cluster size distribution — Pie chart */}
      <div>
        <p className="mb-1 text-xs font-semibold text-muted-foreground">Cluster Size Distribution</p>
        <ResponsiveContainer width="100%" height={220}>
          <PieChart>
            <Pie
              data={sizeData}
              dataKey="count"
              nameKey="label"
              cx="50%"
              cy="50%"
              outerRadius={80}
              label={({ label, pct }) => `${formatClusterLabel(label)} (${pct.toFixed(0)}%)`}
              labelLine={{ strokeWidth: 1 }}
            >
              {sizeData.map((_, i) => (
                <Cell key={i} fill={categoricalColors[i % categoricalColors.length]} />
              ))}
            </Pie>
            <Tooltip formatter={(value: number, name: string) => [formatNumber(value), formatClusterLabel(name)]} />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Radar profile */}
      <div className="md:col-span-2">
        <p className="mb-1 text-xs font-semibold text-muted-foreground">Cluster Profile Radar</p>
        <ResponsiveContainer width="100%" height={320}>
          <RadarChart data={radarData} cx="50%" cy="45%" outerRadius="65%">
            <PolarGrid />
            <PolarAngleAxis
              dataKey="feature"
              tick={{ fontSize: 10 }}
              tickFormatter={(v: string) => RADAR_LABELS[v] ?? v}
            />
            <PolarRadiusAxis tick={{ fontSize: 8 }} />
            <Tooltip
              labelFormatter={(v: string) => RADAR_LABELS[v] ?? v}
              formatter={(value: number, name: string) => [value.toFixed(3), formatClusterLabel(name)]}
            />
            {result.profiles.map((p, i) => (
              <Radar
                key={p.label}
                name={p.label}
                dataKey={p.label}
                stroke={categoricalColors[i % categoricalColors.length]}
                fill={categoricalColors[i % categoricalColors.length]}
                fillOpacity={0.12}
              />
            ))}
            <Legend
              verticalAlign="bottom"
              wrapperStyle={{ fontSize: 10, paddingTop: 8 }}
              formatter={(value: string) => formatClusterLabel(value)}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Cluster separation scatter — CV vs Mean Demand, bubble = count */}
      <div className="md:col-span-2">
        <p className="mb-1 text-xs font-semibold text-muted-foreground">
          Cluster Separation (CV vs Mean Demand)
        </p>
        <ResponsiveContainer width="100%" height={300}>
          <ScatterChart margin={{ top: 10, right: 30, bottom: 10, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="mean_demand"
              name="Mean Demand"
              tick={{ fontSize: 10 }}
              label={{ value: "Mean Demand", position: "insideBottom", offset: -2, fontSize: 10 }}
            />
            <YAxis
              type="number"
              dataKey="cv_demand"
              name="CV"
              tick={{ fontSize: 10 }}
              label={{ value: "CV Demand", angle: -90, position: "insideLeft", fontSize: 10 }}
            />
            <ZAxis type="number" dataKey="count" range={[80, 800]} name="DFUs" />
            <Tooltip
              cursor={{ strokeDasharray: "3 3" }}
              formatter={(value: number, name: string) => [
                name === "DFUs" ? value.toLocaleString() : value.toFixed(3),
                name,
              ]}
              labelFormatter={() => ""}
            />
            {result.profiles.map((p, i) => (
              <Scatter
                key={p.label}
                name={formatClusterLabel(p.label)}
                data={[{ mean_demand: p.mean_demand, cv_demand: p.cv_demand, count: p.count, label: p.label }]}
                fill={categoricalColors[i % categoricalColors.length]}
                stroke={categoricalColors[i % categoricalColors.length]}
                strokeWidth={2}
                fillOpacity={0.7}
              />
            ))}
            <Legend
              verticalAlign="bottom"
              wrapperStyle={{ fontSize: 10, paddingTop: 4 }}
            />
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Feature importance (horizontal bar) */}
      {featureImportance.length > 0 && (
        <div>
          <p className="mb-1 text-xs font-semibold text-muted-foreground">Feature Importance (Variance Ratio)</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={featureImportance} layout="vertical" margin={{ left: 80 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis type="number" domain={[0, "auto"]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
              <YAxis type="category" dataKey="feature" width={75} tick={{ fontSize: 10 }} />
              <Tooltip formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, "Importance"]} />
              <Bar dataKey="variance_ratio" fill={roles.ceiling} name="Importance">
                {featureImportance.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? roles.forecast : roles.reference} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Combined score chart (Silhouette + CH, conditional) */}
      {hasCombinedScores && (
        <div>
          <p className="mb-1 text-xs font-semibold text-muted-foreground">Combined Score (0.5×Sil + 0.5×CH)</p>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={kData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="k" />
              <YAxis domain={[-0.1, 1]} />
              <Tooltip
                formatter={(value: number) => [value.toFixed(4), "Combined"]}
                labelFormatter={(k) => `K = ${k}${Number(k) === result.optimal_k ? " (Optimal)" : ""}`}
              />
              <ReferenceLine
                x={result.optimal_k}
                stroke={roles.error}
                strokeDasharray="5 5"
                label={{ value: `K=${result.optimal_k}`, position: "top", fill: roles.error, fontSize: 11 }}
              />
              <Bar dataKey="combined" name="Combined Score">
                {kData.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={
                      entry.k === result.optimal_k
                        ? roles.forecast
                        : (entry.combined ?? 0) < 0
                          ? roles.error
                          : roles.good
                    }
                    fillOpacity={entry.k === result.optimal_k ? 1 : 0.5}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
      </div>
    </div>
  );
});
