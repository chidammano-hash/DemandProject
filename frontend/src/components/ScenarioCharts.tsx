import { memo } from "react";
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
import type { ClusteringScenarioResult, PCAScatterData } from "@/api/queries";
import { formatNumber, formatCompactNumber, formatClusterLabel } from "@/lib/formatters";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const PIE_COLORS = [
  "#2563EB", "#f59e0b", "#10b981", "#ef4444", "#0891B2",
  "#06b6d4", "#ec4899", "#14b8a6", "#f97316", "#84cc16",
];

const RADAR_LABELS: Record<string, string> = {
  cv_demand: "CV",
  seasonality_strength: "Seasonal",
  trend_slope: "Trend",
  growth_rate: "Growth",
  zero_demand_pct: "Zero %",
};

function silhouetteColor(score: number): string {
  if (score >= 0.71) return "#22c55e";
  if (score >= 0.51) return "#3b82f6";
  if (score >= 0.26) return "#eab308";
  return "#ef4444";
}

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
  const radarColors = ["#2563EB", "#059669", "#D97706", "#0891B2", "#DC2626", "#EA580C"];

  const featureImportance = (result.feature_importance ?? [])
    .slice(0, 10)
    .map((f) => ({ ...f, pct: Math.round(f.variance_ratio * 100) }));

  const hasCombinedScores = result.k_selection_results.combined_scores && result.k_selection_results.combined_scores.length > 0;

  const hasCH = kData.some((d) => d.ch != null);

  // Group PCA scatter points by cluster for coloring
  const pcaByCluster = new Map<number, { pc1: number; pc2: number }[]>();
  if (pcaScatter) {
    for (const pt of pcaScatter.points) {
      const arr = pcaByCluster.get(pt.cluster) ?? [];
      arr.push({ pc1: pt.pc1, pc2: pt.pc2 });
      pcaByCluster.set(pt.cluster, arr);
    }
  }

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
                  stroke="#ef4444"
                  strokeDasharray="5 5"
                  label={{ value: `K=${result.optimal_k}`, position: "top", fill: "#ef4444", fontSize: 10 }}
                />
                <Line type="monotone" dataKey="inertia" stroke="#2563EB" strokeWidth={2} dot={{ r: 3 }} name="WCSS" />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Silhouette Score */}
          <div className="rounded-md border border-input bg-background p-2">
            <p className="mb-1 text-[11px] font-medium text-muted-foreground text-center">
              Silhouette Score <span className="text-[9px] text-red-400">(red=infeasible &lt;5%)</span>
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
                  stroke="#ef4444"
                  strokeDasharray="5 5"
                  label={{ value: `K=${result.optimal_k}`, position: "top", fill: "#ef4444", fontSize: 10 }}
                />
                <Bar dataKey="silhouette" name="Silhouette">
                  {kData.map((entry, i) => (
                    <Cell
                      key={i}
                      fill={entry.feasible ? "#22c55e" : "#fca5a5"}
                      stroke={entry.k === result.optimal_k ? "#1D4ED8" : undefined}
                      strokeWidth={entry.k === result.optimal_k ? 2 : 0}
                    />
                  ))}
                </Bar>
                <Line type="monotone" dataKey="silhouette" stroke="#2563EB" strokeWidth={1.5} dot={{ r: 2 }} name="Trend" />
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
                  stroke="#ef4444"
                  strokeDasharray="5 5"
                  label={{ value: `K=${result.optimal_k}`, position: "top", fill: "#ef4444", fontSize: 10 }}
                />
                {hasCH ? (
                  <>
                    <Bar dataKey="ch" name="CH Score">
                      {kData.map((entry, i) => (
                        <Cell
                          key={i}
                          fill={entry.feasible ? "#22c55e" : "#fca5a5"}
                          stroke={entry.k === result.optimal_k ? "#1D4ED8" : undefined}
                          strokeWidth={entry.k === result.optimal_k ? 2 : 0}
                        />
                      ))}
                    </Bar>
                    <Line type="monotone" dataKey="ch" stroke="#2563EB" strokeWidth={1.5} dot={{ r: 2 }} name="Trend" />
                  </>
                ) : (
                  <Bar dataKey="silhouette" name="N/A" fill="#d1d5db" />
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* ── PCA 2D Scatter ── */}
      {pcaScatter && pcaScatter.points.length > 0 && (
        <div>
          <p className="mb-2 text-sm font-semibold text-foreground">Cluster Visualization (2D PCA)</p>
          <div className="rounded-md border border-input bg-background p-3">
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart margin={{ top: 10, right: 30, bottom: 20, left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="pc1"
                  name="PC1"
                  tick={{ fontSize: 10 }}
                  label={{ value: `PC1 (${pcaScatter.pc1_variance}% variance)`, position: "insideBottom", offset: -10, fontSize: 11 }}
                />
                <YAxis
                  type="number"
                  dataKey="pc2"
                  name="PC2"
                  tick={{ fontSize: 10 }}
                  label={{ value: `PC2 (${pcaScatter.pc2_variance}% variance)`, angle: -90, position: "insideLeft", offset: -5, fontSize: 11 }}
                />
                <Tooltip
                  cursor={{ strokeDasharray: "3 3" }}
                  formatter={(value: number, name: string) => [value.toFixed(2), name]}
                />
                {Array.from(pcaByCluster.entries())
                  .sort(([a], [b]) => a - b)
                  .map(([cluster, points]) => (
                    <Scatter
                      key={cluster}
                      name={`Cluster ${cluster}`}
                      data={points}
                      fill={PIE_COLORS[cluster % PIE_COLORS.length]}
                      fillOpacity={0.6}
                      r={3}
                    />
                  ))}
                <Legend verticalAlign="top" align="right" layout="vertical" wrapperStyle={{ fontSize: 10, paddingLeft: 12 }} />
              </ScatterChart>
            </ResponsiveContainer>
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
                <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
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
                stroke={radarColors[i % radarColors.length]}
                fill={radarColors[i % radarColors.length]}
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
                fill={PIE_COLORS[i % PIE_COLORS.length]}
                stroke={PIE_COLORS[i % PIE_COLORS.length]}
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
              <Bar dataKey="variance_ratio" fill="#0891B2" name="Importance">
                {featureImportance.map((_, i) => (
                  <Cell key={i} fill={i === 0 ? "#2563EB" : "#60A5FA"} />
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
                stroke="#ef4444"
                strokeDasharray="5 5"
                label={{ value: `K=${result.optimal_k}`, position: "top", fill: "#ef4444", fontSize: 11 }}
              />
              <Bar dataKey="combined" name="Combined Score">
                {kData.map((entry, i) => (
                  <Cell
                    key={i}
                    fill={entry.k === result.optimal_k ? "#2563EB" : (entry.combined ?? 0) < 0 ? "#fca5a5" : "#86efac"}
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
