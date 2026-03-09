import {
  ResponsiveContainer,
  LineChart,
  Line,
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
} from "recharts";
import type { ClusteringScenarioResult } from "@/api/queries";
import { formatNumber, formatCompactNumber } from "@/lib/formatters";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
const PIE_COLORS = [
  "#2563EB", "#f59e0b", "#10b981", "#ef4444", "#0891B2",
  "#06b6d4", "#ec4899", "#14b8a6", "#f97316", "#84cc16",
];

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
export function ScenarioCharts({ result }: { result: NonNullable<ClusteringScenarioResult["result"]> }) {
  const kData = result.k_selection_results.k_values.map((k, i) => ({
    k,
    inertia: result.k_selection_results.inertias[i],
    silhouette: result.k_selection_results.silhouette_scores[i],
    ...(result.k_selection_results.gap_stats?.[i] != null ? { gap: result.k_selection_results.gap_stats[i] } : {}),
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

  const hasGapStats = result.k_selection_results.gap_stats && result.k_selection_results.gap_stats.length > 0;

  return (
    <div className="mt-4 grid gap-4 md:grid-cols-2">
      {/* Elbow chart with optimal K marker */}
      <div>
        <p className="mb-1 text-xs font-semibold text-muted-foreground">Elbow (WCSS/Inertia)</p>
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={kData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="k" />
            <YAxis />
            <Tooltip
              formatter={(value: number, name: string) => [
                name === "Inertia" ? formatCompactNumber(value) : value.toFixed(4),
                name,
              ]}
              labelFormatter={(k) => `K = ${k}${Number(k) === result.optimal_k ? " (Optimal)" : ""}`}
            />
            <Legend />
            <ReferenceLine
              x={result.optimal_k}
              stroke="#ef4444"
              strokeDasharray="5 5"
              label={{ value: `Optimal K=${result.optimal_k}`, position: "top", fill: "#ef4444", fontSize: 11 }}
            />
            <Line type="monotone" dataKey="inertia" stroke="#2563EB" name="Inertia" strokeWidth={2} dot={{ r: 3 }} />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* Silhouette chart with quality zones */}
      <div>
        <p className="mb-1 text-xs font-semibold text-muted-foreground">
          Silhouette Score
          <span className="ml-2 text-[10px] font-normal text-muted-foreground">
            ({silhouetteQuality(result.silhouette_score)} — {result.silhouette_score.toFixed(3)})
          </span>
        </p>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart data={kData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="k" />
            <YAxis domain={[0, 1]} />
            <Tooltip
              formatter={(value: number) => [value.toFixed(4), "Silhouette"]}
              labelFormatter={(k) => `K = ${k}${Number(k) === result.optimal_k ? " (Optimal)" : ""}`}
            />
            <ReferenceLine y={0.71} stroke="#22c55e" strokeDasharray="3 3" label={{ value: "Strong", position: "right", fontSize: 9 }} />
            <ReferenceLine y={0.51} stroke="#3b82f6" strokeDasharray="3 3" label={{ value: "Reasonable", position: "right", fontSize: 9 }} />
            <ReferenceLine y={0.26} stroke="#eab308" strokeDasharray="3 3" label={{ value: "Weak", position: "right", fontSize: 9 }} />
            <Bar dataKey="silhouette" name="Silhouette">
              {kData.map((entry, i) => (
                <Cell
                  key={i}
                  fill={entry.k === result.optimal_k ? "#2563EB" : silhouetteColor(entry.silhouette)}
                  stroke={entry.k === result.optimal_k ? "#1D4ED8" : undefined}
                  strokeWidth={entry.k === result.optimal_k ? 2 : 0}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

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
              label={({ label, pct }) => `${label} (${pct.toFixed(0)}%)`}
              labelLine={{ strokeWidth: 1 }}
            >
              {sizeData.map((_, i) => (
                <Cell key={i} fill={PIE_COLORS[i % PIE_COLORS.length]} />
              ))}
            </Pie>
            <Tooltip formatter={(value: number, name: string) => [formatNumber(value), name]} />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Radar profile */}
      <div>
        <p className="mb-1 text-xs font-semibold text-muted-foreground">Cluster Profile Radar</p>
        <ResponsiveContainer width="100%" height={220}>
          <RadarChart data={radarData}>
            <PolarGrid />
            <PolarAngleAxis dataKey="feature" />
            <PolarRadiusAxis />
            {result.profiles.map((p, i) => (
              <Radar
                key={p.label}
                name={p.label}
                dataKey={p.label}
                stroke={radarColors[i % radarColors.length]}
                fill={radarColors[i % radarColors.length]}
                fillOpacity={0.15}
              />
            ))}
            <Legend />
          </RadarChart>
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

      {/* Gap statistic chart (conditional) */}
      {hasGapStats && (
        <div>
          <p className="mb-1 text-xs font-semibold text-muted-foreground">Gap Statistic</p>
          <ResponsiveContainer width="100%" height={220}>
            <LineChart data={kData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="k" />
              <YAxis />
              <Tooltip
                formatter={(value: number) => [value.toFixed(4), "Gap"]}
                labelFormatter={(k) => `K = ${k}${Number(k) === result.optimal_k ? " (Optimal)" : ""}`}
              />
              <ReferenceLine
                x={result.optimal_k}
                stroke="#ef4444"
                strokeDasharray="5 5"
                label={{ value: `K=${result.optimal_k}`, position: "top", fill: "#ef4444", fontSize: 11 }}
              />
              <Line type="monotone" dataKey="gap" stroke="#f59e0b" name="Gap" strokeWidth={2} dot={{ r: 3 }} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}
