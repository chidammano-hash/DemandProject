/**
 * FeatureLabPanel — Feature importance, stability, correlation, and per-cluster breakdown.
 *
 * Sub-sections (inner tabs):
 *   1. Feature Importance — horizontal bar chart sorted by SHAP value, color-coded by category
 *   2. Feature Stability — table with mean rank, rank std, stability badge
 *   3. Feature Correlation — heatmap of top 20 features, |r|>0.9 highlighted
 *   4. Per-Cluster Importance — select a cluster and see its top features
 *   5. Category Legend — always visible summary strip
 */

import { useState, useMemo } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  Cell,
} from "recharts";
import { Sparkles, Layers, GitCompareArrows, FlaskConical } from "lucide-react";

import {
  featureLabKeys,
  fetchFeatureImportance,
  fetchFeatureStability,
  fetchFeatureCorrelation,
  fetchClusterFeatureImportance,
  fetchFeatureCategories,
  fetchClusterProfile,
  clusterEdaKeys,
  STALE,
  type FeatureImportanceRow,
  type FeatureStabilityRow,
} from "@/api/queries";
import { useChartColors } from "@/hooks/useChartColors";
import { formatFixed } from "@/lib/formatters";
import { cn } from "@/lib/utils";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";
import { LoadingElement } from "@/components/LoadingElement";
import { HeatmapGrid } from "@/components/HeatmapGrid";

// ---------------------------------------------------------------------------
// Sub-tab types
// ---------------------------------------------------------------------------
type SubSection = "importance" | "stability" | "correlation" | "per-cluster";

const SUB_TABS: { key: SubSection; label: string; icon: typeof Sparkles }[] = [
  { key: "importance", label: "Feature Importance", icon: Sparkles },
  { key: "stability", label: "Feature Stability", icon: Layers },
  { key: "correlation", label: "Correlation Heatmap", icon: GitCompareArrows },
  { key: "per-cluster", label: "Per-Cluster", icon: FlaskConical },
];

// ---------------------------------------------------------------------------
// Stability badge
// ---------------------------------------------------------------------------
function StabilityBadge({ stability }: { stability: string }) {
  const styles: Record<string, string> = {
    stable: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300",
    moderate: "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300",
    unstable: "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
  };
  return (
    <Badge className={cn("text-[10px] px-2 py-0.5", styles[stability] ?? styles.moderate)}>
      {stability}
    </Badge>
  );
}

// ---------------------------------------------------------------------------
// Default category colors for features without a category color
// ---------------------------------------------------------------------------
const DEFAULT_CATEGORY_COLORS: Record<string, string> = {
  lag: "#2563EB",
  rolling: "#0D9488",
  seasonal: "#D97706",
  calendar: "#0891B2",
  cluster: "#EC4899",
  external: "#84CC16",
  demand: "#f97316",
  other: "#64748B",
};

function getCategoryColor(category: string, colorMap: Map<string, string>): string {
  return colorMap.get(category) ?? DEFAULT_CATEGORY_COLORS[category.toLowerCase()] ?? "#64748B";
}

// ---------------------------------------------------------------------------
// 1. Feature Importance — horizontal bar chart
// ---------------------------------------------------------------------------
function ImportanceChart({
  features,
  catColorMap,
}: {
  features: FeatureImportanceRow[];
  catColorMap: Map<string, string>;
}) {
  const { chartColors } = useChartColors();

  // Top 30 features sorted by SHAP value (descending)
  const chartData = useMemo(
    () =>
      [...features]
        .sort((a, b) => b.shap_value - a.shap_value)
        .slice(0, 30)
        .reverse() // Reverse so highest is at top in horizontal chart
        .map((f) => ({
          feature: f.feature.length > 25 ? f.feature.slice(0, 22) + "..." : f.feature,
          fullName: f.feature,
          shap_value: f.shap_value,
          category: f.category,
          color: getCategoryColor(f.category, catColorMap),
        })),
    [features, catColorMap],
  );

  if (chartData.length === 0) {
    return (
      <p className="text-sm text-muted-foreground text-center py-8">
        No SHAP importance data available.
      </p>
    );
  }

  return (
    <div style={{ height: Math.max(300, chartData.length * 24) }}>
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={chartData} layout="vertical" margin={{ left: 120, right: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} horizontal={false} />
          <XAxis
            type="number"
            tick={{ fontSize: 10, fill: chartColors.axis }}
            tickLine={false}
            label={{ value: "Mean |SHAP|", position: "insideBottom", offset: -5, fontSize: 11, fill: chartColors.axis }}
          />
          <YAxis
            type="category"
            dataKey="feature"
            tick={{ fontSize: 10, fill: chartColors.axis }}
            tickLine={false}
            axisLine={false}
            width={115}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: chartColors.tooltip_bg,
              border: `1px solid ${chartColors.tooltip_border}`,
              borderRadius: "8px",
              fontSize: "12px",
            }}
            formatter={(value: number, _name: string, entry: { payload: { fullName: string; category: string } }) => [
              formatFixed(value, 4),
              `${entry.payload.fullName} (${entry.payload.category})`,
            ]}
          />
          <Bar dataKey="shap_value" radius={[0, 4, 4, 0]} name="SHAP">
            {chartData.map((entry, idx) => (
              <Cell key={idx} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 2. Feature Stability Table
// ---------------------------------------------------------------------------
function StabilityTable({ features }: { features: FeatureStabilityRow[] }) {
  if (features.length === 0) {
    return (
      <p className="text-sm text-muted-foreground text-center py-8">
        No stability data available.
      </p>
    );
  }

  const sorted = useMemo(
    () => [...features].sort((a, b) => a.mean_rank - b.mean_rank),
    [features],
  );

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Feature</TableHead>
            <TableHead className="text-right">Mean Rank</TableHead>
            <TableHead className="text-right">Rank Std</TableHead>
            <TableHead className="w-24">Stability</TableHead>
            <TableHead className="text-right">Folds</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {sorted.map((s) => (
            <TableRow key={s.feature}>
              <TableCell className="font-mono text-xs font-medium">{s.feature}</TableCell>
              <TableCell className="text-right tabular-nums text-sm">
                {formatFixed(s.mean_rank, 1)}
              </TableCell>
              <TableCell className="text-right tabular-nums text-sm">
                {formatFixed(s.rank_std, 2)}
              </TableCell>
              <TableCell>
                <StabilityBadge stability={s.stability} />
              </TableCell>
              <TableCell className="text-right tabular-nums text-xs text-muted-foreground">
                {s.n_folds}
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 3. Feature Correlation Heatmap
// ---------------------------------------------------------------------------
function CorrelationSection() {
  const { data, isLoading } = useQuery({
    queryKey: featureLabKeys.correlation(20),
    queryFn: () => fetchFeatureCorrelation(20),
    staleTime: STALE.FIVE_MIN,
  });

  const correlationScale = useMemo(
    () => (value: number): string => {
      // value is correlation * 100 for the heatmap (scaled to 0-100 range)
      const r = value / 100;
      if (Math.abs(r) >= 0.9) return "#DC2626"; // red — high collinearity
      if (Math.abs(r) >= 0.7) return "#F59E0B"; // amber — moderate
      if (Math.abs(r) >= 0.5) return "#3B82F6"; // blue — weak
      return "#E2E8F0"; // muted — negligible
    },
    [],
  );

  if (isLoading) return <LoadingElement message="Loading correlation data..." />;
  if (!data || !data.features || data.features.length === 0) {
    return (
      <p className="text-sm text-muted-foreground text-center py-8">
        No correlation data available.
      </p>
    );
  }

  // Build a symmetric matrix for the heatmap
  const featureNames = data.features;
  const cellMap = new Map(
    (data.cells ?? []).map((c) => [`${c.feature_a}|${c.feature_b}`, c.correlation]),
  );

  const heatmapRows = featureNames.map((fa) => ({
    label: fa.length > 15 ? fa.slice(0, 12) + "..." : fa,
    values: featureNames.map((fb) => {
      if (fa === fb) return 100; // self-correlation
      const v = cellMap.get(`${fa}|${fb}`) ?? cellMap.get(`${fb}|${fa}`) ?? 0;
      return Math.round(v * 100); // scale to 0-100 for heatmap
    }),
  }));

  const colLabels = featureNames.map((f) =>
    f.length > 8 ? f.slice(0, 6) + ".." : f,
  );

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">
        Top {featureNames.length} features by importance. Red cells indicate |r| &gt; 0.9
        (high collinearity risk). Amber indicates |r| &gt; 0.7.
      </p>
      <HeatmapGrid
        rows={heatmapRows}
        columnLabels={colLabels}
        colorScale={correlationScale}
        valueFormat={(v) => formatFixed(v / 100, 2)}
        showLegend
        minLabel="-1.0"
        maxLabel="1.0"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// 4. Per-Cluster Importance
// ---------------------------------------------------------------------------
function PerClusterSection() {
  const [selectedCluster, setSelectedCluster] = useState(0);
  const { chartColors, trendColors } = useChartColors();

  // Fetch profile to get cluster list (via fetchJson query fetcher — U6.10)
  const { data: profileData } = useQuery({
    queryKey: clusterEdaKeys.profile(),
    queryFn: fetchClusterProfile,
    staleTime: STALE.FIVE_MIN,
  });

  const clusterIds: number[] = useMemo(
    () => (profileData?.rows ?? []).map((r) => r.cluster),
    [profileData],
  );

  const { data, isLoading } = useQuery({
    queryKey: featureLabKeys.clusterImportance(selectedCluster),
    queryFn: () => fetchClusterFeatureImportance(selectedCluster),
    staleTime: STALE.FIVE_MIN,
    enabled: clusterIds.length > 0,
  });

  const chartData = useMemo(
    () =>
      (data?.features ?? [])
        .slice(0, 20)
        .reverse()
        .map((f) => ({
          feature: f.feature.length > 25 ? f.feature.slice(0, 22) + "..." : f.feature,
          fullName: f.feature,
          shap_value: f.shap_value,
        })),
    [data],
  );

  return (
    <div className="space-y-3">
      {/* Cluster selector */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs font-medium text-muted-foreground">Cluster:</span>
        {clusterIds.map((id) => (
          <button
            key={id}
            onClick={() => setSelectedCluster(id)}
            className={cn(
              "rounded-full px-3 py-1 text-xs font-medium transition-all",
              selectedCluster === id
                ? "bg-primary text-primary-foreground shadow-sm"
                : "text-muted-foreground border border-border hover:border-muted-foreground/40 hover:text-foreground",
            )}
          >
            C{id}
          </button>
        ))}
      </div>

      {isLoading ? (
        <LoadingElement message="Loading cluster features..." />
      ) : chartData.length === 0 ? (
        <p className="text-sm text-muted-foreground text-center py-8">
          No per-cluster importance data for cluster {selectedCluster}.
        </p>
      ) : (
        <div style={{ height: Math.max(300, chartData.length * 24) }}>
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ left: 120, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} horizontal={false} />
              <XAxis
                type="number"
                tick={{ fontSize: 10, fill: chartColors.axis }}
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="feature"
                tick={{ fontSize: 10, fill: chartColors.axis }}
                tickLine={false}
                axisLine={false}
                width={115}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: chartColors.tooltip_bg,
                  border: `1px solid ${chartColors.tooltip_border}`,
                  borderRadius: "8px",
                  fontSize: "12px",
                }}
                formatter={(value: number, _name: string, entry: { payload: { fullName: string } }) => [
                  formatFixed(value, 4),
                  entry.payload.fullName,
                ]}
              />
              <Bar dataKey="shap_value" fill={trendColors[1]} radius={[0, 4, 4, 0]} name="SHAP" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Category Legend Strip
// ---------------------------------------------------------------------------
function CategoryLegend({
  categories,
}: {
  categories: Array<{ category: string; color: string; count: number }>;
}) {
  if (categories.length === 0) return null;

  return (
    <div className="flex flex-wrap gap-3">
      {categories.map((cat) => (
        <div
          key={cat.category}
          className="flex items-center gap-2 rounded-lg border border-border/60 px-3 py-1.5"
        >
          <div
            className="h-2.5 w-2.5 rounded-full flex-shrink-0"
            style={{ backgroundColor: cat.color }}
          />
          <span className="text-xs font-medium">{cat.category}</span>
          <span className="text-[10px] text-muted-foreground">({cat.count})</span>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------
export function FeatureLabPanel() {
  const [activeSection, setActiveSection] = useState<SubSection>("importance");

  // Core data queries
  const { data: importanceData, isLoading: importanceLoading } = useQuery({
    queryKey: featureLabKeys.importance(),
    queryFn: () => fetchFeatureImportance(),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: stabilityData, isLoading: stabilityLoading } = useQuery({
    queryKey: featureLabKeys.stability(),
    queryFn: fetchFeatureStability,
    staleTime: STALE.FIVE_MIN,
  });

  const { data: categoriesData } = useQuery({
    queryKey: featureLabKeys.categories(),
    queryFn: fetchFeatureCategories,
    staleTime: STALE.FIVE_MIN,
  });

  const features = importanceData?.features ?? [];
  const stabilityRows = stabilityData?.features ?? [];
  const categories = categoriesData?.categories ?? [];
  const catColorMap = useMemo(
    () => new Map(categories.map((c) => [c.category, c.color])),
    [categories],
  );

  const isLoading = importanceLoading || stabilityLoading;

  if (isLoading) {
    return (
      <div className="p-6">
        <LoadingElement message="Loading feature analysis..." size="md" />
      </div>
    );
  }

  if (features.length === 0 && stabilityRows.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <Sparkles className="h-10 w-10 text-muted-foreground/40 mb-3" />
        <p className="text-sm font-medium text-foreground mb-1">No feature data available</p>
        <p className="text-xs text-muted-foreground mb-3">
          Run a backtest with SHAP enabled to generate feature importance data.
        </p>
        <code className="text-xs bg-muted px-3 py-1 rounded font-mono">make backtest-all</code>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Category legend strip */}
      {categories.length > 0 && (
        <CategoryLegend categories={categories} />
      )}

      {/* Sub-tab selector */}
      <div className="flex items-center gap-0.5 border-b overflow-x-auto">
        {SUB_TABS.map((tab) => {
          const Icon = tab.icon;
          const isActive = activeSection === tab.key;
          return (
            <button
              key={tab.key}
              onClick={() => setActiveSection(tab.key)}
              className={cn(
                "flex items-center gap-1.5 px-3 py-2 text-xs font-medium transition-all border-b-2",
                isActive
                  ? "text-foreground border-primary"
                  : "border-transparent text-muted-foreground hover:text-foreground hover:border-muted-foreground/30",
              )}
            >
              <Icon
                size={13}
                className={cn(
                  "flex-shrink-0",
                  isActive ? "text-primary" : "text-muted-foreground",
                )}
              />
              <span className="whitespace-nowrap">{tab.label}</span>
            </button>
          );
        })}
      </div>

      {/* Section content */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="text-base">
            {SUB_TABS.find((t) => t.key === activeSection)?.label}
          </CardTitle>
          <CardDescription className="text-xs">
            {activeSection === "importance" &&
              "Global feature importance ranked by mean |SHAP| value, color-coded by category."}
            {activeSection === "stability" &&
              "Feature rank consistency across cross-validation folds. Low std = stable feature."}
            {activeSection === "correlation" &&
              "Pairwise correlation heatmap for top features. Red = high collinearity (drop candidate)."}
            {activeSection === "per-cluster" &&
              "Feature importance breakdown for individual clusters."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {activeSection === "importance" && (
            <ImportanceChart features={features} catColorMap={catColorMap} />
          )}
          {activeSection === "stability" && <StabilityTable features={stabilityRows} />}
          {activeSection === "correlation" && <CorrelationSection />}
          {activeSection === "per-cluster" && <PerClusterSection />}
        </CardContent>
      </Card>
    </div>
  );
}
