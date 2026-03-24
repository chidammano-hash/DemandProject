/**
 * ClusterEDAPanel — Cluster exploratory data analysis for LGBM tuning.
 *
 * Sub-sections (inner tabs):
 *   1. Cluster Profile Table (sortable, color-coded accuracy)
 *   2. Error Concentration summary cards
 *   3. Per-cluster demand distribution bar chart
 *   4. Seasonality heatmap (month x cluster)
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
} from "recharts";
import { ArrowUpDown, Layers, AlertTriangle, BarChart2 } from "lucide-react";

import {
  clusterEdaKeys,
  fetchClusterProfile,
  fetchErrorConcentration,
  fetchClusterDistribution,
  fetchSeasonalityHeatmap,
  STALE,
  type ClusterProfileRow,
} from "@/api/queries";
import { useChartColors } from "@/hooks/useChartColors";
import { formatPct, formatFixed, formatInt } from "@/lib/formatters";
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
import { KpiCard } from "@/components/KpiCard";
import { LoadingElement } from "@/components/LoadingElement";
import { HeatmapGrid, makeHeatmapScale } from "@/components/HeatmapGrid";

// ---------------------------------------------------------------------------
// Sub-tab navigation
// ---------------------------------------------------------------------------
type SubSection = "profile" | "errors" | "distribution" | "seasonality";

const SUB_TABS: { key: SubSection; label: string; icon: typeof Layers }[] = [
  { key: "profile", label: "Cluster Profiles", icon: Layers },
  { key: "errors", label: "Error Concentration", icon: AlertTriangle },
  { key: "distribution", label: "Demand Distribution", icon: BarChart2 },
  { key: "seasonality", label: "Seasonality Heatmap", icon: BarChart2 },
];

// ---------------------------------------------------------------------------
// Sort helpers
// ---------------------------------------------------------------------------
type SortKey = keyof ClusterProfileRow;
type SortDir = "asc" | "desc";

function sortRows(rows: ClusterProfileRow[], key: SortKey, dir: SortDir): ClusterProfileRow[] {
  return [...rows].sort((a, b) => {
    const av = a[key] ?? 0;
    const bv = b[key] ?? 0;
    return dir === "asc" ? (av < bv ? -1 : 1) : (av > bv ? -1 : 1);
  });
}

// ---------------------------------------------------------------------------
// Accuracy badge — color-coded by threshold
// ---------------------------------------------------------------------------
function AccuracyBadge({ value }: { value: number | null }) {
  if (value == null) return <span className="text-muted-foreground">--</span>;
  const style =
    value >= 80
      ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300"
      : value >= 60
        ? "bg-amber-100 text-amber-800 dark:bg-amber-900/40 dark:text-amber-300"
        : "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300";
  return (
    <Badge className={cn("text-[10px] font-medium px-2 py-0.5 tabular-nums", style)}>
      {formatPct(value)}
    </Badge>
  );
}

// ---------------------------------------------------------------------------
// 1. Profile Table — sortable
// ---------------------------------------------------------------------------
function ProfileTable({ rows }: { rows: ClusterProfileRow[] }) {
  const [sortKey, setSortKey] = useState<SortKey>("cluster");
  const [sortDir, setSortDir] = useState<SortDir>("asc");

  const sorted = useMemo(() => sortRows(rows, sortKey, sortDir), [rows, sortKey, sortDir]);

  function handleSort(key: SortKey) {
    if (sortKey === key) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortDir("asc");
    }
  }

  const headerCols: { key: SortKey; label: string; align?: string }[] = [
    { key: "cluster", label: "Cluster" },
    { key: "n_dfus", label: "DFUs", align: "text-right" },
    { key: "mean_demand", label: "Mean Demand", align: "text-right" },
    { key: "cv", label: "CV", align: "text-right" },
    { key: "zero_pct", label: "Zero %", align: "text-right" },
    { key: "seasonal_amplitude", label: "Seasonal Amp", align: "text-right" },
    { key: "accuracy_pct", label: "Accuracy", align: "text-right" },
  ];

  // Derived KPIs
  const totalDfus = rows.reduce((s, r) => s + r.n_dfus, 0);
  const avgAccuracy =
    rows.length > 0
      ? rows.reduce((s, r) => s + (r.accuracy_pct ?? 0), 0) / rows.length
      : null;
  const highZeroClusters = rows.filter((r) => r.zero_pct > 50).length;

  return (
    <div className="space-y-4">
      {/* Summary KPIs */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <KpiCard label="Clusters" value={formatInt(rows.length)} size="md" />
        <KpiCard label="Total DFUs" value={formatInt(totalDfus)} size="md" />
        <KpiCard
          label="Avg Accuracy"
          value={avgAccuracy != null ? formatPct(avgAccuracy) : "--"}
          severity={avgAccuracy != null && avgAccuracy >= 70 ? "best" : "warning"}
          size="md"
        />
        <KpiCard
          label="High-Zero Clusters"
          value={formatInt(highZeroClusters)}
          severity={highZeroClusters > 0 ? "warning" : "best"}
          size="md"
        />
      </div>

      {/* Sortable table */}
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              {headerCols.map((col) => (
                <TableHead
                  key={col.key}
                  className={cn("cursor-pointer select-none", col.align)}
                  onClick={() => handleSort(col.key)}
                >
                  <span className="inline-flex items-center gap-1">
                    {col.label}
                    <ArrowUpDown className="h-3 w-3 text-muted-foreground" />
                  </span>
                </TableHead>
              ))}
            </TableRow>
          </TableHeader>
          <TableBody>
            {sorted.map((row) => (
              <TableRow key={row.cluster}>
                <TableCell className="font-mono text-xs">C{row.cluster}</TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {formatInt(row.n_dfus)}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {formatFixed(row.mean_demand, 0)}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {formatFixed(row.cv, 2)}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {formatPct(row.zero_pct)}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {formatFixed(row.seasonal_amplitude, 2)}
                </TableCell>
                <TableCell className="text-right">
                  <AccuracyBadge value={row.accuracy_pct} />
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 2. Error Concentration
// ---------------------------------------------------------------------------
function ErrorSection() {
  const { data, isLoading } = useQuery({
    queryKey: clusterEdaKeys.errorConcentration(),
    queryFn: fetchErrorConcentration,
    staleTime: STALE.FIVE_MIN,
  });

  if (isLoading) return <LoadingElement message="Loading error concentration..." />;
  if (!data) {
    return (
      <p className="text-sm text-muted-foreground text-center py-8">
        No error concentration data available.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <KpiCard
          label="Top 10% DFU Error Share"
          value={formatPct(data.top_10_pct_share)}
          severity={data.top_10_pct_share > 50 ? "warning" : "best"}
          size="md"
          tooltip={{
            title: "Error Concentration",
            description: "Share of total forecast error from the top 10% worst DFUs.",
          }}
        />
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Worst Months</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1.5">
            {(data.worst_months ?? []).length > 0 ? (
              data.worst_months.slice(0, 5).map((m) => (
                <div key={m.month} className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">{m.month}</span>
                  <span className="tabular-nums font-medium">{formatPct(m.error_share)}</span>
                </div>
              ))
            ) : (
              <p className="text-xs text-muted-foreground">No data</p>
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-sm">Worst Clusters</CardTitle>
          </CardHeader>
          <CardContent className="space-y-1.5">
            {(data.worst_clusters ?? []).length > 0 ? (
              data.worst_clusters.slice(0, 5).map((c) => (
                <div key={c.cluster} className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">
                    C{c.cluster}
                    <span className="ml-1 text-muted-foreground/60">
                      ({formatInt(c.n_dfus)} DFUs)
                    </span>
                  </span>
                  <span className="tabular-nums font-medium">{formatPct(c.error_share)}</span>
                </div>
              ))
            ) : (
              <p className="text-xs text-muted-foreground">No data</p>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 3. Demand Distribution — bar chart per selected cluster
// ---------------------------------------------------------------------------
function DistributionSection({ clusters }: { clusters: ClusterProfileRow[] }) {
  const [selectedCluster, setSelectedCluster] = useState<number>(
    clusters.length > 0 ? clusters[0].cluster : 0,
  );
  const { chartColors, trendColors } = useChartColors();

  const { data, isLoading } = useQuery({
    queryKey: clusterEdaKeys.distribution(selectedCluster),
    queryFn: () => fetchClusterDistribution(selectedCluster),
    staleTime: STALE.FIVE_MIN,
    enabled: selectedCluster != null,
  });

  const chartData = useMemo(
    () =>
      (data?.bins ?? []).map((b) => ({
        range: `${formatFixed(b.bin_start, 0)}-${formatFixed(b.bin_end, 0)}`,
        count: b.count,
      })),
    [data],
  );

  return (
    <div className="space-y-3">
      {/* Cluster selector pills */}
      <div className="flex items-center gap-2 flex-wrap">
        <span className="text-xs font-medium text-muted-foreground">Cluster:</span>
        {clusters.map((c) => (
          <button
            key={c.cluster}
            onClick={() => setSelectedCluster(c.cluster)}
            className={cn(
              "rounded-full px-3 py-1 text-xs font-medium transition-all",
              selectedCluster === c.cluster
                ? "bg-primary text-primary-foreground shadow-sm"
                : "text-muted-foreground border border-border hover:border-muted-foreground/40 hover:text-foreground",
            )}
          >
            C{c.cluster}
          </button>
        ))}
      </div>

      {isLoading ? (
        <LoadingElement message="Loading distribution..." />
      ) : chartData.length === 0 ? (
        <p className="text-sm text-muted-foreground text-center py-8">
          No distribution data for cluster {selectedCluster}.
        </p>
      ) : (
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
              <XAxis
                dataKey="range"
                tick={{ fontSize: 10, fill: chartColors.axis }}
                tickLine={false}
                label={{ value: "Demand Range", position: "insideBottom", offset: -5, fontSize: 11, fill: chartColors.axis }}
              />
              <YAxis
                tick={{ fontSize: 10, fill: chartColors.axis }}
                tickLine={false}
                axisLine={false}
                label={{ value: "DFU Count", angle: -90, position: "insideLeft", fontSize: 11, fill: chartColors.axis }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: chartColors.tooltip_bg,
                  border: `1px solid ${chartColors.tooltip_border}`,
                  borderRadius: "8px",
                  fontSize: "12px",
                }}
              />
              <Bar dataKey="count" fill={trendColors[0]} radius={[4, 4, 0, 0]} name="DFUs" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 4. Seasonality Heatmap — month x cluster matrix
// ---------------------------------------------------------------------------
function SeasonalitySection() {
  const { data, isLoading } = useQuery({
    queryKey: clusterEdaKeys.seasonalityHeatmap(),
    queryFn: fetchSeasonalityHeatmap,
    staleTime: STALE.FIVE_MIN,
  });

  const heatmapScale = useMemo(
    () =>
      makeHeatmapScale([
        "#059669", // excellent (95+)
        "#10B981", // good (85+)
        "#F59E0B", // warning (70+)
        "#EF4444", // poor (50+)
        "#991B1B", // critical (<50)
      ]),
    [],
  );

  if (isLoading) return <LoadingElement message="Loading seasonality heatmap..." />;
  if (!data || !data.rows || data.rows.length === 0) {
    return (
      <p className="text-sm text-muted-foreground text-center py-8">
        No seasonality heatmap data available. Run seasonality detection first.
      </p>
    );
  }

  const heatmapRows = data.rows.map((r) => ({
    label: `C${r.cluster}`,
    values: r.values,
  }));

  return (
    <div className="space-y-3">
      <p className="text-xs text-muted-foreground">
        Seasonal demand amplitude by cluster and month. Higher values indicate stronger seasonality patterns.
      </p>
      <HeatmapGrid
        rows={heatmapRows}
        columnLabels={data.months}
        colorScale={heatmapScale}
        valueFormat={(v) => formatFixed(v, 1)}
        showLegend
        minLabel="Low"
        maxLabel="High"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------
export function ClusterEDAPanel() {
  const [activeSection, setActiveSection] = useState<SubSection>("profile");

  const { data: profileData, isLoading: isLoadingProfile } = useQuery({
    queryKey: clusterEdaKeys.profile(),
    queryFn: fetchClusterProfile,
    staleTime: STALE.FIVE_MIN,
  });

  const clusters = profileData?.rows ?? [];

  if (isLoadingProfile) {
    return (
      <div className="p-6">
        <LoadingElement message="Loading cluster profiles..." size="md" />
      </div>
    );
  }

  if (clusters.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <Layers className="h-10 w-10 text-muted-foreground/40 mb-3" />
        <p className="text-sm font-medium text-foreground mb-1">No cluster data available</p>
        <p className="text-xs text-muted-foreground mb-3">
          Run the clustering pipeline to populate cluster profiles.
        </p>
        <code className="text-xs bg-muted px-3 py-1 rounded font-mono">make cluster-all</code>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Sub-tab selector strip */}
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
            {activeSection === "profile" &&
              "Cluster-level summary statistics with sortable columns. Click column headers to sort."}
            {activeSection === "errors" &&
              "Where forecast error concentrates across DFUs, months, and clusters."}
            {activeSection === "distribution" &&
              "Demand distribution histogram for each cluster. Select a cluster to view its distribution."}
            {activeSection === "seasonality" &&
              "Month-by-cluster seasonality amplitude heatmap."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {activeSection === "profile" && <ProfileTable rows={clusters} />}
          {activeSection === "errors" && <ErrorSection />}
          {activeSection === "distribution" && <DistributionSection clusters={clusters} />}
          {activeSection === "seasonality" && <SeasonalitySection />}
        </CardContent>
      </Card>
    </div>
  );
}
