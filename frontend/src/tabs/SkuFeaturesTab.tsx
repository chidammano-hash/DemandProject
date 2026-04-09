/**
 * SKU Features Tab — computed feature explorer with rich visualizations.
 *
 * Layout (top to bottom):
 *  1. Summary Cards row (Total SKUs, Last Computed, Avg CV Demand, Avg Seasonal Amplitude)
 *  2. Distribution Charts row (3 horizontal bar charts: seasonality, variability, trend)
 *  3. Feature Histograms grid (2x3: CV Demand, Seasonal Amp, Trend R2, Zero %, ADI, CAGR)
 *  4. Feature Table (filterable, searchable, sortable, paginated)
 */
import { useState, useMemo, useCallback, useEffect } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import {
  Database,
  Clock,
  TrendingUp,
  Waves,
  ChevronUp,
  ChevronDown,
  Search,
  Play,
  Loader2,
} from "lucide-react";
import {
  skuFeatureKeys,
  fetchSkuFeaturesSummary,
  fetchSkuFeaturesList,
  fetchSkuFeaturesDistributions,
  triggerComputeSkuFeatures,
  STALE_SKU_FEATURES,
} from "@/api/queries/sku-features";
import type {
  SkuFeatureRow,
  SkuFeaturesListParams,
  FeatureDistribution,
} from "@/api/queries/sku-features";
import { useChartColors } from "@/hooks/useChartColors";
import { Skeleton } from "@/components/Skeleton";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const PAGE_SIZE = 50;

const SEASONALITY_OPTIONS = ["", "none", "low", "moderate", "strong"];
const VARIABILITY_OPTIONS = ["", "smooth", "erratic", "intermittent", "lumpy"];
const TREND_OPTIONS = ["", "declining", "flat", "growing"];

const SORTABLE_COLUMNS: { key: string; label: string; align?: "right" }[] = [
  { key: "sku_ck", label: "SKU" },
  { key: "item_id", label: "Item" },
  { key: "loc", label: "Location" },
  { key: "ml_cluster", label: "Cluster", align: "right" },
  { key: "seasonality_profile", label: "Seasonality" },
  { key: "variability_class", label: "Variability" },
  { key: "trend_direction", label: "Trend" },
  { key: "cv_demand", label: "CV Demand", align: "right" },
  { key: "seasonal_amplitude", label: "Seasonal Amp", align: "right" },
  { key: "zero_demand_pct", label: "Zero %", align: "right" },
  { key: "cagr", label: "CAGR", align: "right" },
  { key: "recency_ratio", label: "Recency", align: "right" },
];

const CHART_MARGIN = { top: 4, right: 16, left: 0, bottom: 4 };

const DISTRIBUTION_COLORS: Record<string, string[]> = {
  seasonality_profile: ["#94a3b8", "#60a5fa", "#f59e0b", "#ef4444"],
  variability_class: ["#34d399", "#f97316", "#a78bfa", "#f43f5e"],
  trend_direction: ["#ef4444", "#94a3b8", "#22c55e"],
};

const DISTRIBUTION_TITLES: Record<string, string> = {
  seasonality_profile: "Seasonality Profile",
  variability_class: "Variability Class",
  trend_direction: "Trend Direction",
};

const HISTOGRAM_LABELS: Record<string, string> = {
  cv_demand: "CV Demand",
  seasonal_amplitude: "Seasonal Amplitude",
  trend_r2: "Trend R\u00b2",
  zero_demand_pct: "Zero Demand %",
  adi: "ADI",
  cagr: "CAGR",
};

const HISTOGRAM_FEATURES = ["cv_demand", "seasonal_amplitude", "trend_r2", "zero_demand_pct", "adi", "cagr"] as const;

// Trend direction number -> label mapping
const TREND_LABELS: Record<string, string> = {
  "-1": "declining",
  "0": "flat",
  "1": "growing",
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatNumber(v: number | null | undefined, decimals = 2): string {
  if (v == null) return "\u2014";
  return v.toLocaleString(undefined, {
    minimumFractionDigits: 0,
    maximumFractionDigits: decimals,
  });
}

function formatPct(v: number | null | undefined): string {
  if (v == null) return "\u2014";
  return `${(v * 100).toFixed(1)}%`;
}

function relativeTime(ts: string): string {
  const diff = Date.now() - new Date(ts).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1) return "Just now";
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

function trendLabel(v: number | null): string {
  if (v == null) return "\u2014";
  return TREND_LABELS[String(v)] ?? String(v);
}

function badgeClass(value: string | null, category: "seasonality" | "variability" | "trend"): string {
  if (!value) return "bg-muted text-muted-foreground";
  const map: Record<string, Record<string, string>> = {
    seasonality: {
      none: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
      low: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300",
      moderate: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300",
      strong: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300",
    },
    variability: {
      smooth: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300",
      erratic: "bg-orange-100 text-orange-700 dark:bg-orange-900/40 dark:text-orange-300",
      intermittent: "bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-300",
      lumpy: "bg-rose-100 text-rose-700 dark:bg-rose-900/40 dark:text-rose-300",
    },
    trend: {
      declining: "bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-300",
      flat: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
      growing: "bg-green-100 text-green-700 dark:bg-green-900/40 dark:text-green-300",
    },
  };
  return map[category]?.[value] ?? "bg-muted text-muted-foreground";
}

/** Convert summary.distributions Record<string, number> to chart data array */
function recordToChartData(record: Record<string, number> | undefined): { label: string; count: number }[] {
  if (!record) return [];
  return Object.entries(record).map(([label, count]) => ({ label, count }));
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function SummaryCard({
  icon: Icon,
  label,
  value,
  subtitle,
  isLoading,
}: {
  icon: React.ComponentType<{ className?: string }>;
  label: string;
  value: string;
  subtitle?: string;
  isLoading: boolean;
}) {
  if (isLoading) {
    return (
      <div className="rounded-lg border border-border bg-card p-4 space-y-2">
        <Skeleton className="h-3 w-24" />
        <Skeleton className="h-7 w-16" />
        <Skeleton className="h-2.5 w-32" />
      </div>
    );
  }
  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2 text-muted-foreground">
        <Icon className="h-4 w-4" />
        <span className="text-xs font-medium">{label}</span>
      </div>
      <p className="mt-1 text-2xl font-bold tabular-nums">{value}</p>
      {subtitle && (
        <p className="mt-0.5 text-xs text-muted-foreground">{subtitle}</p>
      )}
    </div>
  );
}

function HorizontalDistribution({
  title,
  data,
  colors,
  isLoading,
}: {
  title: string;
  data: { label: string; count: number }[];
  colors: string[];
  isLoading: boolean;
}) {
  const total = useMemo(() => data.reduce((s, d) => s + d.count, 0), [data]);

  if (isLoading) {
    return (
      <div className="rounded-lg border border-border bg-card p-4 space-y-3">
        <Skeleton className="h-4 w-40" />
        <Skeleton className="h-[180px] w-full" />
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-4">
        <h4 className="text-sm font-medium text-foreground mb-2">{title}</h4>
        <p className="text-xs text-muted-foreground py-8 text-center">No data available</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <h4 className="text-sm font-medium text-foreground mb-2">{title}</h4>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data} layout="vertical" margin={CHART_MARGIN}>
          <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="var(--border)" />
          <XAxis type="number" tick={{ fontSize: 11 }} stroke="var(--muted-foreground)" />
          <YAxis
            dataKey="label"
            type="category"
            width={90}
            tick={{ fontSize: 11 }}
            stroke="var(--muted-foreground)"
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: "6px",
              fontSize: 12,
            }}
            formatter={(value: number) => [
              `${value.toLocaleString()} (${total > 0 ? ((value / total) * 100).toFixed(1) : 0}%)`,
              "Count",
            ]}
          />
          <Bar dataKey="count" radius={[0, 4, 4, 0]}>
            {data.map((_, idx) => (
              <Cell key={idx} fill={colors[idx % colors.length]} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function FeatureHistogram({
  title,
  data,
  color,
  isLoading,
}: {
  title: string;
  data: FeatureDistribution[];
  color: string;
  isLoading: boolean;
}) {
  const chartData = useMemo(
    () =>
      data.map((bin) => ({
        label: formatNumber(bin.bin_start, 1),
        count: bin.count,
        range: `${formatNumber(bin.bin_start, 2)} \u2013 ${formatNumber(bin.bin_end, 2)}`,
      })),
    [data],
  );

  if (isLoading) {
    return (
      <div className="rounded-lg border border-border bg-card p-3 space-y-2">
        <Skeleton className="h-3.5 w-32" />
        <Skeleton className="h-[120px] w-full" />
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="rounded-lg border border-border bg-card p-3">
        <h5 className="text-xs font-medium text-foreground mb-1">{title}</h5>
        <p className="text-xs text-muted-foreground py-6 text-center">No data</p>
      </div>
    );
  }

  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <h5 className="text-xs font-medium text-foreground mb-1">{title}</h5>
      <ResponsiveContainer width="100%" height={120}>
        <BarChart data={chartData} margin={{ top: 2, right: 4, left: -20, bottom: 2 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--border)" />
          <XAxis
            dataKey="label"
            tick={{ fontSize: 9 }}
            stroke="var(--muted-foreground)"
            interval="preserveStartEnd"
          />
          <YAxis tick={{ fontSize: 9 }} stroke="var(--muted-foreground)" />
          <Tooltip
            contentStyle={{
              backgroundColor: "var(--card)",
              border: "1px solid var(--border)",
              borderRadius: "6px",
              fontSize: 11,
            }}
            formatter={(value: number) => [value.toLocaleString(), "SKUs"]}
            labelFormatter={(_label: string, payload: Array<{ payload?: { range?: string } }>) =>
              payload?.[0]?.payload?.range ?? _label
            }
          />
          <Bar dataKey="count" fill={color} radius={[2, 2, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export default function SkuFeaturesTab() {
  const { trendColors } = useChartColors();
  const queryClient = useQueryClient();

  // ---- Detect any already-running compute_sku_features job on mount ----
  const { data: activeJobsData } = useQuery({
    queryKey: ["active-jobs"],
    queryFn: async () => {
      const res = await fetch("/jobs/active");
      if (!res.ok) return { jobs: [] };
      return res.json() as Promise<{ jobs: { job_id: string; job_type: string; status: string }[] }>;
    },
    staleTime: 5_000,
    refetchInterval: 5_000,
  });

  const [computeJobId, setComputeJobId] = useState<string | null>(null);

  // Recover running job on mount / when active jobs change
  useEffect(() => {
    if (!activeJobsData?.jobs || computeJobId) return;
    const running = activeJobsData.jobs.find(
      (j) => j.job_type === "compute_sku_features" && (j.status === "running" || j.status === "queued"),
    );
    if (running) {
      setComputeJobId(running.job_id);
    }
  }, [activeJobsData, computeJobId]);

  // Compute features mutation
  const computeMutation = useMutation({
    mutationFn: () => triggerComputeSkuFeatures(36),
    onSuccess: (data) => {
      setComputeJobId(data.job_id);
    },
  });

  // Poll job status while running
  const { data: jobStatus } = useQuery({
    queryKey: ["jobs", computeJobId],
    queryFn: async () => {
      const res = await fetch(`/jobs/${computeJobId}`);
      if (!res.ok) return null;
      return res.json() as Promise<{ status: string; progress_pct?: number; progress_msg?: string }>;
    },
    enabled: !!computeJobId,
    refetchInterval: (query) => {
      const status = query.state.data?.status;
      if (status === "completed" || status === "failed" || status === "cancelled") return false;
      return 3000;
    },
  });

  // Derived job state
  const jobDone = jobStatus?.status === "completed";
  const jobFailed = jobStatus?.status === "failed";

  // Refresh data when job completes
  useEffect(() => {
    if (jobDone) {
      queryClient.invalidateQueries({ queryKey: ["sku-features"] });
      const timer = setTimeout(() => setComputeJobId(null), 3000);
      return () => clearTimeout(timer);
    }
  }, [jobDone, queryClient]);

  // Filters & pagination state
  const [search, setSearch] = useState("");
  const [seasonalityFilter, setSeasonalityFilter] = useState("");
  const [variabilityFilter, setVariabilityFilter] = useState("");
  const [trendFilter, setTrendFilter] = useState("");
  const [sortBy, setSortBy] = useState("sku_ck");
  const [sortDir, setSortDir] = useState<"asc" | "desc">("asc");
  const [page, setPage] = useState(0);

  const listParams: SkuFeaturesListParams = useMemo(
    () => ({
      limit: PAGE_SIZE,
      offset: page * PAGE_SIZE,
      sort_by: sortBy,
      sort_dir: sortDir,
      seasonality_profile: seasonalityFilter || undefined,
      variability_class: variabilityFilter || undefined,
      trend_direction: trendFilter || undefined,
      search: search || undefined,
    }),
    [page, sortBy, sortDir, seasonalityFilter, variabilityFilter, trendFilter, search],
  );

  // Queries
  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: skuFeatureKeys.summary,
    queryFn: fetchSkuFeaturesSummary,
    staleTime: STALE_SKU_FEATURES.SUMMARY,
  });

  const { data: distributions, isLoading: distLoading } = useQuery({
    queryKey: skuFeatureKeys.distributions,
    queryFn: () => fetchSkuFeaturesDistributions(),
    staleTime: STALE_SKU_FEATURES.DISTRIBUTIONS,
  });

  const { data: listData, isLoading: listLoading } = useQuery({
    queryKey: skuFeatureKeys.list(listParams as Record<string, unknown>),
    queryFn: () => fetchSkuFeaturesList(listParams),
    staleTime: STALE_SKU_FEATURES.LIST,
  });

  const rows: SkuFeatureRow[] = listData?.rows ?? [];
  const totalRows = listData?.total ?? 0;
  const totalPages = Math.ceil(totalRows / PAGE_SIZE);
  const currentPage = page + 1;

  // Derive categorical distribution chart data from summary
  const seasonalityData = useMemo(
    () => recordToChartData(summary?.distributions?.seasonality_profile),
    [summary?.distributions?.seasonality_profile],
  );
  const variabilityData = useMemo(
    () => recordToChartData(summary?.distributions?.variability_class),
    [summary?.distributions?.variability_class],
  );
  const trendData = useMemo(
    () => recordToChartData(summary?.distributions?.trend_direction),
    [summary?.distributions?.trend_direction],
  );

  // Sort handler
  const handleSort = useCallback(
    (col: string) => {
      if (sortBy === col) {
        setSortDir((d) => (d === "asc" ? "desc" : "asc"));
      } else {
        setSortBy(col);
        setSortDir("asc");
      }
      setPage(0);
    },
    [sortBy],
  );

  // Reset pagination when filters change
  const handleFilterChange = useCallback(
    (setter: (v: string) => void) => (e: React.ChangeEvent<HTMLSelectElement>) => {
      setter(e.target.value);
      setPage(0);
    },
    [],
  );

  const handleSearchChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      setSearch(e.target.value);
      setPage(0);
    },
    [],
  );

  // Histogram colors from theme — use trendColors which provides 6 series colors
  const histogramColorList = trendColors.length >= 6
    ? trendColors.slice(0, 6)
    : ["#3b82f6", "#8b5cf6", "#f59e0b", "#22c55e", "#f97316", "#06b6d4"];

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-start justify-between gap-4">
        <div>
          <h2 className="text-lg font-semibold text-foreground">SKU Features</h2>
          <p className="text-sm text-muted-foreground">
            Explore computed demand features across all SKUs — seasonality profiles, variability classes, trend signals, and statistical metrics.
          </p>
        </div>
        <button
          onClick={() => computeMutation.mutate()}
          disabled={computeMutation.isPending || (!!computeJobId && !jobDone && !jobFailed)}
          className="inline-flex items-center gap-2 rounded-md bg-primary px-4 py-2 text-sm font-medium text-primary-foreground shadow-sm hover:bg-primary/90 disabled:opacity-50 disabled:cursor-not-allowed transition-colors shrink-0"
        >
          {computeMutation.isPending || (computeJobId && !jobDone && !jobFailed) ? (
            <>
              <Loader2 className="h-4 w-4 animate-spin" />
              {jobStatus?.progress_msg || "Computing..."}
            </>
          ) : (
            <>
              <Play className="h-4 w-4" />
              Compute Features
            </>
          )}
        </button>
      </div>

      {/* Job status banner */}
      {computeJobId && (
        <div className={`rounded-md border px-4 py-2.5 text-sm ${
          jobDone
            ? "border-green-200 bg-green-50 text-green-800 dark:border-green-800 dark:bg-green-950/30 dark:text-green-300"
            : jobFailed
              ? "border-red-200 bg-red-50 text-red-800 dark:border-red-800 dark:bg-red-950/30 dark:text-red-300"
              : "border-blue-200 bg-blue-50 text-blue-800 dark:border-blue-800 dark:bg-blue-950/30 dark:text-blue-300"
        }`}>
          {jobDone && "Feature computation completed. Refreshing data..."}
          {jobFailed && "Feature computation failed. Check job logs for details."}
          {!jobDone && !jobFailed && (
            <>
              Computing features...
              {jobStatus?.progress_pct != null && ` (${jobStatus.progress_pct}%)`}
              {jobStatus?.progress_msg && ` — ${jobStatus.progress_msg}`}
            </>
          )}
        </div>
      )}

      {/* Mutation error banner */}
      {computeMutation.isError && !computeJobId && (
        <div className="rounded-md border border-red-200 bg-red-50 px-4 py-2.5 text-sm text-red-800 dark:border-red-800 dark:bg-red-950/30 dark:text-red-300">
          Failed to start computation: {computeMutation.error?.message ?? "Unknown error"}
        </div>
      )}

      {/* ================================================================== */}
      {/* SECTION 1: Summary Cards                                           */}
      {/* ================================================================== */}
      <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
        <SummaryCard
          icon={Database}
          label="Total SKUs"
          value={summary ? summary.total_skus.toLocaleString() : "\u2014"}
          isLoading={summaryLoading}
        />
        <SummaryCard
          icon={Clock}
          label="Last Computed"
          value={summary?.last_computed ? relativeTime(summary.last_computed) : "Never"}
          subtitle={summary?.last_computed ? new Date(summary.last_computed).toLocaleString() : undefined}
          isLoading={summaryLoading}
        />
        <SummaryCard
          icon={Waves}
          label="Avg CV Demand"
          value={formatNumber(summary?.averages?.cv_demand)}
          subtitle="Coefficient of variation"
          isLoading={summaryLoading}
        />
        <SummaryCard
          icon={TrendingUp}
          label="Avg Seasonal Amplitude"
          value={formatNumber(summary?.averages?.seasonal_amplitude)}
          subtitle="Peak-to-trough ratio"
          isLoading={summaryLoading}
        />
      </div>

      {/* ================================================================== */}
      {/* SECTION 2: Distribution Charts (3 side by side)                    */}
      {/* ================================================================== */}
      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
        <HorizontalDistribution
          title={DISTRIBUTION_TITLES.seasonality_profile}
          data={seasonalityData}
          colors={DISTRIBUTION_COLORS.seasonality_profile}
          isLoading={summaryLoading}
        />
        <HorizontalDistribution
          title={DISTRIBUTION_TITLES.variability_class}
          data={variabilityData}
          colors={DISTRIBUTION_COLORS.variability_class}
          isLoading={summaryLoading}
        />
        <HorizontalDistribution
          title={DISTRIBUTION_TITLES.trend_direction}
          data={trendData}
          colors={DISTRIBUTION_COLORS.trend_direction}
          isLoading={summaryLoading}
        />
      </div>

      {/* ================================================================== */}
      {/* SECTION 3: Feature Distribution Histograms (2x3 grid)             */}
      {/* ================================================================== */}
      <div>
        <h3 className="mb-2 text-sm font-medium text-foreground">Feature Distributions</h3>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-3">
          {HISTOGRAM_FEATURES.map((key, idx) => (
            <FeatureHistogram
              key={key}
              title={HISTOGRAM_LABELS[key]}
              data={distributions?.features?.[key] ?? []}
              color={histogramColorList[idx % histogramColorList.length]}
              isLoading={distLoading}
            />
          ))}
        </div>
      </div>

      {/* ================================================================== */}
      {/* SECTION 4: Feature Table                                           */}
      {/* ================================================================== */}
      <div className="rounded-lg border border-border bg-card">
        {/* Table header: filters */}
        <div className="flex flex-wrap items-center gap-2 border-b border-border px-4 py-3">
          <h3 className="text-sm font-medium text-foreground mr-auto">
            Feature Table
            {totalRows > 0 && (
              <span className="ml-2 text-xs font-normal text-muted-foreground">
                ({totalRows.toLocaleString()} SKUs)
              </span>
            )}
          </h3>
          <div className="relative">
            <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <input
              type="text"
              placeholder="Search item_id..."
              value={search}
              onChange={handleSearchChange}
              className="h-8 w-44 rounded-md border border-input bg-background pl-8 pr-2.5 text-xs"
            />
          </div>
          <select
            value={seasonalityFilter}
            onChange={handleFilterChange(setSeasonalityFilter)}
            className="h-8 rounded-md border border-input bg-background px-2.5 text-xs"
          >
            <option value="">All Seasonality</option>
            {SEASONALITY_OPTIONS.filter(Boolean).map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
          <select
            value={variabilityFilter}
            onChange={handleFilterChange(setVariabilityFilter)}
            className="h-8 rounded-md border border-input bg-background px-2.5 text-xs"
          >
            <option value="">All Variability</option>
            {VARIABILITY_OPTIONS.filter(Boolean).map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
          <select
            value={trendFilter}
            onChange={handleFilterChange(setTrendFilter)}
            className="h-8 rounded-md border border-input bg-background px-2.5 text-xs"
          >
            <option value="">All Trends</option>
            {TREND_OPTIONS.filter(Boolean).map((v) => (
              <option key={v} value={v}>{v}</option>
            ))}
          </select>
        </div>

        {/* Table body */}
        <div className="overflow-x-auto">
          {listLoading ? (
            <div className="p-4 space-y-3">
              {Array.from({ length: 8 }).map((_, i) => (
                <Skeleton key={i} className="h-6 w-full" />
              ))}
            </div>
          ) : rows.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-16 text-muted-foreground">
              <Database className="h-10 w-10 text-muted-foreground/30 mb-3" />
              <p className="text-sm font-medium">No SKU features found</p>
              <p className="text-xs mt-1 max-w-xs text-center">
                {search || seasonalityFilter || variabilityFilter || trendFilter
                  ? "Try adjusting your filters or search query."
                  : "Run the feature computation pipeline to populate SKU features."}
              </p>
            </div>
          ) : (
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border bg-muted/30">
                  {SORTABLE_COLUMNS.map((col) => (
                    <th
                      key={col.key}
                      className={`cursor-pointer select-none px-3 py-2.5 font-medium text-muted-foreground hover:text-foreground transition-colors ${
                        col.align === "right" ? "text-right" : "text-left"
                      }`}
                      onClick={() => handleSort(col.key)}
                    >
                      <span className="inline-flex items-center gap-1">
                        {col.label}
                        {sortBy === col.key && (
                          sortDir === "asc"
                            ? <ChevronUp className="h-3 w-3" />
                            : <ChevronDown className="h-3 w-3" />
                        )}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr
                    key={row.sku_ck}
                    className="border-b border-border/30 transition-colors hover:bg-muted/20"
                  >
                    <td className="px-3 py-2 font-mono text-xs font-medium">{row.sku_ck}</td>
                    <td className="px-3 py-2 font-mono">{row.item_id}</td>
                    <td className="px-3 py-2">{row.loc}</td>
                    <td className="px-3 py-2 text-right tabular-nums">
                      {row.ml_cluster ?? "\u2014"}
                    </td>
                    <td className="px-3 py-2">
                      {row.seasonality_profile ? (
                        <span className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-medium ${badgeClass(row.seasonality_profile, "seasonality")}`}>
                          {row.seasonality_profile}
                        </span>
                      ) : "\u2014"}
                    </td>
                    <td className="px-3 py-2">
                      {row.variability_class ? (
                        <span className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-medium ${badgeClass(row.variability_class, "variability")}`}>
                          {row.variability_class}
                        </span>
                      ) : "\u2014"}
                    </td>
                    <td className="px-3 py-2">
                      {row.trend_direction != null ? (
                        <span className={`inline-block rounded-full px-2 py-0.5 text-[10px] font-medium ${badgeClass(trendLabel(row.trend_direction), "trend")}`}>
                          {trendLabel(row.trend_direction)}
                        </span>
                      ) : "\u2014"}
                    </td>
                    <td className="px-3 py-2 text-right tabular-nums">{formatNumber(row.cv_demand)}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{formatNumber(row.seasonal_amplitude)}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{formatPct(row.zero_demand_pct)}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{formatPct(row.cagr)}</td>
                    <td className="px-3 py-2 text-right tabular-nums">{formatNumber(row.recency_ratio)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>

        {/* Pagination */}
        {totalPages > 1 && (
          <div className="flex items-center justify-between border-t border-border px-4 py-2.5">
            <span className="text-xs text-muted-foreground">
              Showing {page * PAGE_SIZE + 1}\u2013{Math.min((page + 1) * PAGE_SIZE, totalRows)} of{" "}
              {totalRows.toLocaleString()}
            </span>
            <div className="flex items-center gap-2">
              <button
                disabled={page === 0}
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                className="rounded border border-input px-3 py-1 text-xs disabled:opacity-40 hover:bg-muted transition-colors"
              >
                Previous
              </button>
              <span className="text-xs text-muted-foreground tabular-nums">
                Page {currentPage} of {totalPages}
              </span>
              <button
                disabled={currentPage >= totalPages}
                onClick={() => setPage((p) => p + 1)}
                className="rounded border border-input px-3 py-1 text-xs disabled:opacity-40 hover:bg-muted transition-colors"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
