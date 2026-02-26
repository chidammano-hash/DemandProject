import { useCallback, useMemo, useState } from "react";
import {
  useQuery,
  useMutation,
  useQueryClient,
} from "@tanstack/react-query";
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
import { ChartColumn, Loader2, Trophy } from "lucide-react";

import {
  queryKeys,
  STALE,
  fetchAccuracySlice,
  fetchLagCurve,
  fetchCompetitionConfig,
  fetchCompetitionSummary,
  saveCompetitionConfig,
  runCompetition,
  type CompetitionConfig,
  type ChampionSummary,
  type SliceParams,
  type LagCurveParams,
} from "@/api/queries";
import type { Theme, AccuracyKpis, AccuracySliceRow, LagPoint } from "@/types";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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
import { cn } from "@/lib/utils";
import { ELEMENT_CONFIG } from "@/constants/elements";
import { TREND_COLORS_BY_THEME, CHART_COLORS } from "@/constants/colors";
import { KpiCard } from "@/components/KpiCard";
import { LoadingElement } from "@/components/LoadingElement";
import { titleCase, formatPercent } from "@/lib/formatters";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ACCURACY_KPI_OPTIONS = [
  { key: "accuracy_pct", label: "Accuracy %", format: "pct" },
  { key: "wape",         label: "WAPE %",     format: "pct" },
  { key: "bias",         label: "Bias",        format: "bias" },
  { key: "sum_forecast", label: "\u03A3 Forecast", format: "num" },
  { key: "sum_actual",   label: "\u03A3 Actual",   format: "num" },
  { key: "dfu_count",    label: "DFU Count",  format: "num" },
] as const;

/** Hoisted Recharts margin prop to avoid re-renders from inline objects. */
const CHART_MARGIN = { top: 4, right: 16, left: 0, bottom: 4 };
/** Hoisted dot size prop for Line elements. */
const CHART_DOT_SM = { r: 4 };

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

type AccuracyTabProps = {
  theme: Theme;
};

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function AccuracyTab({ theme }: AccuracyTabProps) {
  const queryClient = useQueryClient();
  const trendColors = TREND_COLORS_BY_THEME[theme];

  // ---- Local state ----------------------------------------------------------
  const [sliceGroupBy, setSliceGroupBy] = useState("cluster_assignment");
  const [sliceLag, setSliceLag] = useState(-1);
  const [sliceModels, setSliceModels] = useState("");
  const [sliceKpis, setSliceKpis] = useState<string[]>(["accuracy_pct", "wape", "bias"]);
  const [lagCurveMetric, setLagCurveMetric] = useState("accuracy_pct");
  const [sliceMonths, setSliceMonths] = useState(12);
  const [commonDfus, setCommonDfus] = useState(false);

  // Competition config is local-mutable (user edits checkboxes/dropdowns)
  const [competitionConfig, setCompetitionConfig] = useState<CompetitionConfig | null>(null);

  // ---- Derived params -------------------------------------------------------
  const monthFrom = useMemo(() => {
    if (sliceGroupBy === "month_start") return "";
    const now = new Date();
    const from = new Date(now.getFullYear(), now.getMonth() - sliceMonths, 1);
    return from.toISOString().slice(0, 10);
  }, [sliceGroupBy, sliceMonths]);

  const needDfuCount = sliceKpis.includes("dfu_count");

  const sliceParams: SliceParams = useMemo(
    () => ({
      group_by: sliceGroupBy,
      lag: sliceLag,
      models: sliceModels,
      month_from: monthFrom,
      common_dfus: commonDfus,
      include_dfu_count: needDfuCount,
    }),
    [sliceGroupBy, sliceLag, sliceModels, monthFrom, commonDfus, needDfuCount],
  );

  const lagCurveParams: LagCurveParams = useMemo(
    () => ({
      models: sliceModels,
      month_from: monthFrom,
      common_dfus: commonDfus,
      include_dfu_count: needDfuCount,
    }),
    [sliceModels, monthFrom, commonDfus, needDfuCount],
  );

  // ---- Data fetching: accuracy slice + lag curve ----------------------------
  const {
    data: slicePayload,
    isLoading: loadingSlice,
  } = useQuery({
    queryKey: queryKeys.accuracySlice(sliceParams),
    queryFn: () => fetchAccuracySlice(sliceParams),
    staleTime: STALE.TWO_MIN,
  });

  const { data: lagPayload } = useQuery({
    queryKey: queryKeys.lagCurve(lagCurveParams),
    queryFn: () => fetchLagCurve(lagCurveParams),
    staleTime: STALE.TWO_MIN,
  });

  const sliceData: AccuracySliceRow[] = slicePayload?.rows ?? [];
  const lagCurveData: LagPoint[] = lagPayload?.by_lag ?? [];
  const commonDfuCount = slicePayload?.common_dfu_count ?? null;
  const dfuCounts = slicePayload?.dfu_counts ?? null;

  // ---- Data fetching: competition config + summary --------------------------
  const { data: configPayload } = useQuery({
    queryKey: queryKeys.competitionConfig(),
    queryFn: fetchCompetitionConfig,
    staleTime: STALE.FIVE_MIN,
    select: (data) => {
      if (data?.config && competitionConfig === null) {
        // Initialise local editable copy on first load
        setCompetitionConfig(data.config);
      }
      return data;
    },
  });

  const availableModels: string[] = configPayload?.available_models ?? [];

  const { data: summaryPayload } = useQuery({
    queryKey: queryKeys.competitionSummary(),
    queryFn: fetchCompetitionSummary,
    staleTime: STALE.FIVE_MIN,
  });

  const championSummary: ChampionSummary | null = summaryPayload?.summary ?? null;

  // ---- Mutations ------------------------------------------------------------
  const saveConfigMutation = useMutation({
    mutationFn: saveCompetitionConfig,
  });

  const runCompetitionMutation = useMutation({
    mutationFn: async (config: CompetitionConfig) => {
      // Save config first, then run
      await saveCompetitionConfig(config);
      return runCompetition();
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.competitionSummary() });
      queryClient.invalidateQueries({ queryKey: queryKeys.accuracySlice(sliceParams) });
    },
  });

  const savingConfig = saveConfigMutation.isPending;
  const runningCompetition = runCompetitionMutation.isPending;

  // ---- Derived / memoised ---------------------------------------------------
  const allModels = useMemo(
    () => Array.from(new Set(sliceData.flatMap((r) => Object.keys(r.by_model)))).sort(),
    [sliceData],
  );

  const lagModels = useMemo(
    () => Array.from(new Set(lagCurveData.flatMap((p) => Object.keys(p.by_model)))).sort(),
    [lagCurveData],
  );

  const activeLagMetric = useMemo(
    () => (sliceKpis.includes(lagCurveMetric) ? lagCurveMetric : sliceKpis[0]),
    [sliceKpis, lagCurveMetric],
  );

  const lagMetricOpt = useMemo(
    () => ACCURACY_KPI_OPTIONS.find((k) => k.key === activeLagMetric),
    [activeLagMetric],
  );

  const chartData = useMemo(() => {
    return lagCurveData.map((p) => {
      const row: Record<string, number | string> = { lag: `Lag ${p.lag}` };
      for (const m of lagModels) {
        const val = p.by_model[m]?.[activeLagMetric as keyof AccuracyKpis];
        if (val !== null && val !== undefined) row[m] = val as number;
      }
      return row;
    });
  }, [lagCurveData, lagModels, activeLagMetric]);

  const yFormatter = useMemo(() => {
    const fmtIsPct = lagMetricOpt?.format === "pct";
    const fmtIsBias = lagMetricOpt?.format === "bias";
    return (v: number) =>
      fmtIsPct
        ? `${Number(v).toFixed(0)}%`
        : fmtIsBias
          ? `${(Number(v) * 100).toFixed(0)}%`
          : Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 });
  }, [lagMetricOpt]);

  const tooltipFormatter = useMemo(() => {
    const fmtIsPct = lagMetricOpt?.format === "pct";
    const fmtIsBias = lagMetricOpt?.format === "bias";
    return (v: number) =>
      fmtIsPct
        ? `${Number(v).toFixed(1)}%`
        : fmtIsBias
          ? `${(Number(v) * 100).toFixed(1)}%`
          : Number(v).toLocaleString(undefined, { maximumFractionDigits: 0 });
  }, [lagMetricOpt]);

  // ---- Callbacks ------------------------------------------------------------
  const handleSliceGroupByChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => setSliceGroupBy(e.target.value),
    [],
  );

  const handleSliceLagChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => setSliceLag(Number(e.target.value)),
    [],
  );

  const handleSliceModelsChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => setSliceModels(e.target.value),
    [],
  );

  const handleSliceMonthsChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => setSliceMonths(Number(e.target.value)),
    [],
  );

  const handleCommonDfusToggle = useCallback(() => setCommonDfus((v) => !v), []);

  const handleLagCurveMetricChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) => setLagCurveMetric(e.target.value),
    [],
  );

  const handleKpiToggle = useCallback((key: string) => {
    setSliceKpis((prev) =>
      prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key],
    );
  }, []);

  const handleCompetingModelToggle = useCallback(
    (model: string) => {
      setCompetitionConfig((prev) => {
        if (!prev) return prev;
        const checked = prev.models.includes(model);
        const next = checked ? prev.models.filter((x) => x !== model) : [...prev.models, model];
        return { ...prev, models: next };
      });
    },
    [],
  );

  const handleMetricChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) =>
      setCompetitionConfig((prev) => (prev ? { ...prev, metric: e.target.value } : prev)),
    [],
  );

  const handleLagChange = useCallback(
    (e: React.ChangeEvent<HTMLSelectElement>) =>
      setCompetitionConfig((prev) => (prev ? { ...prev, lag: e.target.value } : prev)),
    [],
  );

  const handleSaveConfig = useCallback(() => {
    if (!competitionConfig) return;
    saveConfigMutation.mutate(competitionConfig);
  }, [competitionConfig, saveConfigMutation]);

  const handleRunCompetition = useCallback(() => {
    if (!competitionConfig || competitionConfig.models.length < 2) return;
    runCompetitionMutation.mutate(competitionConfig);
  }, [competitionConfig, runCompetitionMutation]);

  // ---- Render ---------------------------------------------------------------
  return (
    <section className="mt-4">
      {/* Accuracy Comparison Card */}
      <Card className="animate-fade-in">
        <CardHeader>
          <div className="flex items-center gap-2">
            <ChartColumn className="h-5 w-5" />
            <CardTitle className="text-base">Accuracy Comparison</CardTitle>
          </div>
          <CardDescription>
            Compare forecast accuracy across models by DFU attribute. Uses pre-aggregated views for fast results.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          {/* ── Filter controls ────────────────────────────────────── */}
          <div className="flex flex-wrap items-end gap-3">
            <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Slice by
              <select
                className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                value={sliceGroupBy}
                onChange={handleSliceGroupByChange}
                disabled={loadingSlice}
              >
                <option value="cluster_assignment">Cluster (Business Label)</option>
                <option value="ml_cluster">Cluster (ML)</option>
                <option value="supplier_desc">Supplier</option>
                <option value="abc_vol">ABC Volume</option>
                <option value="region">Region</option>
                <option value="brand_desc">Brand</option>
                <option value="dfu_execution_lag">Execution Lag</option>
                <option value="month_start">Month</option>
              </select>
            </label>
            <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Lag Filter
              <select
                className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                value={sliceLag}
                onChange={handleSliceLagChange}
                disabled={loadingSlice}
              >
                <option value={-1}>Execution Lag (per DFU)</option>
                <option value={0}>Lag 0 (same month)</option>
                <option value={1}>Lag 1</option>
                <option value={2}>Lag 2</option>
                <option value={3}>Lag 3</option>
                <option value={4}>Lag 4</option>
              </select>
            </label>
            <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              Models (comma-separated, blank = all)
              <input
                className="h-9 w-52 rounded-md border border-input bg-background px-3 text-sm"
                placeholder="e.g. lgbm_global,external"
                value={sliceModels}
                onChange={handleSliceModelsChange}
                disabled={loadingSlice}
              />
            </label>
            {sliceGroupBy !== "month_start" ? (
              <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                KPI Window
                <select
                  className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                  value={sliceMonths}
                  onChange={handleSliceMonthsChange}
                  disabled={loadingSlice}
                >
                  {Array.from({ length: 12 }, (_, idx) => idx + 1).map((m) => (
                    <option key={m} value={m}>
                      {m} month{m > 1 ? "s" : ""}
                    </option>
                  ))}
                </select>
              </label>
            ) : null}
            <label className="flex items-center gap-1.5 self-end pb-1.5 cursor-pointer select-none">
              <input
                type="checkbox"
                className="h-3.5 w-3.5 rounded border-input accent-indigo-700"
                checked={commonDfus}
                onChange={handleCommonDfusToggle}
                disabled={loadingSlice}
              />
              <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground whitespace-nowrap">
                Common DFUs Only
              </span>
            </label>
            {commonDfus && commonDfuCount != null && dfuCounts ? (
              <div className="flex items-center gap-2 self-end pb-1.5 text-xs text-muted-foreground tabular-nums">
                <Badge variant="secondary" className="font-mono text-xs">
                  {commonDfuCount.toLocaleString()} common
                </Badge>
                {Object.entries(dfuCounts).map(([m, cnt]) => (
                  <span key={m} className="font-mono">
                    {m}: {cnt.toLocaleString()}
                  </span>
                ))}
              </div>
            ) : null}
            {loadingSlice ? (
              <div className="flex items-center gap-2">
                <div
                  className={cn(
                    "flex flex-col items-center justify-center rounded-md border px-1.5 py-0.5 animate-pulse-glow",
                    ELEMENT_CONFIG.accuracy.activeColor,
                    ELEMENT_CONFIG.accuracy.glow,
                  )}
                >
                  <span className="text-[9px] font-bold font-mono leading-tight">
                    {ELEMENT_CONFIG.accuracy.symbol}
                  </span>
                </div>
                <span className="text-xs text-muted-foreground">Loading...</span>
              </div>
            ) : null}
          </div>

          {/* ── KPI checkboxes ─────────────────────────────────────── */}
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">KPIs</span>
            {ACCURACY_KPI_OPTIONS.map((opt) => {
              const checked = sliceKpis.includes(opt.key);
              const isLast = sliceKpis.length === 1 && checked;
              return (
                <label key={opt.key} className="flex items-center gap-1.5 text-sm cursor-pointer select-none">
                  <input
                    type="checkbox"
                    className="h-3.5 w-3.5 rounded border-input accent-indigo-700"
                    checked={checked}
                    disabled={isLast}
                    onChange={() => handleKpiToggle(opt.key)}
                  />
                  {opt.label}
                </label>
              );
            })}
          </div>

          {/* ── Model comparison table ─────────────────────────────── */}
          {sliceData.length > 0 ? (
            <div className="space-y-2">
              <p className="text-xs uppercase tracking-wide text-muted-foreground">
                Model Comparison &mdash; {sliceData.length} {sliceGroupBy.replace(/_/g, " ")} bucket(s)
              </p>
              <div className="max-h-[400px] overflow-auto rounded-md border border-input">
                <Table>
                  <TableHeader>
                    <TableRow className="border-muted bg-muted/30">
                      <TableHead className="text-xs sticky left-0 bg-muted/30">
                        {titleCase(sliceGroupBy)}
                      </TableHead>
                      {allModels.flatMap((m) =>
                        ACCURACY_KPI_OPTIONS.filter((k) => sliceKpis.includes(k.key)).map((k) => (
                          <TableHead key={`${m}-${k.key}`} className="text-xs text-right">
                            {m} {k.label}
                          </TableHead>
                        )),
                      )}
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {sliceData.map((row) => {
                      const accValues = allModels
                        .map((m) => row.by_model[m]?.accuracy_pct)
                        .filter((v): v is number => v !== null && v !== undefined);
                      const bestAcc = accValues.length > 0 ? Math.max(...accValues) : null;
                      return (
                        <TableRow key={row.bucket} className="hover:bg-muted/30">
                          <TableCell className="sticky left-0 bg-background font-medium text-sm">
                            {row.bucket}
                          </TableCell>
                          {allModels.flatMap((m) => {
                            const kpi = row.by_model[m];
                            return ACCURACY_KPI_OPTIONS.filter((k) => sliceKpis.includes(k.key)).map((k) => {
                              const val = kpi?.[k.key as keyof AccuracyKpis] as number | null | undefined;
                              const isBestAcc =
                                k.key === "accuracy_pct" &&
                                val !== null &&
                                val !== undefined &&
                                val === bestAcc;
                              const isBadBias =
                                k.key === "bias" &&
                                val !== null &&
                                val !== undefined &&
                                Math.abs(val) > 0.15;
                              let display: string;
                              if (val === null || val === undefined) {
                                display = "-";
                              } else if (k.format === "pct") {
                                display = formatPercent(val);
                              } else if (k.format === "bias") {
                                display = `${(val * 100).toFixed(1)}%`;
                              } else {
                                display = Number(val).toLocaleString(undefined, {
                                  maximumFractionDigits: 0,
                                });
                              }
                              return (
                                <TableCell
                                  key={`${m}-${k.key}`}
                                  className={cn(
                                    "text-right text-sm tabular-nums",
                                    isBestAcc ? "font-bold text-indigo-700 dark:text-indigo-400" : "",
                                    isBadBias ? "text-red-600 dark:text-red-400" : "",
                                  )}
                                >
                                  {isBestAcc && (
                                    <span className="mr-0.5" title="Best accuracy">
                                      &#9733;
                                    </span>
                                  )}
                                  {isBadBias && (
                                    <span className="mr-0.5" title="High bias (|bias| > 15%)">
                                      &#9888;
                                    </span>
                                  )}
                                  {display}
                                </TableCell>
                              );
                            });
                          })}
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
              <p className="text-xs text-muted-foreground">
                &#9733; = best accuracy for that row. &#9888; = |bias| &gt; 15%.
              </p>
            </div>
          ) : loadingSlice ? (
            <LoadingElement config={ELEMENT_CONFIG.accuracy} message="Loading accuracy data..." />
          ) : (
            <p className="text-sm text-muted-foreground">
              No data. Run <code className="rounded bg-muted px-1">make backtest-load</code> to populate the accuracy views.
            </p>
          )}

          {/* ── Lag curve chart ─────────────────────────────────────── */}
          {lagCurveData.length > 0 ? (
            <div className="space-y-2">
              <div className="flex items-center gap-3">
                <p className="text-xs uppercase tracking-wide text-muted-foreground">
                  {lagMetricOpt?.label ?? "KPI"} by Lag Horizon
                </p>
                <select
                  className="h-7 rounded-md border border-input bg-background px-2 text-xs"
                  value={activeLagMetric}
                  onChange={handleLagCurveMetricChange}
                >
                  {ACCURACY_KPI_OPTIONS.filter((k) => sliceKpis.includes(k.key)).map((k) => (
                    <option key={k.key} value={k.key}>
                      {k.label}
                    </option>
                  ))}
                </select>
              </div>
              <ResponsiveContainer width="100%" height={220}>
                <LineChart data={chartData} margin={CHART_MARGIN}>
                  <CartesianGrid strokeDasharray="3 3" stroke={CHART_COLORS[theme].grid} />
                  <XAxis
                    dataKey="lag"
                    tick={{ fontSize: 11, fill: CHART_COLORS[theme].axis }}
                  />
                  <YAxis
                    domain={["auto", "auto"]}
                    tick={{ fontSize: 11, fill: CHART_COLORS[theme].axis }}
                    tickFormatter={yFormatter}
                  />
                  <Tooltip
                    formatter={tooltipFormatter}
                    contentStyle={{
                      backgroundColor: CHART_COLORS[theme].tooltip_bg,
                      borderColor: CHART_COLORS[theme].tooltip_border,
                    }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  {lagModels.map((m, i) => (
                    <Line
                      key={m}
                      type="monotone"
                      dataKey={m}
                      stroke={trendColors[i % trendColors.length]}
                      strokeWidth={2}
                      dot={CHART_DOT_SM}
                      connectNulls
                    />
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          ) : null}
        </CardContent>
      </Card>

      {/* ── Champion Selection panel (feature15) ───────────────────── */}
      <Card className="animate-fade-in mt-4">
        <CardHeader>
          <div className="flex items-center gap-2">
            <Trophy className="h-5 w-5" />
            <CardTitle className="text-base">Champion Selection</CardTitle>
          </div>
          <CardDescription>
            Pick the best model per DFU based on forecast accuracy. Configure which models compete, then run the selection.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          {competitionConfig ? (
            <>
              {/* Competing Models checkboxes */}
              <div className="space-y-3">
                <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Competing Models
                </span>
                <div className="flex flex-wrap gap-3">
                  {availableModels
                    .filter((m) => m !== competitionConfig.champion_model_id && m !== "ceiling")
                    .map((m) => {
                      const checked = competitionConfig.models.includes(m);
                      const isLast = competitionConfig.models.length <= 2 && checked;
                      return (
                        <label key={m} className="flex items-center gap-1.5 text-sm cursor-pointer select-none">
                          <input
                            type="checkbox"
                            className="h-3.5 w-3.5 rounded border-input accent-indigo-700"
                            checked={checked}
                            disabled={isLast || runningCompetition}
                            onChange={() => handleCompetingModelToggle(m)}
                          />
                          <span className="font-mono text-xs">{m}</span>
                        </label>
                      );
                    })}
                </div>
              </div>

              {/* Metric + Lag dropdowns + action buttons */}
              <div className="flex flex-wrap items-end gap-3">
                <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Metric
                  <select
                    className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                    value={competitionConfig.metric}
                    onChange={handleMetricChange}
                    disabled={runningCompetition}
                  >
                    <option value="wape">WAPE (Lowest Wins)</option>
                    <option value="accuracy_pct">Accuracy % (Highest Wins)</option>
                  </select>
                </label>
                <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Lag
                  <select
                    className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                    value={competitionConfig.lag}
                    onChange={handleLagChange}
                    disabled={runningCompetition}
                  >
                    <option value="execution">Execution Lag (per DFU)</option>
                    <option value="0">Lag 0 (same month)</option>
                    <option value="1">Lag 1</option>
                    <option value="2">Lag 2</option>
                    <option value="3">Lag 3</option>
                    <option value="4">Lag 4</option>
                  </select>
                </label>
                <Button
                  variant="outline"
                  size="sm"
                  disabled={savingConfig || runningCompetition}
                  onClick={handleSaveConfig}
                >
                  {savingConfig ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : null}
                  Save Config
                </Button>
                <Button
                  size="sm"
                  disabled={runningCompetition || competitionConfig.models.length < 2}
                  onClick={handleRunCompetition}
                >
                  {runningCompetition ? (
                    <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                  ) : (
                    <Trophy className="mr-1 h-3 w-3" />
                  )}
                  Run Competition
                </Button>
              </div>
            </>
          ) : (
            <p className="text-sm text-muted-foreground">Loading competition config...</p>
          )}

          {/* ── Results summary ──────────────────────────────────────── */}
          {championSummary ? (
            <div className="space-y-3 rounded-lg border bg-muted/40 p-4">
              <div className="flex flex-wrap items-center gap-4">
                <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">Results</span>
                <span className="text-xs text-muted-foreground">
                  Last run: {new Date(championSummary.run_ts).toLocaleString()}
                </span>
              </div>

              {/* Champion KPI cards */}
              <div className="flex flex-wrap gap-4 text-sm">
                <KpiCard
                  label="DFUs Evaluated"
                  value={championSummary.total_dfus.toLocaleString()}
                  sublabel={
                    championSummary.total_dfu_months
                      ? `${championSummary.total_dfu_months.toLocaleString()} DFU-months`
                      : undefined
                  }
                />
                <KpiCard
                  label="Champion Accuracy"
                  value={
                    championSummary.overall_champion_accuracy_pct != null
                      ? `${championSummary.overall_champion_accuracy_pct.toFixed(2)}%`
                      : "-"
                  }
                  colorClass="text-indigo-700 dark:text-indigo-400 midnight:text-indigo-300"
                />
                <KpiCard
                  label="Champion WAPE"
                  value={
                    championSummary.overall_champion_wape != null
                      ? `${championSummary.overall_champion_wape.toFixed(2)}%`
                      : "-"
                  }
                />
                <KpiCard label="Champion Rows" value={championSummary.total_champion_rows.toLocaleString()} />
              </div>

              {/* Ceiling (Oracle) KPI cards */}
              {championSummary.overall_ceiling_accuracy_pct != null && (
                <div className="flex flex-wrap gap-4 text-sm">
                  <KpiCard
                    label="Ceiling Accuracy"
                    sublabel="(oracle)"
                    value={`${championSummary.overall_ceiling_accuracy_pct.toFixed(2)}%`}
                    colorClass="text-emerald-700 dark:text-emerald-400"
                    borderClass="border-emerald-200 dark:border-emerald-800"
                  />
                  <KpiCard
                    label="Ceiling WAPE"
                    sublabel="(oracle)"
                    value={
                      championSummary.overall_ceiling_wape != null
                        ? `${championSummary.overall_ceiling_wape.toFixed(2)}%`
                        : "-"
                    }
                    colorClass="text-emerald-700 dark:text-emerald-400"
                    borderClass="border-emerald-200 dark:border-emerald-800"
                  />
                  {championSummary.total_ceiling_rows != null && (
                    <KpiCard
                      label="Ceiling Rows"
                      value={championSummary.total_ceiling_rows.toLocaleString()}
                      borderClass="border-emerald-200 dark:border-emerald-800"
                    />
                  )}
                  {championSummary.overall_champion_accuracy_pct != null && (
                    <KpiCard
                      label="Gap to Ceiling"
                      value={`${(championSummary.overall_ceiling_accuracy_pct - championSummary.overall_champion_accuracy_pct).toFixed(2)} pp`}
                      colorClass="text-amber-700 dark:text-amber-400"
                      borderClass="border-amber-200 dark:border-amber-800"
                    />
                  )}
                </div>
              )}

              {/* Champion model wins bar chart */}
              <div className="space-y-1.5">
                <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Champion Model Wins (best model per DFU per month, before-the-fact)
                </p>
                {Object.entries(championSummary.model_wins).map(([model, wins]) => {
                  const total = championSummary.total_dfu_months ?? championSummary.total_dfus;
                  const pct = total > 0 ? (wins / total) * 100 : 0;
                  return (
                    <div key={model} className="flex items-center gap-2 text-sm">
                      <span className="w-40 truncate font-mono text-xs text-right">{model}</span>
                      <div className="flex-1 h-5 rounded bg-muted overflow-hidden">
                        <div
                          className="h-full rounded bg-indigo-500 transition-all"
                          style={{ width: `${Math.max(pct, 1)}%` }}
                        />
                      </div>
                      <span className="w-24 text-xs tabular-nums text-muted-foreground">
                        {wins.toLocaleString()} ({pct.toFixed(1)}%)
                      </span>
                    </div>
                  );
                })}
              </div>

              {/* Ceiling model wins bar chart */}
              {championSummary.ceiling_model_wins &&
                Object.keys(championSummary.ceiling_model_wins).length > 0 && (
                  <div className="space-y-1.5">
                    <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      Ceiling Model Wins &mdash; Oracle (best model per DFU per month, after-the-fact)
                    </p>
                    {(() => {
                      const totalCeil = Object.values(championSummary.ceiling_model_wins!).reduce(
                        (a, b) => a + b,
                        0,
                      );
                      return Object.entries(championSummary.ceiling_model_wins!).map(([model, wins]) => {
                        const pct = totalCeil > 0 ? (wins / totalCeil) * 100 : 0;
                        return (
                          <div key={model} className="flex items-center gap-2 text-sm">
                            <span className="w-40 truncate font-mono text-xs text-right">{model}</span>
                            <div className="flex-1 h-5 rounded bg-muted overflow-hidden">
                              <div
                                className="h-full rounded bg-emerald-500 transition-all"
                                style={{ width: `${Math.max(pct, 1)}%` }}
                              />
                            </div>
                            <span className="w-24 text-xs tabular-nums text-muted-foreground">
                              {wins.toLocaleString()} ({pct.toFixed(1)}%)
                            </span>
                          </div>
                        );
                      });
                    })()}
                  </div>
                )}
            </div>
          ) : null}
        </CardContent>
      </Card>
    </section>
  );
}
