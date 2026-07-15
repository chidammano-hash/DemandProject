/**
 * AccuracyBudgetPanel — Accuracy budget analysis: waterfall, gap decomposition,
 * ABC targets, monthly trends, model comparison, and forecast value.
 *
 * Sub-sections (inner tabs):
 *   1. Waterfall — Naive baseline -> ML model -> Oracle ceiling
 *   2. Gap Decomposition — addressable gap breakdown by component
 *   3. ABC Targets — ABC accuracy table with target indicators
 *   4. Monthly Trend — accuracy trend line chart
 *   5. Model Comparison — all models + oracle side-by-side
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
  LineChart,
  Line,
  ReferenceLine,
  Legend,
  Cell,
} from "recharts";
import { TrendingUp, Target, Award, BarChart2, Layers } from "lucide-react";

import {
  accuracyBudgetKeys,
  fetchAccuracyDecomposition,
  fetchAbcBreakdown,
  fetchMonthlyTrend,
  fetchModelComparison,
  fetchForecastValue,
  STALE,
} from "@/api/queries";
import { useChartColors } from "@/hooks/useChartColors";
import { formatPct, formatFixed, formatInt } from "@/lib/formatters";
import { modelLabel } from "@/lib/model-labels";
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

// ---------------------------------------------------------------------------
// Sub-tab types
// ---------------------------------------------------------------------------
type SubSection = "waterfall" | "gap" | "abc" | "trend" | "models";

const SUB_TABS: { key: SubSection; label: string; icon: typeof TrendingUp }[] = [
  { key: "waterfall", label: "Accuracy Waterfall", icon: BarChart2 },
  { key: "gap", label: "Gap Decomposition", icon: Layers },
  { key: "abc", label: "ABC Targets", icon: Target },
  { key: "trend", label: "Monthly Trend", icon: TrendingUp },
  { key: "models", label: "Model Comparison", icon: Award },
];

// Waterfall step colors from the semantic palette: baseline is muted,
// the model is the forecast petrol, the oracle ceiling is teal, the
// addressable gap is warning amber.
function waterfallColors(charts: ReturnType<typeof useChartColors>): Record<string, string> {
  return {
    baseline: charts.fallback[0],
    model: charts.roles.forecast,
    oracle: charts.roles.ceiling,
    gap: charts.roles.warning,
  };
}

// ---------------------------------------------------------------------------
// 1. Waterfall — built from decomposition endpoint
// ---------------------------------------------------------------------------
function WaterfallSection() {
  const colors = useChartColors();
  const { chartColors } = colors;
  const stepColors = waterfallColors(colors);
  const { data, isLoading, isError } = useQuery({
    queryKey: accuracyBudgetKeys.decomposition(),
    queryFn: () => fetchAccuracyDecomposition(),
    staleTime: STALE.FIVE_MIN,
  });

  const { data: valueData } = useQuery({
    queryKey: accuracyBudgetKeys.forecastValue(),
    queryFn: fetchForecastValue,
    staleTime: STALE.FIVE_MIN,
  });

  if (isLoading) return <LoadingElement message="Loading waterfall..." />;
  if (isError || !data || data.current_accuracy == null) {
    return (
      <p className="text-sm text-muted-foreground text-center py-8">
        No waterfall data available.
      </p>
    );
  }

  // Build waterfall steps from decomposition response
  const steps = [
    { label: "Naive Baseline", value: data.naive_baseline ?? 0, type: "baseline" },
    { label: "ML Model", value: data.current_accuracy, type: "model" },
    { label: "Oracle Ceiling", value: data.oracle_ceiling ?? 0, type: "oracle" },
    { label: "Addressable Gap", value: data.addressable_gap ?? 0, type: "gap" },
  ];

  const chartData = steps.map((s) => ({
    label: s.label,
    value: s.value,
    type: s.type,
    fill: stepColors[s.type] ?? colors.fallback[0],
  }));

  // Value-added cards from forecast-value endpoint
  const baselines = valueData?.baselines ?? [];
  const mlAccuracy = valueData?.ml_model?.accuracy;
  const valueAdded = valueData?.value_added;

  return (
    <div className="space-y-4">
      {/* Value-added KPI cards */}
      {baselines.length > 0 && mlAccuracy != null && (
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {baselines.map((b) => {
            const key = b.name === "seasonal_naive" ? "vs_seasonal_naive"
              : b.name === "rolling_3m_avg" ? "vs_rolling_3m" : "vs_flat";
            const delta = valueAdded?.[key];
            return (
              <KpiCard
                key={b.name}
                label={`vs ${b.description}`}
                value={delta != null ? `+${formatPct(delta)}` : "--"}
                sublabel={`${formatPct(b.accuracy)} -> ${formatPct(mlAccuracy)}`}
                severity={delta != null && delta > 0 ? "best" : "warning"}
                size="md"
              />
            );
          })}
        </div>
      )}

      {/* Waterfall bar chart */}
      <div className="h-72">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} vertical={false} />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 11, fill: chartColors.axis }}
              tickLine={false}
            />
            <YAxis
              tick={{ fontSize: 10, fill: chartColors.axis }}
              tickLine={false}
              axisLine={false}
              domain={[0, 100]}
              tickFormatter={(v: number) => formatPct(v)}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: chartColors.tooltip_bg,
                border: `1px solid ${chartColors.tooltip_border}`,
                borderRadius: "8px",
                fontSize: "12px",
              }}
              formatter={(value: number) => [formatPct(value), "Accuracy"]}
            />
            <Bar dataKey="value" radius={[4, 4, 0, 0]} maxBarSize={60} name="Accuracy">
              {chartData.map((entry) => (
                <Cell key={entry.label} fill={entry.fill} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 justify-center">
        {Object.entries(stepColors).map(([key, color]) => (
          <div key={key} className="flex items-center gap-1.5">
            <div className="h-2.5 w-2.5 rounded-sm" style={{ backgroundColor: color }} />
            <span className="text-xs text-muted-foreground capitalize">{key}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 2. Gap Decomposition — from decomposition endpoint
// ---------------------------------------------------------------------------
function GapSection() {
  const { chartColors, trendColors } = useChartColors();
  const { data, isLoading, isError } = useQuery({
    queryKey: accuracyBudgetKeys.decomposition(),
    queryFn: () => fetchAccuracyDecomposition(),
    staleTime: STALE.FIVE_MIN,
  });

  const components = data?.components ?? [];

  if (isLoading) return <LoadingElement message="Loading gap decomposition..." />;
  if (isError || components.length === 0) {
    return (
      <p className="text-sm text-muted-foreground text-center py-8">
        No gap decomposition data available.
      </p>
    );
  }

  const sorted = [...components].sort(
    (a, b) => b.estimated_gain_pp - a.estimated_gain_pp,
  );

  const chartData = sorted.map((c) => ({
    component: c.name,
    gain: c.estimated_gain_pp,
    rationale: c.rationale,
  }));

  return (
    <div className="space-y-4">
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} vertical={false} />
            <XAxis
              dataKey="component"
              tick={{ fontSize: 10, fill: chartColors.axis }}
              tickLine={false}
              interval={0}
              angle={-20}
              textAnchor="end"
              height={60}
            />
            <YAxis
              tick={{ fontSize: 10, fill: chartColors.axis }}
              tickLine={false}
              axisLine={false}
              label={{ value: "Est. Gain (pp)", angle: -90, position: "insideLeft", fontSize: 11, fill: chartColors.axis }}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: chartColors.tooltip_bg,
                border: `1px solid ${chartColors.tooltip_border}`,
                borderRadius: "8px",
                fontSize: "12px",
              }}
              formatter={(value: number) => [`+${formatFixed(value, 1)} pp`, "Estimated Gain"]}
            />
            <Bar dataKey="gain" fill={trendColors[2]} radius={[4, 4, 0, 0]} name="Gap Share" />
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Component detail table */}
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Component</TableHead>
              <TableHead className="text-right">Est. Gain (pp)</TableHead>
              <TableHead>Rationale</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sorted.map((c) => (
              <TableRow key={c.name}>
                <TableCell className="font-medium text-sm">{c.name}</TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  +{formatFixed(c.estimated_gain_pp, 1)}
                </TableCell>
                <TableCell className="text-xs text-muted-foreground max-w-[300px]">
                  {c.rationale}
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
// 3. ABC Accuracy Targets
// ---------------------------------------------------------------------------
function AbcTargetsSection() {
  const { data, isLoading, isError } = useQuery({
    queryKey: accuracyBudgetKeys.abc(),
    queryFn: fetchAbcBreakdown,
    staleTime: STALE.FIVE_MIN,
  });

  const rows = data?.classes ?? [];
  const ABC_TARGETS: Record<string, number> = { A: 80, B: 70, C: 55 };

  if (isLoading) return <LoadingElement message="Loading ABC targets..." />;
  if (isError || rows.length === 0) {
    return (
      <p className="text-sm text-muted-foreground text-center py-8">
        No ABC accuracy target data available.
      </p>
    );
  }

  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead className="w-24">ABC Class</TableHead>
            <TableHead className="text-right">DFUs</TableHead>
            <TableHead className="text-right">Accuracy</TableHead>
            <TableHead className="text-right">Target</TableHead>
            <TableHead className="text-right">WAPE</TableHead>
            <TableHead className="text-right">Bias</TableHead>
            <TableHead className="text-right">Vol Share</TableHead>
            <TableHead className="w-24">Status</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {rows.map((row) => {
            const target = ABC_TARGETS[row.abc_class] ?? 60;
            const accuracy = row.accuracy_pct ?? 0;
            const meetingTarget = accuracy >= target;
            return (
              <TableRow key={row.abc_class}>
                <TableCell className="font-mono text-sm font-bold">{row.abc_class}</TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {formatInt(row.n_dfus)}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm font-medium">
                  {formatPct(row.accuracy_pct)}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm text-muted-foreground">
                  {formatPct(target)}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {formatFixed(row.wape, 2)}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {formatFixed(row.bias, 4)}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {row.volume_share != null ? formatPct(row.volume_share * 100) : "--"}
                </TableCell>
                <TableCell>
                  <Badge
                    className={cn(
                      "text-[10px] px-2 py-0.5",
                      meetingTarget
                        ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300"
                        : "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300",
                    )}
                  >
                    {meetingTarget ? "On Target" : "Below Target"}
                  </Badge>
                </TableCell>
              </TableRow>
            );
          })}
        </TableBody>
      </Table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 4. Monthly Accuracy Trend
// ---------------------------------------------------------------------------
function TrendSection() {
  const { chartColors, trendColors, roles } = useChartColors();
  const { data, isLoading, isError } = useQuery({
    queryKey: accuracyBudgetKeys.monthly(),
    queryFn: fetchMonthlyTrend,
    staleTime: STALE.FIVE_MIN,
  });

  const points = data?.months ?? [];

  if (isLoading) return <LoadingElement message="Loading monthly trend..." />;
  if (isError || points.length === 0) {
    return (
      <p className="text-sm text-muted-foreground text-center py-8">
        No monthly trend data available.
      </p>
    );
  }

  const MONTH_LABELS = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  const chartData = points.map((p) => ({
    month: MONTH_LABELS[p.month] ?? `M${p.month}`,
    accuracy: p.accuracy,
    wape: p.wape,
    flag: p.flag,
  }));

  return (
    <div className="h-72">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
          <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} vertical={false} />
          <XAxis
            dataKey="month"
            tick={{ fontSize: 10, fill: chartColors.axis }}
            tickLine={false}
          />
          <YAxis
            tick={{ fontSize: 10, fill: chartColors.axis }}
            tickLine={false}
            axisLine={false}
            domain={[0, 100]}
            tickFormatter={(v: number) => formatPct(v)}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: chartColors.tooltip_bg,
              border: `1px solid ${chartColors.tooltip_border}`,
              borderRadius: "8px",
              fontSize: "12px",
            }}
            formatter={(value: number, name: string) => [
              name === "accuracy" ? formatPct(value) : formatFixed(value, 2),
              name === "accuracy" ? "Accuracy" : "WAPE",
            ]}
          />
          <Legend wrapperStyle={{ fontSize: "11px" }} />
          <ReferenceLine
            y={70}
            stroke={roles.warning}
            strokeDasharray="6 4"
            label={{ value: "Target 70%", position: "right", fontSize: 10, fill: roles.warning }}
          />
          <Line
            type="monotone"
            dataKey="accuracy"
            stroke={trendColors[0]}
            strokeWidth={2}
            dot={{ r: 3, fill: trendColors[0] }}
            name="Accuracy"
          />
          <Line
            type="monotone"
            dataKey="wape"
            stroke={trendColors[4]}
            strokeWidth={1.5}
            strokeDasharray="4 4"
            dot={false}
            name="WAPE"
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 5. Model Comparison
// ---------------------------------------------------------------------------
function ModelComparisonSection() {
  const { data, isLoading, isError } = useQuery({
    queryKey: accuracyBudgetKeys.models(),
    queryFn: fetchModelComparison,
    staleTime: STALE.FIVE_MIN,
  });

  const models = useMemo(() => data?.models ?? [], [data]);
  const oracleCeiling = data?.oracle_ceiling;

  const sorted = useMemo(
    () => [...models].sort((a, b) => (b.accuracy_pct ?? 0) - (a.accuracy_pct ?? 0)),
    [models],
  );

  if (isLoading) return <LoadingElement message="Loading model comparison..." />;
  if (isError || models.length === 0) {
    return (
      <p className="text-sm text-muted-foreground text-center py-8">
        No model comparison data available.
      </p>
    );
  }

  return (
    <div className="space-y-4">
      {oracleCeiling && (
        <div className="flex items-center gap-3 rounded-md border border-border/60 bg-muted/30 px-3 py-2">
          <div className="text-xs text-muted-foreground">Oracle ceiling</div>
          <div className="text-sm font-semibold tabular-nums">{formatPct(oracleCeiling.accuracy)}</div>
          <div className="text-xs text-muted-foreground">WAPE {formatFixed(oracleCeiling.wape, 2)}</div>
        </div>
      )}
      <div className="overflow-x-auto">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Model</TableHead>
              <TableHead className="text-right">Accuracy</TableHead>
              <TableHead className="text-right">WAPE</TableHead>
              <TableHead className="text-right">Bias</TableHead>
              <TableHead className="text-right">DFUs</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {sorted.map((m, idx) => (
              <TableRow
                key={m.model_id}
                className={cn(idx === 0 && "bg-emerald-50/50 dark:bg-emerald-950/20")}
              >
                <TableCell className="text-sm font-medium">
                  <span className="mr-1">{modelLabel(m.model_id)}</span>
                  <span className="font-mono text-[10px] text-muted-foreground">
                    {m.model_id}
                  </span>
                  {idx === 0 && (
                    <Badge className="ml-2 bg-emerald-100 text-emerald-800 dark:bg-emerald-900/40 dark:text-emerald-300 text-[10px] px-2 py-0.5">
                      Best
                    </Badge>
                  )}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm font-medium">
                  {formatPct(m.accuracy_pct)}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {formatFixed(m.wape, 2)}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {formatFixed(m.bias, 4)}
                </TableCell>
                <TableCell className="text-right tabular-nums text-sm">
                  {formatInt(m.n_dfus)}
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
// Main panel
// ---------------------------------------------------------------------------
export function AccuracyBudgetPanel() {
  const [activeSection, setActiveSection] = useState<SubSection>("waterfall");

  // Top-level KPIs from decomposition endpoint
  const { data, isLoading } = useQuery({
    queryKey: accuracyBudgetKeys.decomposition(),
    queryFn: () => fetchAccuracyDecomposition(),
    staleTime: STALE.FIVE_MIN,
  });

  const kpis = useMemo(() => {
    if (!data) return { baseline: null, model: null, oracle: null, valueAdded: null, addressableGap: null };
    return {
      baseline: data.naive_baseline,
      model: data.current_accuracy,
      oracle: data.oracle_ceiling,
      valueAdded: data.forecast_value_added,
      addressableGap: data.addressable_gap,
    };
  }, [data]);

  if (isLoading) {
    return (
      <div className="p-6">
        <LoadingElement message="Loading accuracy budget..." size="md" />
      </div>
    );
  }

  if (!data || data.current_accuracy == null) {
    return (
      <div className="flex flex-col items-center justify-center py-16 text-center">
        <Target className="h-10 w-10 text-muted-foreground/40 mb-3" />
        <p className="text-sm font-medium text-foreground mb-1">No accuracy budget data</p>
        <p className="text-xs text-muted-foreground mb-3">
          Run backtests to generate accuracy budget analysis.
        </p>
        <code className="text-xs bg-muted px-3 py-1 rounded font-mono">make backtest-all</code>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Summary KPIs */}
      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        <KpiCard
          label="Naive Baseline"
          value={kpis.baseline != null ? formatPct(kpis.baseline) : "--"}
          size="md"
        />
        <KpiCard
          label="ML Model"
          value={kpis.model != null ? formatPct(kpis.model) : "--"}
          severity="best"
          size="md"
        />
        <KpiCard
          label="Oracle Ceiling"
          value={kpis.oracle != null ? formatPct(kpis.oracle) : "--"}
          size="md"
        />
        <KpiCard
          label="Value Added"
          value={kpis.valueAdded != null ? `+${formatPct(kpis.valueAdded)}` : "--"}
          severity={kpis.valueAdded != null && kpis.valueAdded > 0 ? "best" : "warning"}
          size="md"
        />
        <KpiCard
          label="Addressable Gap"
          value={kpis.addressableGap != null ? formatPct(kpis.addressableGap) : "--"}
          severity="warning"
          size="md"
        />
      </div>

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
            {activeSection === "waterfall" &&
              "Accuracy progression: naive baseline, ML model value-add, oracle ceiling, and remaining addressable gap."}
            {activeSection === "gap" &&
              "Decomposition of the addressable accuracy gap by error source."}
            {activeSection === "abc" &&
              "Accuracy by ABC classification with target indicators. Green = meeting target, red = below."}
            {activeSection === "trend" &&
              "Monthly accuracy trend with WAPE overlay. Reference line shows target threshold."}
            {activeSection === "models" &&
              "Side-by-side comparison of all model variants. Best model highlighted."}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {activeSection === "waterfall" && <WaterfallSection />}
          {activeSection === "gap" && <GapSection />}
          {activeSection === "abc" && <AbcTargetsSection />}
          {activeSection === "trend" && <TrendSection />}
          {activeSection === "models" && <ModelComparisonSection />}
        </CardContent>
      </Card>
    </div>
  );
}
