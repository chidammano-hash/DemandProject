import { Loader2, ChevronDown, ChevronRight } from "lucide-react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { useChartColors } from "@/hooks/useChartColors";
import type { ShapTimeframeEntry } from "@/types/shap";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface ShapFeatureRow {
  feature: string;
  value: number;
  selected: boolean;
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ShapPanelProps {
  shapOpen: boolean;
  shapModels: string[];
  activeShapModel: string;
  shapTimeframes: ShapTimeframeEntry[];
  shapTimeframeIdx: number | null;
  shapFeatures: ShapFeatureRow[];
  loadingShap: boolean;
  shapClusters?: string[];
  shapCluster?: string;
  onToggleOpen: () => void;
  onModelChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  onTimeframeChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  onClusterChange?: (e: React.ChangeEvent<HTMLSelectElement>) => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ShapPanel({
  shapOpen,
  shapModels,
  activeShapModel,
  shapTimeframes,
  shapTimeframeIdx,
  shapFeatures,
  loadingShap,
  shapClusters = [],
  shapCluster = "all",
  onToggleOpen,
  onModelChange,
  onTimeframeChange,
  onClusterChange,
}: ShapPanelProps) {
  const { roles, chartColors } = useChartColors();
  const hasMultipleClusters = shapClusters.length > 1 && shapClusters.some((c) => c !== "all");
  return (
    <Card className="mt-4 animate-fade-in">
      <CardHeader className="cursor-pointer select-none" onClick={onToggleOpen}>
        <div className="flex items-center gap-2">
          {shapOpen ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
          )}
          <CardTitle className="text-base">Feature Importance (SHAP)</CardTitle>
        </div>
        <CardDescription>
          Per-timeframe SHAP feature importance from backtests. Shows which features drive forecast
          accuracy for each model. Requires <code className="text-xs">shap_select: true</code> in{" "}
          <code className="text-xs">config/forecasting/forecast_pipeline_config.yaml</code> for
          LightGBM, then re-run the backtest.
        </CardDescription>
      </CardHeader>
      {shapOpen && (
        <CardContent className="space-y-4">
          {shapModels.length === 0 ? (
            <div className="space-y-2 text-sm text-muted-foreground">
              <p className="font-medium">No SHAP outputs found for any model.</p>
              <p>To generate SHAP feature importance data:</p>
              <ol className="list-decimal list-inside space-y-1 ml-2">
                <li>
                  Set <code className="text-xs bg-muted px-1 rounded">shap_select: true</code> under
                  each algorithm in{" "}
                  <code className="text-xs bg-muted px-1 rounded">
                    config/algorithm_config.yaml
                  </code>
                </li>
                <li>
                  Re-run the LightGBM backtest with{" "}
                  <code className="text-xs bg-muted px-1 rounded">make backtest-lgbm</code>
                </li>
                <li>
                  SHAP outputs will appear at{" "}
                  <code className="text-xs bg-muted px-1 rounded">
                    data/backtest/&lt;model&gt;/shap/
                  </code>
                </li>
              </ol>
              <p className="text-xs">
                LightGBM SHAP analysis requires <code>shap&gt;=0.43.0</code>.
              </p>
            </div>
          ) : (
            <>
              {/* ── Selectors ────────────────────────────────────────── */}
              <div className="flex flex-wrap gap-3">
                <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Model
                  <select
                    className="h-9 rounded-md border border-input bg-background px-3 text-sm block"
                    value={activeShapModel}
                    onChange={onModelChange}
                  >
                    {shapModels.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </label>

                <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                  Timeframe
                  <select
                    className="h-9 rounded-md border border-input bg-background px-3 text-sm block"
                    value={shapTimeframeIdx ?? "summary"}
                    onChange={onTimeframeChange}
                  >
                    <option value="summary">All timeframes (average)</option>
                    {shapTimeframes.map((tf) => (
                      <option key={tf.index} value={tf.index}>
                        {tf.label} — cutoff {tf.cutoff_date}
                      </option>
                    ))}
                  </select>
                </label>

                {hasMultipleClusters && shapTimeframeIdx !== null && (
                  <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Cluster
                    <select
                      className="h-9 rounded-md border border-input bg-background px-3 text-sm block"
                      value={shapCluster}
                      onChange={onClusterChange}
                    >
                      {shapClusters.map((c) => (
                        <option key={c} value={c}>
                          {c === "all" ? "All clusters (average importance)" : `Cluster ${c}`}
                        </option>
                      ))}
                    </select>
                  </label>
                )}
              </div>

              {/* ── Chart ────────────────────────────────────────────── */}
              {loadingShap ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground py-6 justify-center">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading SHAP data…
                </div>
              ) : shapFeatures.length === 0 ? (
                <p className="text-sm text-muted-foreground py-4 text-center">
                  No feature data available.
                </p>
              ) : (
                <div>
                  <p className="text-xs text-muted-foreground mb-2">
                    Top 15 features — <span className="text-sky-500 font-medium">■ selected</span>{" "}
                    <span className="text-muted-foreground/60 font-medium">■ dropped</span>
                  </p>
                  <ResponsiveContainer
                    width="100%"
                    height={Math.max(200, shapFeatures.length * 28)}
                  >
                    <BarChart
                      data={shapFeatures}
                      layout="vertical"
                      margin={{ top: 4, right: 40, left: 8, bottom: 4 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                      <XAxis
                        type="number"
                        tick={{ fontSize: 11 }}
                        tickFormatter={(v) => v.toFixed(3)}
                      />
                      <YAxis
                        type="category"
                        dataKey="feature"
                        width={160}
                        tick={{ fontSize: 11 }}
                      />
                      <Tooltip
                        formatter={(v: number) => [v.toFixed(5), "Mean |SHAP|"]}
                        cursor={{ fill: "rgba(0,0,0,0.04)" }}
                      />
                      <Bar dataKey="value" name="Mean |SHAP|" radius={[0, 3, 3, 0]}>
                        {shapFeatures.map((f, i) => (
                          <Cell key={i} fill={f.selected ? roles.forecast : chartColors.grid} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              )}
            </>
          )}
        </CardContent>
      )}
    </Card>
  );
}
