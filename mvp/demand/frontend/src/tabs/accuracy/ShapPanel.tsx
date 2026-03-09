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
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
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
  onToggleOpen: () => void;
  onModelChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  onTimeframeChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
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
  onToggleOpen,
  onModelChange,
  onTimeframeChange,
}: ShapPanelProps) {
  return (
    <Card className="mt-4 animate-fade-in">
      <CardHeader
        className="cursor-pointer select-none"
        onClick={onToggleOpen}
      >
        <div className="flex items-center gap-2">
          {shapOpen ? (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronRight className="h-4 w-4 text-muted-foreground" />
          )}
          <CardTitle className="text-base">Feature Importance (SHAP)</CardTitle>
        </div>
        <CardDescription>
          Per-timeframe SHAP feature importance from SHAP-selected backtests. Run with{" "}
          <code className="text-xs">--shap-select</code> to populate.
        </CardDescription>
      </CardHeader>
      {shapOpen && (
        <CardContent className="space-y-4">
          {shapModels.length === 0 ? (
            <p className="text-sm text-muted-foreground">
              No SHAP outputs found. Run a backtest with <code>--shap-select</code> to generate feature importance data.
            </p>
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
                      <option key={m} value={m}>{m}</option>
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
              </div>

              {/* ── Chart ────────────────────────────────────────────── */}
              {loadingShap ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground py-6 justify-center">
                  <Loader2 className="h-4 w-4 animate-spin" />
                  Loading SHAP data…
                </div>
              ) : shapFeatures.length === 0 ? (
                <p className="text-sm text-muted-foreground py-4 text-center">No feature data available.</p>
              ) : (
                <div>
                  <p className="text-xs text-muted-foreground mb-2">
                    Top 15 features — <span className="text-sky-500 font-medium">■ selected</span>{" "}
                    <span className="text-gray-400 font-medium">■ dropped</span>
                  </p>
                  <ResponsiveContainer width="100%" height={Math.max(200, shapFeatures.length * 28)}>
                    <BarChart
                      data={shapFeatures}
                      layout="vertical"
                      margin={{ top: 4, right: 40, left: 8, bottom: 4 }}
                    >
                      <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                      <XAxis type="number" tick={{ fontSize: 11 }} tickFormatter={(v) => v.toFixed(3)} />
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
                          <Cell
                            key={i}
                            fill={f.selected ? "#6366f1" : "#d1d5db"}
                          />
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
