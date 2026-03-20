/**
 * BiasCorrectionsPanel — F3.1 Forecast Bias Correction Engine
 * Shows portfolio-level bias KPIs, flagged items, and correction history.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, CheckCircle, TrendingUp, Scissors } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  biasKeys,
  fetchBiasCorrectionSummary,
  fetchFlaggedBiasCorrections,
  STALE_EVO,
} from "@/api/queries/evolution";

function KpiCard({
  label,
  value,
  icon,
  warn,
}: {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  warn?: boolean;
}) {
  return (
    <Card className={warn ? "border-amber-400" : ""}>
      <CardHeader className="pb-2 flex flex-row items-center justify-between">
        <CardTitle className="text-sm font-medium text-muted-foreground">{label}</CardTitle>
        <span className={warn ? "text-amber-500" : "text-muted-foreground"}>{icon}</span>
      </CardHeader>
      <CardContent>
        <p className="text-2xl font-bold">{value}</p>
      </CardContent>
    </Card>
  );
}

export function BiasCorrectionsPanel() {
  const [planMonth, setPlanMonth] = useState<string>("");

  const { data: summary, isLoading: sumLoading } = useQuery({
    queryKey: biasKeys.summary(planMonth || undefined),
    queryFn: () => fetchBiasCorrectionSummary(planMonth || undefined),
    staleTime: STALE_EVO.FIVE_MIN,
  });

  const { data: flagged, isLoading: flagLoading } = useQuery({
    queryKey: biasKeys.flagged(planMonth || undefined),
    queryFn: () => fetchFlaggedBiasCorrections(planMonth || undefined),
    staleTime: STALE_EVO.FIVE_MIN,
  });

  const fmtPct = (v: number | null | undefined) =>
    v == null ? "—" : `${(v * 100).toFixed(1)}%`;

  const fmtFactor = (v: number | null | undefined) =>
    v == null ? "—" : v.toFixed(3);

  return (
    <div className="space-y-6 p-4">
      {/* Filter bar */}
      <div className="flex items-center gap-3">
        <label className="text-sm font-medium">Plan Month</label>
        <input
          type="month"
          value={planMonth}
          onChange={(e) => setPlanMonth(e.target.value)}
          className="border rounded px-2 py-1 text-sm"
        />
        {planMonth && (
          <button
            onClick={() => setPlanMonth("")}
            className="text-xs text-muted-foreground underline"
          >
            Clear
          </button>
        )}
      </div>

      {/* KPI cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KpiCard
          label="DFUs Corrected"
          value={sumLoading ? "…" : (summary?.dfu_count ?? 0).toLocaleString()}
          icon={<TrendingUp size={16} />}
        />
        <KpiCard
          label="Avg Correction Factor"
          value={sumLoading ? "…" : fmtFactor(summary?.avg_correction_factor)}
          icon={<CheckCircle size={16} />}
        />
        <KpiCard
          label="Flagged for Review"
          value={sumLoading ? "…" : (summary?.flagged_count ?? 0).toLocaleString()}
          icon={<AlertTriangle size={16} />}
          warn={(summary?.flagged_count ?? 0) > 0}
        />
        <KpiCard
          label="Clipped (Guard Rail)"
          value={sumLoading ? "…" : (summary?.clipped_count ?? 0).toLocaleString()}
          icon={<Scissors size={16} />}
        />
      </div>

      {/* Avg rolling bias */}
      {summary && summary.avg_rolling_bias != null && (
        <Card>
          <CardContent className="pt-4">
            <p className="text-sm text-muted-foreground">
              Avg Rolling 3-Month Bias:{" "}
              <span
                className={
                  Math.abs(summary.avg_rolling_bias) > 0.1
                    ? "font-bold text-amber-600"
                    : "font-bold text-green-600"
                }
              >
                {fmtPct(summary.avg_rolling_bias)}
              </span>
              {summary.last_computed_at && (
                <span className="ml-4 text-xs text-muted-foreground">
                  Computed: {summary.last_computed_at.slice(0, 10)}
                </span>
              )}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Flagged items table */}
      <Card>
        <CardHeader>
          <CardTitle className="text-base">Items Flagged for Review</CardTitle>
        </CardHeader>
        <CardContent className="p-0">
          {flagLoading ? (
            <p className="p-4 text-sm text-muted-foreground">Loading…</p>
          ) : !flagged?.flagged?.length ? (
            <p className="p-4 text-sm text-muted-foreground">No flagged items.</p>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-muted/50 text-xs uppercase text-muted-foreground">
                  <tr>
                    {["Item", "Loc", "Plan Month", "Segment", "Rolling Bias", "Raw Factor", "Clipped Factor", "Clipped?", "Months"].map(
                      (h) => (
                        <th key={h} className="px-3 py-2 text-left font-medium">
                          {h}
                        </th>
                      )
                    )}
                  </tr>
                </thead>
                <tbody>
                  {flagged.flagged.map((row, i) => (
                    <tr
                      key={i}
                      className="border-t hover:bg-muted/30 transition-colors"
                    >
                      <td className="px-3 py-2 font-mono text-xs">{row.item_no}</td>
                      <td className="px-3 py-2 text-xs">{row.loc}</td>
                      <td className="px-3 py-2 text-xs">{row.plan_month?.slice(0, 7) ?? "—"}</td>
                      <td className="px-3 py-2 text-xs">{row.segment_type}</td>
                      <td
                        className={`px-3 py-2 font-medium ${
                          row.rolling_bias_3m > 0.1
                            ? "text-red-600"
                            : row.rolling_bias_3m < -0.1
                            ? "text-blue-600"
                            : "text-foreground"
                        }`}
                      >
                        {fmtPct(row.rolling_bias_3m)}
                      </td>
                      <td className="px-3 py-2 text-xs">{fmtFactor(row.correction_factor_raw)}</td>
                      <td className="px-3 py-2 text-xs">{fmtFactor(row.correction_factor)}</td>
                      <td className="px-3 py-2">
                        {row.correction_was_clipped ? (
                          <span className="text-xs bg-amber-100 text-amber-700 px-1.5 py-0.5 rounded">
                            Yes
                          </span>
                        ) : (
                          <span className="text-xs text-muted-foreground">No</span>
                        )}
                      </td>
                      <td className="px-3 py-2 text-xs">{row.months_of_data}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
