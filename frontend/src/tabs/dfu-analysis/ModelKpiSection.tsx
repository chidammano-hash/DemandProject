import { Card, CardContent } from "@/components/ui/card";

import { useChartColors } from "@/hooks/useChartColors";
import {
  formatNumber,
  formatPercent,
  formatCompactNumber,
} from "@/lib/formatters";
import type { SkuAnalysisKpis, SkuAnalysisPayload } from "@/types";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
export interface ModelKpiSectionProps {
  skuData: SkuAnalysisPayload;
  skuKpis: Record<string, SkuAnalysisKpis>;
  skuKpiMonths: number;
  setSkuKpiMonths: (v: number) => void;
  skuVisibleSeries: Set<string>;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function ModelKpiSection({
  skuData,
  skuKpis,
  skuKpiMonths,
  setSkuKpiMonths,
  skuVisibleSeries,
}: ModelKpiSectionProps) {
  const { trendColors } = useChartColors();

  if (Object.keys(skuKpis).length === 0) return null;

  const visibleModels = skuData.models.filter(
    (m) => skuVisibleSeries.has(`forecast_${m}`) && skuKpis[m],
  );

  if (visibleModels.length === 0) return null;

  return (
    <details className="group rounded-md border border-input bg-background" open>
      <summary className="cursor-pointer select-none px-3 py-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground hover:text-foreground flex items-center gap-2">
        <span>Forecast KPIs</span>
        <span className="ml-1 text-xs text-muted-foreground group-open:hidden">+ expand</span>
        {/* KPI Window dropdown inline */}
        <span className="ml-auto flex items-center gap-1.5" onClick={(e) => e.stopPropagation()}>
          <span className="text-[10px] font-medium text-muted-foreground">Window</span>
          <select
            className="h-7 rounded border border-input bg-background px-2 text-xs"
            value={skuKpiMonths}
            onChange={(e) => setSkuKpiMonths(Number(e.target.value))}
            onClick={(e) => e.stopPropagation()}
          >
            {Array.from({ length: 12 }, (_, i) => i + 1).map((m) => (
              <option key={m} value={m}>{m} mo</option>
            ))}
          </select>
        </span>
      </summary>
      <div className="border-t border-input px-3 py-3">
        <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
          {visibleModels.map((model) => {
            const kpi = skuKpis[model];
            const colorIdx = skuData.models.indexOf(model) + 1;
            return (
              <Card
                key={model}
                className="border-muted bg-muted/20 shadow-none"
              >
                <CardContent className="pt-4">
                  <div className="flex items-center gap-2 mb-2">
                    <span
                      className="inline-block h-3 w-3 rounded-full"
                      style={{
                        backgroundColor:
                          trendColors[colorIdx % trendColors.length],
                      }}
                    />
                    <p className="text-xs uppercase tracking-wide text-muted-foreground font-semibold">
                      {model}
                    </p>
                    <span className="text-xs text-muted-foreground ml-auto">
                      {kpi.months_covered} mo
                    </span>
                  </div>
                  <div className="grid grid-cols-5 gap-2">
                    <div>
                      <p className="text-xs uppercase text-muted-foreground">
                        Accuracy
                      </p>
                      <p className="text-sm font-semibold tabular-nums">
                        {formatPercent(kpi.accuracy_pct)}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs uppercase text-muted-foreground">
                        WAPE
                      </p>
                      <p className="text-sm font-semibold tabular-nums">
                        {formatPercent(kpi.wape)}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs uppercase text-muted-foreground">
                        Bias
                      </p>
                      <p className="text-sm font-semibold tabular-nums">
                        {formatNumber(kpi.bias)}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs uppercase text-muted-foreground">
                        Fcst
                      </p>
                      <p className="text-sm font-semibold tabular-nums">
                        {formatCompactNumber(kpi.sum_forecast)}
                      </p>
                    </div>
                    <div>
                      <p className="text-xs uppercase text-muted-foreground">
                        Actual
                      </p>
                      <p className="text-sm font-semibold tabular-nums">
                        {formatCompactNumber(kpi.sum_actual)}
                      </p>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
    </details>
  );
}
