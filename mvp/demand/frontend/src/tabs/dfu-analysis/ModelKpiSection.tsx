import { Card, CardContent } from "@/components/ui/card";

import { useChartColors } from "@/hooks/useChartColors";
import {
  formatNumber,
  formatPercent,
  formatCompactNumber,
} from "@/lib/formatters";
import type { DfuAnalysisKpis, DfuAnalysisPayload } from "@/types";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
export interface ModelKpiSectionProps {
  dfuData: DfuAnalysisPayload;
  dfuKpis: Record<string, DfuAnalysisKpis>;
  dfuKpiMonths: number;
  dfuVisibleSeries: Set<string>;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function ModelKpiSection({
  dfuData,
  dfuKpis,
  dfuKpiMonths,
  dfuVisibleSeries,
}: ModelKpiSectionProps) {
  const { trendColors } = useChartColors();

  if (Object.keys(dfuKpis).length === 0) return null;

  const visibleModels = dfuData.models.filter(
    (m) => dfuVisibleSeries.has(`forecast_${m}`) && dfuKpis[m],
  );

  if (visibleModels.length === 0) return null;

  return (
    <div className="space-y-2">
      <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
        Model KPIs ({dfuKpiMonths}-month window)
      </span>
      <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-3">
        {visibleModels.map((model) => {
          const kpi = dfuKpis[model];
          const colorIdx = dfuData.models.indexOf(model) + 1;
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
  );
}
