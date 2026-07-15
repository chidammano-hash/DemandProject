import { Card, CardContent } from "@/components/ui/card";

import { useChartColors } from "@/hooks/useChartColors";
import {
  formatNumber,
  formatPercent,
  formatCompactNumber,
} from "@/lib/formatters";
import { modelLabel } from "@/lib/model-labels";
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

type ChampionEvidence = {
  composition: string;
  explanation: string;
  reconciliation?: string;
  mismatch?: boolean;
};

function formatMonth(month: string): string {
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    year: "numeric",
    timeZone: "UTC",
  }).format(new Date(`${month.slice(0, 10)}T00:00:00Z`));
}

function formatMix(mix: { model: string; weight: number }[]): string {
  return mix
    .slice()
    .sort((left, right) => right.weight - left.weight)
    .map(({ model, weight }) => `${Math.round(weight * 100)}% ${modelLabel(model)}`)
    .join(" + ");
}

function championEvidence(
  skuData: SkuAnalysisPayload,
  months: number,
): ChampionEvidence | null {
  const rows = skuData.model_monthly.champion?.slice(0, months) ?? [];
  if (rows.length === 0) return null;

  if (rows.length === 1) {
    const month = rows[0].month;
    const mix = skuData.champion_mix_by_month?.[month];
    if (mix?.length) {
      const components = mix.flatMap(({ model, weight }) => {
        const modelRow = skuData.model_monthly[model]?.find((row) => row.month === month);
        return modelRow ? [{ forecast: modelRow.forecast, weight }] : [];
      });
      const completeComponents = components.length === mix.length;
      const weightedForecast = completeComponents
        ? components.reduce(
            (sum, component) => sum + component.forecast * component.weight,
            0,
          )
        : null;
      const mismatch = weightedForecast !== null
        && Math.abs(weightedForecast - rows[0].forecast) > 0.01;
      const expression = completeComponents
        ? components
            .map(
              (component) =>
                `${Math.round(component.weight * 100)}% × ${formatCompactNumber(component.forecast)}`,
            )
            .join(" + ")
        : null;

      return {
        composition: formatMix(mix),
        explanation: `Blended forecast for ${formatMonth(month)}; accuracy is calculated after combining these models.`,
        reconciliation: expression && weightedForecast !== null
          ? mismatch
            ? `Blend mismatch: stored champion ${formatCompactNumber(rows[0].forecast)}, weighted components ${formatCompactNumber(weightedForecast)}.`
            : `Blend verified: ${expression} = ${formatCompactNumber(rows[0].forecast)}.`
          : undefined,
        mismatch,
      };
    }

    const source = skuData.champion_source_by_month?.[month];
    if (source && source !== "ensemble") {
      return {
        composition: `Selected ${modelLabel(source)}`,
        explanation: `Single-model champion for ${formatMonth(month)}; its KPI should match that model for the same month.`,
      };
    }
  }

  return {
    composition: "Per-month governed routing",
    explanation: `Aggregate champion KPI across ${rows.length} months; the selected model or blend can change each month.`,
  };
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
            const evidence = model === "champion"
              ? championEvidence(skuData, skuKpiMonths)
              : null;
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
                      {model === "champion" ? "Champion" : modelLabel(model)}
                    </p>
                    <span className="text-xs text-muted-foreground ml-auto">
                      {kpi.months_covered} mo
                    </span>
                  </div>
                  {/* Wrap on narrow cards: a rigid 5-up grid overflows and the
                      uppercase labels/values collide across cells. */}
                  <div className="grid grid-cols-2 gap-x-3 gap-y-2 sm:grid-cols-3 xl:grid-cols-5">
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
                  {evidence && (
                    <div className={`mt-3 rounded-md border px-2.5 py-2 ${
                      evidence.mismatch
                        ? "border-red-500/40 bg-red-500/10"
                        : "border-amber-500/30 bg-amber-500/5"
                    }`}>
                      <p className="text-xs font-semibold text-foreground">
                        {evidence.composition}
                      </p>
                      <p className="mt-0.5 text-[11px] leading-4 text-muted-foreground">
                        {evidence.explanation}
                      </p>
                      {evidence.reconciliation && (
                        <p className={`mt-1 text-[11px] font-medium leading-4 ${
                          evidence.mismatch
                            ? "text-red-600 dark:text-red-400"
                            : "text-emerald-600 dark:text-emerald-400"
                        }`}>
                          {evidence.reconciliation}
                        </p>
                      )}
                    </div>
                  )}
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
    </details>
  );
}
