import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import { useChartColors } from "@/hooks/useChartColors";
import { SKU_SALES_COLORS, skuModelColor } from "@/constants/colors";
import { formatNumber, formatCompactNumber } from "@/lib/formatters";
import type { SkuAnalysisPayload } from "@/types";
import { modelLabel, formatChampionLabel } from "@/lib/model-labels";
import {
  PROD_FORECAST_COLOR,
  AI_CHAMPION_COLOR,
  CHART_MARGIN,
  DESELECT_OPACITY,
  STAGING_COLORS,
  STAGING_FALLBACK_COLOR,
  DQ_ORIG_COLOR,
  TOOLTIP_LABELS,
} from "./colors";
import type { SupplySeriesDef } from "./measures";

export interface UnifiedChartProps {
  mergedData: Record<string, unknown>[];
  hasRightAxis: boolean;
  models: SkuAnalysisPayload["models"];
  skuVisibleSeries: Set<string>;
  hiddenDemand: Set<string>;
  selectedModel: string | null;
  hasProdForecast: boolean;
  hasAiChampion: boolean;
  aiChampionLineHidden: boolean;
  /** Dominant champion source model (e.g. "nbeats") — labels the champion line. */
  championDominantSource?: string | null;
  stagingModelIds: string[];
  hiddenStaging: Set<string>;
  hiddenStagingPills: Set<string>;
  backtestModelIds: string[];
  hiddenBacktest: Set<string>;
  hasSupplyData: boolean;
  availableSupply: SupplySeriesDef[];
  hiddenSupply: Set<string>;
  showSupply: (key: string) => boolean;
  showCorrections: boolean;
  activeCorrectionSeries: string[];
  hasSs: boolean;
  ss: number | null;
  ropUnits: number | null;
}

/** Recharts render for the Item Analysis unified demand/supply chart.
 *
 * Extracted verbatim from UnifiedChartPanel; a pure read-only view of the
 * already-merged data and visibility state. */
export function UnifiedChart({
  mergedData,
  hasRightAxis,
  models,
  skuVisibleSeries,
  hiddenDemand,
  selectedModel,
  hasProdForecast,
  hasAiChampion,
  aiChampionLineHidden,
  championDominantSource,
  stagingModelIds,
  hiddenStaging,
  hiddenStagingPills,
  backtestModelIds,
  hiddenBacktest,
  hasSupplyData,
  availableSupply,
  hiddenSupply,
  showSupply,
  showCorrections,
  activeCorrectionSeries,
  hasSs,
  ss,
  ropUnits,
}: UnifiedChartProps) {
  const { chartColors } = useChartColors();
  return (
    <div className="h-[400px] overflow-x-auto overflow-y-hidden pb-2 [scrollbar-gutter:stable]">
      <div
        className="h-full"
        style={{ minWidth: `${Math.max(800, mergedData.length * 40)}px` }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={mergedData} margin={CHART_MARGIN}>
            <CartesianGrid strokeDasharray="3 3" stroke={chartColors.grid} />
            <XAxis dataKey="month" tick={{ fill: chartColors.axis, fontSize: 11 }} />
            <YAxis
              yAxisId="left"
              width={78}
              tickFormatter={formatCompactNumber}
              tick={{ fill: chartColors.axis, fontSize: 11 }}
              label={{ value: "Units", angle: -90, position: "insideLeft", fontSize: 10, offset: 10 }}
            />
            {hasRightAxis && (
              <YAxis
                yAxisId="right"
                orientation="right"
                tick={{ fill: chartColors.axis, fontSize: 11 }}
                tickFormatter={(v: number) => `${Number(v).toFixed(0)}d`}
                label={{ value: "Days", angle: 90, position: "insideRight", fontSize: 10, offset: 10 }}
              />
            )}
            <Tooltip
              contentStyle={{
                backgroundColor: chartColors.tooltip_bg,
                borderColor: chartColors.tooltip_border,
              }}
              formatter={(value: number, name: string, entry?: { payload?: Record<string, unknown> }) => {
                let label = TOOLTIP_LABELS[name] ?? name;
                // Resolve staging/backtest model names to readable labels
                if (name.startsWith("staging_")) {
                  const mid = name.slice("staging_".length);
                  label = `${modelLabel(mid)} (forecast)`;
                } else if (name.startsWith("backtest_")) {
                  const mid = name.slice("backtest_".length);
                  label = `${modelLabel(mid)} (backtest)`;
                } else if (name === "champion") {
                  // Show that month's blend mix on the champion row, e.g.
                  // "champion (40% NBEATS, 35% LGBM, 25% Chronos)"; falls back to
                  // the single source model, then a bare "champion".
                  label = formatChampionLabel(
                    entry?.payload?.champion_mix as { model: string; weight: number }[] | undefined,
                    entry?.payload?.champion_source as string | undefined,
                  );
                }
                if (name === "dos" || name === "avg_lead_time")
                  return [`${Number(value).toFixed(1)} days`, label];
                return [
                  formatNumber(Number.isFinite(Number(value)) ? Number(value) : null),
                  label,
                ];
              }}
            />

            {/* ---- Demand lines ---- */}
            {skuVisibleSeries.has("tothist_dmd") && !hiddenDemand.has("tothist_dmd") && (
              <Line
                type="monotone"
                dataKey="tothist_dmd"
                yAxisId="left"
                name="tothist_dmd"
                stroke={SKU_SALES_COLORS.tothist_dmd}
                strokeWidth={2.5}
                dot={false}
                activeDot={{ r: 4 }}
              />
            )}
            {skuVisibleSeries.has("sales_qty") && !hiddenDemand.has("sales_qty") && (
              <Line
                type="monotone"
                dataKey="sales_qty"
                yAxisId="left"
                name="sales_qty"
                stroke={SKU_SALES_COLORS.sales_qty}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
              />
            )}
            {skuVisibleSeries.has("qty_shipped") && !hiddenDemand.has("qty_shipped") && (
              <Line
                type="monotone"
                dataKey="qty_shipped"
                yAxisId="left"
                name="qty_shipped"
                stroke={SKU_SALES_COLORS.qty_shipped}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
              />
            )}
            {skuVisibleSeries.has("qty_ordered") && !hiddenDemand.has("qty_ordered") && (
              <Line
                type="monotone"
                dataKey="qty_ordered"
                yAxisId="left"
                name="qty_ordered"
                stroke={SKU_SALES_COLORS.qty_ordered}
                strokeWidth={2}
                dot={false}
                activeDot={{ r: 4 }}
              />
            )}
            {models
              .filter((m) => skuVisibleSeries.has(`forecast_${m}`) && !hiddenDemand.has(`forecast_${m}`))
              .map((model, idx) => {
                const isSelected = selectedModel === model;
                const isOtherSelected = selectedModel !== null && selectedModel !== model;
                // Label the champion line with its dominant source model, e.g.
                // "champion (N-BEATS)" (per-month mix shown in the tooltip).
                const seriesName =
                  model === "champion" && championDominantSource
                    ? formatChampionLabel(null, championDominantSource)
                    : model;
                return (
                  <Line
                    key={model}
                    type="monotone"
                    dataKey={`forecast_${model}`}
                    yAxisId="left"
                    name={seriesName}
                    stroke={skuModelColor(model, idx)}
                    strokeWidth={isSelected ? 3 : model === "champion" ? 2.5 : 1.5}
                    strokeDasharray={model === "champion" ? undefined : "5 3"}
                    dot={false}
                    style={{ opacity: isOtherSelected ? DESELECT_OPACITY : 1 }}
                    activeDot={{ r: isSelected ? 7 : 4 }}
                  />
                );
              })}
            {hasProdForecast && skuVisibleSeries.has("production_forecast") && !hiddenDemand.has("production_forecast") && (
              <Line
                type="monotone"
                dataKey="production_forecast"
                yAxisId="left"
                name="production_forecast"
                stroke={PROD_FORECAST_COLOR}
                strokeWidth={2.5}
                strokeDasharray="6 3"
                dot={false}
                activeDot={{ r: 5 }}
              />
            )}
            {hasAiChampion && skuVisibleSeries.has("ai_champion") && !aiChampionLineHidden && (
              <Line
                type="monotone"
                dataKey="ai_champion"
                yAxisId="left"
                name="ai_champion"
                stroke={AI_CHAMPION_COLOR}
                strokeWidth={2.5}
                strokeDasharray="2 2"
                dot={false}
                connectNulls
                activeDot={{ r: 5 }}
              />
            )}

            {/* ---- Staging forecast lines (future) ---- */}
            {stagingModelIds
              .filter((mid) => !hiddenStaging.has(mid) && !hiddenStagingPills.has(mid))
              .map((mid) => {
                const key = `staging_${mid}`;
                const color = STAGING_COLORS[mid] ?? STAGING_FALLBACK_COLOR;
                return (
                  <Line
                    key={key}
                    type="monotone"
                    dataKey={key}
                    yAxisId="left"
                    name={key}
                    stroke={color}
                    strokeWidth={1.5}
                    strokeDasharray="4 3"
                    dot={false}
                    connectNulls
                    activeDot={{ r: 3 }}
                  />
                );
              })}

            {/* ---- Backtest lines (past, out-of-sample) — dotted, same color
                   as the model's forecast line so a model's past fit and forward
                   forecast read as one series across the timeline. ---- */}
            {backtestModelIds
              .filter((mid) => !hiddenBacktest.has(mid))
              .map((mid) => {
                const key = `backtest_${mid}`;
                const color = STAGING_COLORS[mid] ?? STAGING_FALLBACK_COLOR;
                return (
                  <Line
                    key={key}
                    type="monotone"
                    dataKey={key}
                    yAxisId="left"
                    name={key}
                    stroke={color}
                    strokeWidth={1.5}
                    strokeDasharray="1 3"
                    dot={false}
                    connectNulls
                    activeDot={{ r: 3 }}
                  />
                );
              })}

            {/* ---- Supply lines ---- */}
            {hasSupplyData &&
              availableSupply
                .filter((s) => !hiddenSupply.has(s.key))
                .map((s) => (
                  <Line
                    key={s.key}
                    type="monotone"
                    dataKey={s.key}
                    yAxisId={s.axis}
                    name={s.key}
                    stroke={s.color}
                    strokeWidth={s.strokeWidth ?? 2}
                    strokeDasharray={s.dashArray}
                    dot={false}
                    connectNulls
                    activeDot={{ r: 3 }}
                  />
                ))}

            {/* ---- DQ correction original-value lines ---- */}
            {showCorrections &&
              activeCorrectionSeries.map((origKey) => (
                <Line
                  key={origKey}
                  type="monotone"
                  dataKey={origKey}
                  yAxisId="left"
                  name={origKey}
                  stroke={DQ_ORIG_COLOR}
                  strokeWidth={2}
                  strokeDasharray="4 3"
                  dot={{ r: 3, fill: DQ_ORIG_COLOR }}
                  connectNulls={false}
                  activeDot={{ r: 5 }}
                />
              ))}

            {/* ---- Reference lines ---- */}
            {hasSs && showSupply("safety_stock") && (
              <ReferenceLine
                yAxisId="left"
                y={ss!}
                stroke="#8b5cf6"
                strokeDasharray="6 3"
                strokeWidth={1.5}
                label={{
                  value: `SS ${ss!.toFixed(0)}u`,
                  position: "insideTopLeft",
                  fontSize: 10,
                  fill: "#8b5cf6",
                }}
              />
            )}
            {ropUnits != null && showSupply("safety_stock") && (
              <ReferenceLine
                yAxisId="left"
                y={ropUnits}
                stroke="#f97316"
                strokeDasharray="4 2"
                strokeWidth={1.5}
                label={{
                  value: `ROP ${ropUnits.toFixed(0)}u`,
                  position: "insideBottomLeft",
                  fontSize: 10,
                  fill: "#f97316",
                }}
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
