/**
 * Performance KPIs section for Portfolio Analysis — hero accuracy / WAPE / bias
 * KPIs plus forecast & actual volume, with model + window selectors.
 * Extracted verbatim from AggregateAnalysisTab.
 */
import { KpiCard } from "@/components/KpiCard";
import { Skeleton } from "@/components/Skeleton";
import { CollapsibleSection } from "@/components/CollapsibleSection";
import { cn } from "@/lib/utils";

import { formatNumberCompact, trendDirection } from "./aggregateShared";

interface DashboardKpi {
  accuracy_pct?: number | null;
  wape_pct?: number | null;
  bias_pct?: number | null;
  total_forecast: number | null;
  total_actual: number | null;
  deltas?: {
    accuracy_pct?: number | null;
    wape_pct?: number | null;
    bias_pct?: number | null;
  } | null;
}

interface KpiCardsSectionProps {
  kpi: DashboardKpi | undefined;
  isLoading: boolean;
  kpiModel: string;
  kpiWindow: number;
  kpiOptions: number[];
  heatmapModels: string[] | undefined;
  trendData: unknown;
  onKpiModelChange: (model: string) => void;
  onKpiWindowChange: (window: number) => void;
}

export function KpiCardsSection({
  kpi,
  isLoading,
  kpiModel,
  kpiWindow,
  kpiOptions,
  heatmapModels,
  trendData,
  onKpiModelChange,
  onKpiWindowChange,
}: KpiCardsSectionProps) {
  return (
    <CollapsibleSection
      title="Performance KPIs"
      headerRight={
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 text-[10px]">
            <span className="text-muted-foreground font-medium">Model</span>
            <select
              className="h-6 rounded border border-input bg-background px-1.5 text-[10px]"
              value={kpiModel}
              onChange={(e) => onKpiModelChange(e.target.value)}
            >
              {(heatmapModels ?? ["external"]).map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>
          <div className="flex items-center gap-1">
            {kpiOptions.map((w) => (
              <button key={w} onClick={() => onKpiWindowChange(w)} className={cn("rounded px-2 py-0.5 text-[10px] transition-colors", kpiWindow === w ? "bg-primary/10 text-primary font-medium" : "text-muted-foreground hover:bg-muted/50")}>
                {w}mo
              </button>
            ))}
          </div>
        </div>
      }
    >
      {isLoading ? (
        <div className="grid grid-cols-5 gap-3">
          {Array.from({ length: 5 }).map((_, i) => <Skeleton key={i} className="h-20" />)}
        </div>
      ) : kpi ? (
        <div className="grid grid-cols-5 gap-3">
          {/* UX-2: enriched hero KPIs with target thresholds + sparkline when trend data present. */}
          <KpiCard
            label="Accuracy %"
            size="lg"
            value={kpi.accuracy_pct != null ? `${kpi.accuracy_pct.toFixed(1)}%` : "N/A"}
            trend={kpi.deltas?.accuracy_pct != null ? { delta: kpi.deltas.accuracy_pct, direction: trendDirection(kpi.deltas.accuracy_pct), goodDirection: "up", unit: "pp", period: `prev ${kpiWindow}mo` } : undefined}
            severity={kpi.accuracy_pct != null ? (kpi.accuracy_pct >= 90 ? "best" : kpi.accuracy_pct >= 80 ? "neutral" : "warning") : "neutral"}
            target={{ value: ">= 90%", label: "Target" }}
            sparkline={(() => {
              const rows = (trendData as { trend?: Array<{ accuracy_pct?: number | null }> } | undefined)?.trend;
              return Array.isArray(rows) ? rows.slice(-12).map((p) => Number(p.accuracy_pct ?? 0)) : undefined;
            })()}
          />
          <KpiCard
            label="WAPE %"
            size="lg"
            value={kpi.wape_pct != null ? `${kpi.wape_pct.toFixed(1)}%` : "N/A"}
            // U6.1: display the TRUE delta sign; color by goodDirection ("down" = lower WAPE is better).
            trend={kpi.deltas?.wape_pct != null ? { delta: kpi.deltas.wape_pct, direction: trendDirection(kpi.deltas.wape_pct), goodDirection: "down", unit: "pp", period: `prev ${kpiWindow}mo` } : undefined}
            severity={kpi.wape_pct != null ? (kpi.wape_pct <= 10 ? "best" : kpi.wape_pct <= 20 ? "neutral" : "warning") : "neutral"}
            target={{ value: "<= 10%", label: "Target" }}
            sparkline={(() => {
              const rows = (trendData as { trend?: Array<{ wape_pct?: number | null }> } | undefined)?.trend;
              return Array.isArray(rows) ? rows.slice(-12).map((p) => Number(p.wape_pct ?? 0)) : undefined;
            })()}
          />
          <KpiCard
            label="Bias %"
            size="lg"
            value={kpi.bias_pct != null ? `${kpi.bias_pct.toFixed(1)}%` : "N/A"}
            // U6.1: show the TRUE signed bias change. "Good" = moving toward zero,
            // so when current bias is positive lower is better ("down"), and when
            // negative higher is better ("up").
            trend={kpi.deltas?.bias_pct != null ? { delta: kpi.deltas.bias_pct, direction: trendDirection(kpi.deltas.bias_pct), goodDirection: (kpi.bias_pct ?? 0) >= 0 ? "down" : "up", unit: "pp", period: `prev ${kpiWindow}mo` } : undefined}
            severity={kpi.bias_pct != null ? (Math.abs(kpi.bias_pct) <= 5 ? "best" : Math.abs(kpi.bias_pct) <= 15 ? "neutral" : "warning") : "neutral"}
            target={{ value: "+/- 5%", label: "Target" }}
          />
          <KpiCard
            label="Forecast Vol"
            value={formatNumberCompact(kpi.total_forecast)}
            severity="neutral"
          />
          <KpiCard
            label="Actual Vol"
            value={formatNumberCompact(kpi.total_actual)}
            severity="neutral"
          />
        </div>
      ) : null}
    </CollapsibleSection>
  );
}
