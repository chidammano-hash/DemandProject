import { KpiCard } from "@/components/KpiCard";
import { CollapsibleSection } from "@/components/CollapsibleSection";
import type { ChampionSummary } from "@/api/queries";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ChampionPanelProps {
  championSummary: ChampionSummary | null;
}

// ---------------------------------------------------------------------------
// Component — results-only (config controls moved to Jobs tab)
// ---------------------------------------------------------------------------

export function ChampionPanel({ championSummary }: ChampionPanelProps) {
  if (!championSummary) return null;

  return (
    <CollapsibleSection title="Champion Selection Results">
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
              label="SKUs Evaluated"
              value={championSummary.total_skus.toLocaleString()}
              sublabel={
                championSummary.total_sku_months
                  ? `${championSummary.total_sku_months.toLocaleString()} DFU-months`
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
              colorClass="text-blue-700 dark:text-blue-400"
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
              const total = championSummary.total_sku_months ?? championSummary.total_skus;
              const pct = total > 0 ? (wins / total) * 100 : 0;
              return (
                <div key={model} className="flex items-center gap-2 text-sm">
                  <span className="w-40 truncate font-mono text-xs text-right">{model}</span>
                  <div className="flex-1 h-5 rounded bg-muted overflow-hidden">
                    <div
                      className="h-full rounded bg-blue-500 transition-all"
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
    </CollapsibleSection>
  );
}
