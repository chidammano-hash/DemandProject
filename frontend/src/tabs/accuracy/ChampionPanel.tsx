import { KpiCard } from "@/components/KpiCard";
import { CollapsibleSection } from "@/components/CollapsibleSection";
import type { ChampionSummary } from "@/api/queries";
import { modelLabel } from "@/lib/model-labels";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ChampionPanelProps {
  championSummary: ChampionSummary | null;
}

const STRATEGY_LABELS: Record<string, string> = {
  per_cluster: "Per-cluster routing",
  single_model: "Single-model routing",
  ensemble: "Weighted ensemble",
};

function strategyLabel(strategy: string): string {
  const fallback = strategy.replace(/_/g, " ");
  return STRATEGY_LABELS[strategy] ?? fallback.charAt(0).toUpperCase() + fallback.slice(1);
}

// ---------------------------------------------------------------------------
// Component — results-only (config controls moved to Jobs tab)
// ---------------------------------------------------------------------------

export function ChampionPanel({ championSummary }: ChampionPanelProps) {
  if (!championSummary) return null;

  return (
    <CollapsibleSection title="Promoted Champion Results">
      <div className="space-y-4 rounded-lg border bg-muted/40 p-4">
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
            <span className="text-sm font-semibold">
              Experiment #{championSummary.experiment_id} · {championSummary.experiment_label}
            </span>
            <span className="rounded-full border px-2 py-0.5 text-xs text-muted-foreground">
              {strategyLabel(championSummary.strategy)}
            </span>
          </div>
          <div className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted-foreground">
            <span>Promoted: {new Date(championSummary.run_ts).toLocaleString()}</span>
            <span>
              Winner artifact: <span className="font-mono">{championSummary.artifact_name}</span>
            </span>
          </div>
          <p className="text-xs text-muted-foreground">
            Counts and routing composition come from this experiment&apos;s validated winner
            artifact. Accuracy and ceiling metrics come from the same promoted experiment record.
          </p>
        </div>

        {/* Champion KPI cards */}
        <div className="flex flex-wrap gap-4 text-sm">
          <KpiCard
            label="Routed DFUs"
            value={championSummary.total_dfus.toLocaleString()}
            sublabel={
              championSummary.total_dfu_months
                ? `${championSummary.total_dfu_months.toLocaleString()} DFU-months`
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
          <KpiCard
            label="Champion Rows"
            value={championSummary.total_champion_rows.toLocaleString()}
          />
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
            {championSummary.gap_bps != null && (
              <KpiCard
                label="Gap to Ceiling"
                value={`${(championSummary.gap_bps / 100).toFixed(2)} pp`}
                colorClass="text-amber-700 dark:text-amber-400"
                borderClass="border-amber-200 dark:border-amber-800"
              />
            )}
          </div>
        )}

        {/* Promoted winner composition */}
        <div className="space-y-1.5">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Promoted routing composition (winner per DFU-month)
          </p>
          {Object.entries(championSummary.model_wins).map(([model, wins]) => {
            const total = championSummary.total_dfu_months ?? championSummary.total_dfus;
            const pct = total > 0 ? (wins / total) * 100 : 0;
            return (
              <div key={model} className="flex items-center gap-2 text-sm">
                <span className="w-40 truncate text-right text-xs">{modelLabel(model)}</span>
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
      </div>
    </CollapsibleSection>
  );
}
