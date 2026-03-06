import { Loader2, Trophy } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { KpiCard } from "@/components/KpiCard";
import type { CompetitionConfig, ChampionSummary } from "@/api/queries";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface ChampionPanelProps {
  competitionConfig: CompetitionConfig | null;
  availableModels: string[];
  championSummary: ChampionSummary | null;
  savingConfig: boolean;
  runningCompetition: boolean;
  onCompetingModelToggle: (model: string) => void;
  onMetricChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  onLagChange: (e: React.ChangeEvent<HTMLSelectElement>) => void;
  onSaveConfig: () => void;
  onRunCompetition: () => void;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export function ChampionPanel({
  competitionConfig,
  availableModels,
  championSummary,
  savingConfig,
  runningCompetition,
  onCompetingModelToggle,
  onMetricChange,
  onLagChange,
  onSaveConfig,
  onRunCompetition,
}: ChampionPanelProps) {
  return (
    <Card className="animate-fade-in mt-4">
      <CardHeader>
        <div className="flex items-center gap-2">
          <Trophy className="h-5 w-5" />
          <CardTitle className="text-base">Champion Selection</CardTitle>
        </div>
        <CardDescription>
          Pick the best model per DFU based on forecast accuracy. Configure which models compete, then run the selection.
        </CardDescription>
      </CardHeader>
      <CardContent className="space-y-5">
        {competitionConfig ? (
          <>
            {/* Competing Models checkboxes */}
            <div className="space-y-3">
              <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Competing Models
              </span>
              <div className="flex flex-wrap gap-3">
                {availableModels
                  .filter((m) => m !== competitionConfig.champion_model_id && m !== "ceiling")
                  .map((m) => {
                    const checked = competitionConfig.models.includes(m);
                    const isLast = competitionConfig.models.length <= 2 && checked;
                    return (
                      <label key={m} className="flex items-center gap-1.5 text-sm cursor-pointer select-none">
                        <input
                          type="checkbox"
                          className="h-3.5 w-3.5 rounded border-input accent-indigo-700"
                          checked={checked}
                          disabled={isLast || runningCompetition}
                          onChange={() => onCompetingModelToggle(m)}
                        />
                        <span className="font-mono text-xs">{m}</span>
                      </label>
                    );
                  })}
              </div>
            </div>

            {/* Metric + Lag dropdowns + action buttons */}
            <div className="flex flex-wrap items-end gap-3">
              <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Metric
                <select
                  className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                  value={competitionConfig.metric}
                  onChange={onMetricChange}
                  disabled={runningCompetition}
                >
                  <option value="wape">WAPE (Lowest Wins)</option>
                  <option value="accuracy_pct">Accuracy % (Highest Wins)</option>
                </select>
              </label>
              <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                Lag
                <select
                  className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                  value={competitionConfig.lag}
                  onChange={onLagChange}
                  disabled={runningCompetition}
                >
                  <option value="execution">Execution Lag (per DFU)</option>
                  <option value="0">Lag 0 (same month)</option>
                  <option value="1">Lag 1</option>
                  <option value="2">Lag 2</option>
                  <option value="3">Lag 3</option>
                  <option value="4">Lag 4</option>
                </select>
              </label>
              <Button
                variant="outline"
                size="sm"
                disabled={savingConfig || runningCompetition}
                onClick={onSaveConfig}
              >
                {savingConfig ? <Loader2 className="mr-1 h-3 w-3 animate-spin" /> : null}
                Save Config
              </Button>
              <Button
                size="sm"
                disabled={runningCompetition || competitionConfig.models.length < 2}
                onClick={onRunCompetition}
              >
                {runningCompetition ? (
                  <Loader2 className="mr-1 h-3 w-3 animate-spin" />
                ) : (
                  <Trophy className="mr-1 h-3 w-3" />
                )}
                Run Competition
              </Button>
            </div>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">Loading competition config...</p>
        )}

        {/* ── Results summary ──────────────────────────────────────── */}
        {championSummary ? (
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
                label="DFUs Evaluated"
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
                colorClass="text-indigo-700 dark:text-indigo-400"
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
                const total = championSummary.total_dfu_months ?? championSummary.total_dfus;
                const pct = total > 0 ? (wins / total) * 100 : 0;
                return (
                  <div key={model} className="flex items-center gap-2 text-sm">
                    <span className="w-40 truncate font-mono text-xs text-right">{model}</span>
                    <div className="flex-1 h-5 rounded bg-muted overflow-hidden">
                      <div
                        className="h-full rounded bg-indigo-500 transition-all"
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
        ) : null}
      </CardContent>
    </Card>
  );
}
