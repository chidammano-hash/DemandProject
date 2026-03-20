/**
 * ChampionConfigPanel — Champion model competition configuration.
 * Renders inline within the champion_select job card in JobGroupsPanel.
 * Lets users configure competing models, metric, lag, then save config
 * and/or submit a champion_select job.
 */
import { useCallback, useEffect, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Loader2, Trophy } from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  queryKeys,
  STALE,
  fetchCompetitionConfig,
  saveCompetitionConfig,
  submitJob,
  type CompetitionConfig,
} from "@/api/queries";

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export interface ChampionConfigPanelProps {
  /** Called after a champion_select job is submitted (for invalidation) */
  onJobSubmitted?: () => void;
}

export function ChampionConfigPanel({ onJobSubmitted }: ChampionConfigPanelProps) {
  const queryClient = useQueryClient();
  const [config, setConfig] = useState<CompetitionConfig | null>(null);

  // ── Fetch current config ─────────────────────────────────────────────────
  const { data: configPayload } = useQuery({
    queryKey: queryKeys.competitionConfig(),
    queryFn: fetchCompetitionConfig,
    staleTime: STALE.FIVE_MIN,
  });

  // Sync fetched config into local state (one-time init)
  useEffect(() => {
    if (configPayload?.config && config === null) {
      setConfig(configPayload.config);
    }
  }, [configPayload, config]);

  const availableModels: string[] = configPayload?.available_models ?? [];

  // ── Mutations ────────────────────────────────────────────────────────────
  const saveConfigMutation = useMutation({ mutationFn: saveCompetitionConfig });

  const saveAndRunMutation = useMutation({
    mutationFn: async (cfg: CompetitionConfig) => {
      await saveCompetitionConfig(cfg);
      return submitJob("champion_select", {}, "Champion Selection");
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.competitionSummary() });
      queryClient.invalidateQueries({ queryKey: queryKeys.activeJobs() });
      queryClient.invalidateQueries({ queryKey: queryKeys.jobStats() });
      onJobSubmitted?.();
    },
  });

  // ── Handlers ─────────────────────────────────────────────────────────────
  const handleModelToggle = useCallback((model: string) => {
    setConfig((prev) => {
      if (!prev) return prev;
      const checked = prev.models.includes(model);
      return { ...prev, models: checked ? prev.models.filter((x) => x !== model) : [...prev.models, model] };
    });
  }, []);

  const handleMetricChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setConfig((prev) => (prev ? { ...prev, metric: e.target.value } : prev));
  }, []);

  const handleLagChange = useCallback((e: React.ChangeEvent<HTMLSelectElement>) => {
    setConfig((prev) => (prev ? { ...prev, lag: e.target.value } : prev));
  }, []);

  const handleSaveOnly = useCallback(() => {
    if (config) saveConfigMutation.mutate(config);
  }, [config, saveConfigMutation]);

  const handleSaveAndRun = useCallback(() => {
    if (config && config.models.length >= 2) saveAndRunMutation.mutate(config);
  }, [config, saveAndRunMutation]);

  const saving = saveConfigMutation.isPending;
  const running = saveAndRunMutation.isPending;

  // ── Render ───────────────────────────────────────────────────────────────
  if (!config) {
    return <p className="text-sm text-muted-foreground">Loading config...</p>;
  }

  return (
    <div className="space-y-3">
      {/* Competing Models checkboxes */}
      <div className="space-y-1.5">
        <span className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Competing Models
        </span>
        <div className="flex flex-wrap gap-3">
          {availableModels
            .filter((m) => m !== config.champion_model_id && m !== "ceiling")
            .map((m) => {
              const checked = config.models.includes(m);
              const isLast = config.models.length <= 2 && checked;
              return (
                <label key={m} className="flex items-center gap-1.5 text-sm cursor-pointer select-none">
                  <input
                    type="checkbox"
                    className="h-3.5 w-3.5 rounded border-input accent-blue-600"
                    checked={checked}
                    disabled={isLast || running}
                    onChange={() => handleModelToggle(m)}
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
            value={config.metric}
            onChange={handleMetricChange}
            disabled={running}
          >
            <option value="wape">WAPE (Lowest Wins)</option>
            <option value="accuracy_pct">Accuracy % (Highest Wins)</option>
          </select>
        </label>
        <label className="space-y-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
          Lag
          <select
            className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
            value={config.lag}
            onChange={handleLagChange}
            disabled={running}
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
          size="sm"
          disabled={saving || running || config.models.length < 2}
          onClick={handleSaveAndRun}
        >
          {(saving || running) ? (
            <Loader2 className="mr-1 h-3 w-3 animate-spin" />
          ) : (
            <Trophy className="mr-1 h-3 w-3" />
          )}
          Save &amp; Run
        </Button>
        <Button
          variant="ghost"
          size="sm"
          disabled={saving || running}
          onClick={handleSaveOnly}
          title="Save configuration without running competition"
        >
          Save only
        </Button>
      </div>
    </div>
  );
}
