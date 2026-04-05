/**
 * PipelineConfigPanel -- View and edit the master forecast pipeline configuration.
 *
 * Shows the algorithm roster with lifecycle flags, clustering settings,
 * backtest/tuning params, champion strategy, and production forecast config.
 */
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Save, Settings2, ChevronDown, ChevronRight } from "lucide-react";

import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { LoadingElement } from "@/components/LoadingElement";

import {
  fetchPipelineConfig, updatePipelineConfig,
  pipelineConfigKeys, type PipelineConfig, type PipelineAlgorithm,
} from "@/api/queries/unified-model-tuning";
import { MODEL_LABELS, MODEL_TYPE_COLORS, modelLabel } from "@/lib/model-labels";

/** Derive a display label for an algorithm ID. */
function algoLabel(id: string, algo?: PipelineAlgorithm): string {
  const raw = algo as unknown as Record<string, unknown> | undefined;
  if (raw?.display_name) {
    return raw.display_name as string;
  }
  return modelLabel(id);
}

export function PipelineConfigPanel() {
  const queryClient = useQueryClient();
  const [pendingChanges, setPendingChanges] = useState<Record<string, unknown>>({});
  const [saving, setSaving] = useState(false);
  const [expandedAlgo, setExpandedAlgo] = useState<string | null>(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const { data: config, isLoading, error } = useQuery({
    queryKey: pipelineConfigKeys.config,
    queryFn: fetchPipelineConfig,
    staleTime: 30_000,
  });

  const saveMutation = useMutation({
    mutationFn: (values: Record<string, unknown>) => updatePipelineConfig(values),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: pipelineConfigKeys.config });
      setPendingChanges({});
      setSaving(false);
    },
    onError: () => setSaving(false),
  });

  if (isLoading) return <LoadingElement message="Loading pipeline config..." />;
  if (error || !config) return <div className="p-4 text-red-500">Failed to load pipeline config</div>;

  const hasPending = Object.keys(pendingChanges).length > 0;

  function setField(path: string, value: unknown) {
    setPendingChanges(prev => ({ ...prev, [path]: value }));
  }

  function handleSave() {
    if (!hasPending) return;
    setSaving(true);
    saveMutation.mutate(pendingChanges);
  }

  function getEffective(path: string, fallback: unknown): unknown {
    return path in pendingChanges ? pendingChanges[path] : fallback;
  }

  const algorithms = config.algorithms || {};
  const clustering = config.clustering || { enabled: true };
  const backtest = config.backtest || {};
  const champion = config.champion || {};
  const prodForecast = config.production_forecast || {};
  const sampling = config.backtest_sampling || {};

  // Group algorithms by type
  const algosByType: Record<string, [string, PipelineAlgorithm][]> = {};
  for (const [id, algo] of Object.entries(algorithms)) {
    const a = algo as PipelineAlgorithm;
    const t = a.type || "other";
    if (!algosByType[t]) algosByType[t] = [];
    algosByType[t].push([id, a]);
  }

  const typeOrder = ["tree", "foundation", "statistical", "deep_learning"];
  const sortedTypes = typeOrder.filter(t => algosByType[t]);
  // Append any types not in the standard order
  for (const t of Object.keys(algosByType)) {
    if (!sortedTypes.includes(t)) sortedTypes.push(t);
  }

  return (
    <div className="space-y-6">
      {/* Save bar */}
      {hasPending && (
        <div className="sticky top-0 z-10 flex items-center justify-between rounded-lg border border-yellow-300 bg-yellow-50 px-4 py-2 dark:border-yellow-700 dark:bg-yellow-900/30">
          <span className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
            {Object.keys(pendingChanges).length} unsaved change(s)
          </span>
          <div className="flex gap-2">
            <Button variant="outline" size="sm" onClick={() => setPendingChanges({})}>
              Discard
            </Button>
            <Button size="sm" onClick={handleSave} disabled={saving}>
              <Save className="mr-1 h-3.5 w-3.5" />
              {saving ? "Saving..." : "Save Changes"}
            </Button>
          </div>
        </div>
      )}

      {/* Forecast Settings — always visible */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {/* Production Forecast */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Forecast Settings</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs">Horizon (months)</label>
              <Input
                type="number" className="h-7 w-20 text-xs"
                value={getEffective("production_forecast.horizon_months", prodForecast.horizon_months) as number}
                onChange={(e) => setField("production_forecast.horizon_months", Number(e.target.value))}
                min={6} max={36}
              />
            </div>
            <div className="flex items-center justify-between">
              <label className="text-xs">Min History (months)</label>
              <Input
                type="number" className="h-7 w-20 text-xs"
                value={getEffective("production_forecast.min_history_months", prodForecast.min_history_months) as number}
                onChange={(e) => setField("production_forecast.min_history_months", Number(e.target.value))}
                min={3} max={24}
              />
            </div>
            <div className="flex items-center justify-between">
              <label className="text-xs">Cold-Start Model</label>
              <Input
                className="h-7 w-[120px] text-xs"
                value={getEffective("production_forecast.cold_start_model_id", prodForecast.cold_start_model_id) as string}
                onChange={(e) => setField("production_forecast.cold_start_model_id", e.target.value)}
              />
            </div>
            <div className="flex items-center justify-between">
              <label className="text-xs">New Product Min Data (months)</label>
              <Input
                type="number" className="h-7 w-20 text-xs"
                value={getEffective("production_forecast.cold_start_min_months", prodForecast.cold_start_min_months) as number}
                onChange={(e) => setField("production_forecast.cold_start_min_months", Number(e.target.value))}
                min={1} max={12}
              />
            </div>
          </CardContent>
        </Card>

        {/* Champion Selection */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Champion Selection</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs">Strategy</label>
              <Select
                value={getEffective("champion.strategy", champion.strategy) as string}
                onValueChange={(v) => setField("champion.strategy", v)}
              >
                <SelectTrigger className="h-7 w-[140px] text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {["expanding", "rolling", "decay", "ensemble", "meta_learner", "hybrid_warmup", "adaptive_ensemble"].map(s => (
                    <SelectItem key={s} value={s}>{s}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-center justify-between">
              <label className="text-xs">Default Model</label>
              <Input
                className="h-7 w-[140px] text-xs"
                value={getEffective("champion.fallback_model_id", champion.fallback_model_id) as string}
                onChange={(e) => setField("champion.fallback_model_id", e.target.value)}
              />
            </div>
            <div className="flex items-center justify-between">
              <label className="text-xs">Metric</label>
              <Select
                value={getEffective("champion.metric", champion.metric) as string}
                onValueChange={(v) => setField("champion.metric", v)}
              >
                <SelectTrigger className="h-7 w-[140px] text-xs">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="accuracy_pct">accuracy_pct</SelectItem>
                  <SelectItem value="wape">wape</SelectItem>
                </SelectContent>
              </Select>
            </div>
          </CardContent>
        </Card>

        {/* Clustering */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Clustering</CardTitle>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between">
              <label className="text-xs">Enabled</label>
              <Switch
                checked={getEffective("clustering.enabled", clustering.enabled) as boolean}
                onCheckedChange={(v) => setField("clustering.enabled", v)}
              />
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Active Models — card grid grouped by type, always visible */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Settings2 className="h-4 w-4" /> Active Models
          </CardTitle>
          <CardDescription>
            Toggle models and configure lifecycle flags. Changes take effect on next pipeline run.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-5">
          {sortedTypes.map(type => (
            <div key={type}>
              <div className="mb-2">
                <Badge variant="outline" className={`text-[10px] ${MODEL_TYPE_COLORS[type] || ""}`}>
                  {type.replace(/_/g, " ")}
                </Badge>
              </div>
              <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
                {algosByType[type].map(([id, algo]) => {
                  const a = algo as PipelineAlgorithm;
                  const isEnabled = getEffective(`algorithms.${id}.enabled`, a.enabled) as boolean;
                  const isExpanded = expandedAlgo === id;
                  return (
                    <div
                      key={id}
                      className={`rounded-lg border p-3 transition-colors ${
                        isEnabled ? "bg-background" : "bg-muted/40 opacity-70"
                      }`}
                    >
                      {/* Header row: name + enable switch */}
                      <div className="flex items-center justify-between">
                        <span className="text-sm font-medium">{algoLabel(id, a)}</span>
                        <Switch
                          checked={isEnabled}
                          onCheckedChange={(v) => setField(`algorithms.${id}.enabled`, v)}
                        />
                      </div>
                      {/* Configure button */}
                      <button
                        className="mt-2 flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
                        onClick={() => setExpandedAlgo(isExpanded ? null : id)}
                      >
                        {isExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                        Configure
                      </button>
                      {/* Expanded lifecycle toggles + cluster strategy */}
                      {isExpanded && (
                        <div className="mt-3 space-y-2 border-t pt-3">
                          {(["tune", "backtest", "compete", "forecast"] as const).map(flag => (
                            <div key={flag} className="flex items-center justify-between">
                              <label className="text-xs capitalize">{flag}</label>
                              <Switch
                                checked={getEffective(`algorithms.${id}.${flag}`, a[flag]) as boolean}
                                onCheckedChange={(v) => setField(`algorithms.${id}.${flag}`, v)}
                              />
                            </div>
                          ))}
                          {a.cluster_strategy && (
                            <div className="flex items-center justify-between">
                              <label className="text-xs">Cluster Strategy</label>
                              <Select
                                value={getEffective(`algorithms.${id}.cluster_strategy`, a.cluster_strategy) as string}
                                onValueChange={(v) => setField(`algorithms.${id}.cluster_strategy`, v)}
                              >
                                <SelectTrigger className="h-7 w-[110px] text-xs">
                                  <SelectValue />
                                </SelectTrigger>
                                <SelectContent>
                                  <SelectItem value="per_cluster">per_cluster</SelectItem>
                                  <SelectItem value="global">global</SelectItem>
                                </SelectContent>
                              </Select>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </CardContent>
      </Card>

      {/* Advanced Settings — collapsed by default */}
      <div className="rounded-lg border">
        <button
          className="flex w-full items-center justify-between px-4 py-3 text-left"
          onClick={() => setShowAdvanced(!showAdvanced)}
        >
          <span className="text-sm font-semibold">Advanced Settings</span>
          {showAdvanced ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
        </button>
        {showAdvanced && (
          <div className="grid gap-4 px-4 pb-4 md:grid-cols-2 lg:grid-cols-3">
            {/* Backtest */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Backtest</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-xs">Timeframes</label>
                  <Input
                    type="number" className="h-7 w-20 text-xs"
                    value={getEffective("backtest.n_timeframes", backtest.n_timeframes) as number}
                    onChange={(e) => setField("backtest.n_timeframes", Number(e.target.value))}
                    min={1} max={30}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <label className="text-xs">Data Holdout (months)</label>
                  <Input
                    type="number" className="h-7 w-20 text-xs"
                    value={getEffective("backtest.embargo_months", backtest.embargo_months) as number}
                    onChange={(e) => setField("backtest.embargo_months", Number(e.target.value))}
                    min={0} max={6}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <label className="text-xs">Horizon (months)</label>
                  <Input
                    type="number" className="h-7 w-20 text-xs"
                    value={getEffective("backtest.forecast_horizon", backtest.forecast_horizon) as number}
                    onChange={(e) => setField("backtest.forecast_horizon", Number(e.target.value))}
                    min={1} max={24}
                  />
                </div>
              </CardContent>
            </Card>

            {/* Backtest Sampling */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Backtest Sampling</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-xs">Enabled</label>
                  <Switch
                    checked={getEffective("backtest_sampling.enabled", sampling.enabled) as boolean}
                    onCheckedChange={(v) => setField("backtest_sampling.enabled", v)}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <label className="text-xs">Sample Size (items)</label>
                  <Input
                    type="number" className="h-7 w-20 text-xs"
                    value={getEffective("backtest_sampling.default_target_n", sampling.default_target_n) as number}
                    onChange={(e) => setField("backtest_sampling.default_target_n", Number(e.target.value))}
                    min={100} max={50000} step={100}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <label className="text-xs">Method</label>
                  <Select
                    value={getEffective("backtest_sampling.default_method", sampling.default_method) as string}
                    onValueChange={(v) => setField("backtest_sampling.default_method", v)}
                  >
                    <SelectTrigger className="h-7 w-[120px] text-xs">
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="proportional">proportional</SelectItem>
                      <SelectItem value="equal">equal</SelectItem>
                      <SelectItem value="sqrt">sqrt</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
              </CardContent>
            </Card>

            {/* Tuning */}
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-sm">Tuning</CardTitle>
              </CardHeader>
              <CardContent className="space-y-3">
                <div className="flex items-center justify-between">
                  <label className="text-xs">Tuning Iterations</label>
                  <Input
                    type="number" className="h-7 w-20 text-xs"
                    value={getEffective("tuning.n_trials", config.tuning?.n_trials) as number}
                    onChange={(e) => setField("tuning.n_trials", Number(e.target.value))}
                    min={10} max={500} step={10}
                  />
                </div>
                <div className="flex items-center justify-between">
                  <label className="text-xs">Validation Gap (months)</label>
                  <Input
                    type="number" className="h-7 w-20 text-xs"
                    value={getEffective("tuning.gap_months", config.tuning?.gap_months) as number}
                    onChange={(e) => setField("tuning.gap_months", Number(e.target.value))}
                    min={0} max={6}
                  />
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
