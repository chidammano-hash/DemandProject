/**
 * PipelineConfigPanel -- View and edit the master forecast pipeline configuration.
 *
 * Shows the algorithm roster with lifecycle flags, clustering settings,
 * backtest/tuning params, champion strategy, and production forecast config.
 */
import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Save, Settings2, Workflow } from "lucide-react";

import {
  Card, CardContent, CardDescription, CardHeader, CardTitle,
} from "@/components/ui/card";
import {
  Table, TableBody, TableCell, TableHead, TableHeader, TableRow,
} from "@/components/ui/table";
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

// Fallback algorithm display names — used when config does not include a display_name
const ALGO_LABEL_FALLBACK: Record<string, string> = {
  lgbm_cluster: "LightGBM",
  catboost_cluster: "CatBoost",
  xgboost_cluster: "XGBoost",
  lgbm_cust_enriched: "LightGBM (Cust)",
  catboost_cust_enriched: "CatBoost (Cust)",
  xgboost_cust_enriched: "XGBoost (Cust)",
  chronos: "Chronos T5",
  chronos_bolt: "Chronos Bolt",
  chronos2: "Chronos 2",
  chronos2_enriched: "Chronos 2E",
  bolt_hierarchical: "Bolt Hierarchical",
  mstl: "MSTL",
  nbeats: "N-BEATS",
  nhits: "N-HiTS",
  seasonal_naive: "Seasonal Naive",
  rolling_mean: "Rolling Mean",
};

/** Derive a display label for an algorithm ID. */
function algoLabel(id: string, algo?: PipelineAlgorithm): string {
  // Prefer display_name from config if available, then fallback map, then humanized ID
  const raw = algo as unknown as Record<string, unknown> | undefined;
  if (raw?.display_name) {
    return raw.display_name as string;
  }
  return ALGO_LABEL_FALLBACK[id] || id.replace(/_/g, " ").replace(/\b\w/g, c => c.toUpperCase());
}

const TYPE_COLORS: Record<string, string> = {
  tree: "bg-emerald-100 text-emerald-800 dark:bg-emerald-900 dark:text-emerald-200",
  foundation: "bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200",
  statistical: "bg-amber-100 text-amber-800 dark:bg-amber-900 dark:text-amber-200",
  deep_learning: "bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-200",
};

export function PipelineConfigPanel() {
  const queryClient = useQueryClient();
  const [pendingChanges, setPendingChanges] = useState<Record<string, unknown>>({});
  const [saving, setSaving] = useState(false);

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
  const stages = config.pipeline?.stages || [];

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

      {/* Pipeline Stages */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Workflow className="h-4 w-4" /> Pipeline Stages
          </CardTitle>
          <CardDescription>Execution order and dependencies</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center gap-2 overflow-x-auto pb-2">
            {stages.map((stage: { name: string; description: string; depends_on: string[] }, i: number) => (
              <div key={stage.name} className="flex items-center gap-2">
                {i > 0 && <span className="text-muted-foreground">{"\u2192"}</span>}
                <div className="flex flex-col items-center rounded-lg border px-3 py-2 text-center min-w-[100px]">
                  <span className="text-xs font-semibold capitalize">{stage.name}</span>
                  <span className="text-[10px] text-muted-foreground">{stage.description}</span>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Algorithm Roster */}
      <Card>
        <CardHeader className="pb-3">
          <CardTitle className="flex items-center gap-2 text-base">
            <Settings2 className="h-4 w-4" /> Algorithm Roster
          </CardTitle>
          <CardDescription>
            Toggle lifecycle flags per algorithm. Changes take effect on next pipeline run.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="overflow-x-auto">
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead className="w-[140px]">Algorithm</TableHead>
                  <TableHead className="w-[80px]">Type</TableHead>
                  <TableHead className="w-[70px] text-center">Enabled</TableHead>
                  <TableHead className="w-[70px] text-center">Tune</TableHead>
                  <TableHead className="w-[70px] text-center">Backtest</TableHead>
                  <TableHead className="w-[70px] text-center">Compete</TableHead>
                  <TableHead className="w-[70px] text-center">Forecast</TableHead>
                  <TableHead className="w-[100px]">Cluster</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {Object.entries(algorithms).map(([id, algo]) => {
                  const a = algo as PipelineAlgorithm;
                  return (
                    <TableRow key={id}>
                      <TableCell className="font-medium text-sm">{algoLabel(id, a)}</TableCell>
                      <TableCell>
                        <Badge variant="outline" className={`text-[10px] ${TYPE_COLORS[a.type] || ""}`}>
                          {a.type}
                        </Badge>
                      </TableCell>
                      {(["enabled", "tune", "backtest", "compete", "forecast"] as const).map(flag => (
                        <TableCell key={flag} className="text-center">
                          <Switch
                            checked={getEffective(`algorithms.${id}.${flag}`, a[flag]) as boolean}
                            onCheckedChange={(v) => setField(`algorithms.${id}.${flag}`, v)}
                            className="mx-auto"
                          />
                        </TableCell>
                      ))}
                      <TableCell>
                        {a.cluster_strategy ? (
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
                        ) : (
                          <span className="text-xs text-muted-foreground">{"\u2014"}</span>
                        )}
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </div>
        </CardContent>
      </Card>

      {/* Settings Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
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
              <label className="text-xs">Embargo (months)</label>
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

        {/* Champion */}
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
              <label className="text-xs">Fallback Model</label>
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

        {/* Production Forecast */}
        <Card>
          <CardHeader className="pb-3">
            <CardTitle className="text-sm">Production Forecast</CardTitle>
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
              <label className="text-xs">Cold-Start Floor (months)</label>
              <Input
                type="number" className="h-7 w-20 text-xs"
                value={getEffective("production_forecast.cold_start_min_months", prodForecast.cold_start_min_months) as number}
                onChange={(e) => setField("production_forecast.cold_start_min_months", Number(e.target.value))}
                min={1} max={12}
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
              <label className="text-xs">Target DFUs</label>
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
              <label className="text-xs">Optuna Trials</label>
              <Input
                type="number" className="h-7 w-20 text-xs"
                value={getEffective("tuning.n_trials", config.tuning?.n_trials) as number}
                onChange={(e) => setField("tuning.n_trials", Number(e.target.value))}
                min={10} max={500} step={10}
              />
            </div>
            <div className="flex items-center justify-between">
              <label className="text-xs">CV Gap (months)</label>
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
    </div>
  );
}
