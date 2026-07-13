/**
 * SweepBuilder — modal for launching a Champion Strategy Sweep (tournament).
 *
 * Pick template chips (the candidate axis), the model subset every strategy
 * competes over (config-driven roster + presets), the mode (global / per-segment
 * / both), segmentation axis, objective, and parallel toggle. Shows a live
 * "expands to N candidates" preview that mirrors the backend's expansion
 * arithmetic (candidate count = #templates; the model subset doesn't multiply it).
 * See spec 30.
 */
import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { FlaskConical, X } from "lucide-react";

import {
  championExperimentKeys,
  championSweepKeys,
  createChampionSweep,
  fetchChampionTemplates,
  type ChampionExperimentTemplate,
  type SweepAxis,
  type SweepMode,
  type SweepObjective,
} from "@/api/queries";
import { fetchPipelineConfig, pipelineConfigKeys } from "@/api/queries/unified-model-tuning";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import {
  Select, SelectContent, SelectItem, SelectTrigger, SelectValue,
} from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Input } from "@/components/ui/input";
import { isForecastModelId, modelLabel } from "@/lib/model-labels";
import { cn } from "@/lib/utils";

const MODE_OPTIONS: { value: SweepMode; label: string }[] = [
  { value: "both", label: "Global + per-segment (compare)" },
  { value: "global", label: "Global winner only" },
  { value: "per_segment", label: "Per-segment composite only" },
];

const AXIS_OPTIONS: { value: SweepAxis; label: string }[] = [
  { value: "demand_class", label: "Demand class (promotable)" },
  { value: "ml_cluster", label: "ML cluster (diagnostic)" },
  { value: "abc_xyz", label: "ABC-XYZ (diagnostic)" },
];

const OBJECTIVE_OPTIONS: { value: SweepObjective; label: string }[] = [
  { value: "robust", label: "Robust (penalise lag/month dispersion)" },
  { value: "accuracy", label: "Accuracy" },
  { value: "gap_to_ceiling", label: "Gap to ceiling" },
];

export function SweepBuilder({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient();
  const { data: templatesData } = useQuery({
    queryKey: championExperimentKeys.templates(),
    queryFn: fetchChampionTemplates,
    staleTime: 300_000,
  });
  const templates: ChampionExperimentTemplate[] = templatesData?.templates ?? [];

  // Model roster + type map come from the live pipeline config (not hardcoded).
  const { data: pipelineConfig } = useQuery({
    queryKey: pipelineConfigKeys.config,
    queryFn: fetchPipelineConfig,
    staleTime: 300_000,
  });

  const algos = useMemo(() => pipelineConfig?.algorithms ?? {}, [pipelineConfig?.algorithms]);
  // Roster = every enabled model that competes or is forecastable (covers
  // models that are forecast=true but not compete=true).
  const roster = useMemo(
    () =>
      Object.entries(algos)
        .filter(([id, a]) => isForecastModelId(id) && a.enabled && (a.compete || a.forecast))
        .map(([id, a]) => ({ id, type: a.type })),
    [algos],
  );
  const competing = useMemo(
    () => roster.filter((m) => algos[m.id]?.compete).map((m) => m.id),
    [roster, algos],
  );
  const byType = (t: string) => roster.filter((m) => m.type === t).map((m) => m.id);

  const [label, setLabel] = useState("");
  const [selectedTemplates, setSelectedTemplates] = useState<string[]>([]);
  const [selectedModels, setSelectedModels] = useState<string[]>([]);
  const [mode, setMode] = useState<SweepMode>("both");
  const [axis, setAxis] = useState<SweepAxis>("demand_class");
  const [objective, setObjective] = useState<SweepObjective>("robust");
  const [parallel, setParallel] = useState(false);

  // Default the model subset to the current production champion pair once config loads.
  const championModels = useMemo(
    () =>
      ((pipelineConfig?.champion as { models?: string[] } | undefined)?.models ?? []).filter(
        isForecastModelId,
      ),
    [pipelineConfig],
  );
  useEffect(() => {
    if (selectedModels.length === 0 && championModels.length > 0) {
      setSelectedModels(championModels);
    }
  }, [championModels, selectedModels.length]);

  // Candidate count = templates × 1 model subset. (Per-segment scoring adds no runs.)
  const candidateCount = useMemo(
    () => (selectedModels.length >= 2 ? selectedTemplates.length : 0),
    [selectedTemplates, selectedModels],
  );

  const mutation = useMutation({
    mutationFn: createChampionSweep,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: championSweepKeys.all });
      onClose();
    },
  });

  const toggleTemplate = (id: string) =>
    setSelectedTemplates((s) => (s.includes(id) ? s.filter((t) => t !== id) : [...s, id]));
  const toggleModel = (id: string) =>
    setSelectedModels((s) => (s.includes(id) ? s.filter((m) => m !== id) : [...s, id]));

  const PRESETS: { label: string; models: string[] }[] = [
    { label: "Current champion", models: championModels },
    { label: "N-HiTS + N-BEATS", models: byType("deep_learning") },
    {
      label: "LightGBM + Chronos 2E",
      models: [...byType("tree"), ...byType("foundation")],
    },
    { label: "All competing", models: competing },
  ];

  const canLaunch =
    label.trim().length > 0 && candidateCount > 0 && selectedModels.length >= 2 && !mutation.isPending;

  const launch = () => {
    const models_variants = [selectedModels];
    mutation.mutate({
      label: label.trim(),
      mode,
      segment_axis: axis,
      objective,
      parallel,
      grid_spec: {
        templates: selectedTemplates,
        models_variants,
      },
    });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 p-4">
      <Card className="w-full max-w-2xl max-h-[90vh] overflow-auto">
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            <FlaskConical className="h-4 w-4" /> Run Champion Sweep
          </CardTitle>
          <Button variant="ghost" size="icon" onClick={onClose} aria-label="Close">
            <X className="h-4 w-4" />
          </Button>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="text-xs font-medium" htmlFor="sweep-label">Label</label>
            <Input
              id="sweep-label"
              value={label}
              onChange={(e) => setLabel(e.target.value)}
              placeholder="e.g. June champion tournament"
              className="mt-1 h-8 text-sm"
            />
          </div>

          {/* Template chips */}
          <div>
            <div className="text-xs font-medium mb-1">Candidate strategies (templates)</div>
            <div className="flex flex-wrap gap-1.5">
              {templates.map((t) => (
                <button
                  key={t.id}
                  type="button"
                  onClick={() => toggleTemplate(t.id)}
                  className={cn(
                    "rounded-full border px-2.5 py-1 text-xs transition-colors",
                    selectedTemplates.includes(t.id)
                      ? "border-primary bg-primary text-primary-foreground"
                      : "border-border hover:bg-muted",
                  )}
                  title={t.description}
                >
                  {t.label ?? t.id}
                </button>
              ))}
            </div>
          </div>

          {/* Model subset — which models every strategy competes over */}
          <div>
            <div className="flex items-center justify-between mb-1">
              <div className="text-xs font-medium">
                Model subset <span className="text-muted-foreground">({selectedModels.length} selected — every strategy competes over these)</span>
              </div>
            </div>
            {/* Presets */}
            <div className="flex flex-wrap gap-1.5 mb-2">
              {PRESETS.filter((p) => p.models.length >= 2).map((p) => (
                <button
                  key={p.label}
                  type="button"
                  onClick={() => setSelectedModels(p.models)}
                  className="rounded-md border border-dashed border-border px-2 py-0.5 text-[11px] text-muted-foreground hover:bg-muted"
                >
                  {p.label} ({p.models.length})
                </button>
              ))}
            </div>
            {/* Per-model checklist (config-driven roster) */}
            <div className="flex flex-wrap gap-1.5">
              {roster.map((m) => (
                <button
                  key={m.id}
                  type="button"
                  onClick={() => toggleModel(m.id)}
                  className={cn(
                    "rounded-full border px-2.5 py-1 text-xs transition-colors",
                    selectedModels.includes(m.id)
                      ? "border-primary bg-primary text-primary-foreground"
                      : "border-border hover:bg-muted",
                  )}
                  title={m.type}
                >
                  {modelLabel(m.id)}
                </button>
              ))}
            </div>
            {selectedModels.length < 2 ? (
              <div className="mt-1 text-[11px] text-amber-600">Select at least 2 models.</div>
            ) : null}
          </div>

          {/* Selects */}
          <div className="grid grid-cols-3 gap-3">
            <LabeledSelect label="Mode" value={mode} onChange={(v) => setMode(v as SweepMode)} options={MODE_OPTIONS} />
            <LabeledSelect label="Segment axis" value={axis} onChange={(v) => setAxis(v as SweepAxis)} options={AXIS_OPTIONS} />
            <LabeledSelect label="Objective" value={objective} onChange={(v) => setObjective(v as SweepObjective)} options={OBJECTIVE_OPTIONS} />
          </div>

          <label className="flex items-center gap-2 text-xs">
            <Switch checked={parallel} onCheckedChange={setParallel} />
            Run children in parallel (across model families)
          </label>

          {/* Candidate preview + launch */}
          <div className="flex items-center justify-between border-t pt-3">
            <div className="text-xs text-muted-foreground">
              Expands to <span className="font-semibold text-foreground">{candidateCount}</span> candidate
              {candidateCount === 1 ? "" : "s"}
              {mode !== "global" ? " (per-segment scoring adds no extra runs)" : ""}
            </div>
            <Button size="sm" disabled={!canLaunch} onClick={launch}>
              {mutation.isPending ? "Launching…" : "Launch sweep"}
            </Button>
          </div>
          {mutation.isError ? (
            <div className="text-xs text-red-500">{(mutation.error as Error).message}</div>
          ) : null}
        </CardContent>
      </Card>
    </div>
  );
}

function LabeledSelect({
  label, value, onChange, options,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
}) {
  return (
    <div>
      <div className="text-xs font-medium mb-1">{label}</div>
      <Select value={value} onValueChange={onChange}>
        <SelectTrigger className="h-8 text-xs">
          <SelectValue />
        </SelectTrigger>
        <SelectContent>
          {options.map((o) => (
            <SelectItem key={o.value} value={o.value} className="text-xs">
              {o.label}
            </SelectItem>
          ))}
        </SelectContent>
      </Select>
    </div>
  );
}
