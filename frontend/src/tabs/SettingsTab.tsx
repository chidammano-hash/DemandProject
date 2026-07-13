/**
 * Settings Tab — view and edit all system configuration from the UI.
 *
 * Layout: category sidebar (left) → config list (middle) → field editor (right).
 * Each field has a label, description tooltip, and type-appropriate input.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import {
  Settings2, Save, RotateCcw, Search, ChevronRight, Check, AlertTriangle,
  BarChart3, Package, Truck, Database, CalendarClock, Server,
  Layers, Sliders, ToggleRight, X,
} from "lucide-react";

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/Skeleton";
import { cn } from "@/lib/utils";

import {
  configKeys,
  fetchConfigList,
  fetchConfigDetail,
  updateConfig,
  type ConfigField,
} from "@/api/queries/config";

// ---------------------------------------------------------------------------
// Category icon colors + icons
// ---------------------------------------------------------------------------
const CATEGORY_COLORS: Record<string, string> = {
  forecasting: "text-blue-500",
  inventory: "text-emerald-500",
  operations: "text-amber-500",
  pipeline: "text-purple-500",
  planning: "text-cyan-500",
  system: "text-rose-500",
};

const CATEGORY_ICONS: Record<string, React.FC<{ className?: string }>> = {
  forecasting: BarChart3,
  inventory: Package,
  operations: Truck,
  pipeline: Database,
  planning: CalendarClock,
  system: Server,
};

const GROUP_ICONS: Record<string, React.FC<{ className?: string }>> = {
  _general: Sliders,
  hyperparameters: Sliders,
  models: ToggleRight,
  active_models: ToggleRight,
  algorithms: Layers,
  thresholds: BarChart3,
  scoring: BarChart3,
  schedule: CalendarClock,
  connection: Database,
};

// ---------------------------------------------------------------------------
// Field renderer — renders an appropriate input based on field type
// ---------------------------------------------------------------------------
function FieldInput({
  field,
  value,
  onChange,
}: {
  field: ConfigField;
  value: unknown;
  onChange: (path: string, value: unknown) => void;
}) {
  const inputClass = "h-8 rounded border border-input bg-background px-2 text-sm focus:outline-none focus:ring-1 focus:ring-ring";

  switch (field.type) {
    case "boolean":
      return (
        <button
          type="button"
          role="switch"
          aria-checked={!!value}
          onClick={() => onChange(field.path, !value)}
          className={cn(
            "relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent transition-colors",
            value ? "bg-primary" : "bg-muted",
          )}
        >
          <span
            className={cn(
              "pointer-events-none inline-block h-5 w-5 rounded-full bg-background shadow-lg ring-0 transition-transform",
              value ? "translate-x-5" : "translate-x-0",
            )}
          />
        </button>
      );

    case "select":
      return (
        <select
          className={cn(inputClass, "min-w-[140px]")}
          value={String(value ?? "")}
          onChange={(e) => onChange(field.path, e.target.value)}
        >
          {(field.options ?? []).map((opt) => (
            <option key={opt} value={opt}>{opt}</option>
          ))}
        </select>
      );

    case "integer":
    case "number":
      return (
        <div className="flex items-center gap-1.5">
          <input
            type="number"
            className={cn(inputClass, "w-28 tabular-nums")}
            value={value != null ? String(value) : ""}
            min={field.min}
            max={field.max}
            step={field.step ?? (field.type === "integer" ? 1 : 0.01)}
            onChange={(e) => {
              const v = e.target.value;
              if (v === "") { onChange(field.path, null); return; }
              onChange(field.path, field.type === "integer" ? parseInt(v, 10) : parseFloat(v));
            }}
          />
          {field.unit && <span className="text-xs text-muted-foreground">{field.unit}</span>}
        </div>
      );

    case "array":
      return (
        <input
          type="text"
          className={cn(inputClass, "w-full max-w-xs font-mono text-xs")}
          value={Array.isArray(value) ? JSON.stringify(value) : String(value ?? "[]")}
          onChange={(e) => {
            try { onChange(field.path, JSON.parse(e.target.value)); } catch { /* ignore parse errors while typing */ }
          }}
          title="JSON array — e.g. [0.10, 0.50, 0.90]"
        />
      );

    default: // text
      return (
        <input
          type="text"
          className={cn(inputClass, "w-full max-w-xs")}
          value={String(value ?? "")}
          onChange={(e) => onChange(field.path, e.target.value)}
        />
      );
  }
}

// ---------------------------------------------------------------------------
// Field row — label + description + input
// ---------------------------------------------------------------------------
function FieldRow({
  field,
  value,
  originalValue,
  onChange,
}: {
  field: ConfigField;
  value: unknown;
  originalValue: unknown;
  onChange: (path: string, value: unknown) => void;
}) {
  const isChanged = JSON.stringify(value) !== JSON.stringify(originalValue);

  return (
    <div className={cn(
      "grid grid-cols-[1fr_auto] items-start gap-4 rounded-md px-3 py-2.5 transition-colors",
      isChanged && "bg-amber-50 dark:bg-amber-950/20",
    )}>
      <div className="min-w-0">
        <div className="flex items-center gap-1.5">
          <span className="text-sm font-medium">{field.label}</span>
          {isChanged && <span className="rounded bg-amber-100 px-1 py-0.5 text-[9px] font-semibold text-amber-700 dark:bg-amber-900/50 dark:text-amber-300">MODIFIED</span>}
        </div>
        <p className="mt-0.5 text-xs text-muted-foreground leading-relaxed">{field.description}</p>
        {(field.min != null || field.max != null) && (
          <p className="mt-0.5 text-[10px] text-muted-foreground/60">
            {field.min != null && `Min: ${field.min}`}
            {field.min != null && field.max != null && " · "}
            {field.max != null && `Max: ${field.max}`}
          </p>
        )}
      </div>
      <div className="flex items-center pt-0.5">
        <FieldInput field={field} value={value} onChange={onChange} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Model toggle renderer — compact grid of on/off switches for algorithms
// ---------------------------------------------------------------------------
const MODEL_TYPE_BADGE: Record<string, string> = {
  tree: "bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-300",
  foundation: "bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-300",
  statistical: "bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-300",
  deep_learning: "bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-300",
};

function ModelToggleGrid({
  fields,
  editValues,
  onChange,
}: {
  fields: ConfigField[];
  editValues: Record<string, unknown>;
  onChange: (path: string, value: unknown) => void;
}) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 gap-2.5">
      {fields.map((f) => {
        const isOn = !!editValues[f.path];
        const modelType = f.model_type;
        return (
          <button
            key={f.path}
            type="button"
            onClick={() => onChange(f.path, !isOn)}
            className={cn(
              "group relative flex items-center gap-2.5 rounded-lg border px-3 py-2.5 text-left transition-all duration-200",
              isOn
                ? "border-primary/40 bg-primary/5 shadow-sm ring-1 ring-primary/20"
                : "border-border/50 bg-muted/20 opacity-60 hover:opacity-80 hover:bg-muted/40",
            )}
          >
            {/* Toggle indicator */}
            <div className={cn(
              "relative h-5 w-9 shrink-0 rounded-full transition-colors duration-200",
              isOn ? "bg-primary" : "bg-muted-foreground/20",
            )}>
              <div className={cn(
                "absolute top-0.5 h-4 w-4 rounded-full bg-white shadow-sm transition-transform duration-200",
                isOn ? "translate-x-4" : "translate-x-0.5",
              )} />
            </div>
            <div className="min-w-0 flex-1">
              <div className={cn(
                "text-sm font-medium truncate transition-colors",
                isOn ? "text-foreground" : "text-muted-foreground",
              )}>{f.label}</div>
              <div className="flex items-center gap-1.5 mt-0.5">
                {modelType && (
                  <span className={cn("rounded px-1.5 py-px text-[9px] font-semibold", MODEL_TYPE_BADGE[modelType] || "")}>
                    {modelType === "deep_learning" ? "DL" : modelType}
                  </span>
                )}
                <span className="text-[10px] text-muted-foreground truncate">{f.description}</span>
              </div>
            </div>
            {/* Active indicator dot */}
            {isOn && (
              <div className="absolute top-1.5 right-1.5 h-1.5 w-1.5 rounded-full bg-primary" />
            )}
          </button>
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Group fields by explicit "group" metadata or top-level path prefix
// ---------------------------------------------------------------------------
function groupFields(fields: ConfigField[]): Map<string, ConfigField[]> {
  const groups = new Map<string, ConfigField[]>();
  for (const f of fields) {
    const group = (f as unknown as Record<string, unknown>).group as string | undefined;
    const key = group || (f.path.includes(".") ? f.path.split(".")[0] : "_general");
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key)!.push(f);
  }
  return groups;
}

function formatGroupLabel(key: string): string {
  if (key === "_general") return "General";
  return key
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
}

function GroupIcon({ groupKey }: { groupKey: string }) {
  const Icon = GROUP_ICONS[groupKey] ?? Sliders;
  return <Icon className="h-3.5 w-3.5 text-muted-foreground" />;
}

// ---------------------------------------------------------------------------
// Search highlight helper
// ---------------------------------------------------------------------------
function HighlightText({ text, query }: { text: string; query: string }) {
  if (!query.trim()) return <>{text}</>;
  const regex = new RegExp(`(${query.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")})`, "gi");
  const parts = text.split(regex);
  return (
    <>
      {parts.map((part, i) =>
        regex.test(part) ? (
          <mark key={i} className="bg-yellow-200 dark:bg-yellow-800/60 rounded px-0.5">{part}</mark>
        ) : (
          <span key={i}>{part}</span>
        )
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------
export function SettingsTab() {
  const queryClient = useQueryClient();

  // --- State ---
  const [selectedCategory, setSelectedCategory] = useState<string>("forecasting");
  const [selectedConfig, setSelectedConfig] = useState<string | null>(null);
  const [searchQuery, setSearchQuery] = useState("");
  const [fieldSearch, setFieldSearch] = useState("");
  const [editValues, setEditValues] = useState<Record<string, unknown>>({});
  const [saveStatus, setSaveStatus] = useState<"idle" | "saving" | "saved" | "error">("idle");

  // --- Queries ---
  const listQ = useQuery({
    queryKey: configKeys.list(),
    queryFn: fetchConfigList,
    staleTime: 60_000,
  });

  const detailQ = useQuery({
    queryKey: configKeys.detail(selectedConfig ?? ""),
    queryFn: () => fetchConfigDetail(selectedConfig!),
    enabled: !!selectedConfig,
    staleTime: 30_000,
  });

  // Initialize edit values when detail loads
  useEffect(() => {
    if (detailQ.data) {
      const vals: Record<string, unknown> = {};
      for (const f of detailQ.data.fields) {
        vals[f.path] = f.value;
      }
      setEditValues(vals);
      setSaveStatus("idle");
    }
  }, [detailQ.data]);

  useEffect(() => {
    setFieldSearch("");
  }, [selectedConfig]);

  // Auto-select first config when category changes
  useEffect(() => {
    if (listQ.data) {
      const catConfigs = listQ.data.configs.filter((c) => c.category === selectedCategory);
      if (catConfigs.length > 0 && (!selectedConfig || !catConfigs.some((c) => c.name === selectedConfig))) {
        setSelectedConfig(catConfigs[0].name);
      }
    }
  }, [selectedCategory, listQ.data, selectedConfig]);

  // --- Mutations ---
  const saveMutation = useMutation({
    mutationFn: () => updateConfig(selectedConfig!, editValues),
    onSuccess: () => {
      setSaveStatus("saved");
      queryClient.invalidateQueries({ queryKey: configKeys.detail(selectedConfig!) });
      setTimeout(() => setSaveStatus("idle"), 2000);
    },
    onError: () => setSaveStatus("error"),
  });

  // --- Handlers ---
  const handleFieldChange = useCallback((path: string, value: unknown) => {
    setEditValues((prev) => ({ ...prev, [path]: value }));
    setSaveStatus("idle");
  }, []);

  const handleSave = useCallback(() => {
    setSaveStatus("saving");
    saveMutation.mutate();
  }, [saveMutation]);

  const handleReset = useCallback(() => {
    if (detailQ.data) {
      const vals: Record<string, unknown> = {};
      for (const f of detailQ.data.fields) vals[f.path] = f.value;
      setEditValues(vals);
      setSaveStatus("idle");
    }
  }, [detailQ.data]);

  // --- Derived ---
  const categories = useMemo(() => listQ.data?.categories ?? [], [listQ.data]);
  const configs = useMemo(() => listQ.data?.configs ?? [], [listQ.data]);
  const detail = detailQ.data;

  const filteredConfigs = useMemo(() => {
    let list = configs.filter((c) => c.category === selectedCategory);
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = configs.filter(
        (c) => c.label.toLowerCase().includes(q) || c.description.toLowerCase().includes(q) || c.name.includes(q),
      );
    }
    return list;
  }, [configs, selectedCategory, searchQuery]);

  const changedCount = useMemo(() => {
    if (!detail) return 0;
    return detail.fields.filter((f) => JSON.stringify(editValues[f.path]) !== JSON.stringify(f.value)).length;
  }, [detail, editValues]);

  const fieldGroups = useMemo(() => {
    if (!detail) return new Map<string, ConfigField[]>();
    const query = fieldSearch.trim().toLowerCase();
    const fields = query
      ? detail.fields.filter((field) => {
          const group = (field as unknown as Record<string, unknown>).group;
          return [field.label, field.description, field.path, group]
            .some((value) => String(value ?? "").toLowerCase().includes(query));
        })
      : detail.fields;
    return groupFields(fields);
  }, [detail, fieldSearch]);

  // --- Render ---
  if (listQ.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-10 w-64" />
        <div className="grid grid-cols-3 gap-4">
          <Skeleton className="h-[500px]" />
          <Skeleton className="h-[500px] col-span-2" />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Settings2 className="h-5 w-5 text-muted-foreground" />
          <h1 className="text-xl font-semibold">System Configuration</h1>
        </div>
        <div className="relative">
          <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <input
            type="text"
            placeholder="Search configs..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="h-8 rounded-md border border-input bg-background pl-8 pr-8 text-sm focus:outline-none focus:ring-1 focus:ring-ring w-56"
          />
          {searchQuery && (
            <button
              onClick={() => setSearchQuery("")}
              className="absolute right-2 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
              aria-label="Clear search"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-[200px_1fr] lg:grid-cols-[200px_240px_1fr] gap-4 min-h-[calc(100vh-12rem)]">
        {/* Column 1: Categories */}
        <Card className="h-fit">
          <CardHeader className="p-3 pb-2">
            <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Categories</CardTitle>
          </CardHeader>
          <CardContent className="p-1.5">
            {categories.map((cat) => {
              const count = configs.filter((c) => c.category === cat.key).length;
              const CatIcon = CATEGORY_ICONS[cat.key] ?? Settings2;
              return (
                <button
                  key={cat.key}
                  onClick={() => { setSelectedCategory(cat.key); setSearchQuery(""); }}
                  className={cn(
                    "flex w-full items-center gap-2.5 rounded-md px-3 py-2 text-sm transition-colors",
                    selectedCategory === cat.key && !searchQuery
                      ? "bg-primary/10 font-medium text-primary"
                      : "text-foreground hover:bg-muted",
                  )}
                >
                  <CatIcon className={cn("h-4 w-4 shrink-0", CATEGORY_COLORS[cat.key])} />
                  <span className="flex-1 truncate text-left">{cat.label}</span>
                  <span className="text-[10px] text-muted-foreground tabular-nums">{count}</span>
                </button>
              );
            })}
          </CardContent>
        </Card>

        {/* Column 2: Config list */}
        <Card className="h-fit max-h-[calc(100vh-12rem)] overflow-y-auto">
          <CardHeader className="p-3 pb-2">
            <CardTitle className="text-xs font-medium uppercase tracking-wider text-muted-foreground">
              {searchQuery ? "Search Results" : categories.find((c) => c.key === selectedCategory)?.label ?? "Configs"}
            </CardTitle>
          </CardHeader>
          <CardContent className="p-1">
            {filteredConfigs.length === 0 && (
              <p className="px-3 py-4 text-xs text-muted-foreground text-center">No configs found</p>
            )}
            {filteredConfigs.map((cfg) => (
              <button
                key={cfg.name}
                onClick={() => setSelectedConfig(cfg.name)}
                className={cn(
                  "flex w-full items-center gap-2 rounded-md px-3 py-2 text-left transition-colors",
                  selectedConfig === cfg.name
                    ? "bg-primary/10 font-medium text-primary"
                    : "text-foreground hover:bg-muted",
                )}
              >
                <div className="min-w-0 flex-1">
                  <div className="text-sm truncate">
                    <HighlightText text={cfg.label} query={searchQuery} />
                  </div>
                  <div className="text-[10px] text-muted-foreground truncate">
                    <HighlightText text={cfg.description.slice(0, 60) + "..."} query={searchQuery} />
                  </div>
                </div>
                <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
              </button>
            ))}
          </CardContent>
        </Card>

        {/* Column 3: Field editor */}
        <Card className="max-h-[calc(100vh-12rem)] overflow-y-auto">
          {!selectedConfig ? (
            <CardContent className="flex h-64 items-center justify-center">
              <div className="text-center text-muted-foreground">
                <Settings2 className="mx-auto h-10 w-10 mb-2 opacity-30" />
                <p className="text-sm">Select a configuration to edit</p>
              </div>
            </CardContent>
          ) : detailQ.isLoading ? (
            <CardContent className="p-4 space-y-3">
              <Skeleton className="h-6 w-48" />
              <Skeleton className="h-4 w-full" />
              <Skeleton className="h-32" />
            </CardContent>
          ) : detail ? (
            <>
              <CardHeader className="sticky top-0 z-10 bg-card border-b border-border/50 p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="text-base">{detail.label}</CardTitle>
                    <p className="mt-1 text-xs text-muted-foreground leading-relaxed">{detail.description}</p>
                  </div>
                  <div className="flex items-center gap-2 shrink-0">
                    {changedCount > 0 && (
                      <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[10px] font-semibold text-amber-700 dark:bg-amber-900/50 dark:text-amber-300">
                        {changedCount} unsaved
                      </span>
                    )}
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleReset}
                      disabled={changedCount === 0}
                      className="h-7 text-xs gap-1"
                    >
                      <RotateCcw className="h-3 w-3" /> Discard
                    </Button>
                    <Button
                      size="sm"
                      onClick={handleSave}
                      disabled={changedCount === 0 || saveStatus === "saving"}
                      className="h-7 text-xs gap-1"
                    >
                      {saveStatus === "saving" ? (
                        <>Saving...</>
                      ) : saveStatus === "saved" ? (
                        <><Check className="h-3 w-3" /> Saved</>
                      ) : saveStatus === "error" ? (
                        <><AlertTriangle className="h-3 w-3" /> Error</>
                      ) : (
                        <><Save className="h-3 w-3" /> Save</>
                      )}
                    </Button>
                  </div>
                </div>
                <label className="relative mt-3 block" htmlFor="config-field-search">
                  <span className="sr-only">Search parameters</span>
                  <Search className="absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
                  <input
                    id="config-field-search"
                    type="search"
                    value={fieldSearch}
                    onChange={(event) => setFieldSearch(event.target.value)}
                    placeholder="Search parameters..."
                    className="h-8 w-full rounded-md border border-input bg-background pl-8 pr-3 text-sm focus:outline-none focus:ring-1 focus:ring-ring"
                  />
                </label>
              </CardHeader>
              <CardContent className="p-2">
                {fieldGroups.size === 0 && (
                  <div className="px-3 py-10 text-center text-sm text-muted-foreground">
                    No parameters match “{fieldSearch}”.
                  </div>
                )}
                {Array.from(fieldGroups.entries()).map(([group, fields]) => {
                  const isModelToggleGroup = fields.every((f) => f.type === "model_toggle");
                  return (
                    <div key={group} className="mb-5">
                      {fieldGroups.size > 1 && (
                        <div className="sticky top-[124px] z-[5] bg-card px-3 py-2.5 border-b border-border/30">
                          <div className="flex items-center gap-2">
                            <GroupIcon groupKey={group} />
                            <span className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                              {formatGroupLabel(group)}
                            </span>
                            {isModelToggleGroup && (
                              <span className="ml-auto inline-flex items-center gap-1 rounded-full bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
                                {fields.filter((f) => !!editValues[f.path]).length}/{fields.length} active
                              </span>
                            )}
                          </div>
                        </div>
                      )}
                      {isModelToggleGroup ? (
                        <div className="px-3 py-3">
                          <ModelToggleGrid
                            fields={fields}
                            editValues={editValues}
                            onChange={handleFieldChange}
                          />
                        </div>
                      ) : (
                        <div className="divide-y divide-border/30">
                          {fields.map((field) => (
                            <FieldRow
                              key={field.path}
                              field={field}
                              value={editValues[field.path]}
                              originalValue={field.value}
                              onChange={handleFieldChange}
                            />
                          ))}
                        </div>
                      )}
                    </div>
                  );
                })}
              </CardContent>
            </>
          ) : null}
        </Card>
      </div>
    </div>
  );
}

export default SettingsTab;
