/**
 * ParameterTable -- Collapsible, grouped hyperparameter table with
 * Value / Default / Delta columns.
 * Extracted from ExperimentBuilder for readability.
 */
import { useMemo, useCallback, useState } from "react";
import { ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select";
import type { ParamSpec, ValidationError } from "@/lib/model-params";
import { GROUP_LABELS, formatDelta } from "@/lib/model-params";

export interface ParameterTableProps {
  paramSpecs: ParamSpec[];
  params: Record<string, unknown>;
  defaults: Record<string, unknown>;
  errors: ValidationError[];
  /** Whether the whole table is expanded (controlled externally for template interaction) */
  expanded: boolean;
  onToggleExpand: () => void;
  /** Whether the parent is in "custom" template mode (always show table) */
  alwaysExpanded: boolean;
  changedCount: number;
  onUpdateParam: (key: string, value: unknown) => void;
}

export function ParameterTable({
  paramSpecs,
  params,
  defaults,
  errors,
  expanded,
  onToggleExpand,
  alwaysExpanded,
  changedCount,
  onUpdateParam,
}: ParameterTableProps) {
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({});

  const toggleGroup = useCallback((group: string) => {
    setCollapsedGroups((prev) => ({ ...prev, [group]: !prev[group] }));
  }, []);

  const getError = useCallback(
    (field: string) => errors.find((e) => e.field === field)?.message,
    [errors],
  );

  // Group param specs (filtering by visibleWhen)
  const groupedParams = useMemo(() => {
    const groups: Record<string, ParamSpec[]> = {};
    for (const spec of paramSpecs) {
      if (spec.visibleWhen && !spec.visibleWhen(params)) continue;
      if (!groups[spec.group]) groups[spec.group] = [];
      groups[spec.group].push(spec);
    }
    return groups;
  }, [paramSpecs, params]);

  const showTable = alwaysExpanded || expanded;

  return (
    <div>
      {!alwaysExpanded && !expanded ? (
        <button
          onClick={onToggleExpand}
          className="w-full flex items-center justify-between rounded-md border border-border px-3 py-2 hover:bg-muted/30 transition-colors text-left"
        >
          <span className="text-xs font-medium text-foreground">
            View Parameters ({paramSpecs.length} configured, {changedCount} differ from production)
          </span>
          <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
        </button>
      ) : (
        <div className="flex items-center justify-between mb-2">
          <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
            Hyperparameters
          </p>
          {!alwaysExpanded && (
            <button
              onClick={onToggleExpand}
              className="text-[10px] text-muted-foreground hover:text-foreground underline"
            >
              Collapse
            </button>
          )}
        </div>
      )}

      {showTable &&
        Object.entries(groupedParams).map(([group, specs]) => {
          const isCollapsed = collapsedGroups[group] ?? false;
          return (
            <div key={group} className="border rounded-md mb-2 overflow-hidden">
              <button
                onClick={() => toggleGroup(group)}
                className="w-full flex items-center justify-between px-3 py-2 bg-muted/30 hover:bg-muted/50 transition-colors text-left"
              >
                <span className="text-xs font-medium text-foreground">
                  {GROUP_LABELS[group] ?? group}
                </span>
                {isCollapsed ? (
                  <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
                ) : (
                  <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
                )}
              </button>

              {!isCollapsed && (
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead className="bg-muted/20">
                      <tr>
                        <th className="text-left px-3 py-1.5 font-medium w-1/4">Parameter</th>
                        <th className="text-left px-3 py-1.5 font-medium w-1/4">Value</th>
                        <th className="text-right px-3 py-1.5 font-medium w-1/6">Default</th>
                        <th className="text-right px-3 py-1.5 font-medium w-1/6">Delta</th>
                      </tr>
                    </thead>
                    <tbody>
                      {specs.map((spec) => {
                        const val = params[spec.key];
                        const defVal = defaults[spec.key];
                        const changed = val !== defVal;
                        const disabled = spec.disabledWhen?.(params) ?? false;
                        const fieldError = getError(spec.key);

                        return (
                          <tr
                            key={spec.key}
                            className={cn(
                              "border-t border-border/40",
                              changed && "bg-amber-50/50 dark:bg-amber-900/10",
                              disabled && "opacity-50",
                            )}
                          >
                            <td className="px-3 py-1.5">
                              <span className="font-mono text-xs" title={spec.tooltip}>
                                {spec.label}
                              </span>
                              {disabled && spec.disabledTooltip && (
                                <p className="text-[9px] text-amber-600 dark:text-amber-400">
                                  {spec.disabledTooltip}
                                </p>
                              )}
                            </td>
                            <td className="px-3 py-1.5">
                              {spec.type === "bool" ? (
                                <input
                                  type="checkbox"
                                  checked={val === true}
                                  onChange={(e) => onUpdateParam(spec.key, e.target.checked)}
                                  disabled={disabled}
                                  className="rounded"
                                />
                              ) : spec.type === "select" ? (
                                <Select
                                  value={String(val ?? "")}
                                  onValueChange={(v) => onUpdateParam(spec.key, v)}
                                >
                                  <SelectTrigger className="h-7 text-xs w-full min-w-[100px]">
                                    <SelectValue placeholder="Select..." />
                                  </SelectTrigger>
                                  <SelectContent>
                                    {(spec.options ?? []).map((opt) => (
                                      <SelectItem key={opt} value={opt}>
                                        {opt}
                                      </SelectItem>
                                    ))}
                                  </SelectContent>
                                </Select>
                              ) : (
                                <div>
                                  <input
                                    type="number"
                                    value={val !== undefined ? String(val) : ""}
                                    onChange={(e) => {
                                      const raw = e.target.value;
                                      if (raw === "") {
                                        onUpdateParam(spec.key, "");
                                        return;
                                      }
                                      onUpdateParam(
                                        spec.key,
                                        spec.type === "int" ? parseInt(raw, 10) : parseFloat(raw),
                                      );
                                    }}
                                    step={spec.type === "float" ? "0.001" : "1"}
                                    disabled={disabled}
                                    className={cn(
                                      "w-full h-7 rounded-md border border-input bg-background px-2 text-xs tabular-nums",
                                      "focus:outline-none focus:ring-1 focus:ring-ring",
                                      changed && "font-semibold",
                                      fieldError && "border-red-500",
                                      disabled && "cursor-not-allowed opacity-50",
                                    )}
                                  />
                                  {fieldError && (
                                    <p className="text-[9px] text-red-600 mt-0.5">{fieldError}</p>
                                  )}
                                </div>
                              )}
                            </td>
                            <td className="text-right px-3 py-1.5 tabular-nums text-muted-foreground font-mono">
                              {defVal !== undefined ? String(defVal) : "--"}
                            </td>
                            <td
                              className={cn(
                                "text-right px-3 py-1.5 tabular-nums font-mono",
                                changed
                                  ? "text-amber-700 dark:text-amber-400 font-medium"
                                  : "text-muted-foreground/50",
                              )}
                            >
                              {formatDelta(val, defVal, spec.type)}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          );
        })}
    </div>
  );
}
