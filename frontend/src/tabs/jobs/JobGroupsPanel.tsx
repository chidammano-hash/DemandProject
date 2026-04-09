/**
 * JobGroupsPanel — compact job type list grouped by category.
 * Each row shows the job type, description, and action buttons.
 * Collapsible groups for a cleaner, more scannable layout.
 */
import { useMemo, useState } from "react";
import { Loader2, Zap, Calendar, ChevronDown, ChevronRight } from "lucide-react";
import { cn } from "@/lib/utils";
import type { JobType } from "@/types/jobs";
import { GROUP_CONFIG } from "@/types/jobs";
import { GROUP_ICONS } from "./jobsShared";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------
export interface JobGroupsPanelProps {
  jobTypes: JobType[];
  onSubmit: (typeId: string, params: Record<string, unknown>, label: string) => void;
  onSchedule: (typeId: string) => void;
  submitting: boolean;
  /** Render a custom inline panel for a specific job type (replaces the default Run button row) */
  customCards?: Record<string, React.ReactNode>;
  /** Hide entire job groups by name (e.g., ["clustering"] — managed in dedicated tab) */
  hiddenGroups?: string[];
}

// ---------------------------------------------------------------------------
// JobGroupsPanel
// ---------------------------------------------------------------------------
export function JobGroupsPanel({ jobTypes, onSubmit, onSchedule, submitting, customCards, hiddenGroups }: JobGroupsPanelProps) {
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(() => new Set());

  const hiddenSet = useMemo(() => new Set(hiddenGroups ?? []), [hiddenGroups]);

  const groups = useMemo(() => {
    const seen = new Set<string>();
    return jobTypes.reduce<string[]>((acc, t) => {
      if (!seen.has(t.group) && !hiddenSet.has(t.group)) {
        seen.add(t.group);
        acc.push(t.group);
      }
      return acc;
    }, []);
  }, [jobTypes, hiddenSet]);

  const toggleGroup = (group: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(group)) next.delete(group);
      else next.add(group);
      return next;
    });
  };

  return (
    <div className="divide-y rounded-lg border overflow-hidden">
      {groups.map((group) => {
        const cfg = GROUP_CONFIG[group] || GROUP_CONFIG.clustering;
        const GIcon = GROUP_ICONS[group] || Zap;
        const types = jobTypes.filter((t) => t.group === group);
        const isExpanded = expandedGroups.has(group);

        return (
          <div key={group}>
            {/* Group header — clickable to expand/collapse */}
            <button
              className="w-full flex items-center gap-2.5 px-4 py-2.5 text-left hover:bg-muted/50 transition-colors"
              onClick={() => toggleGroup(group)}
            >
              <div className={cn("rounded-md p-1", cfg.iconBg)}>
                <GIcon className={cn("h-3.5 w-3.5", cfg.color)} />
              </div>
              <span className={cn("text-xs font-semibold uppercase tracking-wider flex-1", cfg.color)}>
                {cfg.label}
              </span>
              <span className="text-[10px] text-muted-foreground mr-1">{types.length} job{types.length !== 1 ? "s" : ""}</span>
              {isExpanded ? (
                <ChevronDown className="h-3.5 w-3.5 text-muted-foreground" />
              ) : (
                <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
              )}
            </button>

            {/* Job type rows */}
            {isExpanded && (
              <div className="divide-y border-t bg-muted/20">
                {types.map((t) => {
                  const custom = customCards?.[t.type_id];
                  if (custom) {
                    return (
                      <div key={t.type_id} className="px-4 py-3 space-y-3">
                        <div>
                          <p className="text-sm font-medium text-foreground">{t.label}</p>
                          <p className="text-xs text-muted-foreground">{t.description}</p>
                        </div>
                        {custom}
                      </div>
                    );
                  }
                  return (
                    <div
                      key={t.type_id}
                      className="flex items-center gap-3 px-4 py-2.5 hover:bg-muted/30 transition-colors"
                    >
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-medium text-foreground truncate">{t.label}</p>
                        <p className="text-xs text-muted-foreground truncate">{t.description}</p>
                      </div>
                      <div className="flex items-center gap-1.5 flex-shrink-0">
                        <button
                          disabled={submitting}
                          onClick={() => onSubmit(t.type_id, t.params_schema || {}, t.label)}
                          className={cn(
                            "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors",
                            submitting
                              ? "bg-muted text-muted-foreground cursor-not-allowed"
                              : "bg-primary text-primary-foreground hover:bg-primary/90",
                          )}
                        >
                          {submitting ? (
                            <Loader2 className="h-3 w-3 animate-spin" />
                          ) : (
                            <Zap className="h-3 w-3" />
                          )}
                          Run
                        </button>
                        <button
                          onClick={() => onSchedule(t.type_id)}
                          className="rounded-md border border-border p-1.5 text-muted-foreground hover:bg-muted/50 hover:text-foreground transition-colors"
                          title="Schedule recurring"
                        >
                          <Calendar className="h-3 w-3" />
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
