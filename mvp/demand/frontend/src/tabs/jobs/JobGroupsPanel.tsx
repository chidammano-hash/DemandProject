/**
 * JobGroupsPanel — grouped job type cards (clustering, backtest, seasonality, champion).
 * Each card can be expanded to reveal Run Now and Schedule buttons.
 */
import { useMemo, useState } from "react";
import { PlayCircle, Loader2, Zap, Calendar } from "lucide-react";
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
}

// ---------------------------------------------------------------------------
// JobGroupsPanel
// ---------------------------------------------------------------------------
export function JobGroupsPanel({ jobTypes, onSubmit, onSchedule, submitting }: JobGroupsPanelProps) {
  const [selectedType, setSelectedType] = useState<string | null>(null);

  const groups = useMemo(() => {
    const seen = new Set<string>();
    return jobTypes.reduce<string[]>((acc, t) => {
      if (!seen.has(t.group)) {
        seen.add(t.group);
        acc.push(t.group);
      }
      return acc;
    }, []);
  }, [jobTypes]);

  return (
    <div className="space-y-5">
      {groups.map((group) => {
        const cfg = GROUP_CONFIG[group] || GROUP_CONFIG.clustering;
        const GIcon = GROUP_ICONS[group] || Zap;
        const types = jobTypes.filter((t) => t.group === group);

        return (
          <div key={group}>
            <div className="flex items-center gap-2 mb-3">
              <div className={cn("rounded-md p-1.5", cfg.iconBg)}>
                <GIcon className={cn("h-3.5 w-3.5", cfg.color)} />
              </div>
              <h4 className={cn("text-xs font-semibold uppercase tracking-wider", cfg.color)}>
                {cfg.label}
              </h4>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              {types.map((t) => {
                const isSelected = selectedType === t.type_id;
                return (
                  <button
                    key={t.type_id}
                    className={cn(
                      "group relative rounded-xl border p-4 text-left transition-all duration-200",
                      isSelected
                        ? cn("ring-2 ring-offset-1", cfg.borderColor, cfg.bgColor, "ring-current")
                        : "border-border bg-card hover:border-primary/30 hover:shadow-sm",
                    )}
                    onClick={() => setSelectedType(isSelected ? null : t.type_id)}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <p className="text-sm font-semibold text-foreground truncate">{t.label}</p>
                        <p className="text-xs text-muted-foreground mt-1 line-clamp-2">
                          {t.description}
                        </p>
                      </div>
                      <div
                        className={cn(
                          "rounded-lg p-1.5 transition-colors",
                          isSelected ? cfg.iconBg : "bg-muted/50",
                        )}
                      >
                        <PlayCircle
                          className={cn(
                            "h-4 w-4",
                            isSelected ? cfg.color : "text-muted-foreground/50",
                          )}
                        />
                      </div>
                    </div>

                    {isSelected && (
                      <div className="mt-3 pt-3 border-t border-border/50 flex gap-2">
                        <button
                          disabled={submitting}
                          onClick={(e) => {
                            e.stopPropagation();
                            onSubmit(t.type_id, t.params_schema || {}, t.label);
                          }}
                          className={cn(
                            "flex-1 rounded-lg px-3 py-2 text-xs font-semibold transition-all duration-200",
                            submitting
                              ? "bg-muted text-muted-foreground cursor-not-allowed"
                              : "bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm",
                          )}
                        >
                          {submitting ? (
                            <span className="inline-flex items-center gap-1.5">
                              <Loader2 className="h-3 w-3 animate-spin" /> Scheduling...
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1.5">
                              <Zap className="h-3 w-3" /> Run Now
                            </span>
                          )}
                        </button>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            onSchedule(t.type_id);
                          }}
                          className="rounded-lg border border-border px-3 py-2 text-xs font-medium text-muted-foreground hover:bg-muted/50 hover:text-foreground transition-colors"
                          title="Schedule recurring"
                        >
                          <Calendar className="h-3 w-3" />
                        </button>
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
}
