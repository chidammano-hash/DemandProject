/**
 * SchedulesPanel — recurring schedules list with cron badges,
 * plus the ScheduleDialog modal for creating new schedules.
 */
import { useState } from "react";
import { Repeat, Trash2, X } from "lucide-react";
import { cn } from "@/lib/utils";
import type { JobType, JobSchedule } from "@/types/jobs";
import { GROUP_CONFIG } from "@/types/jobs";
import { formatTimestamp, getGroupKey } from "./jobsShared";

// ---------------------------------------------------------------------------
// ScheduleDialog — cron/interval builder modal
// ---------------------------------------------------------------------------
interface ScheduleDialogProps {
  jobType: string;
  jobTypes: JobType[];
  onClose: () => void;
  onSubmit: (
    typeId: string,
    cron?: string,
    intervalMin?: number,
    label?: string,
  ) => void;
}

function ScheduleDialog({ jobType, jobTypes, onClose, onSubmit }: ScheduleDialogProps) {
  const [mode, setMode] = useState<"cron" | "interval">("interval");
  const [cron, setCron] = useState("0 2 * * *");
  const [intervalMin, setIntervalMin] = useState(60);
  const typeDef = jobTypes.find((t) => t.type_id === jobType);

  const presets = [
    { label: "Every hour", cron: "0 * * * *", intervalMin: 60 },
    { label: "Every 6 hours", cron: "0 */6 * * *", intervalMin: 360 },
    { label: "Daily at 2 AM", cron: "0 2 * * *", intervalMin: 1440 },
    { label: "Weekly (Mon 2 AM)", cron: "0 2 * * 1", intervalMin: 10080 },
  ];

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md rounded-2xl border border-border bg-card p-6 shadow-xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-base font-semibold text-foreground">Schedule Recurring Job</h3>
            <p className="text-xs text-muted-foreground mt-0.5">{typeDef?.label || jobType}</p>
          </div>
          <button onClick={onClose} className="rounded-md p-1 hover:bg-muted">
            <X className="h-4 w-4 text-muted-foreground" />
          </button>
        </div>

        {/* Presets */}
        <div className="flex flex-wrap gap-2 mb-4">
          {presets.map((p) => (
            <button
              key={p.label}
              onClick={() => {
                setCron(p.cron);
                setIntervalMin(p.intervalMin);
              }}
              className="rounded-full border border-border px-3 py-1 text-xs hover:bg-muted transition-colors"
            >
              {p.label}
            </button>
          ))}
        </div>

        {/* Mode toggle */}
        <div className="flex rounded-lg bg-muted p-0.5 mb-4">
          <button
            className={cn(
              "flex-1 rounded-md py-1.5 text-xs font-medium transition-all",
              mode === "interval" ? "bg-card shadow-sm text-foreground" : "text-muted-foreground",
            )}
            onClick={() => setMode("interval")}
          >
            Interval
          </button>
          <button
            className={cn(
              "flex-1 rounded-md py-1.5 text-xs font-medium transition-all",
              mode === "cron" ? "bg-card shadow-sm text-foreground" : "text-muted-foreground",
            )}
            onClick={() => setMode("cron")}
          >
            Cron Expression
          </button>
        </div>

        {mode === "interval" ? (
          <div className="space-y-2">
            <label className="text-xs font-medium text-foreground">Run every</label>
            <div className="flex items-center gap-2">
              <input
                type="number"
                min={1}
                value={intervalMin}
                onChange={(e) => setIntervalMin(Number(e.target.value))}
                className="h-9 w-24 rounded-md border border-input bg-background px-3 text-sm"
              />
              <span className="text-sm text-muted-foreground">minutes</span>
            </div>
          </div>
        ) : (
          <div className="space-y-2">
            <label className="text-xs font-medium text-foreground">Cron expression</label>
            <input
              type="text"
              value={cron}
              onChange={(e) => setCron(e.target.value)}
              placeholder="0 2 * * *"
              className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm font-mono"
            />
            <p className="text-[10px] text-muted-foreground">
              Format: minute hour day-of-month month day-of-week
            </p>
          </div>
        )}

        <div className="flex justify-end gap-2 mt-6">
          <button
            onClick={onClose}
            className="rounded-lg border border-border px-4 py-2 text-xs font-medium hover:bg-muted"
          >
            Cancel
          </button>
          <button
            onClick={() => {
              if (mode === "cron") {
                onSubmit(jobType, cron, undefined, typeDef?.label);
              } else {
                onSubmit(jobType, undefined, intervalMin, typeDef?.label);
              }
              onClose();
            }}
            className="rounded-lg bg-primary px-4 py-2 text-xs font-semibold text-primary-foreground hover:bg-primary/90 shadow-sm"
          >
            <span className="inline-flex items-center gap-1.5">
              <Repeat className="h-3 w-3" />
              Create Schedule
            </span>
          </button>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SchedulesPanel
// ---------------------------------------------------------------------------
export interface SchedulesPanelProps {
  schedules: JobSchedule[];
  jobTypes: JobType[];
  scheduleDialogType: string | null;
  onOpenDialog: (typeId: string) => void;
  onCloseDialog: () => void;
  onCreateSchedule: (
    typeId: string,
    cron?: string,
    intervalMin?: number,
    label?: string,
  ) => void;
  onDeleteSchedule: (scheduleId: string) => void;
}

export function SchedulesPanel({
  schedules,
  jobTypes,
  scheduleDialogType,
  onCloseDialog,
  onCreateSchedule,
  onDeleteSchedule,
}: SchedulesPanelProps) {
  return (
    <>
      {/* Recurring schedules list */}
      {schedules.length > 0 && (
        <section>
          <h3 className="text-sm font-semibold text-foreground/80 uppercase tracking-wider mb-3 flex items-center gap-2">
            <Repeat className="h-3.5 w-3.5" />
            Active Schedules ({schedules.length})
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {schedules.map((s) => {
              const groupKey = getGroupKey(s.job_type);
              const cfg = GROUP_CONFIG[groupKey] || GROUP_CONFIG.clustering;
              return (
                <div
                  key={s.schedule_id}
                  className={cn("rounded-xl border p-3", cfg.borderColor, cfg.bgColor)}
                >
                  <div className="flex items-center justify-between">
                    <p className="text-xs font-semibold text-foreground">{s.job_label}</p>
                    <button
                      onClick={() => onDeleteSchedule(s.schedule_id)}
                      className="text-muted-foreground hover:text-destructive"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                  <div className="mt-2 flex items-center gap-3 text-[10px] text-muted-foreground">
                    {s.cron_expr && (
                      <span className="font-mono bg-muted rounded px-1.5 py-0.5">
                        {s.cron_expr}
                      </span>
                    )}
                    {s.interval_min && <span>Every {s.interval_min}min</span>}
                    <span>Runs: {s.run_count}</span>
                  </div>
                  {s.next_run_at && (
                    <p className="mt-1 text-[10px] text-muted-foreground">
                      Next: {formatTimestamp(s.next_run_at)}
                    </p>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      )}

      {/* Schedule creation dialog */}
      {scheduleDialogType && (
        <ScheduleDialog
          jobType={scheduleDialogType}
          jobTypes={jobTypes}
          onClose={onCloseDialog}
          onSubmit={onCreateSchedule}
        />
      )}
    </>
  );
}
