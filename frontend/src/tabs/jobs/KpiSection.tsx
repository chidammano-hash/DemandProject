/**
 * KpiSection — 4 KPI summary cards for the Jobs dashboard.
 * Displays Total Jobs, Active Now, Success Rate, and Avg Duration.
 */
import { BarChart3, Zap, CheckCircle2, Timer } from "lucide-react";
import { cn } from "@/lib/utils";
import { formatDuration } from "./jobsShared";

// ---------------------------------------------------------------------------
// KpiCard — reusable card within this section
// ---------------------------------------------------------------------------
function KpiCard({
  label,
  value,
  icon: Icon,
  color,
  subtitle,
}: {
  label: string;
  value: string | number;
  icon: typeof BarChart3;
  color: string;
  subtitle?: string;
}) {
  return (
    <div className="rounded-xl border border-border bg-card p-4 transition-shadow hover:shadow-sm">
      <div className="flex items-center justify-between">
        <div className={cn("rounded-lg p-2", color)}>
          <Icon className="h-4 w-4" />
        </div>
      </div>
      <p className="mt-3 text-2xl font-bold tabular-nums text-foreground">{value}</p>
      <p className="text-xs font-medium text-muted-foreground">{label}</p>
      {subtitle && <p className="mt-0.5 text-[10px] text-muted-foreground/70">{subtitle}</p>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// JobStats shape (mirrors API response subset)
// ---------------------------------------------------------------------------
export interface JobStats {
  total: number;
  active: number;
  completed: number;
  failed: number;
  avg_duration_seconds: number;
  last_24h: { submitted: number };
}

// ---------------------------------------------------------------------------
// KpiSection
// ---------------------------------------------------------------------------
export function KpiSection({ stats }: { stats: JobStats }) {
  const successRate =
    stats.completed + stats.failed > 0
      ? Math.round((stats.completed / (stats.completed + stats.failed)) * 100)
      : 100;

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
      <KpiCard
        label="Total Jobs"
        value={stats.total}
        icon={BarChart3}
        color="bg-muted text-muted-foreground"
        subtitle={stats.last_24h.submitted > 0 ? `${stats.last_24h.submitted} today` : undefined}
      />
      <KpiCard
        label="Active Now"
        value={stats.active}
        icon={Zap}
        color="bg-blue-100 dark:bg-blue-900/50 text-blue-600 dark:text-blue-400"
      />
      <KpiCard
        label="Success Rate"
        value={`${successRate}%`}
        icon={CheckCircle2}
        color="bg-emerald-100 dark:bg-emerald-900/50 text-emerald-600 dark:text-emerald-400"
        subtitle={stats.failed > 0 ? `${stats.failed} failed` : undefined}
      />
      <KpiCard
        label="Avg Duration"
        value={stats.avg_duration_seconds > 0 ? formatDuration(stats.avg_duration_seconds) : "-"}
        icon={Timer}
        color="bg-amber-100 dark:bg-amber-900/50 text-amber-600 dark:text-amber-400"
      />
    </div>
  );
}
