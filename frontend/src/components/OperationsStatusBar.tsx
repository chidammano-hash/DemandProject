import { useMemo, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  Activity,
  AlertTriangle,
  CalendarDays,
  CheckCircle2,
  Loader2,
  LogOut,
  PlayCircle,
  UserRound,
} from "lucide-react";

import {
  fetchPipelineReadiness,
  fetchPlanningDate,
  pipelineReadinessKeys,
} from "@/api/queries/dashboard";
import { queryKeys } from "@/api/queries/core";
import { fetchActiveJobs } from "@/api/queries/jobs";
import { useJobNotification } from "@/context/JobNotificationContext";
import { useAuth } from "@/context/AuthContext";
import { cn } from "@/lib/utils";
import { NAV_ITEMS } from "./AppSidebar";

interface OperationsStatusBarProps {
  activeTab: string;
  onNavigate: (tab: string) => void;
}

function formatPlanningDate(value: string | undefined): string {
  if (!value) return "Planning date";
  const date = new Date(`${value}T00:00:00`);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function StatusPill({
  children,
  className,
  onClick,
  title,
}: {
  children: ReactNode;
  className?: string;
  onClick?: () => void;
  title?: string;
}) {
  const content = (
    <span
      className={cn(
        "inline-flex h-7 items-center gap-1.5 rounded-md border border-border/70 bg-background px-2.5 text-xs text-foreground/80",
        className
      )}
    >
      {children}
    </span>
  );

  if (!onClick) return content;

  return (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className="rounded-md transition-colors hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
    >
      {content}
    </button>
  );
}

export function OperationsStatusBar({
  activeTab,
  onNavigate,
}: OperationsStatusBarProps): JSX.Element {
  const { activeJobCount: localActiveJobCount } = useJobNotification();
  const { user, logout } = useAuth();

  const planningDateQuery = useQuery({
    queryKey: queryKeys.planningDate(),
    queryFn: fetchPlanningDate,
    staleTime: 60_000,
    refetchInterval: 60_000,
  });

  const readinessQuery = useQuery({
    queryKey: pipelineReadinessKeys.readiness,
    queryFn: fetchPipelineReadiness,
    staleTime: 60_000,
    refetchInterval: 60_000,
  });

  const activeJobsQuery = useQuery({
    queryKey: queryKeys.activeJobs(),
    queryFn: fetchActiveJobs,
    staleTime: 15_000,
    refetchInterval: 15_000,
  });

  const activeLabel = useMemo(
    () => NAV_ITEMS.find((item) => item.key === activeTab)?.label ?? activeTab,
    [activeTab]
  );

  const activeJobCount = Math.max(localActiveJobCount, activeJobsQuery.data?.jobs.length ?? 0);
  const staleChecks = readinessQuery.data?.checks.filter((check) => check.status === "stale") ?? [];
  const firstReadinessTarget =
    staleChecks.find((check) => check.action?.kind === "navigate")?.action?.target ?? "dataQuality";
  const isSyncing =
    planningDateQuery.isFetching || readinessQuery.isFetching || activeJobsQuery.isFetching;

  return (
    <div className="border-b border-border/50 bg-background/95 px-4 py-2 md:px-6">
      <div className="mx-auto flex max-w-[1600px] min-w-0 flex-wrap items-center gap-x-2 gap-y-1.5">
        <span className="mr-1 hidden text-[10px] font-medium uppercase tracking-wider text-muted-foreground md:inline">
          Operations
        </span>

        <StatusPill>
          <Activity className="h-3.5 w-3.5 text-primary" strokeWidth={1.7} />
          <span className="font-medium text-foreground">{activeLabel}</span>
        </StatusPill>

        <StatusPill>
          <CalendarDays className="h-3.5 w-3.5 text-muted-foreground" strokeWidth={1.7} />
          <span>{formatPlanningDate(planningDateQuery.data?.planning_date)}</span>
          {planningDateQuery.data?.is_frozen && (
            <span className="rounded bg-amber-500/10 px-1.5 py-0.5 text-[10px] font-medium text-amber-700 dark:text-amber-300">
              Frozen
            </span>
          )}
        </StatusPill>

        <StatusPill
          onClick={() => onNavigate("integration")}
          title="Open workflow monitoring"
          className={
            activeJobCount > 0
              ? "border-sky-500/30 bg-sky-500/10 text-sky-700 dark:text-sky-300"
              : undefined
          }
        >
          <PlayCircle className="h-3.5 w-3.5" strokeWidth={1.7} />
          <span>
            {activeJobCount === 0
              ? "No active jobs"
              : `${activeJobCount} active job${activeJobCount === 1 ? "" : "s"}`}
          </span>
        </StatusPill>

        <StatusPill
          onClick={staleChecks.length > 0 ? () => onNavigate(firstReadinessTarget) : undefined}
          title={staleChecks.length > 0 ? "Open remediation workflow" : undefined}
          className={
            staleChecks.length > 0
              ? "border-amber-500/30 bg-amber-500/10 text-amber-800 dark:text-amber-300"
              : "border-emerald-500/30 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
          }
        >
          {staleChecks.length > 0 ? (
            <AlertTriangle className="h-3.5 w-3.5" strokeWidth={1.7} />
          ) : (
            <CheckCircle2 className="h-3.5 w-3.5" strokeWidth={1.7} />
          )}
          <span>
            {staleChecks.length > 0
              ? `${staleChecks.length} stale stage${staleChecks.length === 1 ? "" : "s"}`
              : "ML inputs current"}
          </span>
        </StatusPill>

        {isSyncing && (
          <span className="inline-flex h-7 items-center gap-1.5 text-xs text-muted-foreground sm:ml-auto">
            <Loader2 className="h-3.5 w-3.5 animate-spin" strokeWidth={1.7} />
            Syncing
          </span>
        )}

        {user && user.user_id !== "anonymous" && (
          <div className={cn("ml-auto flex items-center gap-1.5", isSyncing && "sm:ml-0")}>
            <StatusPill title={`Signed in as ${user.email}`}>
              <UserRound className="h-3.5 w-3.5 text-muted-foreground" strokeWidth={1.7} />
              <span className="hidden max-w-40 truncate lg:inline">{user.display_name || user.email}</span>
              <span className="capitalize text-muted-foreground">{user.role}</span>
            </StatusPill>
            <button
              type="button"
              onClick={logout}
              title="Sign out"
              aria-label="Sign out"
              className="inline-flex h-7 w-7 items-center justify-center rounded-md text-muted-foreground transition-colors hover:bg-muted hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <LogOut className="h-3.5 w-3.5" strokeWidth={1.7} />
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
