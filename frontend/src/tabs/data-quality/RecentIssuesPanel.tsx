/**
 * Recent Issues panel — history entries for failed/error/warn checks with
 * expandable AI-style root cause analysis.
 */
import { useMemo, useState } from "react";
import {
  Activity,
  AlertTriangle,
  ChevronDown,
  ChevronRight,
  Lightbulb,
  Wrench,
} from "lucide-react";

import type { DQHistoryEntry } from "@/api/queries/platform";
import {
  STATUS_ICON,
  SEVERITY_STYLE,
  relativeTime,
  analyzeIssue,
} from "./dqShared";

interface RecentIssuesPanelProps {
  historyEntries: DQHistoryEntry[];
  domainFilter: string | null;
}

export function RecentIssuesPanel({ historyEntries, domainFilter }: RecentIssuesPanelProps) {
  const [issueSeverityFilter, setIssueSeverityFilter] = useState<string | null>(null);
  const [expandedIssue, setExpandedIssue] = useState<string | null>(null);

  const recentIssues = useMemo(() => {
    let list = historyEntries.filter((e) => e.status === "fail" || e.status === "error" || e.status === "warn");
    if (domainFilter) list = list.filter((e) => e.domain === domainFilter);
    if (issueSeverityFilter) list = list.filter((e) => e.severity === issueSeverityFilter);
    return list.slice(0, 50);
  }, [historyEntries, domainFilter, issueSeverityFilter]);

  return (
    <div className="rounded-lg border border-border bg-card p-4">
      <div className="mb-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="h-4 w-4 text-red-500" />
          <h3 className="text-sm font-medium text-foreground">
            Recent Issues ({recentIssues.length})
          </h3>
          {domainFilter && (
            <span className="rounded bg-primary/10 px-2 py-0.5 text-[10px] font-medium text-primary">
              {domainFilter}
            </span>
          )}
        </div>
        <select
          value={issueSeverityFilter ?? ""}
          onChange={(ev) => setIssueSeverityFilter(ev.target.value || null)}
          className="rounded border border-border bg-background px-2 py-1 text-xs text-foreground"
          aria-label="Filter issues by severity"
        >
          <option value="">All severities</option>
          <option value="critical">Critical</option>
          <option value="warning">Warning</option>
          <option value="info">Info</option>
        </select>
      </div>
      {recentIssues.length === 0 ? (
        <div className="py-6 text-center text-sm text-muted-foreground">
          {historyEntries.length === 0
            ? "No check history available. Run checks to see results here."
            : "No recent issues found. All checks are passing."}
        </div>
      ) : (
        <div className="max-h-[600px] space-y-2 overflow-y-auto">
          {recentIssues.map((e, i) => {
            const st = STATUS_ICON[e.status] ?? STATUS_ICON.fail;
            const Icon = st.icon;
            const issueKey = `${e.check_id}-${e.run_ts}-${i}`;
            const isExpanded = expandedIssue === issueKey;
            const analysis = analyzeIssue(e);
            return (
              <div
                key={issueKey}
                className={`rounded-md border px-3 py-2.5 transition-all ${
                  e.status === "fail"
                    ? "border-red-200 bg-red-50/50 dark:border-red-900/40 dark:bg-red-950/20"
                    : e.status === "error"
                      ? "border-orange-200 bg-orange-50/50 dark:border-orange-900/40 dark:bg-orange-950/20"
                      : "border-amber-200 bg-amber-50/50 dark:border-amber-900/40 dark:bg-amber-950/20"
                }`}
              >
                {/* Header row */}
                <button
                  type="button"
                  onClick={() => setExpandedIssue(isExpanded ? null : issueKey)}
                  className="flex w-full items-start justify-between gap-2 text-left"
                >
                  <div className="flex items-center gap-2">
                    {isExpanded
                      ? <ChevronDown className="h-3.5 w-3.5 flex-shrink-0 text-muted-foreground" />
                      : <ChevronRight className="h-3.5 w-3.5 flex-shrink-0 text-muted-foreground" />
                    }
                    <Icon className={`h-4 w-4 flex-shrink-0 ${st.color}`} />
                    <div>
                      <span className="text-sm font-medium">{e.check_name}</span>
                      <span className="ml-2 text-xs text-muted-foreground capitalize">
                        {e.domain} / {e.table_name}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-2 flex-shrink-0">
                    <span className={`rounded px-1.5 py-0.5 text-[10px] font-bold uppercase ${SEVERITY_STYLE[e.severity] ?? SEVERITY_STYLE.low}`}>
                      {e.severity}
                    </span>
                    <span className="text-[10px] text-muted-foreground">
                      {e.run_ts ? relativeTime(e.run_ts) : ""}
                    </span>
                  </div>
                </button>

                {/* Summary (always visible) */}
                <div className="mt-1.5 ml-9 text-xs font-medium text-foreground/80">
                  <Lightbulb className="mr-1 inline h-3 w-3 text-amber-500" />
                  {analysis.summary}
                </div>

                {/* Expanded analysis */}
                {isExpanded && (
                  <div className="mt-3 ml-9 space-y-3 border-t border-border/40 pt-3">
                    {/* Root Cause */}
                    <div>
                      <div className="flex items-center gap-1 text-xs font-semibold text-foreground/70 uppercase tracking-wide">
                        <AlertTriangle className="h-3 w-3" />
                        Root Cause
                      </div>
                      <p className="mt-1 text-xs text-muted-foreground leading-relaxed">
                        {analysis.rootCause}
                      </p>
                    </div>

                    {/* Steps to Fix */}
                    <div>
                      <div className="flex items-center gap-1 text-xs font-semibold text-foreground/70 uppercase tracking-wide">
                        <Wrench className="h-3 w-3" />
                        Steps to Fix
                      </div>
                      <ol className="mt-1 list-decimal list-inside space-y-1">
                        {analysis.fixSteps.map((step, si) => (
                          <li key={si} className="text-xs text-muted-foreground leading-relaxed">
                            {step.includes("SELECT") || step.includes("make ")
                              ? <>{step.split(/(SELECT .+|make \S+)/g).map((part, pi) =>
                                  /^(SELECT |make )/.test(part)
                                    ? <code key={pi} className="rounded bg-muted px-1 py-0.5 font-mono text-[10px]">{part}</code>
                                    : <span key={pi}>{part}</span>
                                )}</>
                              : step
                            }
                          </li>
                        ))}
                      </ol>
                    </div>

                    {/* Raw Details (collapsed) */}
                    <details className="group">
                      <summary className="cursor-pointer text-[10px] text-muted-foreground/60 hover:text-muted-foreground">
                        Raw check details
                      </summary>
                      <pre className="mt-1 rounded bg-muted/50 p-2 text-[10px] font-mono text-muted-foreground overflow-x-auto">
                        {JSON.stringify(e.details, null, 2)}
                      </pre>
                    </details>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
