/**
 * PipelineReadinessBanner — a calm, non-blocking notice that a downstream ML
 * stage is out of sync with the SKU dimension (e.g. clustering has no
 * assignments after a dim_sku reload).
 *
 * It informs and deep-links to where the user fixes it; it never blocks and
 * never runs anything itself. Staleness is derived live by the backend
 * (GET /dashboard/pipeline-readiness) and only fires on a genuine total loss of
 * clustering — so the banner stays quiet during normal operation and clears
 * itself once clustering is re-run and promoted.
 *
 * Drop `<PipelineReadinessBanner />` at the top of any surface — it self-fetches
 * and renders nothing when everything is in sync or while loading.
 */
import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, ArrowRight, X } from "lucide-react";

import {
  fetchPipelineReadiness,
  pipelineReadinessKeys,
  type PipelineReadinessCheck,
} from "@/api/queries";
import { getSeverityConfig } from "@/constants/severity";
import { navigateToTab } from "@/lib/navigation";
import { cn } from "@/lib/utils";

const POLL_MS = 30_000;

export function PipelineReadinessBanner() {
  const [dismissed, setDismissed] = useState<Set<string>>(new Set());

  const { data } = useQuery({
    queryKey: pipelineReadinessKeys.readiness,
    queryFn: fetchPipelineReadiness,
    staleTime: 60_000,
    // While something is stale, re-check periodically so the banner clears itself
    // once the dependent stage is fixed — no manual refresh needed.
    refetchInterval: (query) =>
      query.state.data && !query.state.data.ready ? POLL_MS : false,
  });

  if (!data || data.ready) return null;

  const visible = data.checks.filter(
    (c) => c.status === "stale" && !dismissed.has(c.stage),
  );
  if (visible.length === 0) return null;

  const dismiss = (stage: string) =>
    setDismissed((prev) => new Set(prev).add(stage));

  const runAction = (check: PipelineReadinessCheck) => {
    const action = check.action;
    if (action?.kind === "navigate") {
      // The clustering fix lives in the SKU domain; switch domain alongside tab.
      navigateToTab(action.target, action.target === "clusters" ? "sku" : undefined);
    }
  };

  return (
    <div className="flex flex-col gap-2" role="status" aria-live="polite">
      {visible.map((check) => {
        const cfg = getSeverityConfig(check.severity);
        return (
          <div
            key={check.stage}
            className={cn(
              "flex items-start gap-3 rounded-lg border border-l-4 bg-card p-3",
              cfg.border,
            )}
          >
            <AlertTriangle className={cn("mt-0.5 h-4 w-4 shrink-0", cfg.icon)} />
            <div className="min-w-0 flex-1">
              <p className="text-sm font-medium text-foreground">{check.title}</p>
              <p className="mt-0.5 text-xs text-muted-foreground">{check.detail}</p>
            </div>
            {check.action && (
              <button
                type="button"
                onClick={() => runAction(check)}
                className={cn(
                  "shrink-0 inline-flex items-center gap-1.5 rounded-md px-2.5 py-1.5",
                  "text-xs font-medium transition-colors",
                  "bg-primary/10 text-primary hover:bg-primary/20",
                )}
              >
                {check.action.label}
                <ArrowRight className="h-3.5 w-3.5" />
              </button>
            )}
            <button
              type="button"
              onClick={() => dismiss(check.stage)}
              aria-label="Dismiss"
              className="shrink-0 rounded p-1 text-muted-foreground hover:bg-muted hover:text-foreground"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        );
      })}
    </div>
  );
}
