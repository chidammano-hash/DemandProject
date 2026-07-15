/**
 * Command Center — Exception Feed Card.
 *
 * A single row in the unified exception feed. Shows severity dot + source badge
 * + type label, item/location identity, summary, optional recommendation,
 * financial impact + timestamp, and Accept (AI only) / View Item actions.
 */
import {
  Brain,
  BookOpen,
  ExternalLink,
  Loader2,
  CheckCircle2,
  DollarSign,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { formatCurrency, formatDate } from "@/lib/formatters";
import { getSeverityConfig } from "@/constants/severity";
import {
  type UnifiedException,
  SOURCE_BADGE,
  SEVERITY_BORDER,
  SEVERITY_DOT,
} from "./exceptions";

export function ExceptionFeedCard({
  item,
  onAccept,
  onViewItem,
  acceptPending,
}: {
  item: UnifiedException;
  onAccept: (item: UnifiedException) => void;
  onViewItem: () => void;
  acceptPending?: boolean;
}) {
  const sourceBadge = SOURCE_BADGE[item.source];
  const SourceIcon = item.source === "ai" ? Brain : BookOpen;

  // Subtle severity wash behind the row — low severity stays plain card.
  const severityBgColor: Record<string, string> = {
    critical: "bg-destructive/5",
    high: "bg-severity-high/5",
    medium: "bg-warning/5",
    low: "",
  };

  return (
    <div
      className={cn(
        "rounded-lg border border-l-4 bg-card p-4 shadow-sm transition-all hover:shadow-md",
        SEVERITY_BORDER[item.severity] ?? SEVERITY_BORDER.low,
        severityBgColor[item.severity] ?? ""
      )}
      data-testid="exception-card"
    >
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1 min-w-0 space-y-2">
          {/* Top row: severity dot + source badge + type label */}
          <div className="flex items-center gap-2 flex-wrap">
            <span
              className={cn(
                "h-2.5 w-2.5 rounded-full flex-shrink-0 ring-2 ring-offset-1 ring-offset-card",
                SEVERITY_DOT[item.severity],
                item.severity === "critical" ? "ring-red-200 dark:ring-red-800" : "ring-transparent"
              )}
            />
            <span
              className={cn(
                "text-[10px] font-semibold px-2 py-0.5 rounded-full inline-flex items-center gap-1",
                sourceBadge.className
              )}
            >
              <SourceIcon className="h-2.5 w-2.5" />
              {sourceBadge.label}
            </span>
            <span className="text-xs font-medium text-foreground/70 bg-muted px-2 py-0.5 rounded">
              {item.typeLabel}
            </span>
            <span className={cn(
              "text-2xs font-semibold uppercase tracking-wider px-1.5 py-0.5 rounded",
              getSeverityConfig(item.severity).badge,
            )}>
              {item.severity}
            </span>
          </div>

          {/* Item/Location identity — code pair plus the human-readable product
              name when the dim_item join resolved one (U2.1). */}
          <p className="text-xs font-mono font-bold tracking-wide">
            {item.itemNo} @ {item.location}
          </p>
          {item.itemDesc && (
            <p className="text-xs text-muted-foreground truncate">
              {item.itemDesc}
            </p>
          )}

          {/* Summary */}
          <p className="text-sm leading-relaxed">{item.summary}</p>

          {/* Recommendation */}
          {item.recommendation && (
            <div className="flex items-start gap-1.5 text-xs text-muted-foreground bg-muted/50 rounded-md px-2.5 py-1.5">
              <CheckCircle2 className="h-3 w-3 mt-0.5 shrink-0 text-primary/60" />
              <span className="leading-snug">{item.recommendation}</span>
            </div>
          )}

          {/* Financial impact + timestamp */}
          <div className="flex items-center gap-3 flex-wrap pt-0.5">
            {item.financialImpact != null && item.financialImpact > 0 && (
              <span className="inline-flex items-center gap-1 text-xs font-semibold text-warning bg-warning/10 px-2 py-0.5 rounded">
                <DollarSign className="h-3 w-3" />
                {formatCurrency(item.financialImpact)}
              </span>
            )}
            <span className="text-[10px] text-muted-foreground">
              {formatDate(item.createdAt)}
            </span>
          </div>
        </div>

        {/* Action buttons */}
        <div className="flex flex-col gap-2 flex-shrink-0 pt-0.5">
          {item.source === "ai" && item.status === "open" && (
            <button
              onClick={() => onAccept(item)}
              disabled={acceptPending}
              className="inline-flex items-center justify-center gap-1 text-xs font-medium rounded-md border px-3 py-1.5 bg-success/10 border-success/25 text-success hover:bg-success/20 transition-colors disabled:opacity-50"
            >
              {acceptPending ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <>
                  <CheckCircle2 className="h-3 w-3" />
                  Accept
                </>
              )}
            </button>
          )}
          <button
            onClick={onViewItem}
            className="inline-flex items-center justify-center gap-1 text-xs font-medium rounded-md border px-3 py-1.5 hover:bg-muted transition-colors"
          >
            View Item
            <ExternalLink className="h-3 w-3" />
          </button>
        </div>
      </div>
    </div>
  );
}
