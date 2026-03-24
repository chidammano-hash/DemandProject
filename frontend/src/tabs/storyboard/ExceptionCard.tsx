/**
 * ExceptionCard — compact row-style card for the exception queue list.
 */
import type { StoryboardException } from "@/types/storyboard";
import {
  severityBg,
  severityLabel,
  fmt,
  fmtCurrency,
  daysAgo,
  EXCEPTION_TYPE_COLORS,
  EXCEPTION_TYPE_LABELS,
  STATUS_DOT,
} from "./storyboardShared";

export function ExceptionCard({
  exception,
  isSelected,
  onSelect,
}: {
  exception: StoryboardException;
  isSelected: boolean;
  onSelect: () => void;
}) {
  return (
    <button
      onClick={onSelect}
      className={`w-full text-left px-4 py-3 transition-colors ${
        isSelected
          ? "bg-primary/5 border-l-2 border-l-primary"
          : "hover:bg-muted/50 border-l-2 border-l-transparent"
      }`}
    >
      {/* Row 1: severity + type + status + age */}
      <div className="flex items-center gap-2 mb-1">
        <span
          className={`inline-block h-2 w-2 rounded-full flex-shrink-0 ${severityBg(exception.severity)}`}
          title={`Severity: ${severityLabel(exception.severity)} (${fmt(exception.severity, 2)})`}
        />
        <span
          className={`text-[10px] px-1.5 py-0.5 rounded font-medium border ${
            EXCEPTION_TYPE_COLORS[exception.exception_type] ?? ""
          }`}
        >
          {EXCEPTION_TYPE_LABELS[exception.exception_type] ?? exception.exception_type}
        </span>
        <span className="ml-auto flex items-center gap-1 text-[10px] text-muted-foreground">
          <span className={`h-1.5 w-1.5 rounded-full ${STATUS_DOT[exception.status] ?? "bg-gray-400"}`} />
          {exception.status}
        </span>
        <span className="text-[10px] text-muted-foreground">{daysAgo(exception.generated_at)}</span>
      </div>

      {/* Row 2: Item @ Loc */}
      <p className="text-xs font-medium truncate">
        {exception.item_id} @ {exception.loc}
      </p>

      {/* Row 3: Headline (truncated) */}
      {exception.headline && (
        <p className="text-[11px] text-muted-foreground truncate mt-0.5 leading-snug">
          {exception.headline}
        </p>
      )}

      {/* Row 4: Financial impact */}
      {exception.financial_impact != null && (
        <p className="text-[10px] text-muted-foreground mt-1">
          Impact: <span className="font-medium text-foreground">{fmtCurrency(exception.financial_impact)}</span>
        </p>
      )}
    </button>
  );
}
