/**
 * LagFilterBar -- Segmented control for execution-lag filtering (All | Lag 0-4).
 *
 * Execution lag is the number of months between when the forecast was generated
 * and the target month. Lag 0 = 1-month ahead (most accurate), Lag 4 = 5 months
 * ahead (least accurate).
 */
import { HelpCircle } from "lucide-react";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
export interface LagFilterBarProps {
  /** undefined = "All" (portfolio-level, no lag filter) */
  value: number | undefined;
  onChange: (lag: number | undefined) => void;
}

// ---------------------------------------------------------------------------
// Lag definitions
// ---------------------------------------------------------------------------
interface LagOption {
  label: string;
  subLabel: string;
  value: number | undefined;
}

const LAG_OPTIONS: LagOption[] = [
  { label: "All", subLabel: "Portfolio", value: undefined },
  { label: "Lag 0", subLabel: "1mo", value: 0 },
  { label: "Lag 1", subLabel: "2mo", value: 1 },
  { label: "Lag 2", subLabel: "3mo", value: 2 },
  { label: "Lag 3", subLabel: "4mo", value: 3 },
  { label: "Lag 4", subLabel: "5mo", value: 4 },
];

const TOOLTIP_TEXT =
  "Execution lag is the number of months between when the forecast was generated " +
  "and the target month. Lag 0 = 1-month ahead (most accurate), Lag 4 = 5 months " +
  "ahead (least accurate).";

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------
export function LagFilterBar({ value, onChange }: LagFilterBarProps) {
  return (
    <div className="flex items-center gap-2">
      <div className="flex gap-0.5 rounded-lg border border-border bg-muted/30 p-0.5">
        {LAG_OPTIONS.map((opt) => {
          const isActive =
            opt.value === value ||
            (opt.value === undefined && value === undefined);

          return (
            <button
              key={opt.label}
              onClick={() => onChange(opt.value)}
              className={cn(
                "flex flex-col items-center px-3 py-1 text-xs rounded-md transition-colors min-w-[52px]",
                isActive
                  ? "bg-primary text-primary-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted",
              )}
            >
              <span className="font-medium leading-tight">{opt.label}</span>
              <span
                className={cn(
                  "text-[9px] leading-tight",
                  isActive ? "text-primary-foreground/70" : "text-muted-foreground/60",
                )}
              >
                ({opt.subLabel})
              </span>
            </button>
          );
        })}
      </div>

      <span title={TOOLTIP_TEXT} className="cursor-help">
        <HelpCircle className="h-3.5 w-3.5 text-muted-foreground/50" strokeWidth={1.5} />
      </span>
    </div>
  );
}
