import { AlertTriangle, CheckCircle2, Clock } from "lucide-react";
import { cn } from "@/lib/utils";

interface DataFreshnessProps {
  lastRefreshed?: string | null;  // ISO timestamp
  staleSec?: number;              // Threshold in seconds (default: 3600 = 1 hour)
  source?: string;                // e.g. "Production Forecast", "Safety Stock Targets"
  warnings?: string[];            // Additional warnings (e.g. "12 SKUs missing forecast data")
}

export function DataFreshnessBanner({ lastRefreshed, staleSec = 3600, source, warnings = [] }: DataFreshnessProps) {
  const isStale = lastRefreshed ? (Date.now() - new Date(lastRefreshed).getTime()) / 1000 > staleSec : true;
  const isMissing = !lastRefreshed;

  const ageText = lastRefreshed
    ? formatTimeAgo(new Date(lastRefreshed))
    : "Unknown";

  return (
    <div className={cn(
      "flex items-center gap-2 text-xs px-3 py-1.5 rounded-md mb-2",
      isMissing ? "bg-red-50 text-red-700 dark:bg-red-900/20 dark:text-red-400" :
      isStale ? "bg-amber-50 text-amber-700 dark:bg-amber-900/20 dark:text-amber-400" :
      "bg-emerald-50 text-emerald-700 dark:bg-emerald-900/20 dark:text-emerald-400"
    )}>
      {isMissing ? <AlertTriangle className="h-3.5 w-3.5 shrink-0" /> :
       isStale ? <Clock className="h-3.5 w-3.5 shrink-0" /> :
       <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />}
      <span>
        {source ? `${source}: ` : ""}
        {isMissing ? "No data available" : `Last updated ${ageText}`}
        {isStale && !isMissing ? " (stale)" : ""}
      </span>
      {warnings.map((w, i) => (
        <span key={i} className="ml-2 text-amber-600 dark:text-amber-400">Warning: {w}</span>
      ))}
    </div>
  );
}

function formatTimeAgo(date: Date): string {
  const sec = Math.floor((Date.now() - date.getTime()) / 1000);
  if (sec < 60) return "just now";
  if (sec < 3600) return `${Math.floor(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.floor(sec / 3600)}h ago`;
  return `${Math.floor(sec / 86400)}d ago`;
}
