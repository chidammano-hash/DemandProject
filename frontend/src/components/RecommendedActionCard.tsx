import { ArrowRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface RecommendedActionProps {
  severity: "critical" | "high" | "medium" | "info";
  title: string;
  action: string;
  actionPanel?: string;
  onNavigate?: () => void;
}

export function RecommendedActionCard({ severity, title, action, onNavigate }: RecommendedActionProps) {
  const colors = {
    critical: "border-red-200 bg-red-50 dark:bg-red-900/10",
    high: "border-amber-200 bg-amber-50 dark:bg-amber-900/10",
    medium: "border-blue-200 bg-blue-50 dark:bg-blue-900/10",
    info: "border-border bg-muted/50",
  };

  return (
    <div className={cn("rounded-md border p-3 flex items-start gap-3", colors[severity])}>
      <div className="flex-1">
        <p className="text-xs font-medium">{title}</p>
        <p className="text-xs text-muted-foreground mt-0.5">{"\u2192"} {action}</p>
      </div>
      {onNavigate && (
        <button onClick={onNavigate} className="text-primary hover:underline text-xs shrink-0 flex items-center gap-1">
          Go <ArrowRight className="h-3 w-3" />
        </button>
      )}
    </div>
  );
}
