import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

type LoadingElementProps = {
  /** Tab key (kept for API compat) */
  tabKey?: string;
  /** Legacy config prop (ignored — kept for API compat) */
  config?: unknown;
  message?: string;
  /** Render as an absolute overlay with backdrop blur (for table loading) */
  overlay?: boolean;
  /** Size variant */
  size?: "sm" | "md";
};

export function LoadingElement({ message, overlay, size = "sm" }: LoadingElementProps) {
  const iconSize = size === "md" ? "h-8 w-8" : "h-5 w-5";

  const spinnerEl = (
    <div className="flex flex-col items-center gap-2">
      <Loader2 className={cn("animate-spin text-muted-foreground", iconSize)} />
      {message && <span className="text-xs text-muted-foreground">{message}</span>}
    </div>
  );

  if (overlay) {
    return (
      <div className="absolute inset-0 z-30 flex items-center justify-center rounded-md bg-background/70 backdrop-blur-[1px]">
        {spinnerEl}
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center py-12 gap-3">
      {spinnerEl}
    </div>
  );
}
