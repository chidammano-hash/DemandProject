import { cn } from "@/lib/utils";

type ElementConfig = {
  symbol: string;
  number: number;
  name: string;
  color: string;
  activeColor: string;
  glow: string;
};

type LoadingElementProps = {
  config: ElementConfig;
  message?: string;
  /** Render as an absolute overlay with backdrop blur (for table loading) */
  overlay?: boolean;
  /** Size variant */
  size?: "sm" | "md";
};

export function LoadingElement({ config, message, overlay, size = "sm" }: LoadingElementProps) {
  const tile = (
    <div className="flex flex-col items-center gap-2">
      <div
        className={cn(
          "flex flex-col items-center justify-center rounded-lg border-2 shadow-md animate-pulse-glow",
          config.activeColor,
          config.glow,
          size === "md" ? "px-5 py-2.5" : "px-4 py-2"
        )}
      >
        <span className={cn("leading-none self-start font-mono opacity-50", size === "md" ? "text-xs" : "text-[11px]")}>
          {config.number}
        </span>
        <span className={cn("font-bold leading-tight font-mono", size === "md" ? "text-2xl" : "text-lg")}>
          {config.symbol}
        </span>
        <span className={cn("leading-none opacity-70", size === "md" ? "text-xs" : "text-[11px]")}>Loading</span>
      </div>
      {message && <span className="text-xs text-muted-foreground">{message}</span>}
    </div>
  );

  if (overlay) {
    return (
      <div className="absolute inset-0 z-30 flex items-center justify-center rounded-md bg-background/70 backdrop-blur-[1px]">
        {tile}
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center py-12 gap-3">
      {tile}
    </div>
  );
}
