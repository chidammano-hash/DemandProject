import { cn } from "@/lib/utils";

type LoadingElementProps = {
  /** Tab key (unused after motif removal, kept for API compat) */
  tabKey?: string;
  /** Legacy config prop */
  config?: { symbol: string; number: number; name: string; color: string; activeColor: string; glow: string };
  message?: string;
  /** Render as an absolute overlay with backdrop blur (for table loading) */
  overlay?: boolean;
  /** Size variant */
  size?: "sm" | "md";
};

export function LoadingElement({ config, message, overlay, size = "sm" }: LoadingElementProps) {
  const primary = config?.symbol ?? "?";
  const superscript = config?.number ?? 0;
  const activeCls = config?.activeColor ?? "";
  const glowCls = config?.glow ?? "";

  const tileEl = (
    <div className="flex flex-col items-center gap-2">
      <div
        className={cn(
          "flex flex-col items-center justify-center border-2 shadow-md rounded-lg animate-pulse-glow",
          activeCls,
          glowCls,
          size === "md" ? "px-5 py-2.5" : "px-4 py-2"
        )}
      >
        <span className={cn("leading-none self-start font-mono opacity-50", size === "md" ? "text-xs" : "text-[11px]")}>
          {superscript}
        </span>
        <span className={cn("font-bold leading-tight font-mono", size === "md" ? "text-2xl" : "text-lg")}>
          {primary}
        </span>
        <span className={cn("leading-none opacity-70", size === "md" ? "text-xs" : "text-[11px]")}>Loading</span>
      </div>
      {message && <span className="text-xs text-muted-foreground">{message}</span>}
    </div>
  );

  if (overlay) {
    return (
      <div className="absolute inset-0 z-30 flex items-center justify-center rounded-md bg-background/70 backdrop-blur-[1px]">
        {tileEl}
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center py-12 gap-3">
      {tileEl}
    </div>
  );
}
