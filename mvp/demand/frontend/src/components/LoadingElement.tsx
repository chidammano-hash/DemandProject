import { cn } from "@/lib/utils";
import { useMotif } from "@/context/MotifContext";

type LoadingElementProps = {
  /** Tab key into the motif tiles map (preferred) */
  tabKey?: string;
  /** Legacy config prop */
  config?: { symbol: string; number: number; name: string; color: string; activeColor: string; glow: string };
  message?: string;
  /** Render as an absolute overlay with backdrop blur (for table loading) */
  overlay?: boolean;
  /** Size variant */
  size?: "sm" | "md";
};

export function LoadingElement({ tabKey, config, message, overlay, size = "sm" }: LoadingElementProps) {
  let motif: ReturnType<typeof useMotif> | null = null;
  try {
    motif = useMotif();
  } catch {
    // Outside MotifProvider — use legacy config
  }

  const tile = tabKey && motif ? motif.getTile(tabKey) : null;
  const animationName = motif?.motifConfig.loading.animationName ?? "pulse-glow";
  const statusLabel = motif?.motifConfig.loading.statusLabel ?? "Loading";
  const wrapperClasses = motif?.motifConfig.loading.wrapperClasses ?? "rounded-lg";

  const primary = tile ? tile.primary : config?.symbol ?? "?";
  const superscript = tile ? tile.superscript : config?.number ?? 0;
  const activeCls = tile ? tile.activeClasses : config?.activeColor ?? "";
  const glowCls = tile ? tile.glowClass : config?.glow ?? "";

  const tileEl = (
    <div className="flex flex-col items-center gap-2">
      <div
        className={cn(
          "flex flex-col items-center justify-center border-2 shadow-md",
          wrapperClasses,
          `animate-${animationName}`,
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
        <span className={cn("leading-none opacity-70", size === "md" ? "text-xs" : "text-[11px]")}>{statusLabel}</span>
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
