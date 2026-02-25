import { cn } from "@/lib/utils";
import { useMotif } from "@/context/MotifContext";

type ElementTabProps = {
  /** Tab key into the motif tiles map */
  tabKey?: string;
  /** Legacy config prop — still supported for backward compat */
  config?: { symbol: string; number: number; name: string; color: string; activeColor: string; glow: string };
  isActive: boolean;
  onClick: () => void;
};

export function ElementTab({ tabKey, config, isActive, onClick }: ElementTabProps) {
  let motif: ReturnType<typeof useMotif> | null = null;
  try {
    motif = useMotif();
  } catch {
    // Outside MotifProvider — use legacy config
  }

  // If tabKey is provided and motif context is available, read from motif
  const tile = tabKey && motif ? motif.getTile(tabKey) : null;
  const radius = motif?.motifConfig.chrome.tileRadius ?? "rounded-xl";

  // Use motif tile if available, otherwise fall back to legacy config
  const primary = tile ? tile.primary : config?.symbol ?? "?";
  const superscript = tile ? tile.superscript : config?.number ?? 0;
  const label = tile ? tile.label : config?.name ?? "";
  const restCls = tile ? tile.restClasses : config?.color ?? "";
  const activeCls = tile ? tile.activeClasses : config?.activeColor ?? "";
  const glowCls = tile ? tile.glowClass : config?.glow ?? "";

  return (
    <button
      role="tab"
      aria-selected={isActive}
      aria-label={label}
      className={cn(
        "group relative flex flex-col items-center justify-center border px-3.5 py-2 min-w-[68px] transition-all duration-200 backdrop-blur-sm",
        radius,
        isActive
          ? activeCls + " " + glowCls + " scale-105 border-opacity-100"
          : restCls + " hover:scale-105 hover:border-opacity-80 border-opacity-40"
      )}
      onClick={onClick}
    >
      {isActive && <span className="absolute -bottom-1 left-1/2 h-1 w-6 -translate-x-1/2 rounded-full bg-current opacity-60" />}
      <span className="text-[11px] leading-none self-end font-mono opacity-50">{superscript}</span>
      <span className="text-xl font-black leading-tight font-mono tracking-tight">{primary}</span>
      <span className="text-[11px] font-medium leading-none tracking-wide uppercase opacity-70">{label}</span>
    </button>
  );
}
