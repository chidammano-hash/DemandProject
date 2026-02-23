import { cn } from "@/lib/utils";

type ElementConfig = {
  symbol: string;
  number: number;
  name: string;
  color: string;
  activeColor: string;
  glow: string;
};

type ElementTabProps = {
  config: ElementConfig;
  isActive: boolean;
  onClick: () => void;
};

export function ElementTab({ config, isActive, onClick }: ElementTabProps) {
  return (
    <button
      role="tab"
      aria-selected={isActive}
      aria-label={config.name}
      className={cn(
        "group relative flex flex-col items-center justify-center rounded-xl border px-3.5 py-2 min-w-[68px] transition-all duration-200 backdrop-blur-sm",
        isActive
          ? config.activeColor + " " + config.glow + " scale-105 border-opacity-100"
          : config.color + " hover:scale-105 hover:border-opacity-80 border-opacity-40"
      )}
      onClick={onClick}
    >
      {isActive && <span className="absolute -bottom-1 left-1/2 h-1 w-6 -translate-x-1/2 rounded-full bg-current opacity-60" />}
      <span className="text-[11px] leading-none self-end font-mono opacity-50">{config.number}</span>
      <span className="text-xl font-black leading-tight font-mono tracking-tight">{config.symbol}</span>
      <span className="text-[11px] font-medium leading-none tracking-wide uppercase opacity-70">{config.name}</span>
    </button>
  );
}
