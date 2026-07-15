import { Sun, Moon, SunMedium } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ColorMode } from "@/hooks/useTheme";

interface ThemeSelectorProps {
  colorMode: ColorMode;
  onModeChange: (mode: ColorMode) => void;
  collapsed?: boolean;
}

const MODE_CYCLE: ColorMode[] = ["light", "soft", "dark"];

function nextMode(current: ColorMode): ColorMode {
  const idx = MODE_CYCLE.indexOf(current);
  return MODE_CYCLE[(idx + 1) % MODE_CYCLE.length];
}

function modeIcon(mode: ColorMode) {
  if (mode === "light") return Sun;
  if (mode === "soft") return SunMedium;
  return Moon;
}

export function ThemeSelector({ colorMode, onModeChange, collapsed }: ThemeSelectorProps) {
  if (collapsed) {
    const Icon = modeIcon(colorMode);
    return (
      <button
        title={`Mode: ${colorMode}`}
        className="flex w-full items-center justify-center rounded-md p-2 text-sidebar-foreground hover:bg-sidebar-hover"
        onClick={() => onModeChange(nextMode(colorMode))}
      >
        <Icon className="h-4 w-4" strokeWidth={1.5} />
      </button>
    );
  }

  return (
    <div className="space-y-2">
      <p className="text-[10px] font-medium uppercase tracking-wider text-sidebar-foreground/50">Appearance</p>
      <div className="flex items-center gap-1">
        <button
          onClick={() => onModeChange("light")}
          className={cn(
            "flex flex-1 items-center justify-center gap-1 rounded-md border py-1.5 text-[10px] transition-colors",
            colorMode === "light"
              ? "border-sidebar-active bg-sidebar-active/10 font-semibold text-sidebar-active"
              : "border-border/40 text-sidebar-foreground/40",
          )}
        >
          <Sun className="h-3 w-3" />
          Light
        </button>
        <button
          onClick={() => onModeChange("soft")}
          className={cn(
            "flex flex-1 items-center justify-center gap-1 rounded-md border py-1.5 text-[10px] transition-colors",
            colorMode === "soft"
              ? "border-sidebar-active bg-sidebar-active/10 font-semibold text-sidebar-active"
              : "border-border/40 text-sidebar-foreground/40",
          )}
        >
          <SunMedium className="h-3 w-3" />
          Soft
        </button>
        <button
          onClick={() => onModeChange("dark")}
          className={cn(
            "flex flex-1 items-center justify-center gap-1 rounded-md border py-1.5 text-[10px] transition-colors",
            colorMode === "dark"
              ? "border-sidebar-active bg-sidebar-active/10 font-semibold text-sidebar-active"
              : "border-border/40 text-sidebar-foreground/40",
          )}
        >
          <Moon className="h-3 w-3" />
          Dark
        </button>
      </div>
    </div>
  );
}
