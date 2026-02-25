import { Wine, BarChart3, Radar, Sun, Moon } from "lucide-react";
import { cn } from "@/lib/utils";
import type { ProductThemeId } from "@/types/theme";
import type { ColorMode } from "@/hooks/useTheme";
import { PRODUCT_THEMES, THEME_ORDER } from "@/constants/themes";

const THEME_ICONS: Record<ProductThemeId, React.ElementType> = {
  "wine-spirits": Wine,
  general: BarChart3,
  obsidian: Radar,
};

interface ThemeSelectorProps {
  themeId: ProductThemeId;
  colorMode: ColorMode;
  onThemeChange: (id: ProductThemeId) => void;
  onModeChange: (mode: ColorMode) => void;
  collapsed?: boolean;
}

export function ThemeSelector({ themeId, colorMode, onThemeChange, onModeChange, collapsed }: ThemeSelectorProps) {
  const currentTheme = PRODUCT_THEMES[themeId];

  if (collapsed) {
    const Icon = THEME_ICONS[themeId];
    return (
      <button
        title={`Theme: ${currentTheme.displayName}`}
        className="flex w-full items-center justify-center rounded-md p-2 text-sidebar-foreground hover:bg-sidebar-hover"
        onClick={() => {
          const idx = THEME_ORDER.indexOf(themeId);
          onThemeChange(THEME_ORDER[(idx + 1) % THEME_ORDER.length]);
        }}
      >
        <Icon className="h-4 w-4" strokeWidth={1.5} />
      </button>
    );
  }

  return (
    <div className="space-y-2">
      <p className="text-[10px] font-medium uppercase tracking-wider text-sidebar-foreground/50">Theme</p>
      <div className="flex gap-1" role="radiogroup" aria-label="Select theme">
        {THEME_ORDER.map((id) => {
          const theme = PRODUCT_THEMES[id];
          const Icon = THEME_ICONS[id];
          const isSelected = themeId === id;
          return (
            <button
              key={id}
              role="radio"
              aria-checked={isSelected}
              onClick={() => onThemeChange(id)}
              className={cn(
                "flex flex-1 flex-col items-center gap-1 rounded-md border p-2 text-[10px] transition-colors",
                isSelected
                  ? "border-sidebar-active bg-sidebar-active/10 text-sidebar-active"
                  : "border-border/40 text-sidebar-foreground/60 hover:border-sidebar-active/40 hover:text-sidebar-foreground",
              )}
            >
              <Icon className="h-4 w-4" strokeWidth={1.5} />
              <span className="truncate">{theme.displayName.split(" ").pop()}</span>
            </button>
          );
        })}
      </div>

      {/* Color mode toggle */}
      <div className="flex items-center gap-1">
        <button
          onClick={() => onModeChange("light")}
          disabled={themeId === "obsidian"}
          className={cn(
            "flex flex-1 items-center justify-center gap-1 rounded-md border py-1.5 text-[10px] transition-colors",
            colorMode === "light" && themeId !== "obsidian"
              ? "border-sidebar-active bg-sidebar-active/10 text-sidebar-active"
              : "border-border/40 text-sidebar-foreground/40",
            themeId === "obsidian" && "cursor-not-allowed opacity-40",
          )}
        >
          <Sun className="h-3 w-3" />
          Light
        </button>
        <button
          onClick={() => onModeChange("dark")}
          disabled={themeId === "obsidian"}
          className={cn(
            "flex flex-1 items-center justify-center gap-1 rounded-md border py-1.5 text-[10px] transition-colors",
            colorMode === "dark" || themeId === "obsidian"
              ? "border-sidebar-active bg-sidebar-active/10 text-sidebar-active"
              : "border-border/40 text-sidebar-foreground/40",
            themeId === "obsidian" && "cursor-not-allowed opacity-40",
          )}
        >
          <Moon className="h-3 w-3" />
          Dark
        </button>
      </div>
    </div>
  );
}
