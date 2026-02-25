import { useCallback, useEffect, useState } from "react";
import type { ProductThemeId, ProductTheme, ThemePalette, ChartThemeConfig } from "@/types/theme";
import { PRODUCT_THEMES, THEME_ORDER } from "@/constants/themes";

export type ColorMode = "light" | "dark";

const THEME_KEY = "ds-product-theme";
const MODE_KEY = "ds-color-mode";

function getInitialThemeId(): ProductThemeId {
  try {
    const stored = localStorage.getItem(THEME_KEY) as ProductThemeId | null;
    if (stored && stored in PRODUCT_THEMES) return stored;
  } catch { /* ignore */ }
  return "general";
}

function getInitialMode(): ColorMode {
  try {
    const stored = localStorage.getItem(MODE_KEY) as ColorMode | null;
    if (stored && (stored === "light" || stored === "dark")) return stored;
  } catch { /* ignore */ }
  return "light";
}

/** Map a ThemePalette to CSS custom properties on the root element */
function applyPalette(palette: ThemePalette) {
  const root = document.documentElement;
  const map: [string, string][] = [
    ["--background", palette.background],
    ["--foreground", palette.foreground],
    ["--card", palette.card],
    ["--card-foreground", palette.cardForeground],
    ["--primary", palette.primary],
    ["--primary-foreground", palette.primaryForeground],
    ["--secondary", palette.secondary],
    ["--secondary-foreground", palette.secondaryForeground],
    ["--muted", palette.muted],
    ["--muted-foreground", palette.mutedForeground],
    ["--accent", palette.accent],
    ["--accent-foreground", palette.accentForeground],
    ["--border", palette.border],
    ["--input", palette.input],
    ["--ring", palette.ring],
    ["--destructive", palette.destructive],
    ["--destructive-foreground", palette.destructiveForeground],
    ["--sidebar-bg", palette.sidebarBg],
    ["--sidebar-foreground", palette.sidebarForeground],
    ["--sidebar-active", palette.sidebarActive],
    ["--sidebar-hover", palette.sidebarHover],
    ["--chart-1", palette.chart1],
    ["--chart-2", palette.chart2],
    ["--chart-3", palette.chart3],
    ["--chart-4", palette.chart4],
    ["--chart-5", palette.chart5],
    ["--chart-6", palette.chart6],
    ["--kpi-best", palette.kpiBest],
    ["--kpi-warning", palette.kpiWarning],
    ["--kpi-ceiling", palette.kpiCeiling],
    ["--bg-gradient-primary", palette.bgGradientPrimary],
    ["--bg-gradient-secondary", palette.bgGradientSecondary],
    ["--bg-gradient-base-start", palette.bgGradientBaseStart],
    ["--bg-gradient-base-mid", palette.bgGradientBaseMid],
    ["--bg-gradient-base-end", palette.bgGradientBaseEnd],
  ];
  for (const [prop, value] of map) {
    root.style.setProperty(prop, value);
  }
}

function getEffectivePalette(theme: ProductTheme, mode: ColorMode): ThemePalette {
  if (mode === "light" && theme.palette.light) return theme.palette.light;
  return theme.palette.dark;
}

function getEffectiveChartConfig(theme: ProductTheme, mode: ColorMode): ChartThemeConfig {
  if (mode === "light" && theme.charts.light) return theme.charts.light;
  return theme.charts.dark;
}

export function useTheme() {
  const [themeId, setThemeId] = useState<ProductThemeId>(getInitialThemeId);
  const [colorMode, setColorModeState] = useState<ColorMode>(getInitialMode);

  const productTheme = PRODUCT_THEMES[themeId];

  // Obsidian always stays "dark" (even "light" mode is elevated dark)
  const effectiveClass = themeId === "obsidian" ? "dark" : colorMode;

  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute("data-transitioning", "true");
    root.setAttribute("data-theme", themeId);
    root.classList.remove("light", "dark", "midnight");
    root.classList.add(effectiveClass);

    const palette = getEffectivePalette(productTheme, colorMode);
    applyPalette(palette);

    localStorage.setItem(THEME_KEY, themeId);
    localStorage.setItem(MODE_KEY, colorMode);

    const timer = setTimeout(() => root.removeAttribute("data-transitioning"), 300);
    return () => clearTimeout(timer);
  }, [themeId, colorMode, productTheme, effectiveClass]);

  const setProductTheme = useCallback((id: ProductThemeId) => {
    setThemeId(id);
    if (id === "obsidian") setColorModeState("dark");
  }, []);

  const setColorMode = useCallback((mode: ColorMode) => {
    if (themeId === "obsidian") return;
    setColorModeState(mode);
  }, [themeId]);

  const toggleColorMode = useCallback(() => {
    if (themeId === "obsidian") return;
    setColorModeState((m) => (m === "light" ? "dark" : "light"));
  }, [themeId]);

  const cycleTheme = useCallback(() => {
    const idx = THEME_ORDER.indexOf(themeId);
    const nextIdx = (idx + 1) % THEME_ORDER.length;
    setProductTheme(THEME_ORDER[nextIdx]);
  }, [themeId, setProductTheme]);

  const chartConfig = getEffectiveChartConfig(productTheme, colorMode);

  return {
    themeId,
    colorMode,
    effectiveClass,
    productTheme,
    setProductTheme,
    setColorMode,
    toggleColorMode,
    cycleTheme,
    // Backwards compat for existing components that use theme: "light" | "dark"
    theme: effectiveClass as "light" | "dark",
    setTheme: (t: "light" | "dark") => setColorMode(t),
    trendColors: chartConfig.seriesColors,
    chartColors: {
      grid: chartConfig.gridColor,
      axis: chartConfig.axisColor,
      tooltip: chartConfig.tooltipBg,
    },
  };
}
