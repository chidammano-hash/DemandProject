import { useCallback, useEffect, useState } from "react";
import type { ProductTheme, ThemePalette, ChartThemeConfig } from "@/types/theme";
import { PRODUCT_THEMES } from "@/constants/themes";

export type ColorMode = "light" | "soft" | "dark";

const MODE_KEY = "ds-color-mode";

function getInitialMode(): ColorMode {
  try {
    const stored = localStorage.getItem(MODE_KEY) as ColorMode | null;
    if (stored && (stored === "light" || stored === "soft" || stored === "dark")) return stored;
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
  if (mode === "soft" && theme.palette.soft) return theme.palette.soft;
  return theme.palette.dark;
}

function getEffectiveChartConfig(theme: ProductTheme, mode: ColorMode): ChartThemeConfig {
  if (mode === "light" && theme.charts.light) return theme.charts.light;
  if (mode === "soft" && theme.charts.soft) return theme.charts.soft;
  return theme.charts.dark;
}

export function useTheme() {
  const [colorMode, setColorModeState] = useState<ColorMode>(getInitialMode);

  const productTheme = PRODUCT_THEMES["general"];

  useEffect(() => {
    const root = document.documentElement;
    root.setAttribute("data-transitioning", "true");
    root.setAttribute("data-theme", "general");
    const cssClass = colorMode === "dark" ? "dark" : "light";
    root.classList.remove("light", "dark");
    root.classList.add(cssClass);

    const palette = getEffectivePalette(productTheme, colorMode);
    applyPalette(palette);

    localStorage.setItem(MODE_KEY, colorMode);

    const timer = setTimeout(() => root.removeAttribute("data-transitioning"), 300);
    return () => clearTimeout(timer);
  }, [colorMode, productTheme]);

  const setColorMode = useCallback((mode: ColorMode) => {
    setColorModeState(mode);
  }, []);

  const toggleColorMode = useCallback(() => {
    setColorModeState((m) => {
      if (m === "light") return "soft";
      if (m === "soft") return "dark";
      return "light";
    });
  }, []);

  const chartConfig = getEffectiveChartConfig(productTheme, colorMode);

  return {
    themeId: "general" as const,
    colorMode,
    effectiveClass: colorMode === "dark" ? "dark" : "light",
    productTheme,
    setColorMode,
    toggleColorMode,
    theme: colorMode,
    setTheme: (t: ColorMode) => setColorMode(t),
    trendColors: chartConfig.seriesColors,
    chartColors: {
      grid: chartConfig.gridColor,
      axis: chartConfig.axisColor,
      tooltip: chartConfig.tooltipBg,
    },
  };
}
