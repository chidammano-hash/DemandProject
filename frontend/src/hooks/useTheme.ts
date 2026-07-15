import { useCallback, useEffect, useState } from "react";
import type { ProductTheme, ThemePalette, ChartThemeConfig, TypographyConfig } from "@/types/theme";
import { PRODUCT_THEMES } from "@/constants/themes";
import { CSS_VAR_MAP } from "@/constants/palette";

export type { ColorMode } from "@/constants/palette";
import type { ColorMode } from "@/constants/palette";

const MODE_KEY = "ds-color-mode";
const EXPLICIT_KEY = "ds-color-mode-explicit";

function prefersDark(): boolean {
  try {
    return typeof window !== "undefined"
      && typeof window.matchMedia === "function"
      && window.matchMedia("(prefers-color-scheme: dark)").matches;
  } catch { return false; }
}

function getInitialMode(): ColorMode {
  try {
    const stored = localStorage.getItem(MODE_KEY) as ColorMode | null;
    if (stored && (stored === "light" || stored === "soft" || stored === "dark")) return stored;
  } catch { /* ignore */ }
  // No explicit user choice — follow OS
  return prefersDark() ? "dark" : "light";
}

/** Map a ThemePalette (+ typography) to CSS custom properties on the root element */
function applyPalette(palette: ThemePalette, typography: TypographyConfig) {
  const root = document.documentElement;
  for (const [prop, key] of CSS_VAR_MAP) {
    root.style.setProperty(prop, palette[key]);
  }
  root.style.setProperty("--heading-tracking", typography.headingTracking);
  root.style.setProperty("--kpi-tracking", typography.kpiTracking);
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
    root.setAttribute("data-mode", colorMode);
    // Soft is a first-class mode: it keeps the `light` base class (so `dark:`
    // variants stay off) and adds `soft` for the `soft:` Tailwind variant and
    // the `.soft` CSS fallback block.
    root.classList.remove("light", "soft", "dark");
    if (colorMode === "dark") root.classList.add("dark");
    else if (colorMode === "soft") root.classList.add("light", "soft");
    else root.classList.add("light");

    const palette = getEffectivePalette(productTheme, colorMode);
    applyPalette(palette, productTheme.typography);

    localStorage.setItem(MODE_KEY, colorMode);

    const timer = setTimeout(() => root.removeAttribute("data-transitioning"), 300);
    return () => clearTimeout(timer);
  }, [colorMode, productTheme]);

  // Listen for OS prefers-color-scheme changes — only when the user has not
  // made an explicit choice via the mode toggle.
  useEffect(() => {
    if (typeof window === "undefined" || typeof window.matchMedia !== "function") return;
    const mql = window.matchMedia("(prefers-color-scheme: dark)");
    const handler = (e: MediaQueryListEvent) => {
      try {
        if (localStorage.getItem(EXPLICIT_KEY) === "1") return;
      } catch { /* ignore */ }
      setColorModeState(e.matches ? "dark" : "light");
    };
    if (typeof mql.addEventListener === "function") {
      mql.addEventListener("change", handler);
      return () => mql.removeEventListener("change", handler);
    }
    // Safari fallback
    mql.addListener(handler);
    return () => mql.removeListener(handler);
  }, []);

  const setColorMode = useCallback((mode: ColorMode) => {
    try { localStorage.setItem(EXPLICIT_KEY, "1"); } catch { /* ignore */ }
    setColorModeState(mode);
  }, []);

  const toggleColorMode = useCallback(() => {
    try { localStorage.setItem(EXPLICIT_KEY, "1"); } catch { /* ignore */ }
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
