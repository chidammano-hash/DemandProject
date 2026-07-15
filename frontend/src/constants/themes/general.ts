import type { ProductTheme, ChartThemeConfig } from "@/types/theme";
import { PALETTE, type ColorMode } from "@/constants/palette";

/**
 * The runtime ProductTheme. All color values derive from the single palette
 * source (`constants/palette.ts`) — no literals here.
 */

function chartConfig(mode: ColorMode): ChartThemeConfig {
  const c = PALETTE[mode].charts;
  return {
    seriesColors: [...c.series],
    gridColor: c.grid,
    axisColor: c.axis,
    tooltipBg: c.tooltipBg,
    heatmapScale: [...c.heatmapScale],
  };
}

export const generalTheme: ProductTheme = {
  id: "general",
  displayName: "Supply Chain Command Center",
  tagline: "End-to-End Supply Chain Intelligence",
  description:
    "Control-room indigo: a planner's control room with a strict semantic color language — the same concept always wears the same color.",
  supportedModes: ["light", "soft", "dark"],
  defaultMode: "light",
  palette: {
    light: PALETTE.light.core,
    soft: PALETTE.soft.core,
    dark: PALETTE.dark.core,
  },
  sidebar: {
    activeIndicator: "pill",
    iconStrokeWidth: 1.5,
    sectionLabelStyle: "uppercase",
    hoverEffect: "bg",
  },
  cards: {
    borderRadius: "0.625rem",
    shadow: "shadow-sm",
    borderStyle: "solid",
    hoverEffect: "none",
  },
  charts: {
    light: chartConfig("light"),
    soft: chartConfig("soft"),
    dark: chartConfig("dark"),
  },
  typography: {
    headingWeight: 600,
    headingTracking: "-0.01em",
    bodyWeight: 400,
    kpiWeight: 700,
    kpiTracking: "-0.02em",
  },
  logo: {
    icon: "BarChart3",
  },
};
