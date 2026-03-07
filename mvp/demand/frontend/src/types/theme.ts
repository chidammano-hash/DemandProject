// ---------------------------------------------------------------------------
// Product theme type system (Feature 36)
// ---------------------------------------------------------------------------

export type ProductThemeId = "general";

export interface ThemePalette {
  background: string;
  foreground: string;
  card: string;
  cardForeground: string;
  primary: string;
  primaryForeground: string;
  secondary: string;
  secondaryForeground: string;
  muted: string;
  mutedForeground: string;
  accent: string;
  accentForeground: string;
  border: string;
  input: string;
  ring: string;
  destructive: string;
  destructiveForeground: string;
  sidebarBg: string;
  sidebarForeground: string;
  sidebarActive: string;
  sidebarHover: string;
  chart1: string;
  chart2: string;
  chart3: string;
  chart4: string;
  chart5: string;
  chart6: string;
  kpiBest: string;
  kpiWarning: string;
  kpiCeiling: string;
  bgGradientPrimary: string;
  bgGradientSecondary: string;
  bgGradientBaseStart: string;
  bgGradientBaseMid: string;
  bgGradientBaseEnd: string;
}

export interface SidebarThemeConfig {
  activeIndicator: "bar" | "pill" | "glow";
  iconStrokeWidth: 1 | 1.5 | 2;
  sectionLabelStyle: "uppercase" | "capitalize" | "hidden";
  hoverEffect: "bg" | "glow" | "subtle";
}

export interface CardThemeConfig {
  borderRadius: string;
  shadow: string;
  borderStyle: "solid" | "none" | "subtle";
  hoverEffect: "lift" | "glow" | "none";
}

export interface ChartThemeConfig {
  seriesColors: string[];
  gridColor: string;
  axisColor: string;
  tooltipBg: string;
  heatmapScale: string[];
}

export interface TypographyConfig {
  headingWeight: 500 | 600 | 700;
  headingTracking: string;
  bodyWeight: 400;
  kpiWeight: 700;
  kpiTracking: string;
}

export interface ProductTheme {
  id: ProductThemeId;
  displayName: string;
  tagline: string;
  description: string;
  supportedModes: ("light" | "soft" | "dark")[];
  defaultMode: "light" | "soft" | "dark";
  palette: {
    light?: ThemePalette;
    soft?: ThemePalette;
    dark: ThemePalette;
  };
  sidebar: SidebarThemeConfig;
  cards: CardThemeConfig;
  charts: {
    light?: ChartThemeConfig;
    soft?: ChartThemeConfig;
    dark: ChartThemeConfig;
  };
  typography: TypographyConfig;
  logo: {
    icon: string;
    gradient?: string;
  };
}

// Sidebar navigation types
export type SidebarSection = "overview" | "demand" | "supply" | "intelligence" | "system";

export interface SidebarItem {
  key: string;
  label: string;
  icon: string; // Lucide icon name
  section: SidebarSection;
  shortcut?: string;
}

// Global filter types
export interface GlobalFilters {
  brand: string[];
  category: string[];
  market: string[];
  channel: string[];
  item: string[];
  location: string[];
  timeGrain: "month" | "quarter";
}

// Dashboard types
export interface DashboardKpis {
  accuracy_pct: number | null;
  wape_pct: number | null;
  bias_pct: number | null;
  total_forecast: number | null;
  total_actual: number | null;
  weeks_of_supply: number | null;
  window_months: number;
  deltas: {
    accuracy_pct: number | null;
    wape_pct: number | null;
    bias_pct: number | null;
  };
}

export type AlertType = "oos_risk" | "bias_drift" | "low_accuracy" | "demand_spike" | "allocation_shortage" | "scenario_complete" | "job_complete";
export type AlertSeverity = "critical" | "high" | "medium" | "low";

export interface Alert {
  id: string;
  type: AlertType;
  severity: AlertSeverity;
  title: string;
  detail: string;
  count?: number;
  source_tab?: string; // PL-013: destination tab when alert is clicked
}

export interface Mover {
  item_description: string;
  delta: number;
  pct_change: number;
  direction: "up" | "down";
}

export interface HeatmapRow {
  label: string;
  values: number[];
}

export interface DistinctValuesPayload {
  column: string;
  values: string[];
  total: number;
}
