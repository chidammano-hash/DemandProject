// Tile variant determines the visual layout shape
export type TileVariant = "periodic" | "card" | "badge" | "emblem";

// Base fields every tile must supply regardless of motif
export interface TileConfigBase {
  primary: string;           // Big centered glyph (element symbol, emoji, kanji)
  superscript: string | number; // Top-right label (atomic number, year, etc)
  label: string;             // Bottom label below the glyph
  restClasses: string;       // Tailwind classes for inactive state
  activeClasses: string;     // Tailwind classes for active state
  glowClass: string;         // Tailwind shadow class for glow effect
  tagline?: string;          // One-line flavor text
}

export interface TileConfig extends TileConfigBase {
  variant: TileVariant;
}

export type AnimationName =
  | "pulse-glow"     // periodic: indigo box-shadow pulse
  | "orbit-spin"     // space: orbiting ring
  | "flame-flicker"  // f1: red/orange brightness flicker
  | "zen-breathe"    // zen: slow opacity scale
  | "pour-shimmer"   // spirits: shimmer sweep
  ;

export interface LoadingAnimationConfig {
  animationName: AnimationName;
  wrapperClasses: string;
  statusLabel?: string;
}

export interface MotifChrome {
  appName: string;
  appTagline: string;
  logoSvgPath: string | null;
  bgOverlay: string;
  tileRadius: string;
}

export type MotifId = "periodic" | "spirits" | "space" | "f1" | "zen";

export type ColorMode = "light" | "dark" | "midnight";

/**
 * Full UI palette: overrides the CSS custom properties in index.css.
 * HSL fields use the Tailwind convention: "H S% L%" (no hsl() wrapper).
 * Chart/KPI/gradient fields use hex or rgba strings.
 */
export interface MotifPalette {
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

export interface MotifThemeConfig {
  id: MotifId;
  displayName: string;
  description: string;
  previewTile: TileConfig;
  tiles: Record<string, TileConfig>;
  loading: LoadingAnimationConfig;
  chrome: MotifChrome;
  /** Per-color-mode CSS variable overrides. Omit for default (periodic). */
  palette?: Partial<Record<ColorMode, Partial<MotifPalette>>>;
}
