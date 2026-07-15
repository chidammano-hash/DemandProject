/**
 * Color math for the design-token system.
 *
 * Used by `constants/palette.ts` to derive HSL-triplet CSS vars from the hex
 * chart series, and by `paletteSync.test.ts` to enforce the WCAG contrast
 * gates. Pure functions, no DOM.
 */

/** "245 55% 48%" -> { h, s, l } (s/l as 0-1 fractions) */
export function parseHslTriplet(triplet: string): { h: number; s: number; l: number } {
  const m = triplet.trim().match(/^([\d.]+)\s+([\d.]+)%\s+([\d.]+)%$/);
  if (!m) throw new Error(`Invalid HSL triplet: "${triplet}"`);
  return { h: Number(m[1]), s: Number(m[2]) / 100, l: Number(m[3]) / 100 };
}

/** "#RRGGBB" -> { r, g, b } (0-1 fractions) */
export function parseHex(hex: string): { r: number; g: number; b: number } {
  const m = hex.trim().match(/^#([0-9a-fA-F]{6})$/);
  if (!m) throw new Error(`Invalid hex color: "${hex}"`);
  const n = parseInt(m[1], 16);
  return { r: ((n >> 16) & 0xff) / 255, g: ((n >> 8) & 0xff) / 255, b: (n & 0xff) / 255 };
}

function hslToRgb(h: number, s: number, l: number): { r: number; g: number; b: number } {
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const x = c * (1 - Math.abs(((h / 60) % 2) - 1));
  const m = l - c / 2;
  const sector = Math.floor(((h % 360) + 360) % 360 / 60);
  const [r, g, b] = [
    [c, x, 0], [x, c, 0], [0, c, x], [0, x, c], [x, 0, c], [c, 0, x],
  ][sector] ?? [0, 0, 0];
  return { r: r + m, g: g + m, b: b + m };
}

/** "#RRGGBB" -> "H S% L%" rounded to integers, for CSS-var emission. */
export function hexToHslTriplet(hex: string): string {
  const { r, g, b } = parseHex(hex);
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  const l = (max + min) / 2;
  const d = max - min;
  let h = 0;
  let s = 0;
  if (d !== 0) {
    s = d / (1 - Math.abs(2 * l - 1));
    if (max === r) h = 60 * (((g - b) / d) % 6);
    else if (max === g) h = 60 * ((b - r) / d + 2);
    else h = 60 * ((r - g) / d + 4);
  }
  if (h < 0) h += 360;
  return `${Math.round(h)} ${Math.round(s * 100)}% ${Math.round(l * 100)}%`;
}

function channelToLinear(c: number): number {
  return c <= 0.04045 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

function luminanceOfRgb({ r, g, b }: { r: number; g: number; b: number }): number {
  return 0.2126 * channelToLinear(r) + 0.7152 * channelToLinear(g) + 0.0722 * channelToLinear(b);
}

/** WCAG relative luminance for "#RRGGBB" or "H S% L%". */
export function relativeLuminance(color: string): number {
  if (color.startsWith("#")) return luminanceOfRgb(parseHex(color));
  const { h, s, l } = parseHslTriplet(color);
  return luminanceOfRgb(hslToRgb(h, s, l));
}

/** WCAG contrast ratio between two colors (hex or HSL triplet). */
export function contrastRatio(a: string, b: string): number {
  const la = relativeLuminance(a);
  const lb = relativeLuminance(b);
  const [hi, lo] = la >= lb ? [la, lb] : [lb, la];
  return (hi + 0.05) / (lo + 0.05);
}
