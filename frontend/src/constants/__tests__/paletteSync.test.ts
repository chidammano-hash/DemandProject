import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import { PALETTE, CSS_VAR_MAP, type ColorMode } from "@/constants/palette";
import { contrastRatio } from "@/lib/color";

/**
 * The permanent anti-drift gate for the design-token system:
 *  1. index.css fallback blocks mirror constants/palette.ts verbatim.
 *  2. The chart series stays 8 colors and every semantic role is a member.
 *  3. No raw hex inside the CSS token blocks (HSL triplets only; the
 *     bg-gradient base stops are the sanctioned exception).
 *  4. WCAG contrast gates hold in every mode.
 */

const MODES: ColorMode[] = ["light", "soft", "dark"];

const css = readFileSync(resolve(__dirname, "../../index.css"), "utf8");

/** Extract `--token: value;` pairs from the block that starts at `selector {`. */
function cssBlock(selector: string): Record<string, string> {
  const start = css.indexOf(`${selector} {`);
  expect(start, `selector "${selector}" present in index.css`).toBeGreaterThanOrEqual(0);
  const end = css.indexOf("}", start);
  const body = css.slice(start, end);
  const vars: Record<string, string> = {};
  for (const m of body.matchAll(/(--[\w-]+):\s*([^;]+);/g)) {
    vars[m[1]] = m[2].trim();
  }
  return vars;
}

const BLOCK_BY_MODE: Record<ColorMode, string> = {
  light: ":root",
  soft: ".soft",
  dark: ".dark",
};

describe("index.css fallbacks mirror PALETTE", () => {
  for (const mode of MODES) {
    it(`${BLOCK_BY_MODE[mode]} block matches PALETTE.${mode}.core`, () => {
      const block = cssBlock(BLOCK_BY_MODE[mode]);
      for (const [cssVar, key] of CSS_VAR_MAP) {
        expect(block[cssVar], `${cssVar} in ${BLOCK_BY_MODE[mode]}`).toBe(
          PALETTE[mode].core[key],
        );
      }
    });
  }

  it("token blocks contain no raw hex (HSL triplets only)", () => {
    for (const mode of MODES) {
      const block = cssBlock(BLOCK_BY_MODE[mode]);
      for (const [cssVar, value] of Object.entries(block)) {
        if (cssVar.startsWith("--bg-gradient")) continue; // sanctioned: gradient stops
        expect(value, `${cssVar} in ${BLOCK_BY_MODE[mode]}`).not.toMatch(/#[0-9a-fA-F]{3,8}/);
      }
    }
  });
});

describe("chart series contract", () => {
  for (const mode of MODES) {
    const { series, roles, heatmapScale, fallback } = PALETTE[mode].charts;

    it(`${mode}: series has exactly 8 colors, all hex`, () => {
      expect(series).toHaveLength(8);
      for (const c of series) expect(c).toMatch(/^#[0-9A-Fa-f]{6}$/);
      expect(new Set(series).size).toBe(8);
    });

    it(`${mode}: every semantic role is a member of the series`, () => {
      for (const [role, color] of Object.entries(roles)) {
        expect(series, `role "${role}"`).toContain(color);
      }
    });

    it(`${mode}: heatmap has 5 stops, fallback has 6`, () => {
      expect(heatmapScale).toHaveLength(5);
      expect(fallback).toHaveLength(6);
    });
  }
});

describe("WCAG contrast gates", () => {
  for (const mode of MODES) {
    const core = PALETTE[mode].core;
    const { series } = PALETTE[mode].charts;

    it(`${mode}: text tokens reach 4.5:1 on their surfaces`, () => {
      const pairs: Array<[string, string, string]> = [
        ["foreground/background", core.foreground, core.background],
        ["foreground/card", core.foreground, core.card],
        ["cardForeground/card", core.cardForeground, core.card],
        ["mutedForeground/background", core.mutedForeground, core.background],
        ["mutedForeground/card", core.mutedForeground, core.card],
        ["mutedForeground/muted", core.mutedForeground, core.muted],
        ["secondaryForeground/secondary", core.secondaryForeground, core.secondary],
        ["sidebarForeground/sidebarBg", core.sidebarForeground, core.sidebarBg],
        ["primaryForeground/primary", core.primaryForeground, core.primary],
        ["destructiveForeground/destructive", core.destructiveForeground, core.destructive],
        ["successForeground/success", core.successForeground, core.success],
        ["warningForeground/warning", core.warningForeground, core.warning],
        ["infoForeground/info", core.infoForeground, core.info],
        ["severityHighForeground/severityHigh", core.severityHighForeground, core.severityHigh],
      ];
      for (const [label, fg, bg] of pairs) {
        expect(
          contrastRatio(fg, bg),
          `${mode} ${label} >= 4.5:1`,
        ).toBeGreaterThanOrEqual(4.5);
      }
    });

    it(`${mode}: semantic accents reach 3:1 as large text/graphics on both surfaces`, () => {
      const accents: Array<[string, string]> = [
        ["primary", core.primary],
        ["destructive", core.destructive],
        ["success", core.success],
        ["warning", core.warning],
        ["info", core.info],
        ["severityHigh", core.severityHigh],
        ["kpiBest", core.kpiBest],
        ["kpiWarning", core.kpiWarning],
        ["kpiCeiling", core.kpiCeiling],
        ["sidebarActive", core.sidebarActive],
      ];
      for (const [label, color] of accents) {
        expect(
          contrastRatio(color, core.background),
          `${mode} ${label}/background >= 3:1`,
        ).toBeGreaterThanOrEqual(3);
        expect(
          contrastRatio(color, core.card),
          `${mode} ${label}/card >= 3:1`,
        ).toBeGreaterThanOrEqual(3);
      }
    });

    it(`${mode}: all 8 chart series colors reach 3:1 vs background and card`, () => {
      for (const [i, color] of series.entries()) {
        expect(
          contrastRatio(color, core.background),
          `${mode} series[${i}] ${color} vs background >= 3:1`,
        ).toBeGreaterThanOrEqual(3);
        expect(
          contrastRatio(color, core.card),
          `${mode} series[${i}] ${color} vs card >= 3:1`,
        ).toBeGreaterThanOrEqual(3);
      }
    });

    it(`${mode}: series pairs stay distinguishable (contrast or hue separation)`, () => {
      // Two series colors are "confusable" when they have BOTH near-equal
      // luminance (contrast < 1.2) AND near-equal hue (< 30 degrees apart).
      const hueOf = (hex: string) => {
        const n = parseInt(hex.slice(1), 16);
        const r = ((n >> 16) & 0xff) / 255;
        const g = ((n >> 8) & 0xff) / 255;
        const b = (n & 0xff) / 255;
        const max = Math.max(r, g, b);
        const min = Math.min(r, g, b);
        const d = max - min;
        if (d === 0) return 0;
        let h: number;
        if (max === r) h = 60 * (((g - b) / d) % 6);
        else if (max === g) h = 60 * ((b - r) / d + 2);
        else h = 60 * ((r - g) / d + 4);
        return (h + 360) % 360;
      };
      for (let i = 0; i < series.length; i++) {
        for (let j = i + 1; j < series.length; j++) {
          const lumSep = contrastRatio(series[i], series[j]);
          let hueSep = Math.abs(hueOf(series[i]) - hueOf(series[j]));
          if (hueSep > 180) hueSep = 360 - hueSep;
          expect(
            lumSep >= 1.2 || hueSep >= 30,
            `${mode} series[${i}] ${series[i]} vs series[${j}] ${series[j]} (contrast ${lumSep.toFixed(2)}, hue ${hueSep.toFixed(0)}deg)`,
          ).toBe(true);
        }
      }
    });
  }
});
