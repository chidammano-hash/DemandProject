import { describe, it, expect } from "vitest";
import { renderHook } from "@testing-library/react";
import React from "react";
import { ThemeProvider } from "@/context/ThemeContext";
import { PALETTE } from "@/constants/palette";
import type { Theme } from "@/types";
import { buildEChartsThemeDefaults, useEChartsThemeDefaults } from "@/components/echartsTheme";
import { useChartColors } from "@/hooks/useChartColors";

function makeWrapper(theme: Theme) {
  return ({ children }: { children: React.ReactNode }) =>
    React.createElement(ThemeProvider, { value: { theme }, children });
}

describe("buildEChartsThemeDefaults", () => {
  it("derives color/axis/split-line/tooltip defaults from the light palette", () => {
    const { result } = renderHook(() => useChartColors(), { wrapper: makeWrapper("light") });
    const defaults = buildEChartsThemeDefaults(result.current);

    const light = PALETTE.light.charts;
    expect(defaults.color).toEqual([...light.series]);
    expect(defaults.textStyle.color).toBe(light.axis);
    expect(defaults.axisLine.lineStyle.color).toBe(light.axis);
    expect(defaults.axisLabel.color).toBe(light.axis);
    expect(defaults.splitLine.lineStyle.color).toBe(light.grid);
    expect(defaults.tooltip.backgroundColor).toBe(light.tooltipBg);
    expect(defaults.tooltip.borderColor).toBe(light.tooltipBorder);
    expect(defaults.tooltip.textStyle.color).toBe(light.axis);
  });

  it("derives distinct defaults per mode (dark differs from light)", () => {
    const { result: lightResult } = renderHook(() => useChartColors(), { wrapper: makeWrapper("light") });
    const { result: darkResult } = renderHook(() => useChartColors(), { wrapper: makeWrapper("dark") });

    const lightDefaults = buildEChartsThemeDefaults(lightResult.current);
    const darkDefaults = buildEChartsThemeDefaults(darkResult.current);

    expect(lightDefaults.color).not.toEqual(darkDefaults.color);
    expect(lightDefaults.textStyle.color).not.toBe(darkDefaults.textStyle.color);
    expect(lightDefaults.splitLine.lineStyle.color).not.toBe(darkDefaults.splitLine.lineStyle.color);
  });

  it("matches the soft-mode palette", () => {
    const { result } = renderHook(() => useChartColors(), { wrapper: makeWrapper("soft") });
    const defaults = buildEChartsThemeDefaults(result.current);
    const soft = PALETTE.soft.charts;

    expect(defaults.color).toEqual([...soft.series]);
    expect(defaults.tooltip.backgroundColor).toBe(soft.tooltipBg);
  });
});

describe("useEChartsThemeDefaults", () => {
  it("returns the same shape as the pure builder for the active theme", () => {
    const { result } = renderHook(() => useEChartsThemeDefaults(), { wrapper: makeWrapper("dark") });
    const dark = PALETTE.dark.charts;

    expect(result.current.color).toEqual([...dark.series]);
    expect(result.current.axisLine.lineStyle.color).toBe(dark.axis);
    expect(result.current.tooltip.borderColor).toBe(dark.tooltipBorder);
  });
});
