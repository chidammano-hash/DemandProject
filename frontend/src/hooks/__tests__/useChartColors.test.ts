import { describe, it, expect } from "vitest";
import { renderHook } from "@testing-library/react";
import React from "react";
import { useChartColors } from "@/hooks/useChartColors";
import { ThemeProvider } from "@/context/ThemeContext";
import type { Theme } from "@/types";

function makeWrapper(theme: Theme) {
  return ({ children }: { children: React.ReactNode }) =>
    React.createElement(ThemeProvider, { value: { theme }, children });
}

describe("useChartColors", () => {
  it("returns chartColors object with expected keys for light theme", () => {
    const { result } = renderHook(() => useChartColors(), {
      wrapper: makeWrapper("light"),
    });
    const { chartColors } = result.current;
    expect(chartColors).toBeDefined();
    expect(chartColors).toHaveProperty("grid");
    expect(chartColors).toHaveProperty("axis");
    expect(chartColors).toHaveProperty("tooltip_bg");
    expect(chartColors).toHaveProperty("tooltip_border");
  });

  it("returns trendColors array with expected length for light theme", () => {
    const { result } = renderHook(() => useChartColors(), {
      wrapper: makeWrapper("light"),
    });
    const { trendColors } = result.current;
    expect(trendColors).toBeDefined();
    expect(Array.isArray(trendColors)).toBe(true);
    expect(trendColors.length).toBeGreaterThan(0);
  });

  it("returns the theme from ThemeContext", () => {
    const { result } = renderHook(() => useChartColors(), {
      wrapper: makeWrapper("dark"),
    });
    expect(result.current.theme).toBe("dark");
  });

  it("chartColors values are non-empty strings (valid color strings)", () => {
    const { result } = renderHook(() => useChartColors(), {
      wrapper: makeWrapper("light"),
    });
    const { chartColors } = result.current;
    for (const val of Object.values(chartColors)) {
      expect(typeof val).toBe("string");
      expect(val.length).toBeGreaterThan(0);
    }
  });

  it("trendColors entries are non-empty strings", () => {
    const { result } = renderHook(() => useChartColors(), {
      wrapper: makeWrapper("light"),
    });
    for (const color of result.current.trendColors) {
      expect(typeof color).toBe("string");
      expect(color.length).toBeGreaterThan(0);
    }
  });

  it("dark theme returns different chartColors than light theme", () => {
    const { result: lightResult } = renderHook(() => useChartColors(), {
      wrapper: makeWrapper("light"),
    });
    const { result: darkResult } = renderHook(() => useChartColors(), {
      wrapper: makeWrapper("dark"),
    });
    expect(lightResult.current.chartColors.grid).not.toBe(darkResult.current.chartColors.grid);
  });

  it("dark theme returns different trendColors than light theme", () => {
    const { result: lightResult } = renderHook(() => useChartColors(), {
      wrapper: makeWrapper("light"),
    });
    const { result: darkResult } = renderHook(() => useChartColors(), {
      wrapper: makeWrapper("dark"),
    });
    expect(lightResult.current.trendColors).not.toEqual(darkResult.current.trendColors);
  });

  it("soft theme returns theme value of 'soft'", () => {
    const { result } = renderHook(() => useChartColors(), {
      wrapper: makeWrapper("soft"),
    });
    expect(result.current.theme).toBe("soft");
  });

  it("soft theme returns its own distinct chartColors", () => {
    const { result: softResult } = renderHook(() => useChartColors(), {
      wrapper: makeWrapper("soft"),
    });
    const { result: lightResult } = renderHook(() => useChartColors(), {
      wrapper: makeWrapper("light"),
    });
    expect(softResult.current.chartColors.grid).not.toBe(lightResult.current.chartColors.grid);
  });

  it("trendColors has 8 entries for all themes (the categorical series)", () => {
    for (const theme of ["light", "dark", "soft"] as Theme[]) {
      const { result } = renderHook(() => useChartColors(), {
        wrapper: makeWrapper(theme),
      });
      expect(result.current.trendColors).toHaveLength(8);
    }
  });

  it("exposes the mode-tuned categorical palette as okabeIto (compat alias)", () => {
    const { result } = renderHook(() => useChartColors(), {
      wrapper: makeWrapper("light"),
    });
    const { okabeIto, series } = result.current;
    expect(Array.isArray(okabeIto)).toBe(true);
    expect(okabeIto).toHaveLength(8);
    expect(okabeIto).toEqual(series);
  });

  it("okabeIto/series are mode-tuned (dark differs from light)", () => {
    const themes: Theme[] = ["light", "dark"];
    const [light, dark] = themes.map((t) => {
      const { result } = renderHook(() => useChartColors(), {
        wrapper: makeWrapper(t),
      });
      return result.current.okabeIto;
    });
    expect(light).not.toEqual(dark);
  });

  it("roles carry the fixed semantic contract and are members of the series", () => {
    for (const theme of ["light", "dark", "soft"] as Theme[]) {
      const { result } = renderHook(() => useChartColors(), {
        wrapper: makeWrapper(theme),
      });
      const { roles, series } = result.current;
      expect(roles.forecast).toBe(roles.champion);
      for (const color of Object.values(roles)) {
        expect(series).toContain(color);
      }
    }
  });

  it("heatmap has 5 stops in every theme", () => {
    for (const theme of ["light", "dark", "soft"] as Theme[]) {
      const { result } = renderHook(() => useChartColors(), {
        wrapper: makeWrapper(theme),
      });
      expect(result.current.heatmap).toHaveLength(5);
    }
  });
});
