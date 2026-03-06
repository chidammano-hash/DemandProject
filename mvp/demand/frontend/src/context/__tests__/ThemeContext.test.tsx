import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { renderHook } from "@testing-library/react";
import React from "react";
import { ThemeProvider, useThemeContext } from "@/context/ThemeContext";
import type { Theme } from "@/types";

function makeWrapper(theme: Theme) {
  return ({ children }: { children: React.ReactNode }) =>
    React.createElement(ThemeProvider, { value: { theme } }, children);
}

describe("ThemeProvider", () => {
  it("renders children without crashing", () => {
    render(
      <ThemeProvider value={{ theme: "light" }}>
        <div>child content</div>
      </ThemeProvider>
    );
    expect(screen.getByText("child content")).toBeInTheDocument();
  });

  it("renders multiple children", () => {
    render(
      <ThemeProvider value={{ theme: "dark" }}>
        <span>first</span>
        <span>second</span>
      </ThemeProvider>
    );
    expect(screen.getByText("first")).toBeInTheDocument();
    expect(screen.getByText("second")).toBeInTheDocument();
  });
});

describe("useThemeContext", () => {
  it("returns a defined context value within provider", () => {
    const { result } = renderHook(() => useThemeContext(), {
      wrapper: makeWrapper("light"),
    });
    expect(result.current).toBeDefined();
  });

  it("returns theme 'light' when provider is set to light", () => {
    const { result } = renderHook(() => useThemeContext(), {
      wrapper: makeWrapper("light"),
    });
    expect(result.current.theme).toBe("light");
  });

  it("returns theme 'dark' when provider is set to dark", () => {
    const { result } = renderHook(() => useThemeContext(), {
      wrapper: makeWrapper("dark"),
    });
    expect(result.current.theme).toBe("dark");
  });

  it("returns theme 'soft' when provider is set to soft", () => {
    const { result } = renderHook(() => useThemeContext(), {
      wrapper: makeWrapper("soft"),
    });
    expect(result.current.theme).toBe("soft");
  });

  it("throws an error when used outside ThemeProvider", () => {
    // renderHook without wrapper — no provider in context
    expect(() => {
      renderHook(() => useThemeContext());
    }).toThrow("useThemeContext must be used within ThemeProvider");
  });

  it("context value object has a 'theme' property", () => {
    const { result } = renderHook(() => useThemeContext(), {
      wrapper: makeWrapper("light"),
    });
    expect(Object.keys(result.current)).toContain("theme");
  });
});
