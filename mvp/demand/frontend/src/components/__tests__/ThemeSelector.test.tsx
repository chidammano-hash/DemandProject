import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ThemeSelector } from "@/components/ThemeSelector";
import type { ProductThemeId } from "@/types/theme";
import type { ColorMode } from "@/hooks/useTheme";

describe("ThemeSelector", () => {
  const defaultProps = {
    themeId: "general" as ProductThemeId,
    colorMode: "light" as ColorMode,
    onThemeChange: vi.fn(),
    onModeChange: vi.fn(),
    collapsed: false,
  };

  it("renders 3 theme options", () => {
    render(<ThemeSelector {...defaultProps} />);
    const radioButtons = screen.getAllByRole("radio");
    expect(radioButtons.length).toBe(3);
  });

  it("selected theme has aria-checked", () => {
    render(<ThemeSelector {...defaultProps} themeId="general" />);
    const radioButtons = screen.getAllByRole("radio");
    // Find the one that is checked
    const checkedRadio = radioButtons.find(
      (btn) => btn.getAttribute("aria-checked") === "true"
    );
    expect(checkedRadio).toBeDefined();
    // The other two should not be checked
    const unchecked = radioButtons.filter(
      (btn) => btn.getAttribute("aria-checked") !== "true"
    );
    expect(unchecked.length).toBe(2);
  });

  it("clicking a theme calls onThemeChange", () => {
    const onThemeChange = vi.fn();
    render(<ThemeSelector {...defaultProps} onThemeChange={onThemeChange} />);
    const radioButtons = screen.getAllByRole("radio");
    // Click the first radio (wine-spirits is first in THEME_ORDER)
    fireEvent.click(radioButtons[0]);
    expect(onThemeChange).toHaveBeenCalledTimes(1);
  });

  it("mode toggle buttons work - clicking Light calls onModeChange('light')", () => {
    const onModeChange = vi.fn();
    render(<ThemeSelector {...defaultProps} onModeChange={onModeChange} colorMode="dark" />);
    fireEvent.click(screen.getByText("Light"));
    expect(onModeChange).toHaveBeenCalledWith("light");
  });

  it("mode toggle buttons work - clicking Dark calls onModeChange('dark')", () => {
    const onModeChange = vi.fn();
    render(<ThemeSelector {...defaultProps} onModeChange={onModeChange} colorMode="light" />);
    fireEvent.click(screen.getByText("Dark"));
    expect(onModeChange).toHaveBeenCalledWith("dark");
  });

  it("obsidian disables mode toggle", () => {
    render(<ThemeSelector {...defaultProps} themeId="obsidian" colorMode="dark" />);
    const lightBtn = screen.getByText("Light").closest("button");
    const darkBtn = screen.getByText("Dark").closest("button");
    expect(lightBtn).toBeDisabled();
    expect(darkBtn).toBeDisabled();
  });

  it("renders radiogroup with accessible label", () => {
    render(<ThemeSelector {...defaultProps} />);
    expect(screen.getByRole("radiogroup", { name: "Select theme" })).toBeInTheDocument();
  });

  it("collapsed mode renders a single button instead of radiogroup", () => {
    render(<ThemeSelector {...defaultProps} collapsed={true} />);
    // In collapsed mode there should be no radiogroup
    expect(screen.queryByRole("radiogroup")).not.toBeInTheDocument();
    // There should be a single button with a title
    const buttons = screen.getAllByRole("button");
    expect(buttons.length).toBe(1);
  });

  it("collapsed button cycles theme on click", () => {
    const onThemeChange = vi.fn();
    render(
      <ThemeSelector
        {...defaultProps}
        collapsed={true}
        themeId="general"
        onThemeChange={onThemeChange}
      />
    );
    const button = screen.getByRole("button");
    fireEvent.click(button);
    // general is at index 1 in THEME_ORDER [wine-spirits, general, obsidian]
    // So (1+1) % 3 = 2, which is "obsidian"
    expect(onThemeChange).toHaveBeenCalledWith("obsidian");
  });
});
