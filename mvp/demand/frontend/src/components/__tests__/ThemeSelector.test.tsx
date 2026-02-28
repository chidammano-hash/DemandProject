import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { ThemeSelector } from "@/components/ThemeSelector";
import type { ColorMode } from "@/hooks/useTheme";

describe("ThemeSelector", () => {
  const defaultProps = {
    colorMode: "light" as ColorMode,
    onModeChange: vi.fn(),
    collapsed: false,
  };

  it("renders light, soft, and dark toggle buttons", () => {
    render(<ThemeSelector {...defaultProps} />);
    expect(screen.getByText("Light")).toBeInTheDocument();
    expect(screen.getByText("Soft")).toBeInTheDocument();
    expect(screen.getByText("Dark")).toBeInTheDocument();
  });

  it("renders Appearance label", () => {
    render(<ThemeSelector {...defaultProps} />);
    expect(screen.getByText("Appearance")).toBeInTheDocument();
  });

  it("clicking Light calls onModeChange('light')", () => {
    const onModeChange = vi.fn();
    render(<ThemeSelector {...defaultProps} onModeChange={onModeChange} colorMode="dark" />);
    fireEvent.click(screen.getByText("Light"));
    expect(onModeChange).toHaveBeenCalledWith("light");
  });

  it("clicking Soft calls onModeChange('soft')", () => {
    const onModeChange = vi.fn();
    render(<ThemeSelector {...defaultProps} onModeChange={onModeChange} colorMode="light" />);
    fireEvent.click(screen.getByText("Soft"));
    expect(onModeChange).toHaveBeenCalledWith("soft");
  });

  it("clicking Dark calls onModeChange('dark')", () => {
    const onModeChange = vi.fn();
    render(<ThemeSelector {...defaultProps} onModeChange={onModeChange} colorMode="light" />);
    fireEvent.click(screen.getByText("Dark"));
    expect(onModeChange).toHaveBeenCalledWith("dark");
  });

  it("collapsed mode renders a single button", () => {
    render(<ThemeSelector {...defaultProps} collapsed={true} />);
    const buttons = screen.getAllByRole("button");
    expect(buttons.length).toBe(1);
  });

  it("collapsed button cycles light → soft on click", () => {
    const onModeChange = vi.fn();
    render(
      <ThemeSelector
        {...defaultProps}
        collapsed={true}
        colorMode="light"
        onModeChange={onModeChange}
      />
    );
    fireEvent.click(screen.getByRole("button"));
    expect(onModeChange).toHaveBeenCalledWith("soft");
  });

  it("collapsed button cycles soft → dark on click", () => {
    const onModeChange = vi.fn();
    render(
      <ThemeSelector
        {...defaultProps}
        collapsed={true}
        colorMode="soft"
        onModeChange={onModeChange}
      />
    );
    fireEvent.click(screen.getByRole("button"));
    expect(onModeChange).toHaveBeenCalledWith("dark");
  });

  it("collapsed button cycles dark → light on click", () => {
    const onModeChange = vi.fn();
    render(
      <ThemeSelector
        {...defaultProps}
        collapsed={true}
        colorMode="dark"
        onModeChange={onModeChange}
      />
    );
    fireEvent.click(screen.getByRole("button"));
    expect(onModeChange).toHaveBeenCalledWith("light");
  });
});
