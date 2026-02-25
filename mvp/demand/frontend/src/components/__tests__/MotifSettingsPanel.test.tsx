import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";

// Register all motifs before importing the component that calls getAllMotifs().
import "@/constants/motifs";
import { MotifSettingsPanel } from "@/components/MotifSettingsPanel";

describe("MotifSettingsPanel", () => {
  it("renders all 5 motif options", () => {
    render(
      <MotifSettingsPanel currentMotifId="periodic" onSelect={vi.fn()} />,
    );

    // Each motif renders a button with an aria-label like "Switch to <displayName> theme style"
    expect(
      screen.getByLabelText("Switch to Periodic Table theme style"),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText("Switch to The Cellar theme style"),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText("Switch to Deep Space theme style"),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText("Switch to Formula 1 theme style"),
    ).toBeInTheDocument();
    expect(
      screen.getByLabelText("Switch to Zen Garden theme style"),
    ).toBeInTheDocument();
  });

  it("marks current motif as pressed (aria-pressed)", () => {
    render(
      <MotifSettingsPanel currentMotifId="spirits" onSelect={vi.fn()} />,
    );

    const spiritsBtn = screen.getByLabelText(
      "Switch to The Cellar theme style",
    );
    expect(spiritsBtn.getAttribute("aria-pressed")).toBe("true");

    const periodicBtn = screen.getByLabelText(
      "Switch to Periodic Table theme style",
    );
    expect(periodicBtn.getAttribute("aria-pressed")).toBe("false");

    const spaceBtn = screen.getByLabelText(
      "Switch to Deep Space theme style",
    );
    expect(spaceBtn.getAttribute("aria-pressed")).toBe("false");
  });

  it("calls onSelect when clicking a motif", () => {
    const onSelect = vi.fn();
    render(
      <MotifSettingsPanel currentMotifId="periodic" onSelect={onSelect} />,
    );

    const f1Btn = screen.getByLabelText("Switch to Formula 1 theme style");
    fireEvent.click(f1Btn);
    expect(onSelect).toHaveBeenCalledTimes(1);
    expect(onSelect).toHaveBeenCalledWith("f1");
  });

  it("renders display name and description for each motif", () => {
    render(
      <MotifSettingsPanel currentMotifId="periodic" onSelect={vi.fn()} />,
    );

    expect(screen.getByText("Periodic Table")).toBeInTheDocument();
    expect(screen.getByText("The Cellar")).toBeInTheDocument();
    expect(screen.getByText("Deep Space")).toBeInTheDocument();
    expect(screen.getByText("Formula 1")).toBeInTheDocument();
    expect(screen.getByText("Zen Garden")).toBeInTheDocument();
  });

  it("renders the Theme Style heading", () => {
    render(
      <MotifSettingsPanel currentMotifId="periodic" onSelect={vi.fn()} />,
    );

    expect(screen.getByText("Theme Style")).toBeInTheDocument();
  });
});
