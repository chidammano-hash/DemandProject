import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import type { ReactNode } from "react";

// Register all motifs before any usage.
import "@/constants/motifs";
import { getMotif } from "@/constants/motifs";
import { MotifProvider } from "@/context/MotifContext";
import { ElementTab } from "@/components/ElementTab";
import type { UseMotifThemeReturn } from "@/hooks/useMotifTheme";
import type { TileConfig } from "@/types/motif";

/**
 * Wrapper that provides a MotifContext with the "periodic" motif for testing
 * components that call useMotif().
 */
function MotifWrapper({ children }: { children: ReactNode }) {
  const motifConfig = getMotif("periodic");
  const value: UseMotifThemeReturn = {
    motifId: "periodic",
    motifConfig,
    setMotif: vi.fn(),
    cycleMotif: vi.fn(),
    getTile: (key: string): TileConfig =>
      motifConfig.tiles[key] ?? motifConfig.previewTile,
  };
  return <MotifProvider value={value}>{children}</MotifProvider>;
}

describe("ElementTab", () => {
  it("renders with tabKey using motif context", () => {
    render(
      <MotifWrapper>
        <ElementTab tabKey="explorer" isActive={false} onClick={vi.fn()} />
      </MotifWrapper>,
    );

    // The periodic motif explorer tile: primary="Dx", superscript=1, label="Explorer"
    expect(screen.getByText("Dx")).toBeInTheDocument();
    expect(screen.getByText("1")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Explorer" })).toBeInTheDocument();
  });

  it("renders with legacy config prop", () => {
    const legacyConfig = {
      symbol: "Lg",
      number: 42,
      name: "Legacy",
      color: "bg-gray-100 text-gray-800 border-gray-200",
      activeColor: "bg-gray-200 text-gray-900 border-gray-300",
      glow: "shadow-md",
    };

    render(
      <MotifWrapper>
        <ElementTab
          config={legacyConfig}
          isActive={false}
          onClick={vi.fn()}
        />
      </MotifWrapper>,
    );

    expect(screen.getByText("Lg")).toBeInTheDocument();
    expect(screen.getByText("42")).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: "Legacy" })).toBeInTheDocument();
  });

  it("shows active indicator when isActive is true", () => {
    const { container } = render(
      <MotifWrapper>
        <ElementTab tabKey="accuracy" isActive={true} onClick={vi.fn()} />
      </MotifWrapper>,
    );

    const tab = screen.getByRole("tab");
    expect(tab.getAttribute("aria-selected")).toBe("true");
    // The active indicator is a span with specific classes inside the button
    const indicator = container.querySelector(
      "span.absolute.-bottom-1",
    );
    expect(indicator).toBeInTheDocument();
  });

  it("does not show active indicator when isActive is false", () => {
    const { container } = render(
      <MotifWrapper>
        <ElementTab tabKey="accuracy" isActive={false} onClick={vi.fn()} />
      </MotifWrapper>,
    );

    const tab = screen.getByRole("tab");
    expect(tab.getAttribute("aria-selected")).toBe("false");
    const indicator = container.querySelector(
      "span.absolute.-bottom-1",
    );
    expect(indicator).not.toBeInTheDocument();
  });

  it("calls onClick when clicked", () => {
    const onClick = vi.fn();
    render(
      <MotifWrapper>
        <ElementTab tabKey="clusters" isActive={false} onClick={onClick} />
      </MotifWrapper>,
    );

    fireEvent.click(screen.getByRole("tab"));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("renders different motif tile content for different tab keys", () => {
    const { rerender } = render(
      <MotifWrapper>
        <ElementTab tabKey="sales" isActive={false} onClick={vi.fn()} />
      </MotifWrapper>,
    );

    // Periodic motif: sales tile has primary="Sa"
    expect(screen.getByText("Sa")).toBeInTheDocument();

    rerender(
      <MotifWrapper>
        <ElementTab tabKey="forecast" isActive={false} onClick={vi.fn()} />
      </MotifWrapper>,
    );

    // Periodic motif: forecast tile has primary="Fc"
    expect(screen.getByText("Fc")).toBeInTheDocument();
  });
});
