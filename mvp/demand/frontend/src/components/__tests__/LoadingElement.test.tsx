import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import type { ReactNode } from "react";

// Register all motifs before any usage.
import "@/constants/motifs";
import { getMotif } from "@/constants/motifs";
import { MotifProvider } from "@/context/MotifContext";
import { LoadingElement } from "@/components/LoadingElement";
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

describe("LoadingElement", () => {
  it("renders with tabKey inside MotifProvider", () => {
    render(
      <MotifWrapper>
        <LoadingElement tabKey="explorer" />
      </MotifWrapper>,
    );

    // The periodic motif explorer tile: primary="Dx"
    expect(screen.getByText("Dx")).toBeInTheDocument();
    // The periodic motif loading statusLabel is "Loading"
    expect(screen.getByText("Loading")).toBeInTheDocument();
  });

  it("renders with legacy config prop without MotifProvider", () => {
    const legacyConfig = {
      symbol: "Lg",
      number: 99,
      name: "Legacy",
      color: "bg-gray-100 text-gray-800 border-gray-200",
      activeColor: "bg-gray-200 text-gray-900 border-gray-300",
      glow: "shadow-md",
    };

    // Rendering without MotifProvider -- LoadingElement catches the error
    // from useMotif and falls back to legacy config.
    render(<LoadingElement config={legacyConfig} />);

    expect(screen.getByText("Lg")).toBeInTheDocument();
    expect(screen.getByText("99")).toBeInTheDocument();
    // Without a MotifProvider, statusLabel defaults to "Loading"
    expect(screen.getByText("Loading")).toBeInTheDocument();
  });

  it("shows overlay when overlay prop is true", () => {
    const { container } = render(
      <MotifWrapper>
        <LoadingElement tabKey="accuracy" overlay />
      </MotifWrapper>,
    );

    // The overlay wrapper has position absolute and backdrop-blur
    const overlayDiv = container.querySelector(".absolute.inset-0");
    expect(overlayDiv).toBeInTheDocument();
  });

  it("does not show overlay wrapper when overlay is false/undefined", () => {
    const { container } = render(
      <MotifWrapper>
        <LoadingElement tabKey="accuracy" />
      </MotifWrapper>,
    );

    const overlayDiv = container.querySelector(".absolute.inset-0");
    expect(overlayDiv).not.toBeInTheDocument();
  });

  it("shows message when provided", () => {
    render(
      <MotifWrapper>
        <LoadingElement tabKey="sales" message="Fetching data..." />
      </MotifWrapper>,
    );

    expect(screen.getByText("Fetching data...")).toBeInTheDocument();
  });

  it("does not show message when not provided", () => {
    render(
      <MotifWrapper>
        <LoadingElement tabKey="sales" />
      </MotifWrapper>,
    );

    // No extra text element beyond the tile content
    expect(screen.queryByText("Fetching data...")).not.toBeInTheDocument();
  });

  it("uses motif-specific animation and status label", () => {
    // Create a wrapper with the "f1" motif to verify motif-specific values
    const f1Config = getMotif("f1");
    const f1Value: UseMotifThemeReturn = {
      motifId: "f1",
      motifConfig: f1Config,
      setMotif: vi.fn(),
      cycleMotif: vi.fn(),
      getTile: (key: string): TileConfig =>
        f1Config.tiles[key] ?? f1Config.previewTile,
    };

    function F1Wrapper({ children }: { children: ReactNode }) {
      return <MotifProvider value={f1Value}>{children}</MotifProvider>;
    }

    render(
      <F1Wrapper>
        <LoadingElement tabKey="explorer" />
      </F1Wrapper>,
    );

    // F1 motif: explorer primary is "Q3", statusLabel is "Lights Out"
    expect(screen.getByText("Q3")).toBeInTheDocument();
    expect(screen.getByText("Lights Out")).toBeInTheDocument();
  });

  it("renders different tile content for different tab keys", () => {
    render(
      <MotifWrapper>
        <LoadingElement tabKey="clusters" />
      </MotifWrapper>,
    );

    // Periodic motif: clusters tile primary="Cl"
    expect(screen.getByText("Cl")).toBeInTheDocument();
  });
});
