import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import { LoadingElement } from "@/components/LoadingElement";

describe("LoadingElement", () => {
  it("renders with default values when no config provided", () => {
    render(<LoadingElement />);

    expect(screen.getByText("?")).toBeInTheDocument();
    expect(screen.getByText("Loading")).toBeInTheDocument();
  });

  it("renders with legacy config prop", () => {
    const legacyConfig = {
      symbol: "Lg",
      number: 99,
      name: "Legacy",
      color: "bg-gray-100 text-gray-800 border-gray-200",
      activeColor: "bg-gray-200 text-gray-900 border-gray-300",
      glow: "shadow-md",
    };

    render(<LoadingElement config={legacyConfig} />);

    expect(screen.getByText("Lg")).toBeInTheDocument();
    expect(screen.getByText("99")).toBeInTheDocument();
    expect(screen.getByText("Loading")).toBeInTheDocument();
  });

  it("shows overlay when overlay prop is true", () => {
    const { container } = render(<LoadingElement overlay />);

    const overlayDiv = container.querySelector(".absolute.inset-0");
    expect(overlayDiv).toBeInTheDocument();
  });

  it("does not show overlay wrapper when overlay is false/undefined", () => {
    const { container } = render(<LoadingElement />);

    const overlayDiv = container.querySelector(".absolute.inset-0");
    expect(overlayDiv).not.toBeInTheDocument();
  });

  it("shows message when provided", () => {
    render(<LoadingElement message="Fetching data..." />);

    expect(screen.getByText("Fetching data...")).toBeInTheDocument();
  });

  it("does not show message when not provided", () => {
    render(<LoadingElement />);

    expect(screen.queryByText("Fetching data...")).not.toBeInTheDocument();
  });

  it("applies pulse-glow animation class", () => {
    const { container } = render(<LoadingElement />);

    const animatedEl = container.querySelector(".animate-pulse-glow");
    expect(animatedEl).toBeInTheDocument();
  });
});
