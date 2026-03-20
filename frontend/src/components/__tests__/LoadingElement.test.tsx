import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";

import { LoadingElement } from "@/components/LoadingElement";

describe("LoadingElement", () => {
  it("renders spinner without crashing", () => {
    const { container } = render(<LoadingElement />);
    const spinner = container.querySelector(".animate-spin");
    expect(spinner).toBeInTheDocument();
  });

  it("accepts legacy config prop without error", () => {
    const legacyConfig = { symbol: "Lg", number: 99, name: "Legacy", color: "", activeColor: "", glow: "" };
    // Should not throw — config is accepted but ignored
    render(<LoadingElement config={legacyConfig} />);
    const { container } = render(<LoadingElement config={legacyConfig} />);
    expect(container.querySelector(".animate-spin")).toBeInTheDocument();
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

  it("renders larger spinner for md size", () => {
    const { container } = render(<LoadingElement size="md" />);
    const spinner = container.querySelector(".h-8.w-8");
    expect(spinner).toBeInTheDocument();
  });
});
