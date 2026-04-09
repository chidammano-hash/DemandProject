import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { DataFreshnessBanner } from "../DataFreshnessBanner";

describe("DataFreshnessBanner", () => {
  it("renders red/missing state when no lastRefreshed", () => {
    render(<DataFreshnessBanner source="Test Source" />);
    expect(screen.getByText(/Test Source: No data available/)).toBeInTheDocument();
  });

  it("renders green/fresh state for recent timestamp", () => {
    const recent = new Date(Date.now() - 30 * 1000).toISOString(); // 30 seconds ago
    render(<DataFreshnessBanner lastRefreshed={recent} source="Safety Stock" />);
    expect(screen.getByText(/Safety Stock: Last updated just now/)).toBeInTheDocument();
  });

  it("renders amber/stale state for old timestamp", () => {
    const old = new Date(Date.now() - 2 * 3600 * 1000).toISOString(); // 2 hours ago
    render(<DataFreshnessBanner lastRefreshed={old} staleSec={3600} source="Test" />);
    expect(screen.getByText(/Test: Last updated 2h ago \(stale\)/)).toBeInTheDocument();
  });

  it("renders warnings when provided", () => {
    const recent = new Date(Date.now() - 30 * 1000).toISOString();
    render(
      <DataFreshnessBanner
        lastRefreshed={recent}
        source="Projection"
        warnings={["Using fallback forecast (no ML model)"]}
      />,
    );
    expect(screen.getByText(/Using fallback forecast/)).toBeInTheDocument();
  });

  it("renders stale for null lastRefreshed", () => {
    render(<DataFreshnessBanner lastRefreshed={null} source="Queue" />);
    expect(screen.getByText(/Queue: No data available/)).toBeInTheDocument();
  });

  it("formats minutes ago correctly", () => {
    const fiveMinAgo = new Date(Date.now() - 5 * 60 * 1000).toISOString();
    render(<DataFreshnessBanner lastRefreshed={fiveMinAgo} staleSec={86400} />);
    expect(screen.getByText(/Last updated 5m ago/)).toBeInTheDocument();
  });

  it("formats days ago correctly", () => {
    const twoDaysAgo = new Date(Date.now() - 2 * 86400 * 1000).toISOString();
    render(<DataFreshnessBanner lastRefreshed={twoDaysAgo} staleSec={3600} />);
    expect(screen.getByText(/Last updated 2d ago/)).toBeInTheDocument();
  });
});
