import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";

import { PipelineReadinessBanner } from "../PipelineReadinessBanner";
import { TestQueryWrapper } from "@/tabs/__tests__/test-utils";

const fetchPipelineReadiness = vi.fn();

vi.mock("@/api/queries", () => ({
  fetchPipelineReadiness: (...a: unknown[]) => fetchPipelineReadiness(...a),
  pipelineReadinessKeys: { readiness: ["dashboard", "pipeline-readiness"] },
}));

const navigateToTab = vi.fn();
vi.mock("@/lib/navigation", () => ({
  navigateToTab: (...a: unknown[]) => navigateToTab(...a),
}));

const STALE = {
  ready: false,
  checks: [
    {
      stage: "clustering",
      status: "stale",
      severity: "high",
      title: "Clustering needs to be re-run",
      detail: "No SKUs have a cluster assignment.",
      action: { kind: "navigate", target: "clusters", label: "Open Clustering" },
    },
  ],
};

describe("PipelineReadinessBanner", () => {
  beforeEach(() => {
    fetchPipelineReadiness.mockReset();
    navigateToTab.mockReset();
  });

  it("renders nothing when everything is in sync", async () => {
    fetchPipelineReadiness.mockResolvedValue({ ready: true, checks: [] });
    const { container } = render(<PipelineReadinessBanner />, { wrapper: TestQueryWrapper });
    await waitFor(() => expect(fetchPipelineReadiness).toHaveBeenCalled());
    expect(container.querySelector("[role='status']")).toBeNull();
  });

  it("shows a stale check with a deep-link action", async () => {
    fetchPipelineReadiness.mockResolvedValue(STALE);
    render(<PipelineReadinessBanner />, { wrapper: TestQueryWrapper });
    expect(await screen.findByText("Clustering needs to be re-run")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Open Clustering/ })).toBeInTheDocument();
  });

  it("navigates to the Clustering tab when the action is clicked", async () => {
    fetchPipelineReadiness.mockResolvedValue(STALE);
    render(<PipelineReadinessBanner />, { wrapper: TestQueryWrapper });
    fireEvent.click(await screen.findByRole("button", { name: /Open Clustering/ }));
    expect(navigateToTab).toHaveBeenCalledWith("clusters", "sku");
  });

  it("hides the banner when dismissed", async () => {
    fetchPipelineReadiness.mockResolvedValue(STALE);
    render(<PipelineReadinessBanner />, { wrapper: TestQueryWrapper });
    await screen.findByText("Clustering needs to be re-run");
    fireEvent.click(screen.getByRole("button", { name: "Dismiss" }));
    await waitFor(() =>
      expect(screen.queryByText("Clustering needs to be re-run")).toBeNull(),
    );
  });
});
