import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

// ---------------------------------------------------------------------------
// Mock global fetch
// ---------------------------------------------------------------------------
const mockSummary = {
  total_open: 14,
  total_investigating: 5,
  avg_severity: 0.62,
  by_type: [
    { exception_type: "stockout_risk", open_count: 6, avg_severity: 0.8 },
    { exception_type: "forecast_bias", open_count: 4, avg_severity: 0.5 },
  ],
  top_items: [
    { item_no: "ITEM001", loc: "LOC1", exception_count: 3 },
  ],
};

const mockList = {
  total: 2,
  rows: [
    {
      exception_id: "EXC-001",
      exception_type: "stockout_risk",
      item_no: "ITEM001",
      loc: "LOC1",
      severity: 0.85,
      financial_impact: 12500,
      headline: "Stockout risk detected for high-velocity item",
      supporting_data: { current_dos: 2.5, target_dos: 15 },
      status: "open",
      assigned_to: null,
      generated_at: "2026-03-01T00:00:00",
      expires_at: null,
      month_start: "2026-03-01",
    },
    {
      exception_id: "EXC-002",
      exception_type: "forecast_bias",
      item_no: "ITEM002",
      loc: "LOC2",
      severity: 0.45,
      financial_impact: null,
      headline: "Systematic over-forecast detected",
      supporting_data: null,
      status: "investigating",
      assigned_to: "planner@example.com",
      generated_at: "2026-03-02T00:00:00",
      expires_at: null,
      month_start: "2026-03-01",
    },
  ],
};

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn((url: string) => {
      if (url.includes("/summary")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockSummary),
        });
      }
      if (url.includes("/storyboard/exceptions")) {
        return Promise.resolve({
          ok: true,
          json: () => Promise.resolve(mockList),
        });
      }
      return Promise.resolve({
        ok: true,
        json: () => Promise.resolve({}),
      });
    })
  );
});

const StoryboardTab = (await import("@/tabs/StoryboardTab")).default;

describe("StoryboardTab", () => {
  it("renders without crashing", async () => {
    render(
      <TestQueryWrapper>
        <StoryboardTab />
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(document.body.textContent).toBeDefined();
    });
  });

  it("renders Storyboard heading", async () => {
    render(
      <TestQueryWrapper>
        <StoryboardTab />
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText(/Exception Triage/i)).toBeDefined();
    });
  });

  it("renders summary KPI cards after data loads", async () => {
    render(
      <TestQueryWrapper>
        <StoryboardTab />
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText(/Total Open/i)).toBeDefined();
    });
    await waitFor(() => {
      expect(screen.getAllByText(/Investigating/i).length).toBeGreaterThan(0);
    });
  });

  it("renders exception queue section", async () => {
    render(
      <TestQueryWrapper>
        <StoryboardTab />
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(screen.getByText(/Exception Queue/i)).toBeDefined();
    });
  });

  it("renders placeholder when no exception is selected", async () => {
    render(
      <TestQueryWrapper>
        <StoryboardTab />
      </TestQueryWrapper>
    );
    await waitFor(() => {
      expect(
        screen.getByText(/Select an exception from the queue to investigate/i)
      ).toBeDefined();
    });
  });
});
