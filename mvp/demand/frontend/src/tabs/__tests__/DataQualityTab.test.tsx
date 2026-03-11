import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", () => ({
  fetchDQDashboard: vi.fn().mockResolvedValue({ domains: [] }),
  fetchDQChecks: vi.fn().mockResolvedValue({ checks: [] }),
  fetchDQFreshness: vi.fn().mockResolvedValue({ tables: [] }),
  dqKeys: { dashboard: ["dq", "dashboard"], checks: ["dq", "checks"], freshness: ["dq", "freshness"] },
  STALE_PLATFORM: 300000,
}));

describe("DataQualityTab", () => {
  it("renders without crashing", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    expect(screen.getByText("Data Quality & Observability")).toBeInTheDocument();
  });

  it("shows empty state when no checks run", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    expect(screen.getByText(/No data quality checks have been run yet/)).toBeInTheDocument();
  });

  it("renders pipeline freshness section", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    expect(screen.getByText("Pipeline Freshness")).toBeInTheDocument();
  });

  it("renders check catalog section", async () => {
    const { default: DataQualityTab } = await import("../DataQualityTab");
    render(
      <TestQueryWrapper>
        <DataQualityTab />
      </TestQueryWrapper>
    );
    expect(screen.getByText(/Check Catalog/)).toBeInTheDocument();
  });
});
