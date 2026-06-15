import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

import { PipelineLineageSection } from "../PipelineLineageSection";
import { TestQueryWrapper } from "../../__tests__/test-utils";
import { fetchBatches, type LoadBatch } from "../../../api/queries";

vi.mock("../../../api/queries", () => ({
  lineageKeys: { batches: ["lineage", "batches"] },
  STALE_LINEAGE: 0,
  fetchBatches: vi.fn(),
}));

const mockedFetch = vi.mocked(fetchBatches);

function batch(overrides: Partial<LoadBatch> = {}): LoadBatch {
  return {
    batch_id: 1,
    domain: "sales",
    source_file: "s.csv",
    source_hash: "h",
    row_count_in: 100,
    row_count_out: 95,
    status: "completed",
    started_at: "2026-06-14T00:00:00Z",
    completed_at: "2026-06-14T00:01:00Z",
    error_message: null,
    ...overrides,
  };
}

function renderSection() {
  return render(
    <TestQueryWrapper>
      <PipelineLineageSection />
    </TestQueryWrapper>,
  );
}

describe("PipelineLineageSection", () => {
  beforeEach(() => {
    mockedFetch.mockReset();
    mockedFetch.mockResolvedValue({
      batches: [
        batch({ batch_id: 1, domain: "sales" }),
        batch({ batch_id: 2, domain: "customer_demand", row_count_out: 7 }),
      ],
      total: 2,
    });
  });

  it("lists batches including customer_demand", async () => {
    renderSection();
    expect(await screen.findByText("customer_demand")).toBeInTheDocument();
    expect(screen.getByText("sales")).toBeInTheDocument();
  });

  it("filters by status", async () => {
    renderSection();
    await screen.findByText("sales");
    fireEvent.change(screen.getByLabelText("Filter lineage by status"), {
      target: { value: "failed" },
    });
    await waitFor(() =>
      expect(mockedFetch).toHaveBeenLastCalledWith(undefined, "failed", 50),
    );
  });

  it("filters by domain", async () => {
    renderSection();
    await screen.findByText("sales");
    fireEvent.change(screen.getByLabelText("Filter lineage by domain"), {
      target: { value: "customer_demand" },
    });
    await waitFor(() =>
      expect(mockedFetch).toHaveBeenLastCalledWith("customer_demand", undefined, 50),
    );
  });

  it("shows the error message on a failed batch", async () => {
    mockedFetch.mockResolvedValue({
      batches: [batch({ batch_id: 9, status: "failed", error_message: "load failed: bad row" })],
      total: 1,
    });
    renderSection();
    expect(await screen.findByText("load failed: bad row")).toBeInTheDocument();
  });
});
