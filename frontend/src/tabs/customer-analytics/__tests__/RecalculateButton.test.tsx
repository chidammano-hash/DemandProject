/**
 * RecalculateButton — submits a refresh_customer_analytics job and, on
 * completion, invalidates the customer-analytics queries so panels repaint.
 */
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TestQueryWrapper } from "../../__tests__/test-utils";
import { RecalculateButton } from "../RecalculateButton";
import {
  triggerRecalculateCustomerAnalytics,
} from "@/api/queries/customer-analytics";
import { fetchActiveJobs, fetchJobDetail } from "@/api/queries/jobs";

vi.mock("@/api/queries/customer-analytics", () => ({
  triggerRecalculateCustomerAnalytics: vi.fn(),
}));

vi.mock("@/api/queries/jobs", () => ({
  fetchActiveJobs: vi.fn().mockResolvedValue({ jobs: [] }),
  fetchJobDetail: vi.fn(),
}));

const mockTrigger = vi.mocked(triggerRecalculateCustomerAnalytics);
const mockActive = vi.mocked(fetchActiveJobs);
const mockDetail = vi.mocked(fetchJobDetail);

function renderButton() {
  return render(
    <TestQueryWrapper>
      <RecalculateButton />
    </TestQueryWrapper>,
  );
}

describe("RecalculateButton", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockActive.mockResolvedValue({ jobs: [] });
  });

  it("renders the idle Recalculate button", () => {
    renderButton();
    expect(screen.getByRole("button", { name: /recalculate/i })).toBeInTheDocument();
  });

  it("submits a job and shows progress, then a completion banner", async () => {
    mockTrigger.mockResolvedValue({ job_id: "job-ca-9", status: "queued" });
    mockDetail.mockResolvedValue({
      job_id: "job-ca-9",
      job_type: "refresh_customer_analytics",
      status: "completed",
      progress_pct: 100,
      progress_msg: "Done",
    } as never);

    renderButton();
    await userEvent.click(screen.getByRole("button", { name: /recalculate/i }));

    await waitFor(() => expect(mockTrigger).toHaveBeenCalledTimes(1));
    await waitFor(() =>
      expect(screen.getByText(/recalculation completed/i)).toBeInTheDocument(),
    );
  });

  it("surfaces a failure banner when the trigger errors", async () => {
    mockTrigger.mockRejectedValue(new Error("boom"));

    renderButton();
    await userEvent.click(screen.getByRole("button", { name: /recalculate/i }));

    await waitFor(() =>
      expect(screen.getByText(/failed to start recalculation/i)).toBeInTheDocument(),
    );
  });
});
