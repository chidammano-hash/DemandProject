import { describe, expect, it } from "vitest";
import { render, screen } from "@testing-library/react";

import { RecentJobsCard } from "../RecentJobsCard";
import type { Job } from "@/types/jobs";

function makeJob(overrides: Partial<Job>): Job {
  return {
    job_id: "job_20260714_073317_d36a5c97",
    job_type: "generate_forecast",
    job_label: "Generate Forecast: champion",
    status: "completed",
    params: {},
    result: null,
    error: null,
    submitted_at: new Date().toISOString(),
    started_at: new Date().toISOString(),
    completed_at: new Date().toISOString(),
    progress_pct: 100,
    progress_msg: null,
    pid: null,
    ...overrides,
  };
}

describe("RecentJobsCard", () => {
  it("shows the unique ID tail, not the shared job_YYYY prefix", () => {
    render(<RecentJobsCard recentJobs={[makeJob({})]} />);

    expect(screen.getByText("d36a5c97")).toBeInTheDocument();
    expect(screen.queryByText("job_2026")).not.toBeInTheDocument();
    expect(screen.getByTitle("job_20260714_073317_d36a5c97")).toBeInTheDocument();
  });

  it("renders distinct IDs for jobs submitted the same year", () => {
    render(
      <RecentJobsCard
        recentJobs={[
          makeJob({ job_id: "job_20260714_073317_d36a5c97" }),
          makeJob({ job_id: "job_20260714_091502_1f2e3a4b" }),
        ]}
      />
    );

    expect(screen.getByText("d36a5c97")).toBeInTheDocument();
    expect(screen.getByText("1f2e3a4b")).toBeInTheDocument();
  });
});
