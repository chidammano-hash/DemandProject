import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";

import { TestQueryWrapper } from "../../../tabs/__tests__/test-utils";
import { ScanPanel } from "../ScanPanel";
import { planScan } from "../../../api/queries/integration_chain";

vi.mock("../../../api/queries/integration_chain", () => ({
  planScan: vi.fn(),
}));

const mockedPlanScan = vi.mocked(planScan);

function renderPanel(onPlanned = vi.fn()): void {
  render(
    <TestQueryWrapper>
      <ScanPanel onPlanned={onPlanned} />
    </TestQueryWrapper>,
  );
}

describe("ScanPanel", () => {
  beforeEach(() => {
    mockedPlanScan.mockReset();
  });

  it("asks follow-up questions when the planner is unsure", async () => {
    mockedPlanScan.mockResolvedValue({
      plan_id: "plan-1",
      provider: "codex",
      model: "gpt-5.5",
      status: "questions",
      confidence: 0.61,
      explanation: "Need one more answer.",
      risk_flags: ["active_job_overlap"],
      questions: [
        {
          id: "active_job_overlap",
          prompt: "Wait or queue behind the active job?",
          answer_type: "choice",
          options: ["Wait", "Queue"],
          required: true,
          reason: "A domain is already running.",
        },
      ],
      recommended_chain: [],
      evidence: [],
      scanned_at: "2026-07-10T10:00:00Z",
      changes: [],
      proposed_chain: [],
    });

    renderPanel();
    fireEvent.click(screen.getByRole("button", { name: /scan data\/input\/ for changed files/i }));
    expect(await screen.findByText("Wait or queue behind the active job?")).toBeInTheDocument();
    expect(mockedPlanScan.mock.calls[0][0]).toEqual({ answers: [] });
  });

  it("submits answers and hands off a final plan", async () => {
    const onPlanned = vi.fn();
    mockedPlanScan
      .mockResolvedValueOnce({
        plan_id: "plan-1",
        provider: "codex",
        model: "gpt-5.5",
        status: "questions",
        confidence: 0.61,
        explanation: "Need one more answer.",
        risk_flags: ["active_job_overlap"],
        questions: [
          {
            id: "active_job_overlap",
            prompt: "Wait or queue behind the active job?",
            answer_type: "choice",
            options: ["Wait", "Queue"],
            required: true,
            reason: "A domain is already running.",
          },
        ],
        recommended_chain: [],
        evidence: [],
        scanned_at: "2026-07-10T10:00:00Z",
        changes: [],
        proposed_chain: [],
      })
      .mockResolvedValueOnce({
        plan_id: "plan-2",
        provider: "codex",
        model: "gpt-5.5",
        status: "planned",
        confidence: 0.93,
        explanation: "Queue behind the job and keep the chain conservative.",
        risk_flags: [],
        questions: [],
        recommended_chain: [{ step: 1, domain: "sales", mode: "delta", slice: null }],
        evidence: [],
        scanned_at: "2026-07-10T10:00:00Z",
        changes: [],
        proposed_chain: [],
      });

    renderPanel(onPlanned);
    fireEvent.click(screen.getByRole("button", { name: /scan data\/input\/ for changed files/i }));
    const select = await screen.findByRole("combobox");
    fireEvent.change(select, { target: { value: "Queue" } });
    fireEvent.click(screen.getByRole("button", { name: /refine plan/i }));

    await waitFor(() => expect(onPlanned).toHaveBeenCalledTimes(1));
    expect(mockedPlanScan.mock.calls[1][0]).toEqual({
      answers: [{ question_id: "active_job_overlap", answer: "Queue" }],
    });
    expect(onPlanned.mock.calls[0][0].status).toBe("planned");
  });
});
