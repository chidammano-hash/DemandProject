import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("recharts");

vi.mock("@/api/queries", () => ({
  tuningChatKeys: {
    sessions: () => ["tuning-chat-sessions"],
    session: (id: string) => ["tuning-chat-session", id],
    runStatus: (sid: string, rid: number) => ["tuning-chat-run-status", sid, rid],
  },
  lgbmTuningKeys: {
    runs: (p?: Record<string, unknown>) => ["lgbm-tuning-runs", p],
    run: (id: number) => ["lgbm-tuning-run", id],
    compare: (b: number, c: number) => ["lgbm-tuning-compare", b, c],
    comparisons: (n?: number) => ["lgbm-tuning-comparisons", n],
  },
  STALE: { FOREVER: Infinity, TEN_MIN: 600000, FIVE_MIN: 300000, TWO_MIN: 120000, ONE_MIN: 60000, THIRTY_SEC: 30000, NONE: 0 },
  fetchChatSessions: vi.fn().mockResolvedValue({ sessions: [] }),
  fetchChatSession: vi.fn().mockResolvedValue({
    session: { session_id: "abc-123", title: "Test Session", status: "active" },
    messages: [
      { message_id: 1, session_id: "abc-123", role: "user", content: "What should I try?", message_type: "text", metadata: null, created_at: "2026-03-23T10:00:00Z" },
      { message_id: 2, session_id: "abc-123", role: "assistant", content: "Based on your runs...", message_type: "text", metadata: null, created_at: "2026-03-23T10:00:05Z" },
    ],
  }),
  createChatSession: vi.fn().mockResolvedValue({ session: { session_id: "new-123", title: "New Session" } }),
  sendTuningChatMessage: vi.fn().mockResolvedValue({ messages: [] }),
  confirmTuningRun: vi.fn().mockResolvedValue({ run_id: 7, status: "started", strategy_label: "test" }),
  fetchRunStatus: vi.fn().mockResolvedValue({ run_id: 7, status: "running", elapsed_seconds: 30 }),
  // Other barrel exports that might be referenced
  queryKeys: {},
  STALE_INSIGHTS: 300000,
  insightKeys: { all: () => ["insights"] },
}));

describe("TuningChatPanel", () => {
  it("renders the input area with placeholder", async () => {
    const { TuningChatPanel } = await import("../lgbm-tuning/TuningChatPanel");
    render(
      <TestQueryWrapper>
        <TuningChatPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(screen.getByPlaceholderText(/Ask about runs/i)).toBeInTheDocument();
    });
  });

  it("auto-creates a session on mount", async () => {
    const { createChatSession } = await import("@/api/queries");
    const { TuningChatPanel } = await import("../lgbm-tuning/TuningChatPanel");
    render(
      <TestQueryWrapper>
        <TuningChatPanel />
      </TestQueryWrapper>,
    );
    await waitFor(() => {
      expect(createChatSession).toHaveBeenCalled();
    });
  });
});

describe("RecommendationCard", () => {
  it("renders parameter overrides", async () => {
    const { RecommendationCard } = await import("../lgbm-tuning/RecommendationCard");
    const rec = {
      strategy_label: "test_v1",
      description: "Increase regularization",
      overrides: { learning_rate: 0.03, reg_lambda: 3.0 },
      expected_impact: "+0.5% accuracy",
      risk_assessment: "Low risk",
      base_on_run_id: null,
    };
    render(
      <RecommendationCard
        recommendation={rec}
        messageId={1}
        sessionId="abc-123"
        onConfirm={vi.fn()}
        onReject={vi.fn()}
      />,
    );
    expect(screen.getByText("test_v1")).toBeInTheDocument();
    expect(screen.getByText("learning_rate")).toBeInTheDocument();
    expect(screen.getByText("0.03")).toBeInTheDocument();
    expect(screen.getByText(/Confirm & Run/i)).toBeInTheDocument();
    expect(screen.getByText(/Reject/i)).toBeInTheDocument();
  });

  it("shows confirmed state", async () => {
    const { RecommendationCard } = await import("../lgbm-tuning/RecommendationCard");
    const rec = {
      strategy_label: "test_v1",
      description: "Test",
      overrides: { lr: 0.03 },
      expected_impact: "Test",
      risk_assessment: "Low",
      base_on_run_id: null,
    };
    render(
      <RecommendationCard
        recommendation={rec}
        messageId={1}
        sessionId="abc-123"
        onConfirm={vi.fn()}
        onReject={vi.fn()}
        isConfirmed
      />,
    );
    expect(screen.getByText(/Confirmed/i)).toBeInTheDocument();
  });

  it("embeds RunStatusCard inline when confirmedRunId is provided", async () => {
    const { RecommendationCard } = await import("../lgbm-tuning/RecommendationCard");
    const rec = {
      strategy_label: "test_v1",
      description: "Test",
      overrides: { lr: 0.03 },
      expected_impact: "Test",
      risk_assessment: "Low",
      base_on_run_id: null,
    };
    render(
      <TestQueryWrapper>
        <RecommendationCard
          recommendation={rec}
          messageId={1}
          sessionId="abc-123"
          onConfirm={vi.fn()}
          onReject={vi.fn()}
          isConfirmed
          confirmedRunId={9}
        />
      </TestQueryWrapper>,
    );
    // Should show inline RunStatusCard instead of static badge
    expect(screen.getByText(/Run #9/i)).toBeInTheDocument();
    // Should NOT show the static "Confirmed — run started" badge
    expect(screen.queryByText(/Confirmed — run started/i)).not.toBeInTheDocument();
  });
});

describe("RunStatusCard", () => {
  it("renders running state", async () => {
    const { RunStatusCard } = await import("../lgbm-tuning/RunStatusCard");
    render(
      <TestQueryWrapper>
        <RunStatusCard
          sessionId="abc-123"
          runId={7}
          messageType="run_started"
        />
      </TestQueryWrapper>,
    );
    expect(screen.getByText(/Run #7/i)).toBeInTheDocument();
  });

  it("renders completed state with results", async () => {
    const { RunStatusCard } = await import("../lgbm-tuning/RunStatusCard");
    render(
      <TestQueryWrapper>
        <RunStatusCard
          sessionId="abc-123"
          runId={7}
          messageType="run_completed"
          completedResult={{ accuracy_pct: 71.79, wape: 28.21, bias: -0.012 }}
        />
      </TestQueryWrapper>,
    );
    expect(screen.getByText(/Run #7/i)).toBeInTheDocument();
    expect(screen.getByText("71.79%")).toBeInTheDocument();
  });

  it("renders failed state with error", async () => {
    const { RunStatusCard } = await import("../lgbm-tuning/RunStatusCard");
    render(
      <TestQueryWrapper>
        <RunStatusCard
          sessionId="abc-123"
          runId={7}
          messageType="run_failed"
          errorMessage="Backtest timed out"
        />
      </TestQueryWrapper>,
    );
    expect(screen.getByText(/Run #7/i)).toBeInTheDocument();
    expect(screen.getByText(/Backtest timed out/i)).toBeInTheDocument();
  });
});
