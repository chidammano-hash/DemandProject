import { describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, within } from "@testing-library/react";

import type { NamedPipelinePreset } from "@/api/queries/jobs";
import type { PipelineReadiness } from "@/api/queries/dashboard";
import { derivePipelineRunState, PipelineBuilderPanel } from "@/tabs/jobs/PipelineBuilderPanel";
import type { Job, JobType } from "@/types/jobs";

vi.mock("lucide-react", () => {
  const Stub = () => <span />;
  return {
    AlertCircle: Stub,
    ChevronDown: Stub,
    ChevronUp: Stub,
    Loader2: Stub,
    Play: Stub,
    RotateCcw: Stub,
  };
});

const MOCK_JOB_TYPES: JobType[] = [
  {
    type_id: "compute_sku_features",
    label: "Compute SKU Features",
    description: "",
    group: "features",
    params_schema: {},
  },
  {
    type_id: "cluster_pipeline",
    label: "Clustering Pipeline",
    description: "",
    group: "clustering",
    params_schema: {},
  },
  {
    type_id: "tune_stale_clusters",
    label: "Tune LightGBM",
    description: "",
    group: "tuning",
    params_schema: {},
  },
  {
    type_id: "backtest_lgbm",
    label: "LightGBM",
    description: "",
    group: "backtest",
    params_schema: {},
  },
  {
    type_id: "backtest_nhits",
    label: "N-HiTS",
    description: "",
    group: "backtest",
    params_schema: {},
  },
  {
    type_id: "backtest_nbeats",
    label: "N-BEATS",
    description: "",
    group: "backtest",
    params_schema: {},
  },
  {
    type_id: "backtest_mstl",
    label: "MSTL",
    description: "",
    group: "backtest",
    params_schema: {},
  },
  {
    type_id: "backtest_chronos2_enriched",
    label: "Chronos 2E",
    description: "",
    group: "backtest",
    params_schema: {},
  },
  {
    type_id: "champion_select",
    label: "Governed Champion Refresh",
    description: "",
    group: "champion",
    params_schema: {},
  },
  {
    type_id: "governed_champion_refresh",
    label: "Governed Champion Refresh",
    description: "",
    group: "champion",
    params_schema: {},
  },
  {
    type_id: "train_production_model",
    label: "Train Production Models",
    description: "",
    group: "forecast",
    params_schema: {},
  },
  {
    type_id: "generate_production_forecast",
    label: "Generate Release Candidate",
    description: "",
    group: "forecast",
    params_schema: {},
  },
  {
    type_id: "prepare_forecast_snapshot_contenders",
    label: "Select Snapshot Contenders",
    description: "",
    group: "forecast",
    params_schema: {},
  },
  {
    type_id: "archive_forecast_snapshot",
    label: "Archive Snapshot",
    description: "",
    group: "forecast",
    params_schema: {},
  },
  {
    type_id: "refresh_forecast_snapshot_kpis",
    label: "Calculate Snapshot KPIs",
    description: "",
    group: "forecast",
    params_schema: {},
  },
  {
    type_id: "cleanup_forecast_staging",
    label: "Clean Forecast Staging",
    description: "",
    group: "forecast",
    params_schema: {},
  },
  // Negative guard: a stale registry response must not create a selectable model or workflow.
  {
    type_id: "backtest_catboost",
    label: "CatBoost Backtest",
    description: "",
    group: "backtest",
    params_schema: {},
  },
];

const PIPELINES: NamedPipelinePreset[] = [
  {
    name: "clustering-refresh",
    description: "Refresh features and promote current clusters.",
    steps: ["compute_sku_features", "cluster_pipeline"],
  },
  {
    name: "model-refresh",
    description: "Run and load the complete retained model roster.",
    steps: [
      "tune_stale_clusters",
      "backtest_lgbm",
      "backtest_nhits",
      "backtest_nbeats",
      "backtest_mstl",
      "backtest_chronos2_enriched",
    ],
  },
  {
    name: "champion-refresh",
    description: "Select and atomically assign the champion from the governed roster.",
    steps: ["governed_champion_refresh"],
  },
  {
    name: "forecast-publish",
    description: "Train models and build one immutable release candidate.",
    steps: [
      "train_production_model",
      "generate_production_forecast",
      "prepare_forecast_snapshot_contenders",
    ],
  },
  {
    name: "forecast-snapshot-bundle",
    description: "Archive champion plus three contenders, then clean staging.",
    steps: [
      "prepare_forecast_snapshot_contenders",
      "archive_forecast_snapshot",
      "cleanup_forecast_staging",
    ],
  },
  {
    name: "period-roll",
    description: "Score the prior month, archive the current month, then clean staging.",
    steps: [
      "refresh_forecast_snapshot_kpis",
      "prepare_forecast_snapshot_contenders",
      "archive_forecast_snapshot",
      "cleanup_forecast_staging",
    ],
  },
  { name: "data-refresh", description: "General ETL.", steps: ["etl_pipeline"] },
  { name: "inventory-refresh", description: "Inventory calculations.", steps: ["compute_eoq"] },
  { name: "full-refresh", description: "Everything.", steps: ["etl_pipeline"] },
];

const STALE_CHAMPION_READINESS: PipelineReadiness = {
  ready: false,
  checks: [
    {
      stage: "champion",
      status: "stale",
      severity: "high",
      title: "Promoted champion predates the current clusters",
      detail: "Refresh the five-model roster before publishing.",
      action: null,
    },
  ],
};

function pipelineJob(overrides: Partial<Job> = {}): Job {
  return {
    job_id: "job-1",
    job_type: "backtest_lgbm",
    job_label: "[model-refresh 2/6] LightGBM",
    status: "running",
    params: {
      __pipeline_label: "model-refresh",
      __pipeline_step: 2,
      __pipeline_total_steps: 6,
    },
    result: null,
    error: null,
    submitted_at: "2026-07-12T10:00:00Z",
    started_at: "2026-07-12T10:00:01Z",
    completed_at: null,
    progress_pct: 30,
    progress_msg: "Backtesting LightGBM",
    pid: 42,
    ...overrides,
  };
}

function renderPanel(overrides: Partial<React.ComponentProps<typeof PipelineBuilderPanel>> = {}) {
  const onRun = vi.fn();
  render(
    <PipelineBuilderPanel
      jobTypes={MOCK_JOB_TYPES}
      pipelines={PIPELINES}
      onRun={onRun}
      {...overrides}
    />
  );
  return { onRun };
}

describe("PipelineBuilderPanel", () => {
  it("renders only server-defined forecasting lifecycle workflows", () => {
    renderPanel();

    expect(screen.getByText("Forecast Pipelines")).toBeDefined();
    expect(screen.getByText("5 workflows")).toBeDefined();
    expect(screen.getByText("1. Prepare Features & Clusters")).toBeDefined();
    expect(screen.getByText("2. Refresh Five-Model Roster")).toBeDefined();
    expect(screen.queryByText("3. Select & Assign Champion")).toBeNull();
    expect(screen.getByText("3. Build Release Candidate")).toBeDefined();
    expect(screen.getByText("4. Archive Forecast Snapshot")).toBeDefined();
    expect(screen.getByText("Period Roll · Score Prior + Archive Current")).toBeDefined();
    expect(screen.queryByText("General ETL.")).toBeNull();
    expect(screen.queryByText("Inventory calculations.")).toBeNull();
  });

  it("shows exactly the retained five base models in the model refresh workflow", () => {
    renderPanel();
    const card = screen.getByText("2. Refresh Five-Model Roster").closest("article");
    expect(card).not.toBeNull();
    const scoped = within(card!);
    expect(scoped.getByText("LightGBM")).toBeDefined();
    expect(scoped.getByText("N-HiTS")).toBeDefined();
    expect(scoped.getByText("N-BEATS")).toBeDefined();
    expect(scoped.getByText("MSTL")).toBeDefined();
    expect(scoped.getByText("Chronos 2E")).toBeDefined();
    expect(scoped.queryByText("CatBoost Backtest")).toBeNull();
  });

  it("launches the canonical server preset by name", () => {
    const { onRun } = renderPanel();
    const card = screen.getByText("3. Build Release Candidate").closest("article");
    fireEvent.click(within(card!).getByRole("button", { name: "Run" }));
    expect(onRun).toHaveBeenCalledWith("forecast-publish");
  });

  it("shows current server job progress and disables duplicate launch", () => {
    renderPanel({ jobs: [pipelineJob()] });
    const card = screen.getByText("2. Refresh Five-Model Roster").closest("article");
    const scoped = within(card!);
    expect(scoped.getAllByText("Running")).toHaveLength(2);
    expect(scoped.getByText("Step 2 of 6")).toBeDefined();
    expect(scoped.getByText("Backtesting LightGBM")).toBeDefined();
    expect((scoped.getByRole("button", { name: "Running" }) as HTMLButtonElement).disabled).toBe(
      true
    );
  });

  it("reconciles a launch by exact pipeline id instead of a reused label", () => {
    const state = derivePipelineRunState(
      "model-refresh",
      [
        pipelineJob({
          job_id: "other-run",
          pipeline_id: "pipe_other",
          submitted_at: "2026-07-12T12:00:00Z",
          status: "running",
        }),
        pipelineJob({
          job_id: "target-run",
          pipeline_id: "pipe_target",
          pipeline_step: 6,
          status: "completed",
          completed_at: "2026-07-12T11:00:00Z",
        }),
      ],
      { name: "model-refresh", pipelineId: "pipe_target", submitting: false }
    );

    expect(state.status).toBe("completed");
    expect(state.step).toBe(6);
  });

  it("shows the returned pipeline reference while the first step is queued", () => {
    renderPanel({
      launch: { name: "forecast-publish", pipelineId: "pipe_abc123", submitting: false },
    });
    const card = screen.getByText("3. Build Release Candidate").closest("article");
    const scoped = within(card!);
    expect(scoped.getAllByText("Queued")).toHaveLength(2);
    expect(scoped.getByText("pipe_abc123")).toBeDefined();
  });

  it("allows rerun after the final step completes", () => {
    renderPanel({
      jobs: [
        pipelineJob({
          job_label: "[model-refresh 6/6] Chronos 2E",
          status: "completed",
          params: {
            __pipeline_label: "model-refresh",
            __pipeline_step: 6,
            __pipeline_total_steps: 6,
          },
          completed_at: "2026-07-12T11:00:00Z",
          progress_msg: "Done",
        }),
      ],
    });
    const card = screen.getByText("2. Refresh Five-Model Roster").closest("article");
    const scoped = within(card!);
    expect(scoped.getByText("Completed")).toBeDefined();
    expect(scoped.getByRole("button", { name: /Run again/ })).toBeDefined();
  });

  it("describes a failed pipeline relaunch as a restart from step one", () => {
    renderPanel({
      jobs: [pipelineJob({ status: "failed", error: "Backtest failed" })],
    });
    const card = screen.getByText("2. Refresh Five-Model Roster").closest("article");
    const scoped = within(card!);
    expect(scoped.getByRole("button", { name: "Restart from step 1" })).toBeDefined();
    expect(scoped.getByText(/creates a new workflow from step 1/i)).toBeDefined();
  });

  it("blocks publishing until the user assigns a completed experiment in Champion", () => {
    renderPanel({ readiness: STALE_CHAMPION_READINESS });

    const modelCard = screen.getByText("2. Refresh Five-Model Roster").closest("article");
    const publishCard = screen.getByText("3. Build Release Candidate").closest("article");
    const modelScoped = within(modelCard!);
    const publishScoped = within(publishCard!);

    expect(modelScoped.queryByText(/resolves the current champion readiness issue/i)).toBeNull();
    expect((modelScoped.getByRole("button", { name: "Run" }) as HTMLButtonElement).disabled).toBe(
      false
    );
    expect(screen.queryByText("3. Select & Assign Champion")).toBeNull();
    expect(
      publishScoped.getByText(/select and assign a completed experiment in champion/i)
    ).toBeDefined();
    expect(publishScoped.getByText("Prerequisite")).toBeDefined();
    expect(publishScoped.queryByText("Ready")).toBeNull();
    expect(
      (publishScoped.getByRole("button", { name: "Prerequisite required" }) as HTMLButtonElement)
        .disabled
    ).toBe(true);
  });

  it("does not call an unchecked downstream workflow ready while prerequisites load", () => {
    renderPanel({ readinessLoading: true });
    const publishCard = screen.getByText("3. Build Release Candidate").closest("article");
    const scoped = within(publishCard!);

    expect(scoped.getByText("Checking")).toBeDefined();
    expect(
      (scoped.getByRole("button", { name: "Checking prerequisites" }) as HTMLButtonElement).disabled
    ).toBe(true);
  });
});

describe("derivePipelineRunState", () => {
  it("does not mark a completed intermediate step as a completed workflow", () => {
    const state = derivePipelineRunState("model-refresh", [
      pipelineJob({ status: "completed", completed_at: "2026-07-12T10:05:00Z" }),
    ]);
    expect(state.status).toBe("queued");
    expect(state.message).toBe("Advancing to the next step");
  });

  it("surfaces a failed step for retry", () => {
    const state = derivePipelineRunState("model-refresh", [
      pipelineJob({ status: "failed", error: "Backtest failed" }),
    ]);
    expect(state.status).toBe("failed");
    expect(state.message).toBe("Backtest failed");
  });
});
