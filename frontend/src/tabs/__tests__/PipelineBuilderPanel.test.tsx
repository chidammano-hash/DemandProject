import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { PipelineBuilderPanel } from "@/tabs/jobs/PipelineBuilderPanel";
import type { JobType } from "@/types/jobs";

vi.mock("lucide-react", () => {
  const Stub = () => <span />;
  return {
    Play: Stub, Plus: Stub, Trash2: Stub, ArrowRight: Stub, Save: Stub,
    ChevronDown: Stub, ChevronUp: Stub,
    Clock: Stub, Loader2: Stub, CheckCircle2: Stub, XCircle: Stub, Square: Stub,
    Network: Stub, TrendingUp: Stub, Activity: Stub, Trophy: Stub,
    Sparkles: Stub, BarChart2: Stub, Package: Stub, Boxes: Stub,
  };
});

const MOCK_JOB_TYPES: JobType[] = [
  { type_id: "etl_pipeline", label: "Data Ingestion Pipeline", description: "", group: "etl", params_schema: {} },
  { type_id: "compute_sku_features", label: "Compute SKU Features", description: "", group: "features", params_schema: {} },
  { type_id: "cluster_pipeline", label: "Full Clustering Pipeline", description: "", group: "clustering", params_schema: {} },
  { type_id: "backtest_lgbm", label: "LGBM Backtest", description: "", group: "backtest", params_schema: {} },
  { type_id: "backtest_catboost", label: "CatBoost Backtest", description: "", group: "backtest", params_schema: {} },
  { type_id: "backtest_xgboost", label: "XGBoost Backtest", description: "", group: "backtest", params_schema: {} },
  { type_id: "backtest_load_model", label: "Load Backtest Results", description: "", group: "backtest_load", params_schema: {} },
  { type_id: "compute_safety_stock", label: "Safety Stock", description: "", group: "inventory", params_schema: {} },
  { type_id: "compute_eoq", label: "EOQ Targets", description: "", group: "inventory", params_schema: {} },
  { type_id: "train_production_model", label: "Train Production Model", description: "", group: "forecast", params_schema: {} },
  { type_id: "generate_production_forecast", label: "Production Forecast", description: "", group: "forecast", params_schema: {} },
  { type_id: "refresh_forecast_views", label: "Refresh Forecast Views", description: "", group: "forecast", params_schema: {} },
  { type_id: "refresh_customer_analytics", label: "Recalculate Customer Analytics", description: "", group: "forecast", params_schema: {} },
  { type_id: "compute_replenishment_plan", label: "Replenishment Plan", description: "", group: "replenishment", params_schema: {} },
  { type_id: "champion_select", label: "Champion Selection", description: "", group: "champion", params_schema: {} },
  { type_id: "seasonality_pipeline", label: "Seasonality Detection", description: "", group: "seasonality", params_schema: {} },
  { type_id: "classify_abc_xyz", label: "ABC-XYZ Classification", description: "", group: "inventory", params_schema: {} },
  { type_id: "compute_variability", label: "Demand Variability", description: "", group: "inventory", params_schema: {} },
  { type_id: "compute_demand_signals", label: "Demand Signals", description: "", group: "inventory", params_schema: {} },
  { type_id: "generate_storyboard", label: "Storyboard Exceptions", description: "", group: "ai", params_schema: {} },
  { type_id: "prepare_forecast_snapshot_contenders", label: "Prepare Forecast Snapshot Contenders", description: "", group: "forecast", params_schema: {} },
  { type_id: "archive_forecast_snapshot", label: "Archive Forecast Snapshot", description: "", group: "forecast", params_schema: {} },
  { type_id: "cleanup_forecast_staging", label: "Clean Forecast Staging", description: "", group: "forecast", params_schema: {} },
];

const storage = new Map<string, string>();
Object.defineProperty(window, "localStorage", {
  value: {
    getItem: vi.fn((key: string) => storage.get(key) ?? null),
    setItem: vi.fn((key: string, value: string) => storage.set(key, value)),
    removeItem: vi.fn((key: string) => storage.delete(key)),
    clear: vi.fn(() => storage.clear()),
  },
});

function expandPanel() {
  fireEvent.click(screen.getByText("Pipeline Builder").closest("button")!);
}

describe("PipelineBuilderPanel", () => {
  const onSubmit = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    window.localStorage.clear();
  });

  it("renders collapsed by default with bundle count badge", () => {
    render(<PipelineBuilderPanel jobTypes={MOCK_JOB_TYPES} onSubmit={onSubmit} />);
    expect(screen.getByText("Pipeline Builder")).toBeDefined();
    expect(screen.getByText("9 bundles")).toBeDefined();
    expect(screen.queryByText("Delta Data Load")).toBeNull();
  });

  it("renders pre-built default bundle buttons when expanded", () => {
    render(<PipelineBuilderPanel jobTypes={MOCK_JOB_TYPES} onSubmit={onSubmit} />);
    expandPanel();
    expect(screen.getByText("Delta Data Load")).toBeDefined();
    expect(screen.getByText("Forecast Feature Prep")).toBeDefined();
    expect(screen.getByText("Core Tree Backtests")).toBeDefined();
    expect(screen.getByText("Inventory Refresh")).toBeDefined();
    expect(screen.getByText("Forecast Snapshot Archive")).toBeDefined();
  });

  it("submits the forecast snapshot selection, archive, and cleanup bundle", () => {
    render(<PipelineBuilderPanel jobTypes={MOCK_JOB_TYPES} onSubmit={onSubmit} />);
    expandPanel();
    const bundle = screen.getByText("Forecast Snapshot Archive").closest("div.rounded-md")!;
    fireEvent.click(bundle.querySelector("button")!);
    expect(onSubmit).toHaveBeenCalledTimes(1);
    const call = onSubmit.mock.calls[0][0];
    expect(call.label).toBe("Forecast Snapshot Archive");
    expect(call.steps.map((step: { type: string }) => step.type)).toEqual([
      "prepare_forecast_snapshot_contenders",
      "archive_forecast_snapshot",
      "cleanup_forecast_staging",
    ]);
  });

  it("calls onSubmit with correct steps when a default bundle Run button is clicked", () => {
    render(<PipelineBuilderPanel jobTypes={MOCK_JOB_TYPES} onSubmit={onSubmit} />);
    expandPanel();
    const bundle = screen.getByText("Delta Data Load").closest("div.rounded-md")!;
    fireEvent.click(bundle.querySelector("button")!);
    expect(onSubmit).toHaveBeenCalledTimes(1);
    const call = onSubmit.mock.calls[0][0];
    expect(call.label).toBe("Delta Data Load");
    expect(call.steps.length).toBe(1);
    expect(call.steps[0].type).toBe("etl_pipeline");
    expect(call.steps[0].params.mode).toBe("refresh");
  });

  it("custom builder: Add Step appends step to list", () => {
    render(<PipelineBuilderPanel jobTypes={MOCK_JOB_TYPES} onSubmit={onSubmit} />);
    expandPanel();
    const select = screen.getByRole("combobox");
    fireEvent.change(select, { target: { value: "compute_safety_stock" } });
    fireEvent.click(screen.getByText("Add"));
    expect(screen.getAllByText("Safety Stock").length).toBeGreaterThanOrEqual(1);
  });

  it("custom builder: Run Custom Bundle is disabled with no steps", () => {
    render(<PipelineBuilderPanel jobTypes={MOCK_JOB_TYPES} onSubmit={onSubmit} />);
    expandPanel();
    const nameInput = screen.getByPlaceholderText("Pipeline name…");
    fireEvent.change(nameInput, { target: { value: "My Pipeline" } });
    const runBtn = screen.getByText("Run Custom Bundle");
    expect(runBtn.closest("button")?.disabled).toBe(true);
  });

  it("custom builder: submits bundle with 2+ steps and a name", () => {
    render(<PipelineBuilderPanel jobTypes={MOCK_JOB_TYPES} onSubmit={onSubmit} />);
    expandPanel();
    const select = screen.getByRole("combobox");
    // Add step 1
    fireEvent.change(select, { target: { value: "compute_safety_stock" } });
    fireEvent.click(screen.getByText("Add"));
    // Add step 2
    fireEvent.change(select, { target: { value: "compute_eoq" } });
    fireEvent.click(screen.getByText("Add"));
    // Name the pipeline
    const nameInput = screen.getByPlaceholderText("Pipeline name…");
    fireEvent.change(nameInput, { target: { value: "My Pipeline" } });
    // Submit
    fireEvent.click(screen.getByText("Run Custom Bundle"));
    expect(onSubmit).toHaveBeenCalledTimes(1);
    const call = onSubmit.mock.calls[0][0];
    expect(call.label).toBe("My Pipeline");
    expect(call.steps).toHaveLength(2);
    expect(call.steps[0].type).toBe("compute_safety_stock");
    expect(call.steps[1].type).toBe("compute_eoq");
  });

  it("custom builder: saves and runs a user bundle", () => {
    render(<PipelineBuilderPanel jobTypes={MOCK_JOB_TYPES} onSubmit={onSubmit} />);
    expandPanel();
    const select = screen.getByRole("combobox");
    fireEvent.change(select, { target: { value: "compute_safety_stock" } });
    fireEvent.click(screen.getByText("Add"));
    const nameInput = screen.getByPlaceholderText("Pipeline name…");
    fireEvent.change(nameInput, { target: { value: "Monthly Inventory Check" } });
    fireEvent.click(screen.getByText("Save Bundle"));

    expect(screen.getByText("Saved Bundles")).toBeDefined();
    expect(screen.getByText("Monthly Inventory Check")).toBeDefined();

    const runButtons = screen.getAllByText("Run");
    fireEvent.click(runButtons[runButtons.length - 1]);
    expect(onSubmit).toHaveBeenCalledTimes(1);
    expect(onSubmit.mock.calls[0][0].label).toBe("Monthly Inventory Check");
    expect(onSubmit.mock.calls[0][0].steps[0].type).toBe("compute_safety_stock");
  });
});
