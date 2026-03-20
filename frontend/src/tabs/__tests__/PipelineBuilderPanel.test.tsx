import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { PipelineBuilderPanel } from "@/tabs/jobs/PipelineBuilderPanel";
import type { JobType } from "@/types/jobs";

vi.mock("lucide-react", () => {
  const Stub = () => <span />;
  return {
    Play: Stub, Plus: Stub, Trash2: Stub, ArrowRight: Stub,
    ChevronDown: Stub, ChevronUp: Stub,
    Clock: Stub, Loader2: Stub, CheckCircle2: Stub, XCircle: Stub, Square: Stub,
    Network: Stub, TrendingUp: Stub, Activity: Stub, Trophy: Stub,
    Sparkles: Stub, BarChart2: Stub, Package: Stub, Boxes: Stub,
  };
});

const MOCK_JOB_TYPES: JobType[] = [
  { type_id: "cluster_pipeline", label: "Full Clustering Pipeline", description: "", group: "clustering", params_schema: {} },
  { type_id: "backtest_lgbm", label: "LGBM Backtest", description: "", group: "backtest", params_schema: {} },
  { type_id: "compute_safety_stock", label: "Safety Stock", description: "", group: "inventory", params_schema: {} },
  { type_id: "compute_eoq", label: "EOQ Targets", description: "", group: "inventory", params_schema: {} },
  { type_id: "generate_production_forecast", label: "Production Forecast", description: "", group: "forecast", params_schema: {} },
  { type_id: "compute_replenishment_plan", label: "Replenishment Plan", description: "", group: "replenishment", params_schema: {} },
  { type_id: "champion_select", label: "Champion Selection", description: "", group: "champion", params_schema: {} },
  { type_id: "seasonality_pipeline", label: "Seasonality Detection", description: "", group: "seasonality", params_schema: {} },
  { type_id: "classify_abc_xyz", label: "ABC-XYZ Classification", description: "", group: "inventory", params_schema: {} },
  { type_id: "compute_variability", label: "Demand Variability", description: "", group: "inventory", params_schema: {} },
  { type_id: "compute_demand_signals", label: "Demand Signals", description: "", group: "inventory", params_schema: {} },
  { type_id: "generate_storyboard", label: "Storyboard Exceptions", description: "", group: "ai", params_schema: {} },
];

function expandPanel() {
  fireEvent.click(screen.getByText("Pipeline Builder").closest("button")!);
}

describe("PipelineBuilderPanel", () => {
  const onSubmit = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders collapsed by default with template count badge", () => {
    render(<PipelineBuilderPanel jobTypes={MOCK_JOB_TYPES} onSubmit={onSubmit} />);
    expect(screen.getByText("Pipeline Builder")).toBeDefined();
    expect(screen.getByText("3 templates")).toBeDefined();
    // Templates not visible until expanded
    expect(screen.queryByText("Full S&OP Refresh")).toBeNull();
  });

  it("renders pre-built template buttons when expanded", () => {
    render(<PipelineBuilderPanel jobTypes={MOCK_JOB_TYPES} onSubmit={onSubmit} />);
    expandPanel();
    expect(screen.getByText("Full S&OP Refresh")).toBeDefined();
    expect(screen.getByText("Inventory Refresh")).toBeDefined();
    expect(screen.getByText("Weekly Data Refresh")).toBeDefined();
  });

  it("calls onSubmit with correct steps when a template Run button is clicked", () => {
    render(<PipelineBuilderPanel jobTypes={MOCK_JOB_TYPES} onSubmit={onSubmit} />);
    expandPanel();
    const runButtons = screen.getAllByText("Run");
    fireEvent.click(runButtons[0]); // First template: Full S&OP Refresh
    expect(onSubmit).toHaveBeenCalledTimes(1);
    const call = onSubmit.mock.calls[0][0];
    expect(call.label).toBe("Full S&OP Refresh");
    expect(call.steps.length).toBe(5);
    expect(call.steps[0].type).toBe("cluster_pipeline");
  });

  it("custom builder: Add Step appends step to list", () => {
    render(<PipelineBuilderPanel jobTypes={MOCK_JOB_TYPES} onSubmit={onSubmit} />);
    expandPanel();
    const select = screen.getByRole("combobox");
    fireEvent.change(select, { target: { value: "compute_safety_stock" } });
    fireEvent.click(screen.getByText("Add"));
    expect(screen.getAllByText("Safety Stock").length).toBeGreaterThanOrEqual(1);
  });

  it("custom builder: Run Custom Pipeline is disabled with fewer than 2 steps", () => {
    render(<PipelineBuilderPanel jobTypes={MOCK_JOB_TYPES} onSubmit={onSubmit} />);
    expandPanel();
    // Add only 1 step
    const select = screen.getByRole("combobox");
    fireEvent.change(select, { target: { value: "compute_safety_stock" } });
    fireEvent.click(screen.getByText("Add"));
    const nameInput = screen.getByPlaceholderText("Pipeline name…");
    fireEvent.change(nameInput, { target: { value: "My Pipeline" } });
    const runBtn = screen.getByText("Run Custom Pipeline");
    expect(runBtn.closest("button")?.disabled).toBe(true);
  });

  it("custom builder: submits pipeline with 2+ steps and a name", () => {
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
    fireEvent.click(screen.getByText("Run Custom Pipeline"));
    expect(onSubmit).toHaveBeenCalledTimes(1);
    const call = onSubmit.mock.calls[0][0];
    expect(call.label).toBe("My Pipeline");
    expect(call.steps).toHaveLength(2);
    expect(call.steps[0].type).toBe("compute_safety_stock");
    expect(call.steps[1].type).toBe("compute_eoq");
  });
});
