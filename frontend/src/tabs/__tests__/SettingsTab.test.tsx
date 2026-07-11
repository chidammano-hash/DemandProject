import { describe, it, expect, vi } from "vitest";
import { render, screen, waitFor, fireEvent } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries/config", () => ({
  configKeys: {
    list: () => ["config-list"],
    detail: (name: string) => ["config-detail", name],
  },
  fetchConfigList: vi.fn().mockResolvedValue({
    categories: [
      { key: "forecasting", label: "Forecasting & Models", description: "Model settings" },
      { key: "inventory", label: "Inventory Planning", description: "Inv settings" },
      { key: "operations", label: "Supply Chain Operations", description: "Ops settings" },
      { key: "pipeline", label: "Data Pipeline", description: "Pipeline settings" },
      { key: "planning", label: "Planning & Collaboration", description: "Planning settings" },
      { key: "system", label: "System & Integration", description: "System settings" },
    ],
    configs: [
      { name: "algorithm_config", label: "Algorithm Configuration", category: "forecasting", description: "Backtest and model training parameters", exists: true },
      { name: "safety_stock_config", label: "Safety Stock", category: "inventory", description: "Safety stock calculation", exists: true },
      { name: "planning_config", label: "Planning Date", category: "planning", description: "Planning date settings", exists: true },
    ],
  }),
  fetchConfigDetail: vi.fn().mockResolvedValue({
    name: "algorithm_config",
    label: "Algorithm Configuration",
    category: "forecasting",
    description: "Backtest and model training parameters for algorithms.",
    fields: [
      { path: "lgbm.recursive", value: true, label: "LGBM Recursive Inference", description: "Enable recursive multi-step forecasting", type: "boolean" },
      { path: "lgbm.hyperparameters.n_estimators", value: 500, label: "LGBM Number of Trees", description: "Maximum boosting rounds", type: "integer", min: 50, max: 5000, step: 50 },
      { path: "lgbm.cluster_strategy", value: "per_cluster", label: "LGBM Cluster Strategy", description: "Training approach", type: "select", options: ["per_cluster", "global"] },
      { path: "algorithms.lgbm_cluster.params.tune_inline", value: false, label: "Tune During Backtest", description: "Run Optuna inside each backtest", type: "boolean", group: "Model: LightGBM" },
    ],
    raw: { lgbm: { recursive: true, hyperparameters: { n_estimators: 500 }, cluster_strategy: "per_cluster" } },
  }),
  updateConfig: vi.fn().mockResolvedValue({ name: "algorithm_config", changed: ["lgbm.recursive"], message: "Updated 1 field(s)" }),
  resetConfig: vi.fn().mockResolvedValue({ name: "algorithm_config", message: "Reset successful" }),
}));

const { SettingsTab } = await import("@/tabs/SettingsTab");

function renderTab() {
  return render(
    <TestQueryWrapper>
      <SettingsTab />
    </TestQueryWrapper>,
  );
}

describe("SettingsTab", () => {
  it("renders without crashing", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("System Configuration")).toBeDefined();
    });
  });

  it("renders category sidebar", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getAllByText("Forecasting & Models").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("Inventory Planning").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("System & Integration").length).toBeGreaterThanOrEqual(1);
    });
  });

  it("renders config list for selected category", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Algorithm Configuration")).toBeDefined();
    });
  });

  it("renders field editor with labels and descriptions", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("LGBM Recursive Inference")).toBeDefined();
      expect(screen.getByText("LGBM Number of Trees")).toBeDefined();
      expect(screen.getByText("LGBM Cluster Strategy")).toBeDefined();
    });
  });

  it("renders field descriptions", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText(/Enable recursive multi-step forecasting/)).toBeDefined();
      expect(screen.getByText(/Maximum boosting rounds/)).toBeDefined();
    });
  });

  it("renders search input", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByPlaceholderText("Search configs...")).toBeDefined();
    });
  });

  it("renders save and discard buttons", async () => {
    renderTab();
    await waitFor(() => {
      expect(screen.getByText("Save")).toBeDefined();
      expect(screen.getByText("Discard")).toBeDefined();
    });
  });

  it("renders boolean toggle for boolean fields", async () => {
    renderTab();
    await waitFor(() => {
      const switches = screen.getAllByRole("switch");
      expect(switches.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("renders select dropdown for select fields", async () => {
    renderTab();
    await waitFor(() => {
      const options = screen.getAllByRole("option");
      expect(options.length).toBeGreaterThanOrEqual(2);
    });
  });

  it("renders number input for integer fields", async () => {
    renderTab();
    await waitFor(() => {
      const numberInputs = document.querySelectorAll('input[type="number"]');
      expect(numberInputs.length).toBeGreaterThanOrEqual(1);
    });
  });

  it("filters parameters by label and YAML path", async () => {
    renderTab();
    const search = await screen.findByPlaceholderText("Search parameters...");

    fireEvent.change(search, { target: { value: "tune_inline" } });

    expect(screen.getByText("Tune During Backtest")).toBeDefined();
    expect(screen.queryByText("LGBM Number of Trees")).toBeNull();
  });
});
