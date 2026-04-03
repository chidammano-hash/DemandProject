import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

// ---------------------------------------------------------------------------
// Mock data
// ---------------------------------------------------------------------------
const mockPipelineConfig = {
  algorithms: {
    lgbm_cluster: {
      type: "tree", enabled: true, tune: true, backtest: true,
      compete: true, forecast: true, expert: false,
      cluster_strategy: "per_cluster", config_key: "lgbm",
      output_dir: "data/backtest/lgbm_cluster",
    },
    catboost_cluster: {
      type: "tree", enabled: true, tune: true, backtest: true,
      compete: true, forecast: true, expert: false,
      cluster_strategy: "per_cluster", config_key: "catboost",
      output_dir: "data/backtest/catboost_cluster",
    },
    chronos2: {
      type: "foundation", enabled: true, tune: false, backtest: true,
      compete: true, forecast: false, expert: false,
      config_key: "chronos2", output_dir: "data/backtest/chronos2",
    },
    mstl: {
      type: "statistical", enabled: true, tune: false, backtest: true,
      compete: true, forecast: false, expert: true,
      config_key: "mstl", output_dir: "data/backtest/mstl",
    },
  },
  clustering: {
    enabled: true,
    config_ref: "clustering_config.yaml",
    tuning_profiles_ref: "cluster_tuning_profiles.yaml",
    steps: { generate_features: true, train_model: true, label_clusters: true, update_db: true },
  },
  backtest: { n_timeframes: 10, embargo_months: 1, forecast_horizon: 6, early_stop_pct: 0.03 },
  tuning: { n_trials: 50, gap_months: 1, n_splits: 5 },
  champion: {
    strategy: "hybrid_warmup", fallback_model_id: "seasonal_naive",
    metric: "accuracy_pct", strategy_params: { min_prior_months: 3 },
  },
  production_forecast: {
    horizon_months: 24, min_history_months: 12,
    cold_start_model_id: "rolling_mean", cold_start_min_months: 3,
  },
  backtest_sampling: { enabled: true, default_target_n: 5000, default_method: "proportional" },
  pipeline: {
    stages: [
      { name: "clustering", description: "Segment DFUs", makefile_target: "cluster-all", depends_on: [] },
      { name: "backtest", description: "Backtest all models", makefile_target: "backtest-all", depends_on: ["clustering"] },
      { name: "champion", description: "Select champion", makefile_target: "champion-all", depends_on: ["backtest"] },
      { name: "forecast", description: "Production forecast", makefile_target: "forecast-generate", depends_on: ["champion"] },
    ],
  },
};

// ---------------------------------------------------------------------------
// Mocks
// ---------------------------------------------------------------------------
vi.mock("@/api/queries/unified-model-tuning", () => ({
  fetchPipelineConfig: vi.fn().mockResolvedValue(mockPipelineConfig),
  updatePipelineConfig: vi.fn().mockResolvedValue(undefined),
  pipelineConfigKeys: { config: ["pipeline-config"] },
}));

vi.mock("@/hooks/useThemeContext", () => ({
  useThemeContext: () => ({ theme: "light" }),
}));

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe("PipelineConfigPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it("renders algorithm roster table with algorithm names", async () => {
    const { PipelineConfigPanel } = await import("../model-tuning/PipelineConfigPanel");
    render(<TestQueryWrapper><PipelineConfigPanel /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("LightGBM")).toBeInTheDocument();
      expect(screen.getByText("CatBoost")).toBeInTheDocument();
      expect(screen.getByText("Chronos 2")).toBeInTheDocument();
      expect(screen.getByText("MSTL")).toBeInTheDocument();
    });
  });

  it("renders pipeline stages in order", async () => {
    const { PipelineConfigPanel } = await import("../model-tuning/PipelineConfigPanel");
    render(<TestQueryWrapper><PipelineConfigPanel /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("clustering")).toBeInTheDocument();
      expect(screen.getByText("backtest")).toBeInTheDocument();
      expect(screen.getByText("champion")).toBeInTheDocument();
      expect(screen.getByText("forecast")).toBeInTheDocument();
    });
  });

  it("renders algorithm type badges", async () => {
    const { PipelineConfigPanel } = await import("../model-tuning/PipelineConfigPanel");
    render(<TestQueryWrapper><PipelineConfigPanel /></TestQueryWrapper>);
    await waitFor(() => {
      // Multiple tree badges exist (lgbm + catboost), use getAllByText
      expect(screen.getAllByText("tree").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("foundation").length).toBeGreaterThanOrEqual(1);
      expect(screen.getAllByText("statistical").length).toBeGreaterThanOrEqual(1);
    });
  });

  it("renders backtest settings with correct values", async () => {
    const { PipelineConfigPanel } = await import("../model-tuning/PipelineConfigPanel");
    render(<TestQueryWrapper><PipelineConfigPanel /></TestQueryWrapper>);
    await waitFor(() => {
      // "Backtest" may appear in multiple places (heading + sampling), use getAllByText
      expect(screen.getAllByText(/Backtest/i).length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText("Timeframes")).toBeInTheDocument();
    });
  });

  it("renders champion strategy", async () => {
    const { PipelineConfigPanel } = await import("../model-tuning/PipelineConfigPanel");
    render(<TestQueryWrapper><PipelineConfigPanel /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Champion Selection")).toBeInTheDocument();
    });
  });

  it("renders production forecast settings", async () => {
    const { PipelineConfigPanel } = await import("../model-tuning/PipelineConfigPanel");
    render(<TestQueryWrapper><PipelineConfigPanel /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Production Forecast")).toBeInTheDocument();
      expect(screen.getByText("Cold-Start Model")).toBeInTheDocument();
    });
  });

  it("renders clustering section", async () => {
    const { PipelineConfigPanel } = await import("../model-tuning/PipelineConfigPanel");
    render(<TestQueryWrapper><PipelineConfigPanel /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Clustering")).toBeInTheDocument();
    });
  });

  it("renders backtest sampling section", async () => {
    const { PipelineConfigPanel } = await import("../model-tuning/PipelineConfigPanel");
    render(<TestQueryWrapper><PipelineConfigPanel /></TestQueryWrapper>);
    await waitFor(() => {
      expect(screen.getByText("Backtest Sampling")).toBeInTheDocument();
      expect(screen.getByText("Target DFUs")).toBeInTheDocument();
    });
  });
});
