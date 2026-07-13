import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { TestQueryWrapper } from "@/tabs/__tests__/test-utils";
import type { SkuAnalysisPayload } from "@/types";

import { ModelKpiSection } from "../ModelKpiSection";

const skuData: SkuAnalysisPayload = {
  mode: "item_location",
  item: "151682",
  location: "1401-BULK",
  points: 36,
  models: ["champion", "lgbm_cluster", "nbeats"],
  series: [],
  model_monthly: {
    champion: [{ month: "2026-06-01", forecast: 37.6667, actual: 48.327 }],
    lgbm_cluster: [{ month: "2026-06-01", forecast: 38.4723, actual: 48.327 }],
    nbeats: [{ month: "2026-06-01", forecast: 36.8611, actual: 48.327 }],
  },
  champion_source_by_month: { "2026-06-01": "ensemble" },
  champion_mix_by_month: {
    "2026-06-01": [
      { model: "nbeats", weight: 0.5 },
      { model: "lgbm_cluster", weight: 0.5 },
    ],
  },
  dfu_attributes: [],
};

describe("ModelKpiSection", () => {
  it("explains why a one-month blended champion KPI differs from every single model", () => {
    render(
      <ModelKpiSection
        skuData={skuData}
        skuKpis={{
          champion: {
            accuracy_pct: 77.94,
            wape: 22.06,
            bias: -0.22,
            sum_forecast: 37.6667,
            sum_actual: 48.327,
            months_covered: 1,
          },
          lgbm_cluster: {
            accuracy_pct: 79.61,
            wape: 20.39,
            bias: -0.2,
            sum_forecast: 38.4723,
            sum_actual: 48.327,
            months_covered: 1,
          },
          nbeats: {
            accuracy_pct: 76.27,
            wape: 23.73,
            bias: -0.24,
            sum_forecast: 36.8611,
            sum_actual: 48.327,
            months_covered: 1,
          },
        }}
        skuKpiMonths={1}
        setSkuKpiMonths={vi.fn()}
        skuVisibleSeries={new Set(["forecast_champion", "forecast_lgbm_cluster", "forecast_nbeats"])}
      />,
      { wrapper: TestQueryWrapper },
    );

    expect(screen.getByText("50% N-BEATS + 50% LightGBM")).toBeInTheDocument();
    expect(
      screen.getByText("Blended forecast for Jun 2026; accuracy is calculated after combining these models."),
    ).toBeInTheDocument();
    expect(
      screen.getByText("Blend verified: 50% × 36.9 + 50% × 38.5 = 37.7."),
    ).toBeInTheDocument();
    expect(screen.getByText("LightGBM")).toBeInTheDocument();
  });

  it("identifies a single-model champion whose KPI should reconcile exactly", () => {
    const singleModelData: SkuAnalysisPayload = {
      ...skuData,
      models: ["champion"],
      model_monthly: {
        champion: [{ month: "2026-06-01", forecast: 38.4723, actual: 48.327 }],
      },
      champion_source_by_month: { "2026-06-01": "lgbm_cluster" },
      champion_mix_by_month: {},
    };

    render(
      <ModelKpiSection
        skuData={singleModelData}
        skuKpis={{
          champion: {
            accuracy_pct: 79.61,
            wape: 20.39,
            bias: -0.2,
            sum_forecast: 38.4723,
            sum_actual: 48.327,
            months_covered: 1,
          },
        }}
        skuKpiMonths={1}
        setSkuKpiMonths={vi.fn()}
        skuVisibleSeries={new Set(["forecast_champion"])}
      />,
      { wrapper: TestQueryWrapper },
    );

    expect(screen.getByText("Selected LightGBM")).toBeInTheDocument();
    expect(
      screen.getByText("Single-model champion for Jun 2026; its KPI should match that model for the same month."),
    ).toBeInTheDocument();
  });

  it("raises a visible integrity warning when the stored blend does not equal its components", () => {
    const inconsistentData: SkuAnalysisPayload = {
      ...skuData,
      model_monthly: {
        ...skuData.model_monthly,
        champion: [{ month: "2026-06-01", forecast: 30, actual: 48.327 }],
      },
    };

    render(
      <ModelKpiSection
        skuData={inconsistentData}
        skuKpis={{
          champion: {
            accuracy_pct: 62.08,
            wape: 37.92,
            bias: -0.38,
            sum_forecast: 30,
            sum_actual: 48.327,
            months_covered: 1,
          },
        }}
        skuKpiMonths={1}
        setSkuKpiMonths={vi.fn()}
        skuVisibleSeries={new Set(["forecast_champion"])}
      />,
      { wrapper: TestQueryWrapper },
    );

    expect(
      screen.getByText("Blend mismatch: stored champion 30, weighted components 37.7."),
    ).toBeInTheDocument();
  });
});
