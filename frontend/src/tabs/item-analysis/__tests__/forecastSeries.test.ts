import { describe, expect, it } from "vitest";

import { collectFutureForecastMonths, mergeItemForecastSeries } from "../forecastSeries";

describe("item forecast series composition", () => {
  it("collects future production, staging, and AI months without truncating the horizon", () => {
    const months = collectFutureForecastMonths({
      historyMonths: ["2026-06-01"],
      productionForecasts: [{ forecast_month: "2026-07-01" }],
      stagingModels: {
        customer_bottom_up: [{ forecast_month: "2026-08-01" }],
      },
      aiRows: [{ forecast_month: "2026-09-01" }],
    });

    expect(months).toEqual(["2026-07-01", "2026-08-01", "2026-09-01"]);
  });

  it("merges staged customer forecasts forward and their backtests over sales history", () => {
    const merged = mergeItemForecastSeries({
      baseSeries: [{ month: "2026-06-01", sales_qty: 100 }],
      futureMonths: ["2026-07-01"],
      timeEnd: "",
      productionForecasts: [],
      stagingModels: {
        customer_bottom_up: [{ forecast_month: "2026-07-01", forecast_qty: 90 }],
        customer_bottom_up_blend: [{ forecast_month: "2026-07-01", forecast_qty: 95 }],
      },
      candidateModels: {
        customer_bottom_up: [{ forecast_month: "2026-06-01", forecast_qty: 92 }],
      },
      aiRows: [],
    });

    expect(merged[0]).toMatchObject({
      month: "2026-06-01",
      sales_qty: 100,
      backtest_customer_bottom_up: 92,
    });
    expect(merged[1]).toMatchObject({
      month: "2026-07-01",
      staging_customer_bottom_up: 90,
      staging_customer_bottom_up_blend: 95,
    });
  });
});
