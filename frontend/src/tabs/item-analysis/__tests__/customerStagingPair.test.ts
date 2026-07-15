import { describe, expect, it } from "vitest";

import type {
  StagingForecastPoint,
  StagingForecastsPayload,
} from "@/api/queries/production-forecast";
import { applyPairedCustomerStaging, pairCustomerStaging } from "../customerStagingPair";

function row(
  sourceModelId: string,
  sourceRunId: string,
  month: string,
  quantity: number
): StagingForecastPoint {
  return {
    source_model_id: sourceModelId,
    source_run_id: sourceRunId,
    forecast_month: month,
    forecast_qty: quantity,
    forecast_qty_lower: null,
    forecast_qty_upper: null,
    horizon_months: 1,
    cluster_id: null,
    lag_source: null,
    generated_at: null,
  };
}

const staging: StagingForecastsPayload = {
  item_id: "ITEM-1",
  loc: "LOC-1",
  models: {
    customer_bottom_up: [
      row("customer_bottom_up", "shadow-current", "2026-07-01", 90),
      row("customer_bottom_up", "shadow-stale", "2026-08-01", 91),
    ],
    customer_bottom_up_blend: [
      row("customer_bottom_up_blend", "blend-current", "2026-07-01", 95),
      row("customer_bottom_up_blend", "blend-stale", "2026-08-01", 96),
    ],
    champion: [row("mstl", "champion-run", "2026-07-01", 100)],
  },
};

describe("customer staging run pairing", () => {
  it("keeps only exact customer run identities without affecting other models", () => {
    const paired = pairCustomerStaging(staging, "blend-current", "shadow-current");

    expect(paired.payload?.models.customer_bottom_up).toHaveLength(1);
    expect(paired.payload?.models.customer_bottom_up_blend).toHaveLength(1);
    expect(paired.payload?.models.champion).toEqual(staging.models.champion);
    expect([...paired.bottomUpMonths]).toEqual(["2026-07"]);
    expect([...paired.blendMonths]).toEqual(["2026-07"]);
  });

  it("removes stale merged values and restores only the exact run/month values", () => {
    const paired = pairCustomerStaging(staging, "blend-current", "shadow-current");
    const result = applyPairedCustomerStaging(
      [
        {
          month: "2026-07-01",
          staging_customer_bottom_up: 999,
          staging_customer_bottom_up_blend: 999,
        },
        {
          month: "2026-08-01",
          staging_customer_bottom_up: 91,
          staging_customer_bottom_up_blend: 96,
        },
      ],
      paired
    );

    expect(result[0]).toMatchObject({
      staging_customer_bottom_up: 90,
      staging_customer_bottom_up_blend: 95,
    });
    expect(result[1]).toEqual({ month: "2026-08-01" });
  });
});
