import { describe, expect, it } from "vitest";

import { skuModelColor } from "../colors";

describe("skuModelColor", () => {
  it.each(["lgbm_global", "lgbm_transfer"])(
    "uses the generic fallback for retired model %s",
    (retiredModel) => {
      expect(skuModelColor(retiredModel, 0)).toBe(skuModelColor("unrecognized_model", 0));
    }
  );
});
