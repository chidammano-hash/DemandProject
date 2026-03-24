import { describe, it, expect } from "vitest";
import { queryKeys, STALE } from "@/api/queries";

describe("queryKeys", () => {
  it("domains returns stable key", () => {
    expect(queryKeys.domains()).toEqual(["domains"]);
  });

  it("domainMeta includes domain name", () => {
    expect(queryKeys.domainMeta("item")).toEqual(["domain-meta", "item"]);
  });

  it("domainPage includes params", () => {
    const key = queryKeys.domainPage("sales", { limit: 50 });
    expect(key[0]).toBe("domain-page");
    expect(key[1]).toBe("sales");
  });

  it("different params produce different keys", () => {
    const k1 = queryKeys.domainPage("item", { limit: 50 });
    const k2 = queryKeys.domainPage("item", { limit: 100 });
    expect(k1).not.toEqual(k2);
  });

  it("forecastModels returns stable key", () => {
    expect(queryKeys.forecastModels()).toEqual(["forecast-models"]);
  });

  it("accuracySlice includes params", () => {
    const key = queryKeys.accuracySlice({ group_by: "cluster" });
    expect(key[0]).toBe("accuracy-slice");
  });

  it("skuAnalysis includes params", () => {
    const key = queryKeys.skuAnalysis({ item: "X", location: "Y" });
    expect(key[0]).toBe("sku-analysis");
  });
});

describe("STALE constants", () => {
  it("FOREVER is Infinity", () => {
    expect(STALE.FOREVER).toBe(Infinity);
  });

  it("time values are in milliseconds", () => {
    expect(STALE.TEN_MIN).toBe(600000);
    expect(STALE.FIVE_MIN).toBe(300000);
    expect(STALE.TWO_MIN).toBe(120000);
    expect(STALE.ONE_MIN).toBe(60000);
    expect(STALE.THIRTY_SEC).toBe(30000);
    expect(STALE.NONE).toBe(0);
  });

  it("values are in decreasing order", () => {
    expect(STALE.TEN_MIN).toBeGreaterThan(STALE.FIVE_MIN);
    expect(STALE.FIVE_MIN).toBeGreaterThan(STALE.TWO_MIN);
    expect(STALE.TWO_MIN).toBeGreaterThan(STALE.ONE_MIN);
    expect(STALE.ONE_MIN).toBeGreaterThan(STALE.THIRTY_SEC);
    expect(STALE.THIRTY_SEC).toBeGreaterThan(STALE.NONE);
  });
});
