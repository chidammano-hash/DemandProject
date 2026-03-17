import { describe, it, expect } from "vitest";
import { buildSearchParams, buildQueryString, buildQuerySuffix } from "@/api/queries/helpers";

describe("buildSearchParams", () => {
  it("converts string values to URLSearchParams", () => {
    const qs = buildSearchParams({ item: "A", loc: "B" });
    expect(qs.get("item")).toBe("A");
    expect(qs.get("loc")).toBe("B");
  });

  it("converts numeric values to strings", () => {
    const qs = buildSearchParams({ page: 1, limit: 50 });
    expect(qs.get("page")).toBe("1");
    expect(qs.get("limit")).toBe("50");
  });

  it("converts boolean values to strings", () => {
    const qs = buildSearchParams({ active: true, deleted: false });
    expect(qs.get("active")).toBe("true");
    expect(qs.get("deleted")).toBe("false");
  });

  it("filters out undefined values", () => {
    const qs = buildSearchParams({ item: "A", loc: undefined });
    expect(qs.has("item")).toBe(true);
    expect(qs.has("loc")).toBe(false);
  });

  it("filters out null values", () => {
    const qs = buildSearchParams({ item: "A", loc: null });
    expect(qs.has("item")).toBe(true);
    expect(qs.has("loc")).toBe(false);
  });

  it("filters out empty string values", () => {
    const qs = buildSearchParams({ item: "A", loc: "" });
    expect(qs.has("item")).toBe(true);
    expect(qs.has("loc")).toBe(false);
  });

  it("returns empty URLSearchParams when all values are empty", () => {
    const qs = buildSearchParams({ a: undefined, b: null, c: "" });
    expect(qs.toString()).toBe("");
  });

  it("preserves zero as a valid value", () => {
    const qs = buildSearchParams({ offset: 0 });
    expect(qs.get("offset")).toBe("0");
  });
});

describe("buildQueryString", () => {
  it("returns query string without leading ?", () => {
    const qs = buildQueryString({ item: "A", page: 1 });
    expect(qs).toContain("item=A");
    expect(qs).toContain("page=1");
    expect(qs).not.toMatch(/^\?/);
  });

  it("returns empty string when all values are empty", () => {
    expect(buildQueryString({ a: undefined })).toBe("");
  });
});

describe("buildQuerySuffix", () => {
  it("returns ?key=val when params have values", () => {
    const suffix = buildQuerySuffix({ page: 1 });
    expect(suffix).toBe("?page=1");
  });

  it("returns empty string when no params have values", () => {
    expect(buildQuerySuffix({ a: undefined, b: null })).toBe("");
  });

  it("returns empty string for empty object", () => {
    expect(buildQuerySuffix({})).toBe("");
  });

  it("handles mixed valid and invalid params", () => {
    const suffix = buildQuerySuffix({ item: "X", loc: undefined, limit: 50 });
    expect(suffix).toMatch(/^\?/);
    expect(suffix).toContain("item=X");
    expect(suffix).toContain("limit=50");
    expect(suffix).not.toContain("loc");
  });
});
