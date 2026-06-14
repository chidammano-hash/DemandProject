/**
 * Regression tests for `aiFvaBacktestKeys` shape and `listFvaBacktestRuns`
 * URL construction. Any change to these keys silently breaks React Query
 * cache invalidation, so we pin the literal shapes here.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  aiFvaBacktestKeys,
  listFvaBacktestRuns,
  startFvaBacktestRun,
} from "@/api/queries/ai-planner-fva-backtest";

describe("aiFvaBacktestKeys", () => {
  it("root is the stable two-segment prefix used for blanket invalidation", () => {
    expect(aiFvaBacktestKeys.root).toEqual(["ai-planner", "fva-backtest"]);
  });

  it("list(undefined) coerces the status arg to null for cache-key stability", () => {
    expect(aiFvaBacktestKeys.list(undefined)).toEqual([
      "ai-planner",
      "fva-backtest",
      "list",
      null,
    ]);
  });

  it("list(status) includes the status string verbatim", () => {
    expect(aiFvaBacktestKeys.list("running")).toEqual([
      "ai-planner",
      "fva-backtest",
      "list",
      "running",
    ]);
  });

  it("detail/summary/byRecommendation/byMonth use the run id as the leaf segment", () => {
    expect(aiFvaBacktestKeys.detail("abc")).toEqual([
      "ai-planner",
      "fva-backtest",
      "detail",
      "abc",
    ]);
    expect(aiFvaBacktestKeys.summary("abc")).toEqual([
      "ai-planner",
      "fva-backtest",
      "summary",
      "abc",
    ]);
    expect(aiFvaBacktestKeys.byRecommendation("abc")).toEqual([
      "ai-planner",
      "fva-backtest",
      "by-recommendation",
      "abc",
    ]);
    expect(aiFvaBacktestKeys.byMonth("abc")).toEqual([
      "ai-planner",
      "fva-backtest",
      "by-month",
      "abc",
    ]);
  });

  it("dfus key includes sort + limit so different drill-down configs cache independently", () => {
    expect(aiFvaBacktestKeys.dfus("abc", "error_reduction", 25)).toEqual([
      "ai-planner",
      "fva-backtest",
      "dfus",
      "abc",
      "error_reduction",
      25,
    ]);
  });
});

// ---------------------------------------------------------------------------
// Fetcher URL construction
// ---------------------------------------------------------------------------

describe("listFvaBacktestRuns / startFvaBacktestRun URLs", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      text: () => Promise.resolve(""),
      json: () => Promise.resolve({}),
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("listFvaBacktestRuns(undefined) GETs the default limit URL with no status filter", async () => {
    await listFvaBacktestRuns(undefined);
    expect(fetchMock).toHaveBeenCalledTimes(1);
    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toBe("/ai-planner/fva-backtest/runs?limit=50");
  });

  it("listFvaBacktestRuns(status, limit) appends both query params", async () => {
    await listFvaBacktestRuns("running", 10);
    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("/ai-planner/fva-backtest/runs");
    expect(url).toContain("limit=10");
    expect(url).toContain("status=running");
  });

  it("startFvaBacktestRun POSTs JSON body with the right headers", async () => {
    await startFvaBacktestRun({ window_months: 10, provider: "ollama" });
    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe("/ai-planner/fva-backtest/runs");
    const reqInit = init as RequestInit;
    expect(reqInit.method).toBe("POST");
    const headers = reqInit.headers as Record<string, string>;
    expect(headers["Content-Type"]).toBe("application/json");
    expect(JSON.parse(reqInit.body as string)).toEqual({
      window_months: 10,
      provider: "ollama",
    });
  });
});
