import { afterEach, describe, expect, it, vi } from "vitest";

import { submitPromote, submitStageForecast } from "@/api/queries/backtest-management";

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("submitPromote", () => {
  it("promotes one explicit immutable source run", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          model_id: "champion",
          promotion_type: "champion",
          plan_version: "2026-07",
          source_run_id: "00000000-0000-0000-0000-000000000111",
          production_run_id: "00000000-0000-0000-0000-000000000222",
          candidate_checksum: "c".repeat(64),
          outgoing_archive_checksum: null,
          rows_promoted: 120,
          dfu_count: 10,
        }),
        { status: 201, headers: { "Content-Type": "application/json" } }
      )
    );
    vi.stubGlobal("fetch", fetchMock);

    await submitPromote("champion", "00000000-0000-0000-0000-000000000111");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(
      "/backtest-management/champion/promote?source_run_id=00000000-0000-0000-0000-000000000111"
    );
    expect(init).toMatchObject({ method: "POST" });
  });
});

describe("submitStageForecast", () => {
  it("stages one explicit immutable source run", async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(
        JSON.stringify({
          model_id: "mstl",
          source_run_id: "00000000-0000-0000-0000-000000000111",
          status: "staged",
          rows_staged: 120,
          dfu_count: 10,
          candidate_checksum: "c".repeat(64),
        }),
        { status: 200, headers: { "Content-Type": "application/json" } }
      )
    );
    vi.stubGlobal("fetch", fetchMock);

    await submitStageForecast("mstl", "00000000-0000-0000-0000-000000000111");

    const [url, init] = fetchMock.mock.calls[0];
    expect(url).toBe(
      "/backtest-management/mstl/stage?source_run_id=00000000-0000-0000-0000-000000000111"
    );
    expect(init).toMatchObject({ method: "POST" });
  });
});
