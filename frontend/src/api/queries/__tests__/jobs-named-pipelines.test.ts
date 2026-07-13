import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { fetchNamedPipelines, runNamedPipeline } from "@/api/queries/jobs";

describe("named forecast pipelines", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => vi.unstubAllGlobals());

  it("loads canonical presets from the server", async () => {
    const payload = {
      pipelines: [
        {
          name: "model-refresh",
          description: "Run the retained roster.",
          steps: ["backtest_lgbm", "backtest_nhits"],
        },
      ],
    };
    fetchMock.mockResolvedValue({ ok: true, json: () => Promise.resolve(payload) });

    await expect(fetchNamedPipelines()).resolves.toEqual(payload);
    expect(fetchMock).toHaveBeenCalledWith("/jobs/pipelines/named", undefined);
  });

  it("launches one preset by its encoded server name", async () => {
    fetchMock.mockResolvedValue({
      ok: true,
      json: () =>
        Promise.resolve({
          pipeline_id: "pipe_123",
          name: "forecast-publish",
          status: "running",
          steps: 3,
        }),
    });

    await runNamedPipeline("forecast-publish");
    expect(fetchMock).toHaveBeenCalledWith("/jobs/pipelines/named/forecast-publish", {
      method: "POST",
    });
  });
});
