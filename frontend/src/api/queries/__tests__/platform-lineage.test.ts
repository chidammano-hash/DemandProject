/**
 * Regression tests (F2.2 / U2.5) for Pipeline Lineage fetcher URLs.
 *
 * The backend medallion router mounts at `/data-quality` with routes
 * `/batches` and `/batches/{id}` (NO `/lineage/` segment). Corrections live at
 * `/data-quality/corrections`. The frontend previously prefixed these with an
 * extra `/lineage/` segment the backend never exposed, producing a hard 404 on
 * every Data Quality tab load. These tests pin the URLs to the mounted paths.
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
  fetchBatches,
  fetchBatchDetail,
  fetchCorrections,
} from "@/api/queries/platform";

describe("Pipeline Lineage fetcher URLs align with mounted backend routes", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    fetchMock.mockResolvedValue({
      ok: true,
      status: 200,
      text: () => Promise.resolve(""),
      json: () => Promise.resolve({ batches: [], total: 0, corrections: [] }),
    });
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("fetchBatches hits /data-quality/batches (no /lineage/ segment)", async () => {
    await fetchBatches(undefined, undefined, 20);
    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("/data-quality/batches");
    expect(url).not.toContain("/lineage/");
    expect(url).toContain("limit=20");
  });

  it("fetchBatchDetail hits /data-quality/batches/{id}", async () => {
    await fetchBatchDetail(7);
    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toBe("/data-quality/batches/7");
    expect(url).not.toContain("/lineage/");
  });

  it("fetchCorrections hits /data-quality/corrections (no /lineage/ segment)", async () => {
    await fetchCorrections("sales");
    const url = fetchMock.mock.calls[0][0] as string;
    expect(url).toContain("/data-quality/corrections");
    expect(url).not.toContain("/lineage/");
  });
});
