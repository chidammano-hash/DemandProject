/**
 * Regression tests (U2.1) — fetchJson must attach the HTTP status to thrown
 * errors so the global formatApiError sanitizer can map them to friendly copy.
 *
 * Before the fix, fetchJson threw `new Error(rawBody)` with no status, so a 404
 * body `{"detail":"Not Found"}` leaked verbatim into the user-facing toast
 * (the message-digit heuristic in extractStatus finds no digits in "Not Found"
 * and the whole sanitization layer is bypassed).
 */
import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { fetchJson as coreFetchJson } from "@/api/queries/core";
import { fetchJson } from "@/api/queries/request";
import { formatApiError, extractStatus } from "@/lib/formatApiError";

describe("fetchJson error handling (U2.1)", () => {
  const fetchMock = vi.fn();

  beforeEach(() => {
    fetchMock.mockReset();
    sessionStorage.clear();
    vi.stubGlobal("fetch", fetchMock);
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  function notFoundResponse() {
    return {
      ok: false,
      status: 404,
      text: () => Promise.resolve('{"detail":"Not Found"}'),
      json: () => Promise.resolve({ detail: "Not Found" }),
    };
  }

  it("keeps the public core export wired to the leaf request helper", () => {
    expect(coreFetchJson).toBe(fetchJson);
  });

  it("attaches the interactive JWT to shared API requests", async () => {
    sessionStorage.setItem("ds_access_token", "planner-token");
    fetchMock.mockResolvedValue({ ok: true, status: 200, json: () => Promise.resolve({ ok: true }) });

    await fetchJson("/forecast/jobs");

    const init = fetchMock.mock.calls[0][1] as RequestInit;
    expect(new Headers(init.headers).get("Authorization")).toBe("Bearer planner-token");
  });

  it("attaches the HTTP status to the thrown error", async () => {
    fetchMock.mockResolvedValue(notFoundResponse());
    let caught: unknown;
    try {
      await fetchJson("/data-quality/lineage/batches");
    } catch (e) {
      caught = e;
    }
    expect(extractStatus(caught)).toBe(404);
  });

  it("does not leak the raw {detail} JSON body into the error message", async () => {
    fetchMock.mockResolvedValue(notFoundResponse());
    let caught: unknown;
    try {
      await fetchJson("/data-quality/lineage/batches");
    } catch (e) {
      caught = e;
    }
    const msg = (caught as Error).message;
    expect(msg.startsWith("{")).toBe(false);
  });

  it("formatApiError maps a 404 fetchJson failure to friendly copy", async () => {
    fetchMock.mockResolvedValue(notFoundResponse());
    let caught: unknown;
    try {
      await fetchJson("/data-quality/lineage/batches");
    } catch (e) {
      caught = e;
    }
    expect(formatApiError(caught)).toBe("That record could not be found.");
  });

  it("formats FastAPI validation details without object coercion", async () => {
    fetchMock.mockResolvedValue({
      ok: false,
      status: 422,
      text: () => Promise.resolve(JSON.stringify({
        detail: [{ loc: ["body", "jobs", 3, "mode"], msg: "Input should be 'onetime', 'delta' or 'file'" }],
      })),
    });

    await expect(fetchJson("/integration/chains")).rejects.toThrow(
      "body.jobs.3.mode: Input should be 'onetime', 'delta' or 'file'",
    );
  });
});
