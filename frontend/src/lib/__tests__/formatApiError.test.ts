import { describe, it, expect } from "vitest";
import { formatApiError, extractStatus, __test__ } from "@/lib/formatApiError";

const { sanitize, GENERIC_FALLBACK, AUTH_EXPIRED, FORBIDDEN } = __test__;

describe("sanitize", () => {
  it("strips python tracebacks", () => {
    const msg =
      "Promotion failed.\nTraceback (most recent call last):\n  File \"/opt/app/foo.py\", line 12, in <module>\n    raise ValueError('x')";
    expect(sanitize(msg)).toBe("Promotion failed.");
  });

  it("strips JS stack frames", () => {
    const msg = "kaboom\n    at doThing (/Users/me/project/src/foo.ts:12:34)\n    at next";
    expect(sanitize(msg)).toBe("kaboom");
  });

  it("strips filesystem paths", () => {
    expect(sanitize("could not read /Users/me/secret/file.env OK")).toBe(
      "could not read [path] OK",
    );
  });

  it("strips Postgres DETAIL / HINT", () => {
    const msg =
      "duplicate key value violates unique constraint\nDETAIL: Key (id)=(1) already exists.\nHINT: try another";
    expect(sanitize(msg)).toBe("duplicate key value violates unique constraint");
  });

  it("strips trailing SQL statements", () => {
    const msg =
      "query failed: SELECT a, b, c FROM secret_table WHERE id = 1 AND token = 'abc'";
    expect(sanitize(msg).startsWith("query failed")).toBe(true);
    expect(sanitize(msg)).not.toContain("secret_table");
  });

  it("falls back on empty input", () => {
    expect(sanitize("")).toBe(GENERIC_FALLBACK);
  });

  it("truncates very long messages", () => {
    const long = "x".repeat(500);
    const out = sanitize(long);
    expect(out.length).toBeLessThanOrEqual(240);
    expect(out.endsWith("…")).toBe(true);
  });
});

describe("extractStatus", () => {
  it("reads .status", () => {
    expect(extractStatus({ status: 404 })).toBe(404);
  });

  it("reads .response.status", () => {
    expect(extractStatus({ response: { status: 500 } })).toBe(500);
  });

  it("parses status from message", () => {
    expect(extractStatus(new Error("HTTP 401 unauthorized"))).toBe(401);
  });

  it("returns null for no status", () => {
    expect(extractStatus(new Error("plain"))).toBeNull();
    expect(extractStatus(null)).toBeNull();
  });
});

describe("formatApiError", () => {
  it("maps 401 to session-expired", () => {
    expect(formatApiError({ status: 401 })).toBe(AUTH_EXPIRED);
  });

  it("maps 403 to forbidden", () => {
    expect(formatApiError({ status: 403 })).toBe(FORBIDDEN);
  });

  it("maps 5xx to generic server error", () => {
    expect(formatApiError({ status: 503 })).toMatch(/server/i);
  });

  it("handles strings", () => {
    expect(formatApiError("simple failure")).toBe("simple failure");
  });

  it("handles Error instances", () => {
    expect(formatApiError(new Error("kaboom"))).toBe("kaboom");
  });

  it("handles FastAPI {detail: string}", () => {
    expect(formatApiError({ detail: "bad input" })).toBe("bad input");
  });

  it("handles FastAPI validation shape", () => {
    expect(
      formatApiError({ detail: [{ loc: ["body"], msg: "field required" }] }),
    ).toBe("field required");
  });

  it("falls back for null/undefined", () => {
    expect(formatApiError(null)).toBe(GENERIC_FALLBACK);
    expect(formatApiError(undefined)).toBe(GENERIC_FALLBACK);
  });

  it("sanitizes inside .message", () => {
    const err = new Error("fail at /Users/me/foo.py line 3");
    const out = formatApiError(err);
    expect(out).not.toContain("/Users");
  });
});
