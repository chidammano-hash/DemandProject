/**
 * formatApiError — sanitize an unknown error into a safe, user-facing message.
 *
 * Gen-4 roadmap UX P0. Strips information that a planner should never see:
 *   - stack traces (`at foo (/.../file.ts:12:34)` frames, `Traceback (most recent call last)`)
 *   - internal filesystem paths (/Users/..., /home/..., /opt/..., C:\\)
 *   - Postgres / psycopg internals ("DETAIL:", "HINT:", "CONTEXT:")
 *   - SQL statement fragments ("SELECT ... FROM ...")
 *
 * Callers: `toast.error(formatApiError(e))`.
 */

const PATH_RE = /(?:\/(?:Users|home|opt|var|tmp|root|usr|srv)\/[^\s"']*|[A-Za-z]:\\[^\s"']*)/g;
const PG_DETAIL_RE = /\b(?:DETAIL|HINT|CONTEXT|STATEMENT|LINE \d+):[^\n]*/gi;
const SQL_FRAGMENT_RE = /\b(?:SELECT|INSERT|UPDATE|DELETE|WITH|CREATE|ALTER|DROP)\s+[\s\S]{10,}$/i;
const TRACEBACK_RE = /Traceback \(most recent call last\)[\s\S]*/i;
const STACK_FRAME_RE = /\n\s*at\s[^\n]*/g;

const GENERIC_FALLBACK = "Something went wrong. Please try again.";
const AUTH_EXPIRED = "Your session has expired. Please sign in again.";
const FORBIDDEN = "You don't have permission to do that.";
const NOT_FOUND = "That record could not be found.";
const RATE_LIMITED = "You're doing that too fast. Slow down and retry in a moment.";
const SERVER_ERROR = "The server hit an error. Our team has been notified.";

/** Cap the final message so a novel-length validation dump doesn't leak through. */
const MAX_LEN = 240;

function sanitize(raw: string): string {
  let msg = raw
    .replace(TRACEBACK_RE, "")
    .replace(STACK_FRAME_RE, "")
    .replace(PG_DETAIL_RE, "")
    .replace(PATH_RE, "[path]")
    .replace(SQL_FRAGMENT_RE, "")
    .replace(/\s+/g, " ")
    .trim();

  if (!msg) return GENERIC_FALLBACK;

  if (msg.length > MAX_LEN) {
    msg = `${msg.slice(0, MAX_LEN - 1).trimEnd()}…`;
  }
  return msg;
}

/** Pull an HTTP-style status code off an error-like object, if present. */
export function extractStatus(err: unknown): number | null {
  if (err == null || typeof err !== "object") return null;
  const e = err as Record<string, unknown>;
  if (typeof e.status === "number") return e.status;
  const resp = e.response;
  if (resp && typeof resp === "object" && typeof (resp as Record<string, unknown>).status === "number") {
    return (resp as Record<string, number>).status;
  }
  // Message heuristic: "... 401 ..." / "... 403 ..."
  const message = typeof e.message === "string" ? e.message : "";
  const match = message.match(/\b(4\d\d|5\d\d)\b/);
  if (match) return Number(match[1]);
  return null;
}

/**
 * Convert any error-like value to a safe user-facing string.
 * Handles Error instances, fetch `Response`-wrapped errors, plain strings,
 * `{ detail: "..." }` FastAPI shapes, and arbitrary unknowns.
 */
export function formatApiError(err: unknown): string {
  const status = extractStatus(err);
  if (status === 401) return AUTH_EXPIRED;
  if (status === 403) return FORBIDDEN;
  if (status === 404) return NOT_FOUND;
  if (status === 429) return RATE_LIMITED;
  if (status != null && status >= 500) return SERVER_ERROR;

  if (err == null) return GENERIC_FALLBACK;
  if (typeof err === "string") return sanitize(err);

  if (err instanceof Error) {
    return sanitize(err.message || GENERIC_FALLBACK);
  }

  if (typeof err === "object") {
    const e = err as Record<string, unknown>;
    // FastAPI: { detail: "..." } or { detail: [{ msg: "..." }] }
    const detail = e.detail;
    if (typeof detail === "string") return sanitize(detail);
    if (Array.isArray(detail)) {
      const first = detail[0];
      if (first && typeof first === "object" && typeof (first as Record<string, unknown>).msg === "string") {
        return sanitize(String((first as Record<string, unknown>).msg));
      }
    }
    if (typeof e.message === "string") return sanitize(e.message);
  }

  return GENERIC_FALLBACK;
}

/** Exposed for tests — covers sanitization logic without HTTP mapping. */
export const __test__ = { sanitize, GENERIC_FALLBACK, AUTH_EXPIRED, FORBIDDEN };
