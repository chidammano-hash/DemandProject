/**
 * Shared query helpers to reduce duplication across query modules.
 *
 * - `buildQueryString()` — builds a URL query string from a Record, filtering
 *   out `undefined`, `null`, and empty-string values.
 * - `buildSearchParams()` — same filtering but returns a `URLSearchParams`
 *   instance for callers that need to append additional params.
 */

// ---------------------------------------------------------------------------
// URLSearchParams builder — filters out undefined/null/empty values
// ---------------------------------------------------------------------------

type ParamValue = string | number | boolean | null | undefined;

/**
 * Build a URLSearchParams instance from a flat record.
 * Entries whose value is `undefined`, `null`, or `""` are skipped.
 * Numbers and booleans are stringified automatically.
 */
export function buildSearchParams(
  params: Record<string, ParamValue>,
): URLSearchParams {
  const qs = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    qs.set(key, String(value));
  }
  return qs;
}

/**
 * Build a query string (without leading `?`) from a flat record.
 * Returns `""` when all values are empty.
 *
 * @example
 *   buildQueryString({ item: "A", loc: undefined, page: 1 })
 *   // => "item=A&page=1"
 */
export function buildQueryString(
  params: Record<string, ParamValue>,
): string {
  return buildSearchParams(params).toString();
}

/**
 * Build a full `?key=val&...` suffix, or `""` if the params are all empty.
 * Useful for appending to a base URL.
 *
 * @example
 *   `/api/things${buildQuerySuffix({ page: 1 })}`
 *   // => "/api/things?page=1"
 */
export function buildQuerySuffix(
  params: Record<string, ParamValue>,
): string {
  const qs = buildQueryString(params);
  return qs ? `?${qs}` : "";
}
