const numberFmt = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
const compactNumberFmt = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });

export function formatCompactNumber(value: number | string): string {
  if (typeof value !== "number") {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? compactNumberFmt.format(parsed) : String(value);
  }
  return compactNumberFmt.format(value);
}

export function formatNumber(value: number | null | undefined): string {
  if (value === null || value === undefined) return "-";
  return numberFmt.format(value);
}

export function formatPercent(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "-";
  return `${numberFmt.format(value)}%`;
}

// Case-insensitive sentinel strings that some APIs/MVs emit (as text, not JSON
// null) for empty cells — mirrors the load-time '' / 'null' / 'none' / 'NA' →
// NULL normalization rule. Rendered as "-", same as a genuine null (U6.2).
const NULL_SENTINELS = new Set(["null", "none", "na", "undefined"]);

/** True when a value is null/undefined/empty or a case-insensitive null sentinel string. */
export function isEmptyCell(value: unknown): boolean {
  if (value === null || value === undefined || value === "") return true;
  return typeof value === "string" && NULL_SENTINELS.has(value.trim().toLowerCase());
}

export function formatCell(value: unknown): string {
  if (isEmptyCell(value)) return "-";
  if (typeof value === "number") return formatNumber(value);
  return String(value);
}

// U6.8 — a single deterministic date formatter. Five tabs previously used
// `new Date(x).toLocaleDateString()`, which renders host-locale-dependent
// "4/1/2026" (ambiguous m/d vs d/m). This fixes `en-US` short-month output
// ("Apr 1, 2026"), matching the "MMM" family used by month/chart labels.
// `timeZone: "UTC"` so a date-only ISO string ("2026-04-01", parsed as UTC
// midnight) renders as that same calendar day regardless of the host's
// timezone offset — avoiding the classic off-by-one ("Mar 31") in negative-UTC
// locales.
const dateFmt = new Intl.DateTimeFormat("en-US", {
  year: "numeric",
  month: "short",
  day: "numeric",
  timeZone: "UTC",
});

/** Format an ISO date/timestamp as a deterministic "Apr 1, 2026"; "—" when empty/unparseable. */
export function formatDate(value: string | null | undefined): string {
  if (value === null || value === undefined || value === "") return "—";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return "—";
  return dateFmt.format(d);
}

export function titleCase(value: string): string {
  return value
    .split("_")
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

/** Format a nullable number to fixed decimal places; returns "—" for null/undefined/NaN */
export function formatFixed(value: number | null | undefined, decimals = 1): string {
  if (value == null || isNaN(value as number)) return "—";
  return Number(value).toFixed(decimals);
}

/** Format a nullable number as an integer; returns "—" for null/undefined/NaN */
export function formatInt(value: number | null | undefined): string {
  if (value == null || isNaN(value as number)) return "—";
  return Math.round(Number(value)).toLocaleString();
}

/** Format a nullable number as a percentage with 1 decimal place; returns "—" for null/undefined */
export function formatPct(value: number | null | undefined, decimals = 1): string {
  if (value == null || isNaN(value as number)) return "—";
  return `${Number(value).toFixed(decimals)}%`;
}

/** Format a nullable number as USD currency (compact notation); returns "—" for null/undefined */
export function formatCurrency(value: number | null | undefined): string {
  if (value == null || isNaN(value as number)) return "—";
  const n = Number(value);
  if (Math.abs(n) >= 1_000_000) return `$${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 1_000) return `$${(n / 1_000).toFixed(1)}K`;
  return `$${n.toFixed(0)}`;
}

/**
 * Format a number as compact K/M (e.g. 1234567 → "1.2M", 12345 → "12.3K").
 * Intentionally uses our own thresholds rather than `Intl.NumberFormat`
 * compact notation — Intl rounds more aggressively (12345 → "12K" vs our
 * "12.3K"), and several panels already render this exact format.
 *
 * Replaces 5 inline `fmtNum` definitions across customer-analytics panels.
 */
export function formatCompactKMB(value: number | null | undefined): string {
  if (value == null || isNaN(value as number)) return "—";
  const n = Number(value);
  if (Math.abs(n) >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (Math.abs(n) >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return n.toFixed(0);
}

// ---------------------------------------------------------------------------
// Cluster label formatting — standardized 4-letter code system
// ---------------------------------------------------------------------------
//
// Two-axis taxonomy using 4-letter supply chain velocity codes:
//   [VELOCITY].[BEHAVIOR]  or  [VELOCITY].[B1].[B2].[B3]  for compounds
//
// ┌──────────┬──────┬──────────────────────────────────┐
// │ Axis     │ Code │ Meaning                          │
// ├──────────┼──────┼──────────────────────────────────┤
// │ Velocity │ FAST │ Fast mover — top volume drivers   │
// │          │ MOVR │ Mover — above-average runners     │
// │          │ BASE │ Base demand — core portfolio bulk  │
// │          │ SLOW │ Slow mover — below-average volume  │
// │          │ TAIL │ Long tail — niche / sparse SKUs    │
// ├──────────┼──────┼──────────────────────────────────┤
// │ Pattern  │ SEAS │ Seasonal amplitude                │
// │          │ CYCL │ Non-annual periodicity             │
// │          │ RISE │ Growing / accelerating ↑           │
// │          │ FALL │ Declining / decelerating ↓         │
// │          │ WILD │ High CV / volatile                 │
// │          │ CALM │ Low CV / stable / steady           │
// │          │ RARE │ Intermittent / sparse / dormant    │
// │          │ EVEN │ No dominant signal (balanced)      │
// │          │ FLAT │ No trend, no seasonality           │
// │          │ BUMP │ Lumpy / bursty demand              │
// │          │ NEWW │ Emerging / new product             │
// │          │ EOLP │ End-of-life / phasing out          │
// └──────────┴──────┴──────────────────────────────────┘
//
// Examples:
//   very_high_volume_periodic            → FAST.CYCL
//   high_volume_seasonal_growing         → MOVR.SEAS.RISE
//   medium_volume_steady                 → BASE.CALM
//   low_volume_volatile                  → SLOW.WILD
//   very_low_volume_intermittent         → TAIL.RARE
//   medium_volume_moderate               → BASE.EVEN
//   high_volume_declining                → MOVR.FALL
//   low_volume_seasonal_growing_volatile → SLOW.SEAS.RISE.WILD

const VOLUME_CODE: [RegExp, string][] = [
  [/very[_\s]?high/, "FAST"],
  [/very[_\s]?low/, "TAIL"],
  [/high/, "MOVR"],
  [/medium/, "BASE"],
  [/low/, "SLOW"],
];

const PATTERN_CODE: Record<string, string> = {
  // Seasonality & cycles
  periodic: "CYCL",
  seasonal: "SEAS",
  cyclical: "CYCL",
  // Stability
  moderate: "EVEN",
  steady: "CALM",
  very_steady: "CALM",
  stable: "CALM",
  consistent: "CALM",
  // Volatility
  volatile: "WILD",
  very_volatile: "WILD",
  erratic: "WILD",
  noisy: "WILD",
  // Intermittency
  intermittent: "RARE",
  sparse: "RARE",
  dormant: "RARE",
  // Trend — up
  growing: "RISE",
  increasing: "RISE",
  trending_up: "RISE",
  upward: "RISE",
  accelerating: "RISE",
  // Trend — down
  declining: "FALL",
  decreasing: "FALL",
  trending_down: "FALL",
  downward: "FALL",
  shrinking: "FALL",
  decelerating: "FALL",
  flat: "FLAT",
  // Shape
  lumpy: "BUMP",
  bursty: "BUMP",
  peaky: "BUMP",
  smooth: "CALM",
  // Lifecycle
  new: "NEWW",
  emerging: "NEWW",
  mature: "CALM",
  phasing_out: "EOLP",
  end_of_life: "EOLP",
  // Zero-demand
  zero_heavy: "RARE",
  stockout_prone: "RARE",
  // Disambiguator sub-range splits
  higher_avg: "HIGH",
  lower_avg: "LITE",
};

export function formatClusterLabel(raw: string): string {
  const s = raw.toLowerCase().replace(/-/g, "_");

  // --- Volume axis ---
  let vol = "";
  for (const [re, code] of VOLUME_CODE) {
    if (re.test(s)) { vol = code; break; }
  }

  // --- Pattern axis ---
  const stripped = s
    .replace(/very[_\s]?(high|low)/g, "")
    .replace(/\b(high|medium|low|volume|demand)\b/g, "")
    .replace(/[_\s]+/g, " ")
    .trim();

  const tokens = stripped.split(" ").filter(Boolean);
  const seen = new Set<string>();
  const mapped: string[] = [];
  for (const w of tokens) {
    const m = PATTERN_CODE[w];
    if (m && !seen.has(m)) { seen.add(m); mapped.push(m); }
  }
  // Up to 3 pattern codes for compound behaviors
  const codes = mapped.length > 0
    ? mapped.slice(0, 3)
    : tokens.map((w) => w.slice(0, 4).toUpperCase());

  const parts = vol ? [vol, ...codes] : codes;
  if (parts.length > 0) return parts.join(".");

  return raw.replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase()).slice(0, 22);
}
