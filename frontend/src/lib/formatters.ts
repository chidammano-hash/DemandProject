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

export function formatCell(value: unknown): string {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "number") return formatNumber(value);
  return String(value);
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
