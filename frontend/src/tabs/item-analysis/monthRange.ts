/**
 * U2.20 — Item Analysis FROM/TO range helpers.
 *
 * The history range selectors previously rendered raw `YYYY-MM-01` strings and
 * allowed an inverted (TO < FROM) selection that silently yielded an empty
 * chart. These pure helpers format the labels as "Mon YYYY" and encode the
 * inverted-range guard so each option can be disabled in the opposing select.
 */
const MONTH_NAMES = [
  "Jan", "Feb", "Mar", "Apr", "May", "Jun",
  "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/** Format a `YYYY-MM-01` month string as `"Mon YYYY"` (e.g. "Apr 2023"). */
export function formatMonthLabel(month: string): string {
  const m = /^(\d{4})-(\d{2})-\d{2}$/.exec(month);
  if (!m) return month;
  const year = m[1];
  const monthIdx = Number(m[2]) - 1;
  if (monthIdx < 0 || monthIdx > 11) return month;
  return `${MONTH_NAMES[monthIdx]} ${year}`;
}

/** A TO option is disabled when it falls before the selected FROM bound. */
export function isToDisabled(option: string, from: string): boolean {
  if (!from) return false;
  return option < from;
}

/** A FROM option is disabled when it falls after the selected TO bound. */
export function isFromDisabled(option: string, to: string): boolean {
  if (!to) return false;
  return option > to;
}

/**
 * Clamp forecast months to an explicit TO bound (empty bound = show all).
 * Without this the staged 24-month horizon stretches the axis years past
 * the selected range.
 */
export function clampFutureMonths(futureMonths: string[], to: string): string[] {
  if (!to) return futureMonths;
  return futureMonths.filter((month) => month <= to);
}
