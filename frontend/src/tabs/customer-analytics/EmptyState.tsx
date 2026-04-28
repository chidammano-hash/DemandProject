/**
 * Empty-state placeholder shown when a CA panel's filter combination returns
 * no data. Common case: user picks an item that has zero history in the
 * selected date range. Without this, panels render their decorative chrome
 * (axes, legends, US outline) but no actual content — looks broken even
 * though the API correctly returned an empty result.
 */
interface Props {
  height?: number | string;
  message?: string;
  hint?: string;
}

export function EmptyState({
  height = 360,
  message = "No data for the selected filters",
  hint = "Try a different item or widen the date range",
}: Props) {
  return (
    <div
      className="flex flex-col items-center justify-center text-sm text-muted-foreground gap-1"
      style={{ height }}
    >
      <span className="font-medium">{message}</span>
      {hint && <span className="text-xs">{hint}</span>}
    </div>
  );
}
