import { memo } from "react";

// ---------------------------------------------------------------------------
// TogglePill — reusable pill button for series visibility
// ---------------------------------------------------------------------------
export const TogglePill = memo(function TogglePill({
  label,
  color,
  active,
  onClick,
  dashed,
  ring,
  suffix,
}: {
  label: string;
  color: string;
  active: boolean;
  onClick: () => void;
  dashed?: boolean;
  ring?: boolean;
  suffix?: string;
}) {
  return (
    <button
      onClick={onClick}
      className={[
        "flex items-center gap-1 rounded-full border px-2 py-0.5 text-[11px] transition-opacity",
        active ? "opacity-100" : "opacity-30",
        ring ? "ring-2 ring-primary ring-offset-1" : "",
      ]
        .filter(Boolean)
        .join(" ")}
      style={{ borderColor: color, color: active ? color : undefined }}
    >
      <span
        className="inline-block h-0.5 w-3"
        style={
          dashed
            ? { borderTop: `2px dashed ${color}`, backgroundColor: "transparent" }
            : { backgroundColor: color }
        }
      />
      {label}
      {suffix && (
        <span className="text-[9px] text-muted-foreground">({suffix})</span>
      )}
    </button>
  );
});
