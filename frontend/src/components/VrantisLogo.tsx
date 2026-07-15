/**
 * Vrantis logo mark — a bold angular V on a petrol brand tile.
 *
 * Concept: two upward arms of unequal height (right arm rises higher) suggest
 * a rising demand forecast. The violet accent node at the apex of the right
 * arm represents an AI-driven data insight moment — the same violet the
 * palette reserves for AI-generated chart series (see `constants/palette.ts`).
 *
 * Colors read from CSS vars (`--primary`, `--chart-8`) so the mark recolors
 * per mode instead of carrying its own fixed hexes.
 *
 * Usage:
 *   <VrantisLogo size={32} />               — icon only (collapsed sidebar)
 *   <VrantisLogo size={28} showWordmark />   — icon + "Vrantis" wordmark
 */

interface VrantisLogoProps {
  size?: number;
  showWordmark?: boolean;
  className?: string;
}

export function VrantisLogo({ size = 32, showWordmark = false, className }: VrantisLogoProps) {
  return (
    <span className={`flex items-center gap-2 select-none ${className ?? ""}`}>
      {/* Icon mark */}
      <svg
        width={size}
        height={size}
        viewBox="0 0 36 36"
        fill="none"
        xmlns="http://www.w3.org/2000/svg"
        aria-label="Vrantis"
        role="img"
        style={{ flexShrink: 0 }}
      >
        {/* Rounded background tile — brand petrol, mode-aware via --primary */}
        <rect width="36" height="36" rx="9" fill="hsl(var(--primary))" />

        {/*
          V shape: left arm descends to vertex, right arm rises higher than left.
          Left starts at (7, 13), vertex at (17, 27), right ends at (29, 10).
          Asymmetry (right arm ~3px higher) reads as an upward demand signal.
        */}
        <polyline
          points="7,13 17,27 29,10"
          stroke="white"
          strokeWidth="3.8"
          strokeLinecap="round"
          strokeLinejoin="round"
        />

        {/* Vertex node — convergence of signals */}
        <circle cx="17" cy="27" r="2.2" fill="white" opacity="0.95" />

        {/* Violet accent node at right-arm tip — AI insight moment (ai role) */}
        <circle cx="29" cy="10" r="2.4" fill="hsl(var(--chart-8))" />
        <circle cx="29" cy="10" r="1.1" fill="white" />
      </svg>

      {/* Wordmark */}
      {showWordmark && (
        <span
          className="text-primary"
          style={{
            fontWeight: 700,
            fontSize: `${size * 0.44}px`,
            letterSpacing: "-0.01em",
            lineHeight: 1,
          }}
        >
          Vrantis
        </span>
      )}
    </span>
  );
}
