import { cn } from "@/lib/utils";
import { getAllMotifs } from "@/constants/motifs";
import type { MotifId, MotifThemeConfig, TileConfig } from "@/types/motif";

function MotifTilePreview({ tile }: { tile: TileConfig }) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center rounded border w-8 h-8 shrink-0",
        tile.activeClasses,
        tile.glowClass,
      )}
      aria-hidden="true"
    >
      <span className="text-[7px] leading-none self-end font-mono opacity-50 pr-0.5">
        {tile.superscript}
      </span>
      <span className="text-xs font-black leading-tight font-mono">
        {tile.primary}
      </span>
    </div>
  );
}

export function MotifSettingsPanel({ currentMotifId, onSelect }: { currentMotifId: MotifId; onSelect: (id: MotifId) => void }) {
  const motifs = getAllMotifs();

  return (
    <div className="mt-3 pt-3 border-t border-border">
      <h3 className="text-sm font-semibold text-foreground mb-2">Theme Style</h3>
      <div className="flex flex-col gap-1">
        {motifs.map((m: MotifThemeConfig) => (
          <button
            key={m.id}
            aria-label={`Switch to ${m.displayName} theme style`}
            aria-pressed={currentMotifId === m.id}
            onClick={() => onSelect(m.id)}
            className={cn(
              "flex items-center gap-2 rounded-md border px-2 py-1.5 text-left transition-all duration-150",
              currentMotifId === m.id
                ? "border-primary bg-primary/10 ring-1 ring-primary/40"
                : "border-border hover:border-primary/40 hover:bg-muted/40"
            )}
          >
            <MotifTilePreview tile={m.previewTile} />
            <span className="text-xs font-medium text-foreground">{m.displayName}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
