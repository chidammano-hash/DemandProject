import React, { useState } from "react";
import { cn } from "@/lib/utils";

interface HeatmapGridProps {
  rows: { label: string; values: number[] }[];
  columnLabels: string[];
  colorScale: (value: number) => string;
  valueFormat?: (value: number) => string;
  onCellClick?: (row: string, col: string) => void;
  className?: string;
}

const defaultFormat = (v: number) => `${v.toFixed(1)}%`;

export function HeatmapGrid({
  rows,
  columnLabels,
  colorScale,
  valueFormat = defaultFormat,
  onCellClick,
  className,
}: HeatmapGridProps) {
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);

  if (rows.length === 0) {
    return (
      <div className={cn("flex items-center justify-center py-8 text-sm text-muted-foreground", className)}>
        No data available
      </div>
    );
  }

  return (
    <div className={cn("overflow-x-auto", className)}>
      <div
        className="grid gap-px"
        style={{
          gridTemplateColumns: `120px repeat(${columnLabels.length}, minmax(60px, 1fr))`,
        }}
        role="grid"
        aria-label="Performance heatmap"
      >
        {/* Header row */}
        <div className="text-xs font-medium text-muted-foreground" />
        {columnLabels.map((label) => (
          <div key={label} className="truncate px-1 text-center text-[10px] font-medium text-muted-foreground">
            {label}
          </div>
        ))}

        {/* Data rows */}
        {rows.map((row, rowIdx) => (
          <React.Fragment key={row.label}>
            <div className="flex items-center truncate text-xs font-medium text-card-foreground">
              {row.label}
            </div>
            {row.values.map((value, colIdx) => {
              const bg = colorScale(value);
              const isHovered = hoveredCell?.row === rowIdx && hoveredCell?.col === colIdx;
              return (
                <div
                  key={`${row.label}-${colIdx}`}
                  className={cn(
                    "relative flex items-center justify-center rounded-sm px-1 py-2 text-[11px] font-medium tabular-nums transition-transform",
                    isHovered && "scale-110 z-10 shadow-md",
                    onCellClick && "cursor-pointer",
                  )}
                  style={{ backgroundColor: bg, color: value > 80 ? "#fff" : value < 60 ? "#fff" : "#1a1a1a" }}
                  onMouseEnter={() => setHoveredCell({ row: rowIdx, col: colIdx })}
                  onMouseLeave={() => setHoveredCell(null)}
                  onClick={() => onCellClick?.(row.label, columnLabels[colIdx])}
                  role="gridcell"
                  aria-label={`${row.label}, ${columnLabels[colIdx]}: ${valueFormat(value)}`}
                >
                  {valueFormat(value)}
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>
    </div>
  );
}

/** Default 5-step heatmap color scale using theme heatmap colors */
export function makeHeatmapScale(scale: string[]): (value: number) => string {
  return (value: number) => {
    if (value >= 95) return scale[0]; // excellent
    if (value >= 85) return scale[1]; // good
    if (value >= 70) return scale[2]; // warning
    if (value >= 50) return scale[3]; // poor
    return scale[4]; // critical
  };
}
