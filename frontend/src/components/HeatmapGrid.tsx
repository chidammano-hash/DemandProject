import React, { useMemo, useState } from "react";
import { cn } from "@/lib/utils";

interface HeatmapGridProps {
  rows: { label: string; values: number[]; counts?: number[] }[];
  columnLabels: string[];
  colorScale: (value: number) => string;
  valueFormat?: (value: number) => string;
  onCellClick?: (row: string, col: string) => void;
  className?: string;
  /** Show a gradient legend bar below the heatmap */
  showLegend?: boolean;
  /** Label for the low end of the legend (default: "0%") */
  minLabel?: string;
  /** Label for the high end of the legend (default: "100%") */
  maxLabel?: string;
}

const defaultFormat = (v: number) => `${v.toFixed(1)}%`;

/** Sample the colorScale function at even intervals to build a CSS gradient */
function buildLegendGradient(colorScale: (value: number) => string): string {
  const stops = [0, 25, 50, 75, 100].map((v) => colorScale(v));
  return `linear-gradient(to right, ${stops.join(", ")})`;
}

export function HeatmapGrid({
  rows: rawRows,
  columnLabels: rawColumnLabels,
  colorScale,
  valueFormat = defaultFormat,
  onCellClick,
  className,
  showLegend = false,
  minLabel = "0%",
  maxLabel = "100%",
}: HeatmapGridProps) {
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);

  // ── Prune empty rows/columns (all DFU counts = 0) ─────────────────
  const { rows, columnLabels } = useMemo(() => {
    const hasCounts = rawRows.some((r) => r.counts && r.counts.length > 0);
    if (!hasCounts) return { rows: rawRows, columnLabels: rawColumnLabels };

    // Find columns that have at least one non-zero count across all rows
    const keepCol = rawColumnLabels.map((_, ci) =>
      rawRows.some((r) => (r.counts?.[ci] ?? 0) > 0),
    );
    // Find rows that have at least one non-zero count in a kept column
    const keepRow = rawRows.map((r) =>
      r.counts ? r.counts.some((c, ci) => keepCol[ci] && c > 0) : true,
    );

    const filteredCols = rawColumnLabels.filter((_, ci) => keepCol[ci]);
    const filteredRows = rawRows
      .filter((_, ri) => keepRow[ri])
      .map((r) => ({
        label: r.label,
        values: r.values.filter((_, ci) => keepCol[ci]),
        counts: r.counts?.filter((_, ci) => keepCol[ci]),
      }));

    return { rows: filteredRows, columnLabels: filteredCols };
  }, [rawRows, rawColumnLabels]);

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
              const isHovered = hoveredCell?.row === rowIdx && hoveredCell?.col === colIdx;
              const skuCount = row.counts?.[colIdx];
              const hasCounts = row.counts != null && row.counts.length > 0;
              const isEmpty = hasCounts && (skuCount ?? 0) === 0;
              const bg = isEmpty ? "var(--color-muted, #e5e7eb)" : colorScale(value);
              const formatted = isEmpty ? "" : valueFormat(value);
              const cellLabel = isEmpty
                ? `${row.label}, ${columnLabels[colIdx]}: no data`
                : `${row.label}, ${columnLabels[colIdx]}: ${formatted}${skuCount != null ? `, ${skuCount} SKUs` : ""}`;
              // U7.13 — give sighted hover the same context AT users already get
              // via aria-label, and on a floored "<0%*" cell append the low-base
              // note so a planner learns the marker's meaning without scrolling
              // to the caption below the grid.
              const cellTitle = !isEmpty && formatted.includes("<0%*")
                ? `${cellLabel} — Accuracy = 100 − WAPE; <0%* = tiny actual base, review WAPE`
                : cellLabel;
              return (
                <div
                  key={`${row.label}-${colIdx}`}
                  className={cn(
                    "relative flex items-center justify-center rounded-sm px-1 py-2 text-[11px] font-medium tabular-nums transition-all duration-150 hover:opacity-90",
                    isHovered && !isEmpty && "scale-110 z-10 shadow-md",
                    onCellClick && !isEmpty && "cursor-pointer",
                  )}
                  style={{
                    backgroundColor: bg,
                    color: isEmpty ? "transparent" : value > 80 ? "#fff" : value < 60 ? "#fff" : "#1a1a1a",
                  }}
                  onMouseEnter={() => !isEmpty && setHoveredCell({ row: rowIdx, col: colIdx })}
                  onMouseLeave={() => setHoveredCell(null)}
                  onClick={() => !isEmpty && onCellClick?.(row.label, columnLabels[colIdx])}
                  role="gridcell"
                  aria-label={cellLabel}
                  title={cellTitle}
                >
                  {isEmpty
                    ? "\u00A0"
                    : isHovered && skuCount != null
                      ? <>{valueFormat(value)} <span className="opacity-80">({skuCount.toLocaleString()})</span></>
                      : valueFormat(value)}
                </div>
              );
            })}
          </React.Fragment>
        ))}
      </div>

      {showLegend && (
        <div className="mt-3 flex items-center gap-2 px-1" aria-label="Heatmap legend">
          <span className="text-xs text-muted-foreground tabular-nums">{minLabel}</span>
          <div
            className="h-2 flex-1 rounded-full"
            style={{ background: buildLegendGradient(colorScale) }}
          />
          <span className="text-xs text-muted-foreground tabular-nums">{maxLabel}</span>
        </div>
      )}
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
