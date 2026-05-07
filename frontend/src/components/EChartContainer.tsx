import { memo, useMemo } from "react";
import ReactEChartsCore from "echarts-for-react/lib/core";
import * as echarts from "echarts/core";
import { LineChart } from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";
import type { Theme } from "@/types";
import { CHART_COLORS } from "@/constants/colors";
import { useThemeContext } from "@/context/ThemeContext";

// Register required ECharts modules
echarts.use([
  LineChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  DataZoomComponent,
  CanvasRenderer,
]);

type EChartContainerProps = {
  option: echarts.EChartsCoreOption;
  /** Optional override; defaults to ThemeContext. */
  theme?: Theme;
  height?: string | number;
  className?: string;
  /** Accessible label. Defaults to a generic "Chart" if not provided. */
  ariaLabel?: string;
};

type SeriesShape = {
  name?: unknown;
  type?: unknown;
  data?: unknown;
};

type CategoryAxisShape = {
  type?: unknown;
  data?: unknown;
};

/** Build a short accessible summary + visually-hidden data table from the option. */
function summarize(option: unknown): { label: string; table: string[][] | null } {
  if (!option || typeof option !== "object") {
    return { label: "Chart", table: null };
  }
  const opt = option as Record<string, unknown>;
  const seriesRaw = opt.series;
  const xAxisRaw = opt.xAxis;
  const xAxis: CategoryAxisShape | null =
    xAxisRaw && typeof xAxisRaw === "object"
      ? Array.isArray(xAxisRaw)
        ? ((xAxisRaw[0] as CategoryAxisShape) ?? null)
        : (xAxisRaw as CategoryAxisShape)
      : null;

  const seriesList: SeriesShape[] = Array.isArray(seriesRaw)
    ? (seriesRaw as SeriesShape[])
    : seriesRaw && typeof seriesRaw === "object"
      ? [seriesRaw as SeriesShape]
      : [];

  const seriesNames = seriesList
    .map((s) => (typeof s.name === "string" ? s.name : null))
    .filter((n): n is string => !!n);

  const pointCount = seriesList.reduce((max, s) => {
    const d = Array.isArray(s.data) ? s.data.length : 0;
    return Math.max(max, d);
  }, 0);

  const parts: string[] = ["Chart"];
  if (seriesNames.length > 0) parts.push(`with series ${seriesNames.join(", ")}`);
  if (pointCount > 0) parts.push(`${pointCount} data points`);
  const label = parts.join(", ");

  // Build a small data table fallback (screen-reader only).
  const categories = Array.isArray(xAxis?.data) ? (xAxis!.data as unknown[]) : null;
  if (categories && seriesList.length > 0) {
    const header = ["Category", ...seriesList.map((s, i) => (typeof s.name === "string" ? s.name : `Series ${i + 1}`))];
    const rows: string[][] = [header];
    const rowCount = Math.min(categories.length, 50);
    for (let i = 0; i < rowCount; i++) {
      const row: string[] = [String(categories[i])];
      for (const s of seriesList) {
        const d = Array.isArray(s.data) ? s.data[i] : null;
        if (d == null) {
          row.push("");
        } else if (typeof d === "number" || typeof d === "string") {
          row.push(String(d));
        } else if (typeof d === "object" && d && "value" in (d as object)) {
          row.push(String((d as { value: unknown }).value ?? ""));
        } else {
          row.push("");
        }
      }
      rows.push(row);
    }
    return { label, table: rows };
  }

  return { label, table: null };
}

function EChartContainerInner({
  option,
  theme: themeProp,
  height = 380,
  className,
  ariaLabel,
}: EChartContainerProps) {
  const { theme: ctxTheme } = useThemeContext();
  const theme: Theme = themeProp ?? ctxTheme;
  const colors = CHART_COLORS[theme];
  const isDark = theme === "dark";

  const mergedOption = useMemo(
    () => ({
      backgroundColor: "transparent",
      textStyle: { color: colors.axis },
      ...option,
    }),
    [option, colors.axis],
  );

  const { label, table } = useMemo(() => summarize(option), [option]);
  const resolvedLabel = ariaLabel ?? label;

  const echartsTheme = isDark ? "dark" : undefined;

  return (
    <div role="img" aria-label={resolvedLabel} className={className}>
      <ReactEChartsCore
        echarts={echarts}
        option={mergedOption}
        theme={echartsTheme}
        style={{ height, width: "100%" }}
        notMerge
        lazyUpdate
      />
      {table && (
        <table className="sr-only" aria-label={`${resolvedLabel} data table`}>
          <thead>
            <tr>
              {table[0].map((h, i) => (
                <th key={i} scope="col">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {table.slice(1).map((row, ri) => (
              <tr key={ri}>
                {row.map((cell, ci) =>
                  ci === 0 ? (
                    <th key={ci} scope="row">{cell}</th>
                  ) : (
                    <td key={ci}>{cell}</td>
                  ),
                )}
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export const EChartContainer = memo(EChartContainerInner);
