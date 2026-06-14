/**
 * Modular ECharts entry. Registers only the chart types + components that
 * the Customer Analytics panels actually use, so the bundle ships ~50% less
 * ECharts JS than the default `echarts-for-react` (which pulls in the full
 * library — every chart type, every component).
 *
 * Use `<ModularReactECharts ... />` exactly like the old `<ReactECharts ... />`
 * — same props, just a different component instance.
 */
import { forwardRef, useMemo } from "react";
import ReactEChartsCore from "echarts-for-react/lib/core";
import * as echarts from "echarts/core";
import {
  HeatmapChart,
  SunburstChart,
  TreemapChart,
  SankeyChart,
  ScatterChart,
  BarChart,
  LineChart,
  PieChart,
} from "echarts/charts";
import {
  GridComponent,
  TooltipComponent,
  LegendComponent,
  VisualMapComponent,
  TitleComponent,
  GraphicComponent,
  MarkAreaComponent,
  MarkLineComponent,
  DataZoomComponent,
} from "echarts/components";
import { CanvasRenderer } from "echarts/renderers";

echarts.use([
  HeatmapChart,
  SunburstChart,
  TreemapChart,
  SankeyChart,
  ScatterChart,
  BarChart,
  LineChart,
  PieChart,
  GridComponent,
  TooltipComponent,
  LegendComponent,
  VisualMapComponent,
  TitleComponent,
  GraphicComponent,
  MarkAreaComponent,
  MarkLineComponent,
  DataZoomComponent,
  CanvasRenderer,
]);

// Mirror the prop shape of the old `<ReactECharts ... />` so the call sites
// don't need to change beyond the import line.
type ReactEChartsProps = React.ComponentProps<typeof ReactEChartsCore>;
type Props = Omit<ReactEChartsProps, "echarts">;

/**
 * Compute the props handed to `ReactEChartsCore`. Beyond the sane
 * notMerge/lazyUpdate defaults, this defaults `style.width` to `"100%"`
 * (U5.2): ECharts measures its container's width on mount, and CA panels pass
 * only `{ height }`. Before a flex container settles, the measured width can
 * read 0, collapsing the chart (e.g. the treemap rendering as one rectangle).
 * Callers can still override width/height and the chart flags.
 */
export function mergeEchartsProps<T extends Partial<Props>>(props: T): T & { style: React.CSSProperties } {
  return {
    notMerge: false,
    lazyUpdate: true,
    ...props,
    style: { width: "100%", ...props.style },
  };
}

export const ModularReactECharts = forwardRef<unknown, Props>(function ModularReactECharts(
  props,
  _ref,
) {
  const merged = useMemo(() => mergeEchartsProps(props), [props]);
  return <ReactEChartsCore {...merged} echarts={echarts} />;
});

export default ModularReactECharts;
