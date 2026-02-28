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
  option: echarts.EChartsOption;
  theme: Theme;
  height?: string | number;
  className?: string;
};

function EChartContainerInner({ option, theme, height = 380, className }: EChartContainerProps) {
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

  const echartsTheme = isDark ? "dark" : undefined;

  return (
    <ReactEChartsCore
      echarts={echarts}
      option={mergedOption}
      theme={echartsTheme}
      style={{ height, width: "100%" }}
      className={className}
      notMerge
      lazyUpdate
    />
  );
}

export const EChartContainer = memo(EChartContainerInner);
