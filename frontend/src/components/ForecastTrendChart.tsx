import { EChartContainer } from "@/components/EChartContainer";

interface ForecastTrendChartProps {
  data: { month: string; forecast: number; actual: number }[];
  theme: "light" | "dark";
  chartColors: { grid: string; axis: string; tooltip: string };
  seriesColors: string[];
}

export function ForecastTrendChart({ data, theme, chartColors, seriesColors }: ForecastTrendChartProps) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center py-12 text-sm text-muted-foreground">
        No forecast data available
      </div>
    );
  }

  const months = data.map((d) => d.month);
  const forecasts = data.map((d) => d.forecast);
  const actuals = data.map((d) => d.actual);

  const option = {
    tooltip: {
      trigger: "axis" as const,
      backgroundColor: chartColors.tooltip,
      borderColor: chartColors.grid,
      textStyle: { color: theme === "dark" ? "#e5e5e5" : "#171717", fontSize: 12 },
    },
    legend: {
      data: ["Forecast", "Actual"],
      bottom: 0,
      textStyle: { color: chartColors.axis, fontSize: 11 },
    },
    grid: { top: 16, right: 16, bottom: 36, left: 60, containLabel: false },
    xAxis: {
      type: "category" as const,
      data: months,
      axisLine: { lineStyle: { color: chartColors.grid } },
      axisLabel: { color: chartColors.axis, fontSize: 10 },
    },
    yAxis: {
      type: "value" as const,
      axisLine: { show: false },
      splitLine: { lineStyle: { color: chartColors.grid, type: "dashed" as const } },
      axisLabel: {
        color: chartColors.axis,
        fontSize: 10,
        formatter: (v: number) => {
          if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
          if (v >= 1_000) return `${(v / 1_000).toFixed(0)}K`;
          return String(v);
        },
      },
    },
    series: [
      {
        name: "Forecast",
        type: "line" as const,
        data: forecasts,
        smooth: true,
        areaStyle: { opacity: 0.15 },
        lineStyle: { width: 2, color: seriesColors[0] },
        itemStyle: { color: seriesColors[0] },
        symbol: "none",
      },
      {
        name: "Actual",
        type: "line" as const,
        data: actuals,
        smooth: true,
        lineStyle: { width: 2, color: seriesColors[1], type: "dashed" as const },
        itemStyle: { color: seriesColors[1] },
        symbol: "circle",
        symbolSize: 4,
      },
    ],
  };

  return <EChartContainer option={option} theme={theme} height={260} />;
}
