/**
 * Shared recharts mock for Vitest.
 *
 * Vitest auto-discovers this file when any test calls `vi.mock("recharts")`
 * without a factory argument. Container components render their children
 * inside a <div> with a data-testid; leaf components render null.
 */
import React from "react";

/* ------------------------------------------------------------------ */
/* Container components — render children inside a div                */
/* ------------------------------------------------------------------ */

export const ResponsiveContainer = ({ children }: { children: React.ReactNode }) => (
  <div>{children}</div>
);

export const LineChart = ({ children }: { children: React.ReactNode }) => (
  <div data-testid="line-chart">{children}</div>
);

export const BarChart = ({ children }: { children: React.ReactNode }) => (
  <div data-testid="bar-chart">{children}</div>
);

export const ComposedChart = ({ children }: { children: React.ReactNode }) => (
  <div data-testid="composed-chart">{children}</div>
);

export const PieChart = ({ children }: { children: React.ReactNode }) => (
  <div data-testid="pie-chart">{children}</div>
);

export const RadarChart = ({ children }: { children: React.ReactNode }) => (
  <div data-testid="radar-chart">{children}</div>
);

export const ScatterChart = ({ children }: { children: React.ReactNode }) => (
  <div data-testid="scatter-chart">{children}</div>
);

export const AreaChart = ({ children }: { children: React.ReactNode }) => (
  <div data-testid="area-chart">{children}</div>
);

/* ------------------------------------------------------------------ */
/* Leaf components — render nothing                                   */
/* ------------------------------------------------------------------ */

export const Line = () => null;
export const Bar = () => null;
export const Area = () => null;
export const Cell = () => null;
export const XAxis = () => null;
export const YAxis = () => null;
export const ZAxis = () => null;
export const CartesianGrid = () => null;
export const Tooltip = () => null;
export const Legend = () => null;
export const ReferenceLine = () => null;
export const Pie = () => null;
export const Radar = () => null;
export const Scatter = () => null;
export const PolarGrid = () => null;
export const PolarAngleAxis = () => null;
export const PolarRadiusAxis = () => null;
