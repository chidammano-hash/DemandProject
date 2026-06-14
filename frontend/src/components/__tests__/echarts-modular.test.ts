import { describe, it, expect } from "vitest";
import { mergeEchartsProps } from "../echarts-modular";

/**
 * U5.2 — the 7 Customer-Analytics ECharts panels pass `style={{ height: N }}`
 * with no width. ECharts sizes to its container's measured width; before the
 * flex container settles that width can read 0, collapsing the chart (the
 * flagship Customer Concentration treemap renders as a single rectangle).
 * The wrapper must default `width: "100%"` so every panel paints full-width
 * on first render, while still letting callers override.
 */
describe("mergeEchartsProps (U5.2)", () => {
  it("defaults style.width to 100% when caller omits it", () => {
    const merged = mergeEchartsProps({ style: { height: 360 } });
    expect(merged.style).toMatchObject({ height: 360, width: "100%" });
  });

  it("injects a width even when no style is provided", () => {
    const merged = mergeEchartsProps({});
    expect(merged.style).toMatchObject({ width: "100%" });
  });

  it("does not override an explicit caller width", () => {
    const merged = mergeEchartsProps({ style: { height: 480, width: 600 } });
    expect(merged.style).toMatchObject({ height: 480, width: 600 });
  });

  it("keeps the sane notMerge/lazyUpdate defaults and honors overrides", () => {
    expect(mergeEchartsProps({})).toMatchObject({ notMerge: false, lazyUpdate: true });
    expect(mergeEchartsProps({ notMerge: true }).notMerge).toBe(true);
  });
});
