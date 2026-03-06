import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Package } from "lucide-react";
import { KpiCard } from "@/components/KpiCard";

describe("KpiCard", () => {
  it("renders without crashing", () => {
    render(<KpiCard label="Fill Rate" value="95.2%" />);
  });

  it("displays the value prop", () => {
    render(<KpiCard label="Fill Rate" value="95.2%" />);
    expect(screen.getByText("95.2%")).toBeInTheDocument();
  });

  it("displays the label prop", () => {
    render(<KpiCard label="Fill Rate" value="95.2%" />);
    expect(screen.getByText("Fill Rate")).toBeInTheDocument();
  });

  it("displays the sublabel when provided", () => {
    render(<KpiCard label="Accuracy" value="88%" sublabel="(last 3 months)" />);
    expect(screen.getByText("(last 3 months)")).toBeInTheDocument();
  });

  it("does not render sublabel when not provided", () => {
    const { container } = render(<KpiCard label="Accuracy" value="88%" />);
    // Only the label text should be in the <p> — no extra span
    const labelEl = container.querySelector("p.text-xs");
    expect(labelEl?.querySelectorAll("span").length).toBe(0);
  });

  it("renders TrendingUp icon for upward trend direction", () => {
    const { container } = render(
      <KpiCard
        label="Units Sold"
        value="1,200"
        trend={{ delta: 5.3, direction: "up" }}
      />
    );
    // The trend row renders a +delta text
    expect(screen.getByText(/\+5\.3%/)).toBeInTheDocument();
    // An SVG icon is present in the trend row
    const trendRow = container.querySelector(".flex.items-center.gap-1.text-xs");
    expect(trendRow).not.toBeNull();
    expect(trendRow!.querySelector("svg")).not.toBeNull();
  });

  it("renders TrendingDown icon for downward trend direction", () => {
    const { container } = render(
      <KpiCard
        label="Units Sold"
        value="800"
        trend={{ delta: -3.7, direction: "down" }}
      />
    );
    expect(screen.getByText(/-3\.7%/)).toBeInTheDocument();
    const trendRow = container.querySelector(".flex.items-center.gap-1.text-xs");
    expect(trendRow).not.toBeNull();
    expect(trendRow!.querySelector("svg")).not.toBeNull();
  });

  it("renders Minus icon for flat trend direction", () => {
    const { container } = render(
      <KpiCard
        label="Bias"
        value="0.0%"
        trend={{ delta: 0, direction: "flat" }}
      />
    );
    expect(screen.getByText("0.0% vs prior")).toBeInTheDocument();
    const trendRow = container.querySelector(".flex.items-center.gap-1.text-xs");
    expect(trendRow).not.toBeNull();
    expect(trendRow!.querySelector("svg")).not.toBeNull();
  });

  it("does not render trend row when trend prop is omitted", () => {
    const { container } = render(<KpiCard label="Accuracy" value="90%" />);
    const trendRow = container.querySelector(".flex.items-center.gap-1.text-xs");
    expect(trendRow).toBeNull();
  });

  it("renders sparkline SVG when sparkline with 2+ points is provided", () => {
    const { container } = render(
      <KpiCard label="Sales" value="5,000" sparkline={[100, 200, 150, 300, 250]} />
    );
    const svg = container.querySelector("svg");
    expect(svg).not.toBeNull();
    const polyline = svg!.querySelector("polyline");
    expect(polyline).not.toBeNull();
  });

  it("does not render sparkline when sparkline has fewer than 2 points", () => {
    const { container } = render(
      <KpiCard label="Sales" value="5,000" sparkline={[100]} />
    );
    const svg = container.querySelector("svg");
    expect(svg).toBeNull();
  });

  it("does not render sparkline when sparkline prop is not provided", () => {
    const { container } = render(<KpiCard label="Sales" value="5,000" />);
    const svg = container.querySelector("svg");
    expect(svg).toBeNull();
  });

  it("applies colorClass to the value element", () => {
    const { container } = render(
      <KpiCard label="Revenue" value="$1M" colorClass="text-green-600" />
    );
    const valueEl = container.querySelector("p.text-xl");
    expect(valueEl?.className).toContain("text-green-600");
  });

  it("applies borderClass to the root element", () => {
    const { container } = render(
      <KpiCard label="Revenue" value="$1M" borderClass="border-green-500" />
    );
    const root = container.firstChild as HTMLElement;
    expect(root.className).toContain("border-green-500");
  });

  it("applies kpi-best CSS class on value when severity is 'best'", () => {
    const { container } = render(
      <KpiCard label="Accuracy" value="99%" severity="best" />
    );
    const valueEl = container.querySelector("p.text-xl");
    expect(valueEl?.className).toContain("text-[var(--kpi-best)]");
  });

  it("applies kpi-warning CSS class on value when severity is 'warning'", () => {
    const { container } = render(
      <KpiCard label="Stock Outs" value="12" severity="warning" />
    );
    const valueEl = container.querySelector("p.text-xl");
    expect(valueEl?.className).toContain("text-[var(--kpi-warning)]");
  });

  it("does not apply severity class when severity is 'neutral'", () => {
    const { container } = render(
      <KpiCard label="Neutral" value="50" severity="neutral" />
    );
    const valueEl = container.querySelector("p.text-xl");
    expect(valueEl?.className).not.toContain("text-[var(--kpi-best)]");
    expect(valueEl?.className).not.toContain("text-[var(--kpi-warning)]");
  });

  it("renders icon when icon prop is provided", () => {
    const { container } = render(
      <KpiCard label="Inventory" value="200" icon={Package} />
    );
    // Icon renders as an SVG inside the card header row
    const svg = container.querySelector("svg");
    expect(svg).not.toBeNull();
  });

  it("does not render icon element when icon prop is omitted", () => {
    const { container } = render(<KpiCard label="Inventory" value="200" />);
    // No SVG icon — only check that there's no icon SVG (sparkline is also absent here)
    const icons = container.querySelectorAll("svg");
    expect(icons.length).toBe(0);
  });

  it("renders trend with positive delta showing '+' prefix", () => {
    render(<KpiCard label="Test" value="42" trend={{ delta: 12.5, direction: "up" }} />);
    expect(screen.getByText("+12.5% vs prior")).toBeInTheDocument();
  });

  it("renders trend with negative delta without double minus", () => {
    render(<KpiCard label="Test" value="42" trend={{ delta: -8.0, direction: "down" }} />);
    expect(screen.getByText("-8.0% vs prior")).toBeInTheDocument();
  });
});
