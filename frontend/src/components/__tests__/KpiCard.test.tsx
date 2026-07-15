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
    expect(valueEl?.className).toContain("text-kpi-best");
  });

  it("applies kpi-warning CSS class on value when severity is 'warning'", () => {
    const { container } = render(
      <KpiCard label="Stock Outs" value="12" severity="warning" />
    );
    const valueEl = container.querySelector("p.text-xl");
    expect(valueEl?.className).toContain("text-kpi-warning");
  });

  it("does not apply severity class when severity is 'neutral'", () => {
    const { container } = render(
      <KpiCard label="Neutral" value="50" severity="neutral" />
    );
    const valueEl = container.querySelector("p.text-xl");
    expect(valueEl?.className).not.toContain("text-kpi-best");
    expect(valueEl?.className).not.toContain("text-kpi-warning");
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

  it("renders HelpCircle icon when tooltip prop is provided", () => {
    const { container } = render(
      <KpiCard
        label="Fill Rate"
        value="95%"
        tooltip={{ title: "Fill Rate", description: "% of orders fulfilled on time" }}
      />
    );
    // HelpCircle renders as an SVG inside a span with cursor-help
    const helpSpan = container.querySelector("span.cursor-help");
    expect(helpSpan).not.toBeNull();
    expect(helpSpan!.querySelector("svg")).not.toBeNull();
  });

  it("sets title attribute on tooltip span with title and description", () => {
    const { container } = render(
      <KpiCard
        label="Fill Rate"
        value="95%"
        tooltip={{ title: "Fill Rate", description: "% of orders fulfilled on time" }}
      />
    );
    const helpSpan = container.querySelector("span.cursor-help");
    expect(helpSpan?.getAttribute("title")).toBe("Fill Rate — % of orders fulfilled on time");
  });

  it("includes threshold in title attribute when provided", () => {
    const { container } = render(
      <KpiCard
        label="Fill Rate"
        value="95%"
        tooltip={{ title: "Fill Rate", description: "% of orders fulfilled on time", threshold: "Target: 98%" }}
      />
    );
    const helpSpan = container.querySelector("span.cursor-help");
    expect(helpSpan?.getAttribute("title")).toBe("Fill Rate — % of orders fulfilled on time — Target: 98%");
  });

  it("does not render HelpCircle when tooltip prop is omitted", () => {
    const { container } = render(<KpiCard label="Fill Rate" value="95%" />);
    const helpSpan = container.querySelector("span.cursor-help");
    expect(helpSpan).toBeNull();
  });

  it("renders target sub-line with default label when target prop is provided", () => {
    render(<KpiCard label="Fill Rate" value="95%" target={{ value: "98%" }} />);
    expect(screen.getByText("Target: 98%")).toBeInTheDocument();
  });

  it("renders target sub-line with custom label when target.label is provided", () => {
    render(<KpiCard label="Fill Rate" value="95%" target={{ value: "98%", label: "Goal" }} />);
    expect(screen.getByText("Goal: 98%")).toBeInTheDocument();
  });

  it("does not render target sub-line when target prop is omitted", () => {
    render(<KpiCard label="Fill Rate" value="95%" />);
    // Neither "Target:" nor any muted sub-line should appear
    expect(screen.queryByText(/Target:/)).toBeNull();
  });

  it("renders text-2xl class for size='lg'", () => {
    const { container } = render(
      <KpiCard label="Hero KPI" value="99%" size="lg" />
    );
    const valueEl = container.querySelector("p.tabular-nums");
    expect(valueEl?.className).toContain("text-2xl");
    expect(valueEl?.className).toContain("font-bold");
  });

  it("renders text-base class for size='sm'", () => {
    const { container } = render(
      <KpiCard label="Compact KPI" value="42" size="sm" />
    );
    const valueEl = container.querySelector("p.tabular-nums");
    expect(valueEl?.className).toContain("text-base");
    expect(valueEl?.className).toContain("font-semibold");
  });

  it("renders text-xl class for default size (backward compat)", () => {
    const { container } = render(
      <KpiCard label="Default KPI" value="88%" />
    );
    const valueEl = container.querySelector("p.tabular-nums");
    expect(valueEl?.className).toContain("text-xl");
    expect(valueEl?.className).toContain("font-bold");
  });

  it("renders inset accent span with kpi-best bg for size='lg' severity='best'", () => {
    const { container } = render(
      <KpiCard label="Hero" value="99%" size="lg" severity="best" />
    );
    // Root must NOT carry the old heavy border classes
    const root = container.firstChild as HTMLElement;
    expect(root.className).not.toContain("border-l-4");
    // The inset accent span must be present with the correct token bg class
    const accentSpan = container.querySelector("span[aria-hidden='true']");
    expect(accentSpan).not.toBeNull();
    expect(accentSpan!.className).toContain("bg-kpi-best");
    expect(accentSpan!.className).toContain("rounded-r-full");
  });

  it("does not render accent span for size='md' even with severity", () => {
    const { container } = render(
      <KpiCard label="Normal" value="99%" size="md" severity="best" />
    );
    const root = container.firstChild as HTMLElement;
    expect(root.className).not.toContain("border-l-4");
    const accentSpan = container.querySelector("span[aria-hidden='true']");
    expect(accentSpan).toBeNull();
  });

  it("does not render sparkline for size='sm' even with data", () => {
    const { container } = render(
      <KpiCard label="Compact" value="5,000" size="sm" sparkline={[100, 200, 150, 300, 250]} />
    );
    const svg = container.querySelector("svg");
    expect(svg).toBeNull();
  });

  // U6.1 — the displayed delta sign must always reflect the TRUE movement,
  // decoupled from the good/bad color. For a lower-is-better metric (WAPE),
  // a -1.9 delta is an improvement: it must DISPLAY "-1.9" but be colored GREEN.
  describe("U6.1 goodDirection decouples display-sign from color", () => {
    it("displays the raw (true) delta sign — negative stays negative", () => {
      render(
        <KpiCard label="WAPE %" value="26.1%" trend={{ delta: -1.9, direction: "down", goodDirection: "down", unit: "pp" }} />
      );
      // The true movement (-1.9) is shown, NOT a flipped/positive value.
      expect(screen.getByText("-1.9pp vs prior")).toBeInTheDocument();
      expect(screen.queryByText("+1.9pp vs prior")).toBeNull();
    });

    it("colors an improvement GREEN even when the delta is negative (lower-is-better)", () => {
      const { container } = render(
        <KpiCard label="WAPE %" value="26.1%" trend={{ delta: -1.9, direction: "down", goodDirection: "down", unit: "pp" }} />
      );
      const trendRow = container.querySelector(".flex.items-center.gap-1.text-xs");
      expect(trendRow?.className).toContain("text-kpi-best");
      expect(trendRow?.className).not.toContain("text-kpi-warning");
    });

    it("colors a regression RED when a lower-is-better metric rises (positive delta)", () => {
      const { container } = render(
        <KpiCard label="WAPE %" value="30.0%" trend={{ delta: 2.0, direction: "up", goodDirection: "down", unit: "pp" }} />
      );
      expect(screen.getByText("+2.0pp vs prior")).toBeInTheDocument();
      const trendRow = container.querySelector(".flex.items-center.gap-1.text-xs");
      expect(trendRow?.className).toContain("text-kpi-warning");
    });

    it("colors a higher-is-better metric GREEN on a positive delta", () => {
      const { container } = render(
        <KpiCard label="Accuracy %" value="74%" trend={{ delta: 1.9, direction: "up", goodDirection: "up", unit: "pp" }} />
      );
      expect(screen.getByText("+1.9pp vs prior")).toBeInTheDocument();
      const trendRow = container.querySelector(".flex.items-center.gap-1.text-xs");
      expect(trendRow?.className).toContain("text-kpi-best");
    });

    it("renders a neutral color for a zero delta regardless of goodDirection", () => {
      const { container } = render(
        <KpiCard label="WAPE %" value="26.1%" trend={{ delta: 0, direction: "flat", goodDirection: "down", unit: "pp" }} />
      );
      const trendRow = container.querySelector(".flex.items-center.gap-1.text-xs");
      expect(trendRow?.className).toContain("text-muted-foreground");
    });
  });

  it("renders transition-all class on value for animated value changes", () => {
    const { container } = render(
      <KpiCard label="Test" value="42" />
    );
    const valueEl = container.querySelector("p.tabular-nums");
    expect(valueEl?.className).toContain("transition-all");
    expect(valueEl?.className).toContain("duration-300");
  });
});
