import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { TopMovers } from "@/components/TopMovers";
import type { Mover } from "@/types/theme";

const sampleMovers: Mover[] = [
  { item_description: "Widget A", delta: 5000, pct_change: 12.5, direction: "up" },
  { item_description: "Gadget B", delta: -3200, pct_change: -8.1, direction: "down" },
  { item_description: "Gizmo C", delta: 1500000, pct_change: 45.0, direction: "up" },
];

describe("TopMovers", () => {
  it("renders movers with their descriptions", () => {
    render(<TopMovers movers={sampleMovers} />);
    expect(screen.getByText("Widget A")).toBeInTheDocument();
    expect(screen.getByText("Gadget B")).toBeInTheDocument();
    expect(screen.getByText("Gizmo C")).toBeInTheDocument();
  });

  it("renders movers with direction indicators (formatted deltas)", () => {
    render(<TopMovers movers={sampleMovers} />);
    // Widget A: +5000 = +5.0K
    expect(screen.getByText("+5.0K")).toBeInTheDocument();
    // Gadget B: -3200 = -3.2K (no plus sign for negative)
    expect(screen.getByText("-3.2K")).toBeInTheDocument();
    // Gizmo C: +1500000 = +1.5M
    expect(screen.getByText("+1.5M")).toBeInTheDocument();
  });

  it("renders empty state message when no movers", () => {
    render(<TopMovers movers={[]} />);
    expect(screen.getByText("No movers data")).toBeInTheDocument();
  });

  it("applies custom className", () => {
    const { container } = render(<TopMovers movers={sampleMovers} className="custom-class" />);
    const root = container.firstChild as HTMLElement;
    expect(root.className).toContain("custom-class");
  });

  it("applies className to empty state too", () => {
    const { container } = render(<TopMovers movers={[]} className="empty-cls" />);
    const root = container.firstChild as HTMLElement;
    expect(root.className).toContain("empty-cls");
  });
});
