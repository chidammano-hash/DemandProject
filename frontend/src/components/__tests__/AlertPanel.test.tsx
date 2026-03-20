import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { AlertPanel } from "@/components/AlertPanel";
import type { Alert } from "@/types/theme";

const sampleAlerts: Alert[] = [
  { id: "a1", type: "low_accuracy", severity: "medium", title: "Medium accuracy", detail: "Detail medium", count: 12 },
  { id: "a2", type: "oos_risk", severity: "critical", title: "Critical OOS", detail: "Detail critical", count: 3 },
  { id: "a3", type: "bias_drift", severity: "low", title: "Low bias drift", detail: "Detail low" },
  { id: "a4", type: "demand_spike", severity: "high", title: "High spike", detail: "Detail high", count: 7 },
];

describe("AlertPanel", () => {
  it("renders alerts sorted by severity (critical > high > medium > low)", () => {
    const { container } = render(<AlertPanel alerts={sampleAlerts} />);
    // Get all alert title elements in order
    const titles = container.querySelectorAll("p.text-sm.font-medium");
    const titleTexts = Array.from(titles).map((el) => el.textContent);
    expect(titleTexts).toEqual([
      "Critical OOS",
      "High spike",
      "Medium accuracy",
      "Low bias drift",
    ]);
  });

  it("renders empty state message when no alerts", () => {
    render(<AlertPanel alerts={[]} />);
    expect(screen.getByText("No active alerts")).toBeInTheDocument();
  });

  it("shows count badges when count is provided", () => {
    render(<AlertPanel alerts={sampleAlerts} />);
    // Count badges for alerts that have a count
    expect(screen.getByText("3")).toBeInTheDocument();
    expect(screen.getByText("7")).toBeInTheDocument();
    expect(screen.getByText("12")).toBeInTheDocument();
  });

  it("renders alert title and detail", () => {
    render(<AlertPanel alerts={[sampleAlerts[0]]} />);
    expect(screen.getByText("Medium accuracy")).toBeInTheDocument();
    expect(screen.getByText("Detail medium")).toBeInTheDocument();
  });

  it("does not render count badge when count is not provided", () => {
    const alertWithoutCount: Alert[] = [
      { id: "a1", type: "bias_drift", severity: "low", title: "No count alert", detail: "No count detail" },
    ];
    const { container } = render(<AlertPanel alerts={alertWithoutCount} />);
    // Should not have any count badge span
    const countSpans = container.querySelectorAll("span.rounded-full");
    expect(countSpans.length).toBe(0);
  });

  it("applies custom className", () => {
    const { container } = render(<AlertPanel alerts={sampleAlerts} className="my-class" />);
    const root = container.firstChild as HTMLElement;
    expect(root.className).toContain("my-class");
  });
});
