import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Breadcrumbs } from "@/components/Breadcrumbs";

describe("Breadcrumbs", () => {
  it("renders nothing when items is empty", () => {
    const { container } = render(<Breadcrumbs items={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders a Tab > Item > LOC trail (drill pattern)", () => {
    render(
      <Breadcrumbs
        items={[
          { label: "Item Analysis", onClick: vi.fn() },
          { label: "Item 100320", onClick: vi.fn() },
          { label: "1401-BULK" },
        ]}
      />,
    );
    expect(screen.getByText("Item Analysis")).toBeInTheDocument();
    expect(screen.getByText("Item 100320")).toBeInTheDocument();
    expect(screen.getByText("1401-BULK")).toBeInTheDocument();
  });

  it("marks the final segment with aria-current='page'", () => {
    render(
      <Breadcrumbs
        items={[
          { label: "Cluster Tab", onClick: vi.fn() },
          { label: "High-Volatility Weekly" },
        ]}
      />,
    );
    const terminal = screen.getByText("High-Volatility Weekly");
    expect(terminal).toHaveAttribute("aria-current", "page");
  });

  it("calls onClick when a non-terminal segment is clicked", () => {
    const onClick = vi.fn();
    render(
      <Breadcrumbs
        items={[
          { label: "Model Tuning", onClick },
          { label: "Run abc-123" },
        ]}
      />,
    );
    fireEvent.click(screen.getByRole("button", { name: "Model Tuning" }));
    expect(onClick).toHaveBeenCalledTimes(1);
  });

  it("renders a non-interactive span when segment has no onClick", () => {
    render(
      <Breadcrumbs
        items={[
          { label: "Plain Tab" },
          { label: "Detail" },
        ]}
      />,
    );
    // First segment should NOT be a button since no onClick.
    expect(
      screen.queryByRole("button", { name: "Plain Tab" }),
    ).not.toBeInTheDocument();
  });

  it("wraps output in nav with aria-label 'Breadcrumb'", () => {
    render(
      <Breadcrumbs items={[{ label: "A" }, { label: "B" }]} />,
    );
    expect(screen.getByRole("navigation", { name: "Breadcrumb" })).toBeInTheDocument();
  });
});
