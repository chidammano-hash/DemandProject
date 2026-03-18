import { describe, it, expect } from "vitest";
import { render } from "@testing-library/react";
import { ChartSkeleton, KpiRowSkeleton } from "@/components/ChartSkeleton";

describe("ChartSkeleton", () => {
  it("renders with default height", () => {
    const { container } = render(<ChartSkeleton />);
    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper).toBeInTheDocument();
    expect(wrapper.className).toContain("space-y-2");
    // The large skeleton block should have the default 260px height
    const skeletons = wrapper.querySelectorAll(".animate-shimmer");
    const chartBlock = skeletons[skeletons.length - 1] as HTMLElement;
    expect(chartBlock.style.height).toBe("260px");
  });

  it("renders with custom height", () => {
    const { container } = render(<ChartSkeleton height={400} />);
    const wrapper = container.firstChild as HTMLElement;
    const skeletons = wrapper.querySelectorAll(".animate-shimmer");
    const chartBlock = skeletons[skeletons.length - 1] as HTMLElement;
    expect(chartBlock.style.height).toBe("400px");
  });

  it("renders header row with title placeholder and 3 control placeholders", () => {
    const { container } = render(<ChartSkeleton />);
    const headerRow = container.querySelector(".flex.items-center.justify-between") as HTMLElement;
    expect(headerRow).toBeInTheDocument();
    // Title skeleton + 3 control skeletons = 4 shimmer elements in the header
    const titleSkeleton = headerRow.querySelector(".h-4.w-32");
    expect(titleSkeleton).toBeInTheDocument();
    const controlSkeletons = headerRow.querySelectorAll(".h-6.w-10");
    expect(controlSkeletons.length).toBe(3);
  });

  it("applies custom className", () => {
    const { container } = render(<ChartSkeleton className="p-4" />);
    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper.className).toContain("p-4");
  });
});

describe("KpiRowSkeleton", () => {
  it("renders default 4 KPI cards", () => {
    const { container } = render(<KpiRowSkeleton />);
    const cards = container.querySelectorAll(".rounded-lg.border");
    expect(cards.length).toBe(4);
  });

  it("renders custom count of KPI cards", () => {
    const { container } = render(<KpiRowSkeleton count={2} />);
    const cards = container.querySelectorAll(".rounded-lg.border");
    expect(cards.length).toBe(2);
  });

  it("each card has 3 skeleton lines", () => {
    const { container } = render(<KpiRowSkeleton count={1} />);
    const card = container.querySelector(".rounded-lg.border") as HTMLElement;
    const skeletons = card.querySelectorAll(".animate-shimmer");
    expect(skeletons.length).toBe(3);
  });

  it("grid columns match count", () => {
    const { container } = render(<KpiRowSkeleton count={5} />);
    const grid = container.firstChild as HTMLElement;
    expect(grid.style.gridTemplateColumns).toBe("repeat(5, 1fr)");
  });
});
