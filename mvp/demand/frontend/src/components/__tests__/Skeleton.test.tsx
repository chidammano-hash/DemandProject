import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Skeleton, TableSkeleton } from "@/components/Skeleton";

describe("Skeleton", () => {
  it("renders a div element", () => {
    const { container } = render(<Skeleton />);
    const div = container.firstChild as HTMLElement;
    expect(div).toBeInTheDocument();
    expect(div.tagName).toBe("DIV");
  });

  it("has animate-pulse class", () => {
    const { container } = render(<Skeleton />);
    const div = container.firstChild as HTMLElement;
    expect(div.className).toContain("animate-pulse");
  });

  it("accepts additional className", () => {
    const { container } = render(<Skeleton className="h-10 w-40" />);
    const div = container.firstChild as HTMLElement;
    expect(div.className).toContain("h-10");
    expect(div.className).toContain("w-40");
  });

  it("passes through HTML attributes", () => {
    const { container } = render(<Skeleton data-testid="skel" aria-label="loading" />);
    const div = container.firstChild as HTMLElement;
    expect(div.getAttribute("data-testid")).toBe("skel");
    expect(div.getAttribute("aria-label")).toBe("loading");
  });
});

describe("TableSkeleton", () => {
  it("renders default 8 rows and 6 cols", () => {
    const { container } = render(<TableSkeleton />);
    // 1 header row + 8 data rows = 9 row divs
    const rows = container.querySelectorAll(".flex.gap-2");
    expect(rows.length).toBe(9); // 1 header + 8 body
  });

  it("renders custom rows and cols", () => {
    const { container } = render(<TableSkeleton rows={3} cols={4} />);
    // 1 header + 3 body = 4 row divs
    const rows = container.querySelectorAll(".flex.gap-2");
    expect(rows.length).toBe(4);
  });

  it("header cells match cols count", () => {
    const { container } = render(<TableSkeleton rows={2} cols={5} />);
    // First row is header
    const headerRow = container.querySelector(".flex.gap-2");
    const headerCells = headerRow?.querySelectorAll("div");
    expect(headerCells?.length).toBe(5);
  });
});
