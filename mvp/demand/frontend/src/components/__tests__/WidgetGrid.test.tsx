import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { WidgetGrid, WidgetCard } from "@/components/WidgetGrid";

describe("WidgetGrid", () => {
  it("renders children", () => {
    render(
      <WidgetGrid>
        <div data-testid="child-1">Card 1</div>
        <div data-testid="child-2">Card 2</div>
      </WidgetGrid>
    );
    expect(screen.getByTestId("child-1")).toBeInTheDocument();
    expect(screen.getByTestId("child-2")).toBeInTheDocument();
  });

  it("applies grid class", () => {
    const { container } = render(
      <WidgetGrid>
        <div>Content</div>
      </WidgetGrid>
    );
    const grid = container.firstChild as HTMLElement;
    expect(grid.className).toContain("grid");
  });

  it("applies 12-col classes by default", () => {
    const { container } = render(
      <WidgetGrid>
        <div>Content</div>
      </WidgetGrid>
    );
    const grid = container.firstChild as HTMLElement;
    expect(grid.className).toContain("lg:grid-cols-12");
  });

  it("applies 6-col classes when cols=6", () => {
    const { container } = render(
      <WidgetGrid cols={6}>
        <div>Content</div>
      </WidgetGrid>
    );
    const grid = container.firstChild as HTMLElement;
    expect(grid.className).toContain("lg:grid-cols-6");
  });

  it("applies gap classes based on gap prop", () => {
    const { container: sm } = render(
      <WidgetGrid gap="sm"><div>A</div></WidgetGrid>
    );
    expect((sm.firstChild as HTMLElement).className).toContain("gap-2");

    const { container: md } = render(
      <WidgetGrid gap="md"><div>A</div></WidgetGrid>
    );
    expect((md.firstChild as HTMLElement).className).toContain("gap-4");

    const { container: lg } = render(
      <WidgetGrid gap="lg"><div>A</div></WidgetGrid>
    );
    expect((lg.firstChild as HTMLElement).className).toContain("gap-6");
  });

  it("applies additional className", () => {
    const { container } = render(
      <WidgetGrid className="my-custom">
        <div>Content</div>
      </WidgetGrid>
    );
    const grid = container.firstChild as HTMLElement;
    expect(grid.className).toContain("my-custom");
  });
});

describe("WidgetCard", () => {
  it("renders children", () => {
    render(
      <WidgetCard>
        <p data-testid="card-content">Hello</p>
      </WidgetCard>
    );
    expect(screen.getByTestId("card-content")).toBeInTheDocument();
  });

  it("renders title when provided", () => {
    render(
      <WidgetCard title="My Widget">
        <div>Content</div>
      </WidgetCard>
    );
    expect(screen.getByText("My Widget")).toBeInTheDocument();
  });

  it("renders subtitle when provided", () => {
    render(
      <WidgetCard title="Widget" subtitle="Description here">
        <div>Content</div>
      </WidgetCard>
    );
    expect(screen.getByText("Description here")).toBeInTheDocument();
  });

  it("renders actions when provided", () => {
    render(
      <WidgetCard title="Widget" actions={<button data-testid="action-btn">Go</button>}>
        <div>Content</div>
      </WidgetCard>
    );
    expect(screen.getByTestId("action-btn")).toBeInTheDocument();
  });

  it("applies span class", () => {
    const { container } = render(
      <WidgetCard span={3}>
        <div>Content</div>
      </WidgetCard>
    );
    const card = container.firstChild as HTMLElement;
    expect(card.className).toContain("sm:col-span-3");
  });
});
