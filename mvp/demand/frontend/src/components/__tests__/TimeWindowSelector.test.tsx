import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { TimeWindowSelector } from "@/components/TimeWindowSelector";

describe("TimeWindowSelector", () => {
  const windows = [3, 6, 12];
  const onChange = vi.fn();

  it("renders all window buttons", () => {
    render(<TimeWindowSelector windows={windows} selected={6} onChange={onChange} />);
    expect(screen.getByText("3mo")).toBeInTheDocument();
    expect(screen.getByText("6mo")).toBeInTheDocument();
    expect(screen.getByText("12mo")).toBeInTheDocument();
  });

  it("highlights the selected window with primary styling", () => {
    render(<TimeWindowSelector windows={windows} selected={6} onChange={onChange} />);
    const selectedBtn = screen.getByText("6mo");
    expect(selectedBtn.className).toContain("bg-primary/10");
    expect(selectedBtn.className).toContain("text-primary");
  });

  it("non-selected windows have muted styling", () => {
    render(<TimeWindowSelector windows={windows} selected={6} onChange={onChange} />);
    const nonSelected = screen.getByText("3mo");
    expect(nonSelected.className).toContain("text-muted-foreground");
    expect(nonSelected.className).not.toContain("bg-primary/10");
  });

  it("calls onChange with the clicked window value", () => {
    render(<TimeWindowSelector windows={windows} selected={6} onChange={onChange} />);
    fireEvent.click(screen.getByText("12mo"));
    expect(onChange).toHaveBeenCalledWith(12);
  });

  it("uses custom suffix", () => {
    render(<TimeWindowSelector windows={[1, 7, 30]} selected={7} onChange={onChange} suffix="d" />);
    expect(screen.getByText("1d")).toBeInTheDocument();
    expect(screen.getByText("7d")).toBeInTheDocument();
    expect(screen.getByText("30d")).toBeInTheDocument();
  });

  it("applies custom className to wrapper", () => {
    const { container } = render(
      <TimeWindowSelector windows={windows} selected={3} onChange={onChange} className="mt-4" />,
    );
    const wrapper = container.firstChild as HTMLElement;
    expect(wrapper.className).toContain("mt-4");
  });
});
