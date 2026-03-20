import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { Breadcrumbs } from "../Breadcrumbs";

describe("Breadcrumbs", () => {
  it("renders all items", () => {
    render(<Breadcrumbs items={[{ label: "Home" }, { label: "Page" }, { label: "Sub" }]} />);
    expect(screen.getByText("Home")).toBeDefined();
    expect(screen.getByText("Page")).toBeDefined();
    expect(screen.getByText("Sub")).toBeDefined();
  });

  it("last item is not clickable", () => {
    const onClick = vi.fn();
    render(<Breadcrumbs items={[{ label: "Home", onClick }, { label: "Current" }]} />);
    expect(screen.getByText("Current").tagName).toBe("SPAN");
  });

  it("middle items are clickable", () => {
    const onClick = vi.fn();
    render(<Breadcrumbs items={[{ label: "Home", onClick }, { label: "Current" }]} />);
    fireEvent.click(screen.getByText("Home"));
    expect(onClick).toHaveBeenCalled();
  });
});
