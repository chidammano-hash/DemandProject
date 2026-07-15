import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { Users } from "lucide-react";

import { PageHeader } from "@/components/PageHeader";

describe("PageHeader", () => {
  it("renders the title as an h1 with the contracted classes", () => {
    render(<PageHeader title="Customer Analytics" />);
    const h1 = screen.getByRole("heading", { level: 1, name: "Customer Analytics" });
    expect(h1.className).toContain("text-xl");
    expect(h1.className).toContain("font-semibold");
    expect(h1.className).toContain("tracking-heading");
    expect(h1.className).toContain("text-foreground");
  });

  it("renders the description with the contracted classes", () => {
    render(<PageHeader title="Customer Analytics" description="Move from footprint to risk." />);
    const description = screen.getByText("Move from footprint to risk.");
    expect(description.className).toContain("text-sm");
    expect(description.className).toContain("text-muted-foreground");
    expect(description.className).toContain("max-w-3xl");
  });

  it("does not render a description paragraph when omitted", () => {
    render(<PageHeader title="Customer Analytics" />);
    expect(screen.queryByText(/footprint/)).toBeNull();
  });

  it("renders the icon with h-5 w-5 text-primary when provided", () => {
    const { container } = render(<PageHeader title="Customers" icon={Users} />);
    const svg = container.querySelector("svg");
    expect(svg).not.toBeNull();
    expect(svg!.getAttribute("class")).toContain("h-5");
    expect(svg!.getAttribute("class")).toContain("w-5");
    expect(svg!.getAttribute("class")).toContain("text-primary");
  });

  it("does not render an icon when omitted", () => {
    const { container } = render(<PageHeader title="Customers" />);
    expect(container.querySelector("svg")).toBeNull();
  });

  it("renders the eyebrow with the contracted classes before the title", () => {
    render(<PageHeader title="Customer Analytics" eyebrow="Customer intelligence workspace" />);
    const eyebrow = screen.getByText("Customer intelligence workspace");
    expect(eyebrow.className).toContain("text-2xs");
    expect(eyebrow.className).toContain("uppercase");
    expect(eyebrow.className).toContain("tracking-wider");
    expect(eyebrow.className).toContain("text-muted-foreground");
  });

  it("does not render an eyebrow when omitted", () => {
    render(<PageHeader title="Customer Analytics" />);
    expect(screen.queryByText("Customer intelligence workspace")).toBeNull();
  });

  it("renders actions in a right-aligned, wrapping slot", () => {
    render(<PageHeader title="Customer Analytics" actions={<button type="button">Filters</button>} />);
    const button = screen.getByRole("button", { name: "Filters" });
    const actionsWrapper = button.parentElement as HTMLElement;
    expect(actionsWrapper.className).toContain("flex");
    expect(actionsWrapper.className).toContain("flex-wrap");
  });

  it("does not render an actions wrapper when omitted", () => {
    const { container } = render(<PageHeader title="Customer Analytics" />);
    expect(container.querySelectorAll("button").length).toBe(0);
  });

  it("applies an additional className to the root", () => {
    const { container } = render(<PageHeader title="Customer Analytics" className="mb-6" />);
    const root = container.firstChild as HTMLElement;
    expect(root.className).toContain("mb-6");
  });
});
