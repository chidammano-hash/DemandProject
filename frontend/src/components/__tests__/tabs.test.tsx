import { useState } from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MapPinned, Users, Network } from "lucide-react";

import { TabStrip, type TabStripItem, type TabStripVariant } from "@/components/ui/tabs";

const ITEMS: TabStripItem[] = [
  { key: "overview", label: "Overview", icon: MapPinned, description: "Demand footprint" },
  { key: "customers", label: "Customers", icon: Users, description: "Rank and retain" },
  { key: "segments", label: "Segments", icon: Network, description: "Mix and trends" },
];

/** Controlled harness — mirrors how a real consumer wires value/onValueChange. */
function Harness({ variant, initial = "overview" }: { variant?: TabStripVariant; initial?: string }) {
  const [value, setValue] = useState(initial);
  return (
    <TabStrip aria-label="Test views" value={value} onValueChange={setValue} items={ITEMS} variant={variant} />
  );
}

describe("TabStrip", () => {
  it("renders a tablist with an accessible name", () => {
    render(<Harness />);
    expect(screen.getByRole("tablist", { name: "Test views" })).toBeInTheDocument();
  });

  it("renders one tab per item with the label text", () => {
    render(<Harness />);
    expect(screen.getAllByRole("tab")).toHaveLength(3);
    expect(screen.getByRole("tab", { name: /Overview/ })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Customers/ })).toBeInTheDocument();
    expect(screen.getByRole("tab", { name: /Segments/ })).toBeInTheDocument();
  });

  it("marks the tab matching `value` as aria-selected and the rest as not", () => {
    render(<Harness initial="customers" />);
    expect(screen.getByRole("tab", { name: /Overview/ })).toHaveAttribute("aria-selected", "false");
    expect(screen.getByRole("tab", { name: /Customers/ })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: /Segments/ })).toHaveAttribute("aria-selected", "false");
  });

  it("uses a roving tabindex — only the selected tab is tabbable", () => {
    render(<Harness initial="segments" />);
    expect(screen.getByRole("tab", { name: /Overview/ })).toHaveAttribute("tabindex", "-1");
    expect(screen.getByRole("tab", { name: /Customers/ })).toHaveAttribute("tabindex", "-1");
    expect(screen.getByRole("tab", { name: /Segments/ })).toHaveAttribute("tabindex", "0");
  });

  it("falls back to the first tab as tabbable when `value` matches nothing", () => {
    render(<Harness initial="does-not-exist" />);
    expect(screen.getByRole("tab", { name: /Overview/ })).toHaveAttribute("tabindex", "0");
  });

  it("is controlled — clicking calls onValueChange, and re-rendering with the new value updates selection", () => {
    const onValueChange = vi.fn();
    const { rerender } = render(
      <TabStrip aria-label="Test views" value="overview" onValueChange={onValueChange} items={ITEMS} />,
    );
    fireEvent.click(screen.getByRole("tab", { name: /Customers/ }));
    expect(onValueChange).toHaveBeenCalledWith("customers");
    // Uncontrolled click alone must NOT flip selection — the component owns no state.
    expect(screen.getByRole("tab", { name: /Overview/ })).toHaveAttribute("aria-selected", "true");

    rerender(<TabStrip aria-label="Test views" value="customers" onValueChange={onValueChange} items={ITEMS} />);
    expect(screen.getByRole("tab", { name: /Customers/ })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tab", { name: /Overview/ })).toHaveAttribute("aria-selected", "false");
  });

  describe("keyboard navigation", () => {
    it("ArrowRight moves to and selects the next tab, wrapping at the end", () => {
      render(<Harness initial="segments" />);
      fireEvent.keyDown(screen.getByRole("tab", { name: /Segments/ }), { key: "ArrowRight" });
      expect(screen.getByRole("tab", { name: /Overview/ })).toHaveAttribute("aria-selected", "true");
      expect(screen.getByRole("tab", { name: /Overview/ })).toHaveFocus();
    });

    it("ArrowLeft moves to and selects the previous tab, wrapping at the start", () => {
      render(<Harness initial="overview" />);
      fireEvent.keyDown(screen.getByRole("tab", { name: /Overview/ }), { key: "ArrowLeft" });
      expect(screen.getByRole("tab", { name: /Segments/ })).toHaveAttribute("aria-selected", "true");
      expect(screen.getByRole("tab", { name: /Segments/ })).toHaveFocus();
    });

    it("Home selects the first tab", () => {
      render(<Harness initial="segments" />);
      fireEvent.keyDown(screen.getByRole("tab", { name: /Segments/ }), { key: "Home" });
      expect(screen.getByRole("tab", { name: /Overview/ })).toHaveAttribute("aria-selected", "true");
    });

    it("End selects the last tab", () => {
      render(<Harness initial="overview" />);
      fireEvent.keyDown(screen.getByRole("tab", { name: /Overview/ }), { key: "End" });
      expect(screen.getByRole("tab", { name: /Segments/ })).toHaveAttribute("aria-selected", "true");
    });

    it("ignores unrelated keys", () => {
      render(<Harness initial="overview" />);
      fireEvent.keyDown(screen.getByRole("tab", { name: /Overview/ }), { key: "a" });
      expect(screen.getByRole("tab", { name: /Overview/ })).toHaveAttribute("aria-selected", "true");
    });

    it("supports full click-then-arrow keyboard flow via userEvent", async () => {
      const user = userEvent.setup();
      render(<Harness />);
      await user.click(screen.getByRole("tab", { name: /Overview/ }));
      await user.keyboard("{ArrowRight}");
      expect(screen.getByRole("tab", { name: /Customers/ })).toHaveAttribute("aria-selected", "true");
    });
  });

  describe("variants", () => {
    it.each<TabStripVariant>(["underline", "pills", "segmented", "cards"])(
      "renders the '%s' variant without crashing and applies an active-state class",
      (variant) => {
        render(<Harness variant={variant} initial="customers" />);
        const active = screen.getByRole("tab", { name: /Customers/ });
        const inactive = screen.getByRole("tab", { name: /Overview/ });
        if (variant === "underline") {
          expect(active.className).toContain("border-primary");
          expect(active.className).toContain("font-medium");
          expect(inactive.className).not.toContain("border-primary");
        } else {
          expect(active.className).toContain("bg-primary");
          expect(active.className).toContain("text-primary-foreground");
          expect(inactive.className).not.toContain("bg-primary");
        }
      },
    );

    it("only the 'cards' variant renders the description subtitle", () => {
      render(<Harness variant="cards" />);
      expect(screen.getByText("Demand footprint")).toBeInTheDocument();
    });

    it("non-card variants do not render the description as visible text (falls back to title attribute)", () => {
      render(<Harness variant="underline" />);
      expect(screen.queryByText("Demand footprint")).toBeNull();
      expect(screen.getByRole("tab", { name: /Overview/ })).toHaveAttribute("title", "Demand footprint");
    });
  });

  it("renders the icon for each item", () => {
    render(<Harness />);
    const tab = screen.getByRole("tab", { name: /Overview/ });
    expect(tab.querySelector("svg")).not.toBeNull();
  });

  it("renders a badge when provided", () => {
    const items: TabStripItem[] = [{ key: "a", label: "Alerts", badge: 4 }];
    render(<TabStrip aria-label="Badged" value="a" onValueChange={vi.fn()} items={items} />);
    expect(screen.getByText("4")).toBeInTheDocument();
  });

  it("does not render a badge span when omitted", () => {
    render(<Harness />);
    const tab = screen.getByRole("tab", { name: /Overview/ });
    expect(tab.querySelector(".rounded-full.px-1")).toBeNull();
  });

  it("keeps the tablist horizontally scrollable for narrow screens", () => {
    render(<Harness />);
    expect(screen.getByRole("tablist").className).toContain("overflow-x-auto");
  });

  it("applies a token-based focus-visible ring to every tab", () => {
    render(<Harness />);
    for (const tab of screen.getAllByRole("tab")) {
      expect(tab.className).toContain("focus-visible:ring-2");
      expect(tab.className).toContain("focus-visible:ring-ring");
    }
  });

  it("uses reduced-motion-safe transitions", () => {
    render(<Harness />);
    expect(screen.getByRole("tab", { name: /Overview/ }).className).toContain("motion-reduce:transition-none");
  });

  it("applies an additional className to the tablist container", () => {
    render(
      <TabStrip aria-label="Custom" value="a" onValueChange={vi.fn()} items={[{ key: "a", label: "A" }]} className="mt-4" />,
    );
    expect(screen.getByRole("tablist").className).toContain("mt-4");
  });
});
