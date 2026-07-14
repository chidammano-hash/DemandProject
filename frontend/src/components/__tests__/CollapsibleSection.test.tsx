import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import { CollapsibleSection } from "@/components/CollapsibleSection";

describe("CollapsibleSection", () => {
  it("renders title and children when open (default)", () => {
    render(
      <CollapsibleSection title="Test Section">
        <p>Content here</p>
      </CollapsibleSection>
    );
    expect(screen.getByText("Test Section")).toBeInTheDocument();
    expect(screen.getByText("Content here")).toBeInTheDocument();
  });

  it("hides children when defaultOpen=false", () => {
    render(
      <CollapsibleSection title="Closed Section" defaultOpen={false}>
        <p>Hidden content</p>
      </CollapsibleSection>
    );
    expect(screen.getByText("Closed Section")).toBeInTheDocument();
    expect(screen.queryByText("Hidden content")).not.toBeInTheDocument();
  });

  it("toggles content on header click", () => {
    render(
      <CollapsibleSection title="Toggle Me">
        <p>Toggleable</p>
      </CollapsibleSection>
    );
    expect(screen.getByText("Toggleable")).toBeInTheDocument();

    // Click header to collapse
    fireEvent.click(screen.getByText("Toggle Me"));
    expect(screen.queryByText("Toggleable")).not.toBeInTheDocument();

    // Click again to expand
    fireEvent.click(screen.getByText("Toggle Me"));
    expect(screen.getByText("Toggleable")).toBeInTheDocument();
  });

  it("is keyboard-accessible: focusable header with aria-expanded, toggled by Enter and Space", () => {
    render(
      <CollapsibleSection title="Error Decomposition">
        <p>Pareto table</p>
      </CollapsibleSection>
    );
    const header = screen.getByRole("button", { name: "Error Decomposition section" });
    expect(header).toHaveAttribute("tabindex", "0");
    expect(header).toHaveAttribute("aria-expanded", "true");

    fireEvent.keyDown(header, { key: "Enter" });
    expect(screen.queryByText("Pareto table")).not.toBeInTheDocument();
    expect(header).toHaveAttribute("aria-expanded", "false");

    fireEvent.keyDown(header, { key: " " });
    expect(screen.getByText("Pareto table")).toBeInTheDocument();
    expect(header).toHaveAttribute("aria-expanded", "true");
  });

  it("renders headerRight without toggling when clicked", () => {
    render(
      <CollapsibleSection
        title="With Controls"
        headerRight={<button>Action</button>}
      >
        <p>Body</p>
      </CollapsibleSection>
    );
    expect(screen.getByText("Action")).toBeInTheDocument();
    expect(screen.getByText("Body")).toBeInTheDocument();

    // Clicking the headerRight button should NOT collapse (stopPropagation)
    fireEvent.click(screen.getByText("Action"));
    expect(screen.getByText("Body")).toBeInTheDocument();
  });
});
