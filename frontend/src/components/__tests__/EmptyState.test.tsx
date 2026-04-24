import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { EmptyState } from "@/components/EmptyState";

describe("EmptyState", () => {
  it("renders no-data variant by default with steps", () => {
    render(
      <EmptyState
        title="Nothing here"
        description="No data yet."
        steps={[{ label: "Run load", command: "make load" }]}
      />,
    );
    expect(screen.getByText("Nothing here")).toBeInTheDocument();
    expect(screen.getByText("How to populate")).toBeInTheDocument();
    expect(screen.getByText("make load")).toBeInTheDocument();
    // Non-error variants use role=status
    expect(screen.getByRole("status")).toBeInTheDocument();
  });

  it("renders filtered variant without steps", () => {
    render(
      <EmptyState
        variant="filtered"
        title="No matches"
        description="Try widening your filters."
        steps={[{ label: "x", command: "y" }]}
      />,
    );
    // Steps block is suppressed on filtered variant
    expect(screen.queryByText("How to populate")).not.toBeInTheDocument();
    expect(screen.getByText("No matches")).toBeInTheDocument();
  });

  it("renders error variant with role=alert", () => {
    render(
      <EmptyState
        variant="error"
        title="Something failed"
        description="The server returned an error."
      />,
    );
    expect(screen.getByRole("alert")).toBeInTheDocument();
    expect(screen.getByText("Something failed")).toBeInTheDocument();
  });

  it("fires action callback", async () => {
    const onAction = vi.fn();
    render(
      <EmptyState
        variant="filtered"
        title="Nothing"
        description="No matches for current filters."
        onAction={onAction}
        actionLabel="Reset filters"
      />,
    );
    await userEvent.click(screen.getByRole("button", { name: "Reset filters" }));
    expect(onAction).toHaveBeenCalledTimes(1);
  });
});
