import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { LagFilterBar } from "../LagFilterBar";

describe("LagFilterBar", () => {
  it("explains the selected evaluation horizon", () => {
    render(<LagFilterBar value={2} onChange={vi.fn()} />);

    expect(screen.getByText("Evaluation horizon")).toBeInTheDocument();
    expect(
      screen.getByText("Showing every experiment at a fixed 3-month-ahead horizon."),
    ).toBeInTheDocument();
    expect(screen.getByRole("group", { name: "Evaluation horizon" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Lag 2 (3mo)" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });

  it("describes portfolio mode and reports changes", () => {
    const onChange = vi.fn();
    render(<LagFilterBar value={undefined} onChange={onChange} />);

    expect(
      screen.getByText(
        "Showing each DFU at its assigned production planning lead time.",
      ),
    ).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Portfolio (assigned)" })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    fireEvent.click(screen.getByRole("button", { name: "Lag 4 (5mo)" }));
    expect(onChange).toHaveBeenCalledWith(4);
  });
});
