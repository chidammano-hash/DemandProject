import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import OperationsTab from "../OperationsTab";

vi.mock("@/components/workflows/WorkflowScanPanel", () => ({
  WorkflowScanPanel: () => <div>AI Operations Guide</div>,
}));

vi.mock("../IntegrationTab", () => ({
  default: ({ view }: { view: string }) => <div>Integration {view}</div>,
}));

vi.mock("../JobsTab", () => ({
  default: () => <div>Workflow monitoring library</div>,
}));

describe("OperationsTab", () => {
  it("keeps guided planning, monitoring, and advanced loading in one workspace", () => {
    render(<OperationsTab onNavigateToScenario={vi.fn()} />);

    expect(screen.getByRole("heading", { name: "Workflow Command Center" })).toBeInTheDocument();
    expect(screen.getByText("AI Operations Guide")).toBeInTheDocument();
    expect(screen.getByText("Integration guided")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Workflow Library/i }));
    expect(screen.getByText("Workflow monitoring library")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /Manual Load/i }));
    expect(screen.getByText("Integration manual")).toBeInTheDocument();
  });
});
