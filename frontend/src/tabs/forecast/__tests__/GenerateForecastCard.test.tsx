import { render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { GenerateForecastCard } from "../GenerateForecastCard";

vi.mock("lucide-react", () => {
  const Stub = () => <span />;
  return { Check: Stub, Loader2: Stub, Play: Stub };
});

describe("GenerateForecastCard", () => {
  it("describes champion as promoted routing rather than a retired meta-learner", () => {
    render(
      <GenerateForecastCard
        selectedModel="champion"
        effectiveHorizon={24}
        onHorizonChange={vi.fn()}
        includeCI
        onIncludeCIChange={vi.fn()}
        isSubmitting={false}
        isForecastRunning={false}
        onGenerateForecast={vi.fn()}
        latestVersion={null}
      />
    );

    expect(screen.getByText("Champion (promoted DFU routing)")).toBeDefined();
    expect(screen.queryByText(/meta-learner/i)).toBeNull();
  });

  it("blocks champion generation until a user-selected champion is assigned", () => {
    render(
      <GenerateForecastCard
        selectedModel="champion"
        effectiveHorizon={24}
        onHorizonChange={vi.fn()}
        includeCI
        onIncludeCIChange={vi.fn()}
        isSubmitting={false}
        isForecastRunning={false}
        blockedReason="Select and assign a completed experiment in Champion first."
        onGenerateForecast={vi.fn()}
        latestVersion={null}
      />
    );

    expect(screen.getByRole("button", { name: /generate forecast/i })).toBeDisabled();
    expect(screen.getByText(/select and assign a completed experiment/i)).toBeDefined();
  });
});
