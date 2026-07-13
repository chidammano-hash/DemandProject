import { fireEvent, render, screen } from "@testing-library/react";
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
        candidateReady={false}
        isPromoting={false}
        isSelectedPromoted={false}
        onGenerateForecast={vi.fn()}
        onPromote={vi.fn()}
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
        candidateReady={false}
        isPromoting={false}
        isSelectedPromoted={false}
        blockedReason="Select and assign a completed experiment in Champion first."
        onGenerateForecast={vi.fn()}
        onPromote={vi.fn()}
        latestVersion={null}
      />
    );

    expect(screen.getByRole("button", { name: /generate forecast/i })).toBeDisabled();
    expect(screen.getByText(/select and assign a completed experiment/i)).toBeDefined();
  });

  it("promotes the selected ready single-model candidate", () => {
    const onPromote = vi.fn();
    render(
      <GenerateForecastCard
        selectedModel="mstl"
        effectiveHorizon={24}
        onHorizonChange={vi.fn()}
        includeCI
        onIncludeCIChange={vi.fn()}
        isSubmitting={false}
        isForecastRunning={false}
        candidateReady
        candidateDfuCount={321}
        isPromoting={false}
        isSelectedPromoted={false}
        onGenerateForecast={vi.fn()}
        onPromote={onPromote}
        latestVersion={null}
      />
    );

    const button = screen.getByRole("button", { name: /promote mstl to production/i });
    fireEvent.click(button);
    expect(onPromote).toHaveBeenCalledOnce();
    expect(screen.getByText(/321 dfus staged/i)).toBeDefined();
  });

  it("disables production promotion when the selected candidate is not ready", () => {
    render(
      <GenerateForecastCard
        selectedModel="champion"
        effectiveHorizon={24}
        onHorizonChange={vi.fn()}
        includeCI
        onIncludeCIChange={vi.fn()}
        isSubmitting={false}
        isForecastRunning={false}
        candidateReady={false}
        isPromoting={false}
        isSelectedPromoted={false}
        promotionBlockedReason="Generate the selected model to staging first."
        onGenerateForecast={vi.fn()}
        onPromote={vi.fn()}
        latestVersion={null}
      />
    );

    expect(screen.getByRole("button", { name: /promote champion to production/i })).toBeDisabled();
    expect(screen.getByText(/generate the selected model to staging first/i)).toBeDefined();
  });
});
