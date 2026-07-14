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
        candidateGenerated={false}
        candidateStaged={false}
        isStaging={false}
        isPromoting={false}
        isSelectedPromoted={false}
        onGenerateForecast={vi.fn()}
        onStage={vi.fn()}
        onPromote={vi.fn()}
        latestVersion={null}
      />
    );

    expect(screen.getByText("Champion (promoted DFU routing)")).toBeDefined();
    expect(screen.queryByText(/meta-learner/i)).toBeNull();
  });

  it("identifies a customer bottom-up blend and its gate before staging", () => {
    render(
      <GenerateForecastCard
        selectedModel="champion"
        effectiveHorizon={24}
        onHorizonChange={vi.fn()}
        includeCI
        onIncludeCIChange={vi.fn()}
        isSubmitting={false}
        isForecastRunning={false}
        candidateGenerated
        candidateStaged={false}
        candidateDfuCount={2_048}
        candidateModelId="customer_bottom_up_blend"
        customerBlendLineage={{
          customer_run_id: "customer-run",
          backtest_run_id: "backtest-run",
          backtest_gate: {
            passed: true,
            reason: "passed",
            common_months: 6,
            common_dfus: 2_048,
            champion_wape_pct: 12.4,
            customer_wape_pct: 10.1,
            blend_wape_pct: 9.8,
            blend_wape_degradation_pct: -2.6,
          },
        }}
        isStaging={false}
        isPromoting={false}
        isSelectedPromoted={false}
        onGenerateForecast={vi.fn()}
        onStage={vi.fn()}
        onPromote={vi.fn()}
        latestVersion={null}
      />
    );

    expect(screen.getByText("Customer Bottom-Up Blend")).toBeInTheDocument();
    expect(screen.getByText(/Backtest gate passed/i)).toBeInTheDocument();
    expect(screen.getByText(/Blend WAPE 9.8%/i)).toBeInTheDocument();
    expect(
      screen.getByRole("button", { name: /Promote Customer Bottom-Up Blend to Staging/i })
    ).toBeEnabled();
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
        candidateGenerated={false}
        candidateStaged={false}
        isStaging={false}
        isPromoting={false}
        isSelectedPromoted={false}
        blockedReason="Select and assign a completed experiment in Champion first."
        onGenerateForecast={vi.fn()}
        onStage={vi.fn()}
        onPromote={vi.fn()}
        latestVersion={null}
      />
    );

    expect(screen.getByRole("button", { name: /generate candidate/i })).toBeDisabled();
    expect(screen.getByText(/select and assign a completed experiment/i)).toBeDefined();
  });

  it("stages the selected generated single-model candidate before production", () => {
    const onStage = vi.fn();
    render(
      <GenerateForecastCard
        selectedModel="mstl"
        effectiveHorizon={24}
        onHorizonChange={vi.fn()}
        includeCI
        onIncludeCIChange={vi.fn()}
        isSubmitting={false}
        isForecastRunning={false}
        candidateGenerated
        candidateStaged={false}
        candidateDfuCount={321}
        isStaging={false}
        isPromoting={false}
        isSelectedPromoted={false}
        onGenerateForecast={vi.fn()}
        onStage={onStage}
        onPromote={vi.fn()}
        latestVersion={null}
      />
    );

    fireEvent.click(screen.getByRole("button", { name: /promote mstl to staging/i }));
    expect(onStage).toHaveBeenCalledOnce();
    expect(screen.getByRole("button", { name: /promote mstl to production/i })).toBeDisabled();
    expect(screen.getByText(/321 dfus generated as a draft/i)).toBeDefined();
  });

  it("promotes the selected staged single-model candidate to production", () => {
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
        candidateGenerated
        candidateStaged
        candidateDfuCount={321}
        isStaging={false}
        isPromoting={false}
        isSelectedPromoted={false}
        onGenerateForecast={vi.fn()}
        onStage={vi.fn()}
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
        candidateGenerated={false}
        candidateStaged={false}
        isStaging={false}
        isPromoting={false}
        isSelectedPromoted={false}
        promotionBlockedReason="Promote the selected generated candidate to staging first."
        onGenerateForecast={vi.fn()}
        onStage={vi.fn()}
        onPromote={vi.fn()}
        latestVersion={null}
      />
    );

    expect(screen.getByRole("button", { name: /promote champion to production/i })).toBeDisabled();
    expect(
      screen.getByText(/promote the selected generated candidate to staging first/i)
    ).toBeDefined();
  });
});
