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
});
