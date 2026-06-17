import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AiChampionItemPanel } from "../AiChampionItemPanel";

vi.mock("@/api/queries/ai-champion", () => ({
  AI_CHAMPION_PROVIDERS: [
    { value: "ollama", label: "Ollama (local, $0)" },
    { value: "google", label: "Google Gemini" },
    { value: "anthropic", label: "Anthropic (Opus)" },
    { value: "openai", label: "OpenAI (GPT-4o)" },
  ],
  aiChampionKeys: { saved: (i: string, l: string) => ["ai-champion", "saved", i, l] },
  fetchAiChampionSaved: vi.fn(),
  adjustAiChampion: vi.fn(),
  saveAiChampion: vi.fn(),
}));

import {
  fetchAiChampionSaved,
  adjustAiChampion,
  saveAiChampion,
} from "@/api/queries/ai-champion";

function wrapper(children: React.ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}

const PREVIEW = {
  item_id: "100", loc: "L1", plan_version: "2026-04", provider: "ollama",
  model: "llama3.1:8b", prompt_version: "v1.1.0",
  recommendation_code: "SCALE_UP", rec_pct_change: 15, proposed_qty: null,
  apply_horizon_months: 3, confidence: 0.82, rationale: "Customer concentration rising",
  evidence_keys: ["customer_concentration"],
  months: [
    { forecast_month: "2026-05-01", horizon_months: 1, champion_qty: 100, ai_qty: 115, pct_change: 15 },
  ],
};

describe("AiChampionItemPanel", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    vi.mocked(fetchAiChampionSaved).mockResolvedValue({ total: 0, rows: [] });
  });

  it("renders nothing until both item and loc are set", () => {
    const { container } = render(wrapper(<AiChampionItemPanel itemId="" loc="" />));
    expect(container).toBeEmptyDOMElement();
    expect(fetchAiChampionSaved).not.toHaveBeenCalled();
  });

  it("shows the empty state when nothing is saved", async () => {
    render(wrapper(<AiChampionItemPanel itemId="100" loc="L1" />));
    expect(await screen.findByText(/No saved adjustment for 100-L1/i)).toBeInTheDocument();
  });

  it("renders a previously-saved adjustment with the Saved badge", async () => {
    vi.mocked(fetchAiChampionSaved).mockResolvedValue({
      total: 1,
      rows: [{
        item_id: "100", loc: "L1", forecast_month: "2026-05-01", horizon_months: 1,
        champion_qty: 100, ai_qty: 90, recommendation_code: "SCALE_DOWN",
        pct_change: -10, confidence: 0.7, rationale: "soft demand",
      }],
    });
    render(wrapper(<AiChampionItemPanel itemId="100" loc="L1" />));
    expect(await screen.findByText("Saved")).toBeInTheDocument();
    expect(screen.getByText("SCALE_DOWN")).toBeInTheDocument();
    expect(screen.getByText(/soft demand/)).toBeInTheDocument();
  });

  it("previews on AI Adjust, then saves", async () => {
    vi.mocked(adjustAiChampion).mockResolvedValue(PREVIEW);
    vi.mocked(saveAiChampion).mockResolvedValue({
      item_id: "100", loc: "L1", plan_version: "2026-04", run_id: "r1",
      recommendation_code: "SCALE_UP", saved_months: 1,
    });
    render(wrapper(<AiChampionItemPanel itemId="100" loc="L1" />));
    await screen.findByText(/No saved adjustment/i);

    fireEvent.click(screen.getByRole("button", { name: /AI Adjust/i }));

    expect(await screen.findByText("Preview — not saved")).toBeInTheDocument();
    expect(screen.getByText(/Customer concentration rising/)).toBeInTheDocument();
    expect(adjustAiChampion).toHaveBeenCalledWith({ item_id: "100", loc: "L1", provider: "ollama" });

    fireEvent.click(screen.getByRole("button", { name: /^Save$/i }));
    await waitFor(() => expect(saveAiChampion).toHaveBeenCalledOnce());
  });
});
