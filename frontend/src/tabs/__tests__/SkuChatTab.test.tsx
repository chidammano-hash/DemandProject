import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", () => ({
  skuChatKeys: { config: () => ["sku-chat", "config"] as const },
  fetchSkuChatConfig: vi.fn().mockResolvedValue({
    runtime_provider: "claude",
    auth_mode: "auto",
    models: {
      fast: "claude-haiku-4-5",
      standard: "claude-sonnet-4-6",
      deep: "claude-opus-4-8",
    },
    codex_models: {
      fast: "gpt-5.4-mini",
      standard: "gpt-5.5",
      deep: "gpt-5.5",
    },
    routing: { default_tier: "standard", allow_user_override: true },
    guardrails: {},
    tools: [],
  }),
  streamSkuChat: vi.fn(),
  createSkuChatSession: vi.fn(),
  decideSkuChatAdjustment: vi.fn(),
}));

const { SkuChatTab } = await import("@/tabs/SkuChatTab");

describe("SkuChatTab", () => {
  it("renders header, SKU inputs and composer", () => {
    render(
      <TestQueryWrapper>
        <SkuChatTab />
      </TestQueryWrapper>,
    );
    expect(screen.getByText("SKU Chat")).toBeTruthy();
    expect(screen.getByLabelText("item_id")).toBeTruthy();
    expect(screen.getByLabelText("loc")).toBeTruthy();
    expect(screen.getByLabelText("message")).toBeTruthy();
  });

  it("disables Send until a SKU and a question are set", () => {
    render(
      <TestQueryWrapper>
        <SkuChatTab />
      </TestQueryWrapper>,
    );
    const send = screen.getByRole("button", { name: "Send" }) as HTMLButtonElement;
    expect(send.disabled).toBe(true);
  });

  it("shows the routing config badge once loaded", async () => {
    render(
      <TestQueryWrapper>
        <SkuChatTab />
      </TestQueryWrapper>,
    );
    expect(await screen.findByText(/default tier/i)).toBeTruthy();
  });
});
