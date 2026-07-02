import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", () => ({
  skuChatKeys: { config: () => ["sku-chat", "config"] as const },
  fetchSkuChatConfig: vi.fn().mockResolvedValue({
    auth_mode: "auto",
    models: {},
    routing: { default_tier: "standard", allow_user_override: true },
    guardrails: {},
    tools: [],
  }),
  streamSkuChat: vi.fn(),
  decideSkuChatAdjustment: vi.fn(),
}));

const { SkuChatPanel } = await import("@/tabs/sku-chat/SkuChatPanel");

describe("SkuChatPanel", () => {
  it("renders the composer and an AI Adjust button, enabled when a SKU is set", () => {
    render(
      <TestQueryWrapper>
        <SkuChatPanel itemId="100320" loc="DC1" />
      </TestQueryWrapper>,
    );
    expect(screen.getByLabelText("message")).toBeTruthy();
    const adjust = screen.getByRole("button", { name: /ai adjust/i }) as HTMLButtonElement;
    expect(adjust.disabled).toBe(false);
  });

  it("disables AI Adjust until a SKU is selected", () => {
    render(
      <TestQueryWrapper>
        <SkuChatPanel itemId="" loc="" />
      </TestQueryWrapper>,
    );
    const adjust = screen.getByRole("button", { name: /ai adjust/i }) as HTMLButtonElement;
    expect(adjust.disabled).toBe(true);
  });

  it("renders page suggestion chips", () => {
    render(
      <TestQueryWrapper>
        <SkuChatPanel itemId="" loc="" suggestions={["Why did the forecast miss?"]} />
      </TestQueryWrapper>,
    );
    expect(screen.getByText("Why did the forecast miss?")).toBeTruthy();
  });
});
