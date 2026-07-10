import { describe, it, expect, vi } from "vitest";
import { useState } from "react";
import { render, screen, fireEvent } from "@testing-library/react";
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

const { GlobalChatDrawer } = await import("@/tabs/sku-chat/GlobalChatDrawer");
const { ActiveSkuProvider, usePublishActiveSku } = await import("@/context/ActiveSkuContext");

function Publisher({ item, loc }: { item: string; loc: string }) {
  usePublishActiveSku(item, loc);
  return null;
}

describe("GlobalChatDrawer", () => {
  it("shows the assistant button and opens a page-customized drawer", () => {
    render(
      <TestQueryWrapper>
        <GlobalChatDrawer activeTab="invPlanning" />
      </TestQueryWrapper>,
    );
    const fab = screen.getByRole("button", { name: /open assistant chat/i });
    fireEvent.click(fab);

    // Page title + a page-specific suggestion come from pageChatConfig.
    expect(screen.getByRole("complementary", { name: /assistant chat/i })).toBeTruthy();
    expect(screen.getByText(/Inventory Planning/i)).toBeTruthy();
    expect(screen.getByText(/stockout risk/i)).toBeTruthy();
  });

  it("inherits the SKU the active page publishes, over the global filter", () => {
    render(
      <TestQueryWrapper>
        <ActiveSkuProvider>
          <Publisher item="100320" loc="DC1" />
          <GlobalChatDrawer activeTab="itemAnalysis" />
        </ActiveSkuProvider>
      </TestQueryWrapper>,
    );
    fireEvent.click(screen.getByRole("button", { name: /open assistant chat/i }));
    // The global filter is empty in the test wrapper, so the SKU shown in the
    // drawer header must have come from the page-published active SKU.
    expect(screen.getByText("100320")).toBeTruthy();
    expect(screen.getByText("DC1")).toBeTruthy();
  });

  it("keeps a separate, persistent conversation per page across tab switches", () => {
    function Harness() {
      const [tab, setTab] = useState("invPlanning");
      return (
        <>
          <button
            type="button"
            onClick={() => setTab((t) => (t === "invPlanning" ? "commandCenter" : "invPlanning"))}
          >
            switch
          </button>
          <GlobalChatDrawer activeTab={tab} />
        </>
      );
    }
    render(
      <TestQueryWrapper>
        <Harness />
      </TestQueryWrapper>,
    );

    fireEvent.click(screen.getByRole("button", { name: /open assistant chat/i }));

    // Draft a message on the Inventory Planning page's thread.
    const inv = screen.getByRole("textbox", { name: /message/i });
    fireEvent.change(inv, { target: { value: "draft on inventory" } });
    expect((inv as HTMLTextAreaElement).value).toBe("draft on inventory");

    // Switch to another page — a fresh thread, so the composer is empty.
    fireEvent.click(screen.getByText("switch"));
    expect((screen.getByRole("textbox", { name: /message/i }) as HTMLTextAreaElement).value).toBe("");

    // Return to Inventory Planning — the prior conversation/draft is preserved.
    fireEvent.click(screen.getByText("switch"));
    expect((screen.getByRole("textbox", { name: /message/i }) as HTMLTextAreaElement).value).toBe(
      "draft on inventory",
    );
  });

  it.each(["skuChat", "customerAnalytics"])(
    "is hidden when %s owns its own assistant surface",
    (activeTab) => {
      const { container } = render(
        <TestQueryWrapper>
          <GlobalChatDrawer activeTab={activeTab} />
        </TestQueryWrapper>,
      );
      expect(container.querySelector("button")).toBeNull();
    },
  );
});
