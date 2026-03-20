import { describe, it, expect, vi } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { TestQueryWrapper } from "./test-utils";

vi.mock("@/api/queries", () => ({
  sendChatMessage: vi.fn().mockResolvedValue({
    answer: "Test answer",
    sql: "SELECT 1",
    data: [{ id: 1 }],
    columns: ["id"],
    row_count: 1,
  }),
}));

const ChatPanel = (await import("@/tabs/ChatPanel")).default;

describe("ChatPanel", () => {
  it("renders without crashing", async () => {
    render(
      <TestQueryWrapper>
        <ChatPanel domain="item" />
      </TestQueryWrapper>
    );

    await waitFor(() => {
      expect(document.body.textContent).toBeDefined();
    });
  });
});
