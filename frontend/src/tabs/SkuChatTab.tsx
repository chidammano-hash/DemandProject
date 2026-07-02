// SKU Chatbot tab — standalone, conversational per-SKU assistant.
// Thin wrapper: SKU selector row + the reusable SkuChatPanel.
// Spec: docs/specs/06-ai-platform/07-sku-chatbot.md
import { useState } from "react";
import { SkuChatPanel } from "./sku-chat/SkuChatPanel";

export function SkuChatTab() {
  const [itemId, setItemId] = useState("");
  const [customerGroup, setCustomerGroup] = useState("");
  const [loc, setLoc] = useState("");

  return (
    <div className="flex h-[calc(100vh-9rem)] flex-col gap-3 p-4">
      <div>
        <h1 className="text-lg font-semibold">SKU Chat</h1>
        <p className="text-sm text-muted-foreground">
          Ask about one SKU — sales, forecast, accuracy, inventory, cluster peers.
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-2 rounded border p-3">
        <label className="flex flex-col text-xs">
          item_id
          <input
            aria-label="item_id"
            className="mt-1 w-32 rounded border px-2 py-1 text-sm"
            value={itemId}
            onChange={(e) => setItemId(e.target.value)}
          />
        </label>
        <label className="flex flex-col text-xs">
          customer_group
          <input
            aria-label="customer_group"
            className="mt-1 w-32 rounded border px-2 py-1 text-sm"
            value={customerGroup}
            onChange={(e) => setCustomerGroup(e.target.value)}
          />
        </label>
        <label className="flex flex-col text-xs">
          loc
          <input
            aria-label="loc"
            className="mt-1 w-28 rounded border px-2 py-1 text-sm"
            value={loc}
            onChange={(e) => setLoc(e.target.value)}
          />
        </label>
      </div>

      <div className="min-h-0 flex-1">
        <SkuChatPanel itemId={itemId} loc={loc} customerGroup={customerGroup} />
      </div>
    </div>
  );
}

export default SkuChatTab;
