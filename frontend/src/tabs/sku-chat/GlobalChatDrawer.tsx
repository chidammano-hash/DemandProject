// Global, page-aware chat drawer — present on every tab. A floating button opens
// a slide-out assistant that re-contextualizes to the active page: its focus
// (sent to the agent), suggested prompts, scope (item/location), and title all
// come from the per-page config. Scope prefers the SKU the active page has
// published (e.g. the Item Analysis SKU) and falls back to the global filter.
// Hidden on the standalone SKU Chat tab (which is already a full-page chat).
//
// Persistent per-page threads: one SkuChatPanel is kept ALIVE per thread
// (`tab|item|loc`) and only the active one is shown — inactive threads are
// CSS-hidden, so React preserves each conversation and it resumes when you
// return. The thread key includes the SKU, so a SKU change on a SKU-scoped page
// starts a fresh thread. Retained threads are LRU-capped.
// Spec: docs/specs/06-ai-platform/07-sku-chatbot.md
import { useEffect, useState } from "react";
import { MessageSquare, X } from "lucide-react";
import { useGlobalFilterContext } from "@/context/GlobalFilterContext";
import { useActiveSku } from "@/context/ActiveSkuContext";
import { GroundedCopilotPanel } from "./GroundedCopilotPanel";
import { CHAT_HIDDEN_TABS, getPageChatConfig } from "./pageChatConfig";

export interface GlobalChatDrawerProps {
  activeTab: string;
}

/** Max conversations kept alive in memory; oldest are evicted (LRU). */
const MAX_THREADS = 12;

interface ThreadMeta {
  key: string;
  tab: string;
  itemId: string;
  loc: string;
}

function singleOrEmpty(values: string[] | undefined): string {
  return values && values.length === 1 ? values[0] : "";
}

export function GlobalChatDrawer({ activeTab }: GlobalChatDrawerProps) {
  const [open, setOpen] = useState(false);
  const [everOpened, setEverOpened] = useState(false);
  const [threads, setThreads] = useState<ThreadMeta[]>([]);
  const { filters } = useGlobalFilterContext();
  const activeSku = useActiveSku();

  const cfg = getPageChatConfig(activeTab);
  // The SKU the active page is showing wins over the global filter (e.g. the
  // Item Analysis local selector); fall back to a single-valued global filter.
  const itemId = activeSku ? activeSku.item : singleOrEmpty(filters.item);
  const loc = activeSku ? activeSku.loc : singleOrEmpty(filters.location);
  const threadKey = `${activeTab}|${itemId}|${loc}`;
  const activeThread: ThreadMeta = { key: threadKey, tab: activeTab, itemId, loc };

  // While open, register/promote the active thread (LRU) so its conversation is
  // retained when the user navigates away and back.
  useEffect(() => {
    if (!open || CHAT_HIDDEN_TABS.has(activeTab)) return;
    setThreads((prev) => {
      const next = [
        ...prev.filter((t) => t.key !== threadKey),
        { key: threadKey, tab: activeTab, itemId, loc },
      ];
      return next.length > MAX_THREADS ? next.slice(next.length - MAX_THREADS) : next;
    });
  }, [open, threadKey, activeTab, itemId, loc]);

  if (CHAT_HIDDEN_TABS.has(activeTab)) return null;

  const openDrawer = () => {
    setOpen(true);
    setEverOpened(true);
  };

  // Always render the active thread, even before the effect commits its entry.
  const rendered = threads.some((t) => t.key === threadKey) ? threads : [...threads, activeThread];

  return (
    <>
      {!open && (
        <button
          type="button"
          onClick={openDrawer}
          aria-label="Open assistant chat"
          className="fixed bottom-6 right-6 z-40 flex items-center gap-2 rounded-full bg-primary px-4 py-3 text-sm text-primary-foreground shadow-lg hover:opacity-90"
        >
          <MessageSquare className="h-4 w-4" />
          Assistant
        </button>
      )}

      {/* Kept mounted once opened (hidden when closed) so threads persist across open/close. */}
      {everOpened && (
        <aside
          hidden={!open}
          role="complementary"
          aria-label="Assistant chat"
          className="fixed inset-y-0 right-0 z-50 flex w-[420px] max-w-[92vw] flex-col border-l bg-background shadow-2xl"
        >
          <div className="flex items-center justify-between border-b px-4 py-2">
            <div className="text-sm font-semibold">
              {cfg.title}
              {itemId ? (
                <span className="ml-1 font-normal text-muted-foreground">
                  · <span className="font-mono">{itemId}</span>
                  {loc ? (
                    <>
                      {" @ "}
                      <span className="font-mono">{loc}</span>
                    </>
                  ) : null}
                </span>
              ) : null}
            </div>
            <button
              type="button"
              onClick={() => setOpen(false)}
              aria-label="Close chat"
              className="rounded p-1 hover:bg-muted"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="min-h-0 flex-1 p-3">
            {/* One panel per thread; inactive ones stay mounted but hidden so each
                page's conversation is preserved and resumes on return. */}
            {rendered.map((t) => {
              const tcfg = getPageChatConfig(t.tab);
              const isActive = t.key === threadKey;
              return (
                <div key={t.key} hidden={!isActive} className={isActive ? "h-full" : undefined}>
                  <GroundedCopilotPanel
                    page={t.tab}
                    itemId={t.itemId}
                    loc={t.loc}
                    suggestions={tcfg.suggestions}
                  />
                </div>
              );
            })}
          </div>
        </aside>
      )}
    </>
  );
}

export default GlobalChatDrawer;
