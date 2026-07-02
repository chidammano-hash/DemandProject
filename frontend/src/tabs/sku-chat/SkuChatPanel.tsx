// Reusable SKU chat panel — conversation + composer + streaming, scoped to a SKU
// passed by props. Used by the standalone SkuChatTab and the Item Analysis side chat.
// Also drives the agentic champion-forecast adjustment: an "AI Adjust" button asks
// the agent to propose an adjustment, which streams back as an approval card.
// Spec: docs/specs/06-ai-platform/07-sku-chatbot.md
import { useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import {
  decideSkuChatAdjustment,
  fetchSkuChatConfig,
  skuChatKeys,
  streamSkuChat,
  type ChampionAdjustPreview,
  type SkuChatEvent,
} from "@/api/queries";

const ADJUST_PROMPT =
  "Review this SKU's champion forecast against recent actuals and seasonality. " +
  "If an adjustment is warranted, use the apply_champion_adjustment tool to stage a " +
  "proposal for my approval and explain your reasoning. If no change is warranted, say so.";

interface ToolCall {
  name: string | null;
  input: unknown;
}

interface ChatMessage {
  role: "user" | "assistant";
  text: string;
  tools: ToolCall[];
  meta: { tier: string; model: string } | null;
  cost: number | null;
  error: string | null;
  pending: boolean;
}

type ApprovalStatus = "pending" | "approving" | "rejecting" | "approved" | "rejected" | "error";

interface PendingApproval {
  approval_id: string;
  item_id: string;
  loc: string;
  preview: ChampionAdjustPreview;
  status: ApprovalStatus;
  error?: string;
}

function blankAssistant(): ChatMessage {
  return { role: "assistant", text: "", tools: [], meta: null, cost: null, error: null, pending: true };
}

function applyEvent(msg: ChatMessage, ev: SkuChatEvent): ChatMessage {
  switch (ev.type) {
    case "meta":
      return { ...msg, meta: { tier: ev.tier, model: ev.model } };
    case "text":
      return { ...msg, text: msg.text + ev.chunk };
    case "tool":
      return { ...msg, tools: [...msg.tools, { name: ev.name, input: ev.input }] };
    case "result":
      return { ...msg, text: ev.text ?? msg.text, cost: ev.cost_usd, pending: false };
    case "error":
      return { ...msg, error: ev.message, pending: false };
    default:
      return msg;
  }
}

export interface SkuChatPanelProps {
  itemId: string;
  loc: string;
  customerGroup?: string;
  /** Short description of the current page — sent to the agent for per-page tailoring. */
  pageFocus?: string;
  /** Page-relevant starter prompts shown as chips before the first message. */
  suggestions?: string[];
}

export function SkuChatPanel({
  itemId,
  loc,
  customerGroup = "",
  pageFocus,
  suggestions = [],
}: SkuChatPanelProps) {
  const [input, setInput] = useState("");
  const [deep, setDeep] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [pending, setPending] = useState<PendingApproval[]>([]);
  const [streaming, setStreaming] = useState(false);
  const sessionRef = useRef<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  const { data: config } = useQuery({
    queryKey: skuChatKeys.config(),
    queryFn: fetchSkuChatConfig,
    staleTime: 5 * 60 * 1000,
  });

  const hasSku = Boolean(itemId.trim() && loc.trim());
  const canSend = Boolean(input.trim() && !streaming); // SKU optional — page-aware global chat
  const canAdjust = Boolean(hasSku && !streaming);

  async function send(questionOverride?: string, tierOverride?: "fast" | "standard" | "deep") {
    const question = (questionOverride ?? input).trim();
    if (!question || streaming) return;
    const history = messages
      .filter((m) => !m.error)
      .map((m) => ({ role: m.role, content: m.text }));

    setMessages((prev) => [
      ...prev,
      { role: "user", text: question, tools: [], meta: null, cost: null, error: null, pending: false },
      blankAssistant(),
    ]);
    if (questionOverride === undefined) setInput("");
    setStreaming(true);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      for await (const ev of streamSkuChat(
        {
          question,
          item_id: itemId.trim(),
          loc: loc.trim(),
          customer_group: customerGroup.trim(),
          session_id: sessionRef.current,
          history,
          model_tier: tierOverride ?? (deep ? "deep" : null),
          page_focus: pageFocus ?? null,
        },
        controller.signal,
      )) {
        if (ev.type === "meta" && ev.session_id) sessionRef.current = ev.session_id;
        if (ev.type === "approval_request") {
          setPending((p) => [
            ...p,
            { approval_id: ev.approval_id, item_id: ev.item_id, loc: ev.loc, preview: ev.preview, status: "pending" },
          ]);
          continue;
        }
        setMessages((prev) => {
          const copy = [...prev];
          copy[copy.length - 1] = applyEvent(copy[copy.length - 1], ev);
          return copy;
        });
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Stream failed";
      setMessages((prev) => {
        const copy = [...prev];
        copy[copy.length - 1] = { ...copy[copy.length - 1], error: message, pending: false };
        return copy;
      });
    } finally {
      setStreaming(false);
      abortRef.current = null;
    }
  }

  async function decide(approvalId: string, decision: "approve" | "reject") {
    setPending((p) =>
      p.map((x) =>
        x.approval_id === approvalId ? { ...x, status: decision === "approve" ? "approving" : "rejecting" } : x,
      ),
    );
    try {
      const res = await decideSkuChatAdjustment(approvalId, decision);
      setPending((p) =>
        p.map((x) =>
          x.approval_id === approvalId ? { ...x, status: res.status === "approved" ? "approved" : "rejected" } : x,
        ),
      );
    } catch (err) {
      const message = err instanceof Error ? err.message : "failed";
      setPending((p) =>
        p.map((x) => (x.approval_id === approvalId ? { ...x, status: "error", error: message } : x)),
      );
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-2">
      {config ? (
        <div className="px-1 text-[11px] text-muted-foreground">
          auth <span className="font-mono">{config.auth_mode}</span> · default tier{" "}
          <span className="font-mono">{config.routing.default_tier}</span>
        </div>
      ) : null}

      <div className="flex-1 space-y-3 overflow-y-auto rounded border p-3">
        {messages.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Ask a question about this page — pick a suggestion below to start. When a SKU is in
            scope you can also use &ldquo;AI Adjust&rdquo; to propose a champion-forecast change for
            your approval.
          </p>
        ) : (
          messages.map((m, i) => (
            <div key={i} className={m.role === "user" ? "text-right" : "text-left"}>
              <div
                className={
                  m.role === "user"
                    ? "inline-block rounded bg-primary px-3 py-2 text-sm text-primary-foreground"
                    : "inline-block max-w-[92%] rounded bg-muted px-3 py-2 text-sm"
                }
              >
                {m.meta && m.role === "assistant" ? (
                  <div className="mb-1 text-[10px] text-muted-foreground">
                    {m.meta.tier} · {m.meta.model}
                  </div>
                ) : null}
                {m.tools.length > 0 ? (
                  <div className="mb-1 flex flex-wrap gap-1">
                    {m.tools.map((t, j) => (
                      <span key={j} className="rounded bg-background px-1.5 py-0.5 text-[10px] font-mono">
                        {t.name}
                      </span>
                    ))}
                  </div>
                ) : null}
                <span className="whitespace-pre-wrap">{m.text}</span>
                {m.pending ? <span className="ml-1 animate-pulse">▋</span> : null}
                {m.error ? <div className="mt-1 text-xs text-destructive">{m.error}</div> : null}
                {m.cost != null ? (
                  <div className="mt-1 text-[10px] text-muted-foreground">~${m.cost.toFixed(4)}</div>
                ) : null}
              </div>
            </div>
          ))
        )}
      </div>

      {/* Pending champion-adjustment approvals */}
      {pending.length > 0 ? (
        <div className="space-y-2">
          {pending.map((a) => (
            <div
              key={a.approval_id}
              className="rounded border border-amber-400/60 bg-amber-50/40 p-2 text-xs dark:bg-amber-950/20"
            >
              <div className="font-semibold">
                Proposed champion adjustment · <span className="font-mono">{a.item_id}</span> @{" "}
                <span className="font-mono">{a.loc}</span>
              </div>
              <div className="mt-0.5">
                {a.preview.recommendation_code}
                {a.preview.rec_pct_change != null
                  ? ` · ${a.preview.rec_pct_change > 0 ? "+" : ""}${a.preview.rec_pct_change}%`
                  : ""}
                {a.preview.confidence != null ? ` · conf ${Math.round(a.preview.confidence * 100)}%` : ""}
              </div>
              {a.preview.rationale ? (
                <div className="mt-1 text-muted-foreground">{a.preview.rationale}</div>
              ) : null}
              {a.status === "pending" || a.status === "approving" || a.status === "rejecting" ? (
                <div className="mt-2 flex gap-2">
                  <button
                    type="button"
                    disabled={a.status !== "pending"}
                    onClick={() => void decide(a.approval_id, "approve")}
                    className="rounded bg-primary px-3 py-1 text-primary-foreground disabled:opacity-50"
                  >
                    {a.status === "approving" ? "Applying…" : "Approve & apply"}
                  </button>
                  <button
                    type="button"
                    disabled={a.status !== "pending"}
                    onClick={() => void decide(a.approval_id, "reject")}
                    className="rounded border px-3 py-1 disabled:opacity-50"
                  >
                    Reject
                  </button>
                </div>
              ) : (
                <div
                  className={
                    "mt-1 font-medium " +
                    (a.status === "approved"
                      ? "text-green-600"
                      : a.status === "error"
                        ? "text-destructive"
                        : "text-muted-foreground")
                  }
                >
                  {a.status === "approved"
                    ? "✓ Applied to the champion forecast"
                    : a.status === "rejected"
                      ? "Rejected — no change made"
                      : `Error: ${a.error}`}
                </div>
              )}
            </div>
          ))}
        </div>
      ) : null}

      {suggestions.length > 0 && messages.length === 0 ? (
        <div className="flex flex-wrap gap-1 px-1">
          {suggestions.map((s, i) => (
            <button
              key={i}
              type="button"
              onClick={() => void send(s)}
              disabled={streaming}
              className="rounded-full border px-2 py-1 text-[11px] text-muted-foreground hover:bg-muted disabled:opacity-50"
            >
              {s}
            </button>
          ))}
        </div>
      ) : null}

      <div className="flex items-center justify-between px-1">
        <label className="flex items-center gap-1 text-xs">
          <input type="checkbox" checked={deep} onChange={(e) => setDeep(e.target.checked)} />
          Deep analysis (Opus)
        </label>
        <button
          type="button"
          onClick={() => void send(ADJUST_PROMPT, "deep")}
          disabled={!canAdjust}
          className="rounded border px-2 py-1 text-xs disabled:opacity-50"
        >
          ✨ AI Adjust
        </button>
      </div>

      <div className="flex items-end gap-2">
        <textarea
          aria-label="message"
          className="min-h-[44px] flex-1 resize-none rounded border px-3 py-2 text-sm"
          placeholder={hasSku ? "Ask about this SKU…" : "Select a SKU first"}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              void send();
            }
          }}
        />
        {streaming ? (
          <button
            type="button"
            onClick={() => abortRef.current?.abort()}
            className="rounded border px-4 py-2 text-sm"
          >
            Stop
          </button>
        ) : (
          <button
            type="button"
            onClick={() => void send()}
            disabled={!canSend}
            className="rounded bg-primary px-4 py-2 text-sm text-primary-foreground disabled:opacity-50"
          >
            Send
          </button>
        )}
      </div>
    </div>
  );
}

export default SkuChatPanel;
