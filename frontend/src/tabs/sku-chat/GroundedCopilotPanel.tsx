import { FormEvent, useRef, useState } from "react";
import {
  createCopilotSession,
  runCopilotTurn,
  type CopilotCitation,
} from "@/api/queries/ai-copilot";

interface Message {
  role: "user" | "assistant";
  text: string;
  citations?: CopilotCitation[];
}

export interface GroundedCopilotPanelProps {
  page: string;
  itemId: string;
  loc: string;
  suggestions?: string[];
}

function requestId(): string {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
}

export function GroundedCopilotPanel({
  page,
  itemId,
  loc,
  suggestions = [],
}: GroundedCopilotPanelProps) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const sessionId = useRef<string | null>(null);

  async function send(prompt: string) {
    const question = prompt.trim();
    if (!question || busy) return;
    setMessages((rows) => [...rows, { role: "user", text: question }]);
    setInput("");
    setError("");
    setBusy(true);
    try {
      if (!sessionId.current) {
        const context = itemId && loc ? { page, item_id: itemId, loc } : { page };
        sessionId.current = (await createCopilotSession(context, requestId())).session_id;
      }
      const turn = await runCopilotTurn(sessionId.current, question, requestId());
      setMessages((rows) => [
        ...rows,
        { role: "assistant", text: turn.answer, citations: turn.citations },
      ]);
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Copilot request failed");
    } finally {
      setBusy(false);
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void send(input);
  }

  return (
    <div className="flex h-full min-h-0 flex-col gap-3">
      <div className="flex-1 space-y-3 overflow-y-auto rounded border p-3">
        {messages.length === 0 ? (
          <p className="text-sm text-muted-foreground">
            Ask about the current planning context. Answers are grounded in server-resolved data and
            include traceable evidence.
          </p>
        ) : null}
        {messages.map((message, index) => (
          <div
            key={`${message.role}-${index}`}
            className={message.role === "user" ? "text-right" : "text-left"}
          >
            <div
              className={
                message.role === "user"
                  ? "inline-block rounded bg-primary px-3 py-2 text-sm text-primary-foreground"
                  : "inline-block max-w-[92%] rounded bg-muted px-3 py-2 text-sm"
              }
            >
              {message.text}
            </div>
            {message.citations?.map((citation) => (
              <details key={citation.evidence_id} className="mt-1 text-xs text-muted-foreground">
                <summary className="cursor-pointer">Evidence: {citation.claim}</summary>
                <div className="mt-1 rounded border p-2 font-mono">
                  {citation.source} · {citation.business_key} · {citation.freshness}
                  <br />
                  hash {citation.content_hash}
                </div>
              </details>
            ))}
          </div>
        ))}
        {error ? (
          <p role="alert" className="text-sm text-destructive">
            {error}
          </p>
        ) : null}
      </div>
      {messages.length === 0 && suggestions.length > 0 ? (
        <div className="flex flex-wrap gap-2">
          {suggestions.slice(0, 3).map((suggestion) => (
            <button
              key={suggestion}
              type="button"
              className="rounded border px-2 py-1 text-xs"
              onClick={() => void send(suggestion)}
            >
              {suggestion}
            </button>
          ))}
        </div>
      ) : null}
      <form onSubmit={submit} className="flex gap-2">
        <input
          aria-label="Message grounded Copilot"
          className="min-w-0 flex-1 rounded border bg-background px-3 py-2 text-sm"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          maxLength={4000}
        />
        <button
          type="submit"
          disabled={busy || !input.trim()}
          className="rounded bg-primary px-3 py-2 text-sm text-primary-foreground disabled:opacity-50"
        >
          {busy ? "Working…" : "Send"}
        </button>
      </form>
    </div>
  );
}
