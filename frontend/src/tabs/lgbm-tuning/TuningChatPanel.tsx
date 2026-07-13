import { useState, useRef, useEffect, useCallback, useMemo, type ReactNode } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Send, Loader2, Bot, User, Info, Sparkles } from "lucide-react";
import {
  tuningChatKeys,
  fetchChatSession,
  sendTuningChatMessage,
  createChatSession,
  confirmTuningRun,
  type ChatMessage,
  type TuningRecommendation,
} from "@/api/queries";
import { cn } from "@/lib/utils";
import { RecommendationCard } from "./RecommendationCard";
import { RunStatusCard } from "./RunStatusCard";

// ---------------------------------------------------------------------------
// Lightweight markdown renderer — handles **bold**, `code`, and - lists
// ---------------------------------------------------------------------------

function renderMarkdown(text: string): ReactNode[] {
  const lines = text.split("\n");
  const elements: ReactNode[] = [];
  let listItems: ReactNode[] = [];

  function flushList() {
    if (listItems.length > 0) {
      elements.push(
        <ul key={`list-${elements.length}`} className="space-y-1 my-1.5">
          {listItems}
        </ul>,
      );
      listItems = [];
    }
  }

  function inlineFormat(line: string, key: string | number): ReactNode {
    const parts: ReactNode[] = [];
    const regex = /(\*\*(.+?)\*\*)|(`([^`]+?)`)/g;
    let lastIndex = 0;
    let match: RegExpExecArray | null;

    while ((match = regex.exec(line)) !== null) {
      if (match.index > lastIndex) {
        parts.push(line.slice(lastIndex, match.index));
      }
      if (match[2]) {
        parts.push(
          <strong key={`b-${key}-${match.index}`} className="font-semibold text-foreground">
            {match[2]}
          </strong>,
        );
      } else if (match[4]) {
        parts.push(
          <code key={`c-${key}-${match.index}`} className="px-1 py-0.5 rounded bg-muted text-primary text-xs font-mono">
            {match[4]}
          </code>,
        );
      }
      lastIndex = match.index + match[0].length;
    }
    if (lastIndex < line.length) {
      parts.push(line.slice(lastIndex));
    }
    return parts.length > 0 ? parts : line;
  }

  for (let i = 0; i < lines.length; i++) {
    const line = lines[i];
    const trimmed = line.trim();

    if (/^\s*-\s+/.test(line)) {
      const content = trimmed.replace(/^-\s+/, "");
      listItems.push(
        <li key={`li-${i}`} className="flex gap-1.5 text-sm leading-relaxed">
          <span className="text-primary mt-0.5 shrink-0">&#8226;</span>
          <span>{inlineFormat(content, i)}</span>
        </li>,
      );
      continue;
    }

    flushList();

    if (trimmed === "") {
      elements.push(<div key={`br-${i}`} className="h-1.5" />);
    } else {
      elements.push(
        <p key={`p-${i}`} className="text-sm leading-relaxed">
          {inlineFormat(trimmed, i)}
        </p>,
      );
    }
  }

  flushList();
  return elements;
}

// ---------------------------------------------------------------------------
// Message bubble
// ---------------------------------------------------------------------------

function MessageBubble({ message, sessionId, onConfirm, isConfirming, confirmedRuns, inlineRunIds }: {
  message: ChatMessage;
  sessionId: string;
  onConfirm: (messageId: number) => void;
  isConfirming: boolean;
  confirmedRuns: Map<number, number>;
  inlineRunIds: Set<number>;
}) {
  const isUser = message.role === "user";
  const isSystem = message.role === "system";

  // Recommendation card
  if (message.message_type === "recommendation" && message.metadata) {
    const rec = message.metadata as unknown as TuningRecommendation;
    return (
      <div className="my-3">
        <RecommendationCard
          recommendation={rec}
          messageId={message.message_id}
          sessionId={sessionId}
          onConfirm={onConfirm}
          onReject={() => {}}
          isConfirming={isConfirming}
          isConfirmed={confirmedRuns.has(message.message_id)}
          confirmedRunId={confirmedRuns.get(message.message_id)}
        />
      </div>
    );
  }

  // Run status cards
  if (message.message_type === "run_started" || message.message_type === "run_completed" || message.message_type === "run_failed") {
    const runId = (message.metadata as Record<string, unknown>)?.run_id as number;
    if (runId && inlineRunIds.has(runId)) return null;
    if (runId) {
      return (
        <div className="my-3">
          <RunStatusCard
            sessionId={sessionId}
            runId={runId}
            messageType={message.message_type}
            completedResult={message.metadata as Record<string, unknown> | undefined}
            errorMessage={(message.metadata as Record<string, unknown>)?.error as string | undefined}
          />
        </div>
      );
    }
  }

  const timestamp = new Date(message.created_at).toLocaleTimeString(undefined, {
    hour: "2-digit",
    minute: "2-digit",
  });

  // User message
  if (isUser) {
    return (
      <div className="flex gap-2.5 my-3 justify-end">
        <div className="max-w-[85%]">
          <div className="bg-primary text-primary-foreground rounded-2xl rounded-br-md px-3.5 py-2.5 text-sm leading-relaxed">
            {message.content}
          </div>
          <span className="text-[10px] text-muted-foreground mt-1 block text-right pr-1">
            {timestamp}
          </span>
        </div>
        <div className="flex-shrink-0 w-7 h-7 rounded-full bg-primary/15 flex items-center justify-center mt-1">
          <User className="h-3.5 w-3.5 text-primary" />
        </div>
      </div>
    );
  }

  // System message
  if (isSystem) {
    return (
      <div className="flex justify-center my-2">
        <div className="flex items-center gap-1.5 px-3 py-1 rounded-full bg-muted text-muted-foreground text-xs">
          <Info className="h-3 w-3" />
          {message.content}
        </div>
      </div>
    );
  }

  // Bot message — render with markdown
  return (
    <div className="flex gap-2.5 my-3 justify-start">
      <div className="flex-shrink-0 w-7 h-7 rounded-full bg-primary flex items-center justify-center mt-1 shadow-sm">
        <Bot className="h-3.5 w-3.5 text-primary-foreground" />
      </div>
      <div className="max-w-[85%]">
        <div className="bg-muted/70 border border-border text-foreground rounded-2xl rounded-bl-md px-3.5 py-2.5">
          {renderMarkdown(message.content)}
        </div>
        <span className="text-[10px] text-muted-foreground mt-1 block pl-1">
          {timestamp}
        </span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel — single auto-created session, no session tabs
// ---------------------------------------------------------------------------

const SESSION_KEY = "tuning_chat_session_id";

export function TuningChatPanel() {
  const queryClient = useQueryClient();
  const [activeSessionId, setActiveSessionId] = useState<string | null>(() => {
    try { return localStorage.getItem(SESSION_KEY); } catch { return null; }
  });
  const [inputValue, setInputValue] = useState("");
  const [confirmedRuns, setConfirmedRuns] = useState<Map<number, number>>(new Map());
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Persist active session to localStorage
  useEffect(() => {
    try {
      if (activeSessionId) localStorage.setItem(SESSION_KEY, activeSessionId);
    } catch { /* ignore */ }
  }, [activeSessionId]);

  // Auto-create session on first mount if none exists
  const createMutation = useMutation({
    mutationFn: () => createChatSession("Tuning Session"),
    onSuccess: (data) => {
      setActiveSessionId(data.session.session_id);
    },
  });

  useEffect(() => {
    if (!activeSessionId && !createMutation.isPending) {
      createMutation.mutate();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch session messages
  const { data: sessionData, isLoading: isLoadingSession } = useQuery({
    queryKey: tuningChatKeys.session(activeSessionId ?? ""),
    queryFn: () => fetchChatSession(activeSessionId!),
    enabled: !!activeSessionId,
    refetchInterval: 15_000,
  });

  const messages = useMemo(() => sessionData?.messages ?? [], [sessionData]);

  // On session load, detect recommendation -> run_started pairs to restore confirmed state
  useEffect(() => {
    if (messages.length === 0) return;
    const restored = new Map<number, number>();
    for (let i = 0; i < messages.length; i++) {
      if (messages[i].message_type === "recommendation") {
        for (let j = i + 1; j < messages.length && j <= i + 3; j++) {
          const next = messages[j];
          if (next.message_type === "run_started" && next.metadata) {
            const runId = (next.metadata as Record<string, unknown>).run_id as number;
            if (runId) {
              restored.set(messages[i].message_id, runId);
              break;
            }
          }
          if (next.message_type === "recommendation") break;
        }
      }
    }
    if (restored.size > 0) {
      setConfirmedRuns((prev) => {
        const merged = new Map(prev);
        for (const [k, v] of restored) {
          if (!merged.has(k)) merged.set(k, v);
        }
        return merged;
      });
    }
  }, [messages]);

  // Send message mutation
  const sendMutation = useMutation({
    mutationFn: (content: string) => sendTuningChatMessage(activeSessionId!, content),
    onSuccess: () => {
      queryClient.invalidateQueries({
        queryKey: tuningChatKeys.session(activeSessionId!),
      });
    },
  });

  // Confirm run mutation
  const confirmMutation = useMutation({
    mutationFn: (messageId: number) =>
      confirmTuningRun(activeSessionId!, messageId),
    onSuccess: (data, messageId) => {
      setConfirmedRuns((prev) => new Map(prev).set(messageId, data.run_id));
      queryClient.invalidateQueries({
        queryKey: tuningChatKeys.session(activeSessionId!),
      });
    },
  });

  // Auto-scroll to bottom
  useEffect(() => {
    if (messagesEndRef.current && typeof messagesEndRef.current.scrollIntoView === "function") {
      messagesEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages.length]);

  const handleSend = useCallback(() => {
    const text = inputValue.trim();
    if (!text || sendMutation.isPending || !activeSessionId) return;
    setInputValue("");
    sendMutation.mutate(text);
  }, [inputValue, sendMutation, activeSessionId]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        handleSend();
      }
    },
    [handleSend],
  );

  return (
    <div className="flex flex-col bg-background h-[500px]">
      {/* Messages area */}
      <div className="flex-1 overflow-y-auto px-4 py-3">
        {(!activeSessionId || createMutation.isPending) && (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        )}

        {activeSessionId && isLoadingSession && (
          <div className="flex items-center justify-center h-full">
            <Loader2 className="h-5 w-5 animate-spin text-muted-foreground" />
          </div>
        )}

        {activeSessionId && !isLoadingSession && messages.length === 0 && (
          <div className="flex flex-col items-center justify-center h-full">
            <div className="w-12 h-12 rounded-full bg-primary flex items-center justify-center mb-3 shadow-lg">
              <Sparkles className="h-6 w-6 text-primary-foreground" />
            </div>
            <p className="text-sm font-medium text-foreground mb-1">AI Tuning Advisor</p>
            <p className="text-xs text-muted-foreground mb-4">Ask about your runs or get suggestions</p>
            <div className="flex flex-col gap-2 w-full max-w-[280px]">
              {[
                "What was the best run so far?",
                "Which clusters need improvement?",
                "What should I try next?",
              ].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => setInputValue(suggestion)}
                  className="text-xs px-3 py-2 rounded-lg border border-border hover:border-primary/40 hover:bg-primary/5 text-muted-foreground hover:text-foreground transition-colors text-left"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        )}

        {(() => {
          const inlineRunIds = new Set(confirmedRuns.values());
          return messages.map((msg) => (
            <MessageBubble
              key={msg.message_id}
              message={msg}
              sessionId={activeSessionId!}
              onConfirm={(id) => confirmMutation.mutate(id)}
              isConfirming={confirmMutation.isPending}
              confirmedRuns={confirmedRuns}
              inlineRunIds={inlineRunIds}
            />
          ));
        })()}

        {sendMutation.isPending && (
          <div className="flex items-center gap-2.5 my-3">
            <div className="w-7 h-7 rounded-full bg-primary flex items-center justify-center shadow-sm">
              <Bot className="h-3.5 w-3.5 text-primary-foreground" />
            </div>
            <div className="bg-muted/70 border border-border rounded-2xl rounded-bl-md px-3.5 py-2.5">
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <div className="flex gap-1">
                  <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-bounce" style={{ animationDelay: "0ms" }} />
                  <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-bounce" style={{ animationDelay: "150ms" }} />
                  <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground/50 animate-bounce" style={{ animationDelay: "300ms" }} />
                </div>
                Analyzing...
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input area */}
      <div className="border-t border-border p-3">
        <div className="flex gap-2 items-end bg-muted/50 border border-border rounded-xl px-3 py-2 focus-within:border-ring transition-colors">
          <textarea
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about runs, clusters, or what to try next..."
            disabled={!activeSessionId || sendMutation.isPending}
            rows={1}
            className={cn(
              "flex-1 bg-transparent text-sm resize-none border-none outline-none",
              "text-foreground placeholder:text-muted-foreground/60",
              "disabled:opacity-50 disabled:cursor-not-allowed",
            )}
          />
          <button
            onClick={handleSend}
            disabled={!inputValue.trim() || !activeSessionId || sendMutation.isPending}
            className={cn(
              "p-1.5 rounded-lg transition-all shrink-0",
              inputValue.trim()
                ? "bg-primary hover:bg-primary/90 text-primary-foreground shadow-sm"
                : "text-muted-foreground",
              "disabled:opacity-40 disabled:cursor-not-allowed",
            )}
          >
            {sendMutation.isPending ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>
    </div>
  );
}
