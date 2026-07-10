import { useState } from "react";
import { ArrowUp, Bot, Database, Loader2, Sparkles } from "lucide-react";

import {
  askCustomerAnalytics,
  type CustomerAnalyticsAskResponse,
  type CustomerAnalyticsChatMessage,
  type CustomerAnalyticsFilters,
  type CustomerAnalyticsView,
} from "@/api/queries/customer-analytics";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Input } from "@/components/ui/input";

interface CustomerAnalyticsAssistantProps {
  filters: CustomerAnalyticsFilters;
  activeView: CustomerAnalyticsView;
}

interface ConversationTurn {
  question: string;
  response: CustomerAnalyticsAskResponse;
}

const SUGGESTIONS: Record<CustomerAnalyticsView, string[]> = {
  overview: ["What changed most?", "Where is demand concentrated?"],
  customers: ["Which customers need attention?", "Who drives the most demand?"],
  segments: ["Which segment is weakening?", "Where is growth strongest?"],
  service: ["Why is fill rate under pressure?", "Where is lost demand highest?"],
  behavior: ["What buying patterns stand out?", "Where are cross-sell opportunities?"],
};

function conversationHistory(turns: ConversationTurn[]): CustomerAnalyticsChatMessage[] {
  return turns.slice(-3).flatMap((turn) => [
    { role: "user" as const, content: turn.question },
    { role: "assistant" as const, content: turn.response.answer },
  ]);
}

export function CustomerAnalyticsAssistant({
  filters,
  activeView,
}: CustomerAnalyticsAssistantProps) {
  const [question, setQuestion] = useState("");
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [isAsking, setIsAsking] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submitQuestion = async (nextQuestion: string) => {
    const trimmed = nextQuestion.trim();
    if (trimmed.length < 2 || isAsking) return;

    setIsAsking(true);
    setError(null);
    try {
      const response = await askCustomerAnalytics({
        question: trimmed,
        filters,
        active_view: activeView,
        history: conversationHistory(turns),
      });
      setTurns((current) => [...current, { question: trimmed, response }].slice(-4));
      setQuestion("");
    } catch {
      setError("Customer Intelligence could not answer right now. Check the AI runtime and try again.");
    } finally {
      setIsAsking(false);
    }
  };

  const latest = turns.length > 0 ? turns[turns.length - 1] : undefined;

  return (
    <Card className="overflow-hidden border-primary/20 bg-primary/[0.03]">
      <CardContent className="p-0">
        <div className="grid gap-0 lg:grid-cols-[minmax(0,1fr)_minmax(320px,0.7fr)]">
          <div className="p-4 sm:p-5">
            <div className="mb-3 flex items-start gap-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground shadow-sm">
                <Sparkles className="h-4 w-4" aria-hidden="true" />
              </div>
              <div>
                <div className="flex flex-wrap items-center gap-2">
                  <h3 className="text-sm font-semibold text-foreground">Customer Intelligence</h3>
                  <span className="rounded-full border border-primary/20 bg-background px-2 py-0.5 text-[10px] font-medium text-primary">
                    Grounded in this view
                  </span>
                </div>
                <p className="mt-0.5 text-xs text-muted-foreground">
                  Ask a question about the current filters. Answers use live KPIs and customer rankings.
                </p>
              </div>
            </div>

            <form
              className="flex flex-col gap-2 sm:flex-row"
              onSubmit={(event) => {
                event.preventDefault();
                void submitQuestion(question);
              }}
            >
              <Input
                value={question}
                onChange={(event) => setQuestion(event.target.value)}
                placeholder="Ask about demand, service, or customers…"
                aria-label="Customer Intelligence question"
                className="h-11 bg-background"
                disabled={isAsking}
              />
              <Button
                type="submit"
                className="h-11 shrink-0 px-5"
                disabled={isAsking || question.trim().length < 2}
                aria-label="Ask customer intelligence"
              >
                {isAsking ? (
                  <Loader2 className="h-4 w-4 animate-spin motion-reduce:animate-none" aria-hidden="true" />
                ) : (
                  <ArrowUp className="h-4 w-4" aria-hidden="true" />
                )}
                {isAsking ? "Analyzing" : "Ask"}
              </Button>
            </form>

            <div className="mt-2 flex flex-wrap gap-2" aria-label="Suggested questions">
              {SUGGESTIONS[activeView].map((suggestion) => (
                <button
                  key={suggestion}
                  type="button"
                  onClick={() => void submitQuestion(suggestion)}
                  disabled={isAsking}
                  className="min-h-8 rounded-full border bg-background px-3 text-xs text-muted-foreground transition-colors hover:border-primary/30 hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring disabled:opacity-50"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>

          <div className="border-t bg-background/70 p-4 lg:border-l lg:border-t-0" aria-live="polite">
            {error ? (
              <p className="text-sm text-destructive">{error}</p>
            ) : latest ? (
              <div className="space-y-3">
                <div className="flex items-center gap-2 text-xs font-medium text-foreground">
                  <Bot className="h-4 w-4 text-primary" aria-hidden="true" />
                  Answer
                </div>
                <p className="whitespace-pre-wrap text-sm leading-6 text-foreground">
                  {latest.response.answer}
                </p>
                <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-[11px] text-muted-foreground">
                  <span>{latest.response.provider} · {latest.response.model} · {latest.response.tier}</span>
                  <span className="inline-flex items-center gap-1">
                    <Database className="h-3 w-3" aria-hidden="true" />
                    {latest.response.evidence.length} evidence sets
                  </span>
                </div>
              </div>
            ) : (
              <div className="flex h-full min-h-24 items-center gap-3 text-sm text-muted-foreground">
                <Bot className="h-5 w-5 shrink-0" aria-hidden="true" />
                <p>Your answer will appear here without taking you away from the analysis.</p>
              </div>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
