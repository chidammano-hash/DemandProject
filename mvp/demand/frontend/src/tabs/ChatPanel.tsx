import { useEffect, useRef, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { sendChatMessage } from "@/api/queries";
import type { ChatMessage } from "@/types";
import { formatCell } from "@/lib/formatters";
import { cn } from "@/lib/utils";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Loader2, MessageSquare, Send } from "lucide-react";

type ChatPanelProps = {
  domain: string;
};

export default function ChatPanel({ domain }: ChatPanelProps) {
  const [chatOpen, setChatOpen] = useState(false);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [chatInput, setChatInput] = useState("");
  const chatEndRef = useRef<HTMLDivElement>(null);

  const chatMutation = useMutation({
    mutationFn: (question: string) => sendChatMessage(question, domain),
    onSuccess: (payload) => {
      const assistantMsg: ChatMessage = {
        role: "assistant",
        content: payload.answer || "No answer returned.",
        sql: payload.sql || null,
        data: payload.data || null,
        columns: payload.columns || [],
        row_count: payload.row_count ?? null,
        error: payload.error || null,
      };
      setChatMessages((prev) => [...prev, assistantMsg]);
    },
    onError: (err: Error) => {
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Network error: ${err.message}` },
      ]);
    },
  });

  const chatLoading = chatMutation.isPending;

  function sendChat() {
    const q = chatInput.trim();
    if (!q || chatLoading) return;
    const userMsg: ChatMessage = { role: "user", content: q };
    setChatMessages((prev) => [...prev, userMsg]);
    setChatInput("");
    chatMutation.mutate(q);
  }

  // Auto-scroll to latest message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatMessages]);

  return (
    <section className="mt-4">
      <Card className="animate-fade-in">
        <CardHeader className="cursor-pointer" onClick={() => setChatOpen((v) => !v)}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <MessageSquare className="h-5 w-5" />
              <CardTitle className="text-base">Chat with Planthium</CardTitle>
            </div>
            <Badge variant="outline">{chatOpen ? "Collapse" : "Expand"}</Badge>
          </div>
          <CardDescription>Ask questions about your data in plain English.</CardDescription>
        </CardHeader>
        {chatOpen ? (
          <CardContent className="space-y-3">
            <div className="max-h-[400px] min-h-[120px] overflow-y-auto rounded-md border bg-muted/10 p-3 space-y-3">
              {chatMessages.length === 0 ? (
                <p className="text-sm text-muted-foreground">
                  No messages yet. Try asking: &quot;What are the top 10 items by total sales
                  quantity?&quot;
                </p>
              ) : (
                chatMessages.map((msg, idx) => (
                  <div
                    key={idx}
                    className={cn(
                      "rounded-lg px-3 py-2 text-sm",
                      msg.role === "user"
                        ? "ml-8 bg-indigo-100 text-indigo-900"
                        : "mr-8 bg-white border shadow-sm",
                    )}
                  >
                    <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground mb-1">
                      {msg.role === "user" ? "You" : "Assistant"}
                    </p>
                    <p className="whitespace-pre-wrap">{msg.content}</p>
                    {msg.error ? (
                      <p className="mt-1 text-xs text-red-600">{msg.error}</p>
                    ) : null}
                    {msg.sql ? (
                      <details className="mt-2">
                        <summary className="cursor-pointer text-xs font-medium text-muted-foreground">
                          SQL Query
                        </summary>
                        <pre className="mt-1 overflow-x-auto rounded bg-slate-100 p-2 text-xs">
                          {msg.sql}
                        </pre>
                      </details>
                    ) : null}
                    {msg.data && msg.data.length > 0 && msg.columns && msg.columns.length > 0 ? (
                      <div className="mt-2">
                        <div className="flex items-center gap-2 mb-1">
                          <Badge variant="secondary" className="text-xs">
                            {msg.row_count ?? msg.data.length} row(s)
                          </Badge>
                        </div>
                        <div className="max-h-[200px] overflow-auto rounded border">
                          <Table>
                            <TableHeader>
                              <TableRow>
                                {msg.columns.map((col) => (
                                  <TableHead key={col} className="text-xs whitespace-nowrap">
                                    {col}
                                  </TableHead>
                                ))}
                              </TableRow>
                            </TableHeader>
                            <TableBody>
                              {msg.data.slice(0, 10).map((row, rIdx) => (
                                <TableRow key={rIdx}>
                                  {msg.columns!.map((col) => (
                                    <TableCell key={col} className="text-xs whitespace-nowrap">
                                      {formatCell(row[col])}
                                    </TableCell>
                                  ))}
                                </TableRow>
                              ))}
                            </TableBody>
                          </Table>
                        </div>
                        {(msg.row_count ?? 0) > 10 ? (
                          <p className="mt-1 text-xs text-muted-foreground">
                            Showing 10 of {msg.row_count} rows.
                          </p>
                        ) : null}
                      </div>
                    ) : null}
                  </div>
                ))
              )}
              {chatLoading ? (
                <div className="flex items-center gap-2 text-sm text-muted-foreground">
                  <Loader2 className="h-4 w-4 animate-spin" /> Thinking...
                </div>
              ) : null}
              <div ref={chatEndRef} />
            </div>
            <div className="flex gap-2">
              <Input
                placeholder="Ask a question about your data..."
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendChat();
                  }
                }}
                disabled={chatLoading}
                className="flex-1"
              />
              <Button onClick={sendChat} disabled={chatLoading || !chatInput.trim()} size="sm">
                <Send className="mr-1 h-4 w-4" /> Send
              </Button>
            </div>
          </CardContent>
        ) : null}
      </Card>
    </section>
  );
}
