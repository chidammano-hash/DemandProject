/**
 * Floating AI Tuning Advisor — bottom-right FAB that opens an embedded
 * TuningChatPanel popover.
 */
import { useState } from "react";
import { createPortal } from "react-dom";
import { MessageSquare, X } from "lucide-react";

import { TuningChatPanel } from "../lgbm-tuning/TuningChatPanel";

export function AIAdvisorFAB() {
  const [chatOpen, setChatOpen] = useState(false);

  return createPortal(
    <div className="fixed top-4 right-6 z-50 flex flex-col items-end gap-3">
      {!chatOpen && (
        <button
          onClick={() => setChatOpen(true)}
          title="AI Tuning Advisor"
          className="h-10 w-10 rounded-full bg-primary text-primary-foreground shadow-lg hover:bg-primary/90 transition-all hover:scale-105 flex items-center justify-center"
        >
          <MessageSquare className="h-4.5 w-4.5" />
        </button>
      )}
      {chatOpen && (
        <div className="w-[420px] max-h-[80vh] animate-in slide-in-from-top-4 fade-in duration-200 rounded-2xl border border-border bg-card shadow-2xl overflow-hidden flex flex-col">
          <div className="flex items-center justify-between px-4 py-2.5 border-b border-border bg-muted/30">
            <div className="flex items-center gap-2">
              <MessageSquare className="h-4 w-4 text-primary" />
              <span className="text-sm font-semibold">AI Tuning Advisor</span>
            </div>
            <button
              onClick={() => setChatOpen(false)}
              className="rounded-md p-1 text-muted-foreground hover:text-foreground hover:bg-muted transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <div className="flex-1 overflow-hidden">
            <TuningChatPanel />
          </div>
        </div>
      )}
    </div>,
    document.body,
  );
}
