import { cn } from "@/lib/utils";

const SHORTCUTS = [
  { keys: ["1"], desc: "Explorer tab" },
  { keys: ["2"], desc: "Clusters tab" },
  { keys: ["3"], desc: "DFU Analysis tab" },
  { keys: ["4"], desc: "Accuracy tab" },
  { keys: ["5"], desc: "Market Intel tab" },
  { keys: ["/"], desc: "Focus search" },
  { keys: ["Esc"], desc: "Close panel / help" },
  { keys: ["←"], desc: "Previous page" },
  { keys: ["→"], desc: "Next page" },
  { keys: ["Ctrl", "E"], desc: "Toggle column fields" },
  { keys: ["?"], desc: "Show / hide this help" },
];

export function KeyboardShortcutHelp({ onClose }: { onClose: () => void }) {
  return (
    <div
      className="fixed inset-0 z-[100] flex items-center justify-center bg-black/40 backdrop-blur-sm"
      onClick={onClose}
    >
      <div
        className="w-full max-w-md rounded-xl border border-border bg-card p-6 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <h2 className="text-lg font-bold text-foreground mb-4">Keyboard Shortcuts</h2>
        <div className="space-y-2">
          {SHORTCUTS.map((s) => (
            <div key={s.desc} className="flex items-center justify-between text-sm">
              <span className="text-muted-foreground">{s.desc}</span>
              <div className="flex gap-1">
                {s.keys.map((k) => (
                  <kbd
                    key={k}
                    className={cn(
                      "inline-flex min-w-[24px] items-center justify-center rounded border border-border bg-muted px-1.5 py-0.5 font-mono text-xs text-foreground",
                    )}
                  >
                    {k}
                  </kbd>
                ))}
              </div>
            </div>
          ))}
        </div>
        <p className="mt-4 text-xs text-muted-foreground text-center">
          Press <kbd className="rounded border border-border bg-muted px-1 font-mono text-xs">Esc</kbd> or <kbd className="rounded border border-border bg-muted px-1 font-mono text-xs">?</kbd> to close
        </p>
      </div>
    </div>
  );
}
