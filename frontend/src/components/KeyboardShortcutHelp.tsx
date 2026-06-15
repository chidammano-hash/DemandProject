/**
 * Keyboard shortcut help dialog.
 *
 * Gen-4 UX: migrated from a bespoke `fixed inset-0` div stack to Radix Dialog
 * so the overlay/content get focus trap, aria-modal, Escape-to-close, and
 * focus restoration for free.
 */
import { cn } from "@/lib/utils";
import {
  Dialog,
  DialogContent,
  DialogTitle,
} from "@/components/ui/dialog";
import { NUMERIC_SHORTCUTS } from "@/components/AppSidebar";

interface ShortcutRow {
  keys: string[];
  desc: string;
}

// U6.1 — the numeric tab rows are derived from NAV_ITEMS[].shortcut (via
// NUMERIC_SHORTCUTS) so the help modal labels always match the sidebar hints
// and the live key handler. Only the non-numeric global shortcuts are static.
const NON_NUMERIC_SHORTCUTS: ShortcutRow[] = [
  { keys: ["["], desc: "Toggle sidebar" },
  { keys: ["d"], desc: "Toggle dark / light mode" },
  { keys: ["/"], desc: "Focus search" },
  { keys: ["Esc"], desc: "Close panel / help" },
  { keys: ["←"], desc: "Previous page" },
  { keys: ["→"], desc: "Next page" },
  { keys: ["Ctrl", "E"], desc: "Toggle column fields" },
  { keys: ["?"], desc: "Show / hide this help" },
];

const SHORTCUTS: ShortcutRow[] = [
  ...NUMERIC_SHORTCUTS.map((s) => ({ keys: [s.digit], desc: s.label })),
  ...NON_NUMERIC_SHORTCUTS,
];

export function KeyboardShortcutHelp({ onClose }: { onClose: () => void }) {
  return (
    <Dialog open onOpenChange={(o) => { if (!o) onClose(); }}>
      <DialogContent size="md" className="p-6" hideCloseButton aria-describedby={undefined}>
        <DialogTitle className="mb-4 text-lg font-bold text-foreground">
          Keyboard Shortcuts
        </DialogTitle>
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
      </DialogContent>
    </Dialog>
  );
}
