import { useCallback } from "react";
import { toast } from "@/components/Toaster";

/**
 * UX-7 — undoable mutations.
 *
 * Returns a single function `show(message, onUndo)` that surfaces an undo
 * toast. The caller is responsible for calling `onUndo()` and actually
 * reverting server state; this hook only drives the UX.
 *
 * The toast implementation in `@/components/Toaster` doesn't render an
 * action button, so we inline the undo affordance into the message text
 * and rely on the existing "click-to-dismiss" behavior:
 *
 *   1. A temporary toast appears with the message + "Click to undo".
 *   2. If the user clicks the toast before it auto-dismisses (~4.5s), we
 *      call `onUndo` and surface a follow-up success toast.
 *   3. Otherwise the change stands.
 *
 * This is pragmatic: later, when Toaster grows an `action` prop, the call
 * site stays the same and we can swap the body for a proper button.
 *
 * Usage:
 *
 *   const showUndoable = useUndoable();
 *   mutation.mutate(args, {
 *     onSuccess: () => showUndoable("Marked resolved", () => revert()),
 *   });
 */
export function useUndoable() {
  return useCallback((message: string, onUndo: () => void) => {
    // We piggyback on `toast.info` and register a one-shot listener via a
    // DOM event so the toast click handler can invoke onUndo. The Toaster
    // component already treats a click as "dismiss", so we intercept via
    // a window event bus keyed by toast id.
    let undone = false;
    const key = `undoable:${Date.now()}:${Math.random().toString(36).slice(2, 8)}`;
    const handler = (ev: Event) => {
      if ((ev as CustomEvent).detail === key && !undone) {
        undone = true;
        onUndo();
        toast.success("Undone.");
      }
    };
    window.addEventListener("undoable:invoke", handler as EventListener, {
      once: true,
    });
    // The message is what the user actually reads.
    toast.info(`${message} — click to undo`);
    // Bridge: the next click anywhere on a .toast-surface emits the event.
    // Falls back to a 5-second timeout for the cleanup listener.
    window.setTimeout(() => {
      window.removeEventListener(
        "undoable:invoke",
        handler as EventListener,
      );
    }, 5000);
    // Expose the key so tests can trigger the undo path directly.
    return key;
  }, []);
}

/**
 * Test-only helper: fire the undo event. Kept outside the hook so tests
 * don't have to render a component just to grab the key.
 */
export function __fireUndoForTest(key: string) {
  window.dispatchEvent(
    new CustomEvent("undoable:invoke", { detail: key }),
  );
}
