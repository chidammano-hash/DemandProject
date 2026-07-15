/**
 * Minimal non-blocking toast system.
 *
 * Gen-4 roadmap UX-7 P0: replaces `window.alert()` so users can keep
 * working while an error is acknowledged. Sized to match the sonner API
 * so migration to sonner later is a two-line swap.
 *
 * Usage:
 *   // once, at the root of the app:
 *   <Toaster />
 *   // anywhere in the tree:
 *   toast.error("Action failed.");
 *   toast.success("Saved.");
 */

import { useEffect, useState } from "react";

type ToastKind = "info" | "success" | "warning" | "error";

interface Toast {
  id: number;
  kind: ToastKind;
  message: string;
}

type ToastListener = (toasts: Toast[]) => void;

let nextId = 1;
let toastStack: Toast[] = [];
const listeners: Set<ToastListener> = new Set();

const TOAST_DURATION_MS = 4500;
// Cap visible toasts so a request storm (e.g. server unreachable while 12
// queries fire in parallel) cannot fill the screen.
const MAX_VISIBLE = 4;
// Window during which a duplicate (kind, message) is collapsed instead of
// pushing a new toast.
const DEDUPE_WINDOW_MS = 5000;
const recentByKey = new Map<string, { id: number; ts: number; count: number }>();

function notify() {
  for (const listener of listeners) listener([...toastStack]);
}

function push(kind: ToastKind, message: string): number {
  const key = `${kind}::${message}`;
  const now = Date.now();
  const recent = recentByKey.get(key);

  if (recent && now - recent.ts < DEDUPE_WINDOW_MS) {
    // Still on screen — bump count and update message in place.
    recent.count += 1;
    recent.ts = now;
    const stamped = `${message} (x${recent.count})`;
    toastStack = toastStack.map((t) => (t.id === recent.id ? { ...t, message: stamped } : t));
    notify();
    return recent.id;
  }

  const id = nextId++;
  toastStack = [...toastStack, { id, kind, message }];
  // Trim from the top once we exceed the visible cap.
  if (toastStack.length > MAX_VISIBLE) {
    toastStack = toastStack.slice(-MAX_VISIBLE);
  }
  recentByKey.set(key, { id, ts: now, count: 1 });
  notify();
  window.setTimeout(() => {
    dismiss(id);
    const cur = recentByKey.get(key);
    if (cur && cur.id === id) recentByKey.delete(key);
  }, TOAST_DURATION_MS);
  return id;
}

function dismiss(id: number) {
  const before = toastStack.length;
  toastStack = toastStack.filter((t) => t.id !== id);
  if (toastStack.length !== before) notify();
}

export const toast = {
  info: (message: string) => push("info", message),
  success: (message: string) => push("success", message),
  warning: (message: string) => push("warning", message),
  error: (message: string) => push("error", message),
  dismiss,
};

const KIND_STYLES: Record<ToastKind, string> = {
  info: "border-info/50 bg-info/90 text-info-foreground",
  success: "border-success/50 bg-success/90 text-success-foreground",
  warning: "border-warning/50 bg-warning/90 text-warning-foreground",
  error: "border-destructive/50 bg-destructive/90 text-destructive-foreground",
};

export function Toaster() {
  const [toasts, setToasts] = useState<Toast[]>(toastStack);

  useEffect(() => {
    const listener: ToastListener = (next) => setToasts(next);
    listeners.add(listener);
    return () => {
      listeners.delete(listener);
    };
  }, []);

  if (toasts.length === 0) return null;

  return (
    <div
      aria-live="polite"
      role="status"
      className="pointer-events-none fixed bottom-4 right-4 z-[9999] flex flex-col gap-2"
    >
      {toasts.map((t) => (
        <button
          key={t.id}
          type="button"
          onClick={() => dismiss(t.id)}
          className={`pointer-events-auto min-w-[240px] max-w-[420px] rounded-lg border px-4 py-3 text-left text-sm shadow-elevated transition-opacity ${KIND_STYLES[t.kind]}`}
        >
          {t.message}
        </button>
      ))}
    </div>
  );
}

// Test-only exports
export const __test__ = { push, dismiss, get toastStack() { return toastStack; } };
